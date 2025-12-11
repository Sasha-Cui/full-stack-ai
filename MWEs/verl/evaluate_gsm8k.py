"""
Simple GSM8K evaluation script for verl trained models.
Usage:
    python evaluate_gsm8k.py --model_path Qwen/Qwen2.5-0.5B-Instruct --data_path ~/data/gsm8k/test.parquet
    python evaluate_gsm8k.py --model_path ./checkpoints/verl_tutorial/qwen25_ppo_gsm8k/global_step_435/actor/huggingface --data_path ~/data/gsm8k/test.parquet
    
    # Save results to file:
    python evaluate_gsm8k.py --model_path ... --data_path ... --output results.jsonl
"""
import argparse
import json
import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import numpy as np

# Import the GSM8K scoring function from verl
from verl.utils.reward_score.gsm8k import compute_score, extract_solution




def evaluate_gsm8k(model_path: str, data_path: str, n_samples: int = 1, max_tokens: int = 512, temperature: float = 0.0, output_path: str = None):
    """Evaluate a model on GSM8K with both strict and flexible scoring.
    
    Args:
        output_path: If provided, save results to this JSONL file
    """
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(model=model_path, trust_remote_code=True, gpu_memory_utilization=0.5)
    
    # Load test data
    df = pd.read_parquet(data_path)
    
    # Extract prompts and ground truths
    prompts = []
    prompts_raw = []  # Original question text for saving
    ground_truths = []
    
    for _, row in df.iterrows():
        # In verl GSM8K format, 'prompt' is already a list of message dicts
        # e.g., [{"role": "user", "content": "..."}]
        if 'prompt' in row:
            messages = row['prompt']
            # If it's already a list of messages, use directly
            if isinstance(messages, list):
                prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                raw_prompt = messages[0].get('content', '') if messages else ''
            else:
                # If it's a string, wrap it
                prompt_str = tokenizer.apply_chat_template(
                    [{"role": "user", "content": str(messages)}], 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                raw_prompt = str(messages)
        else:
            # Fallback for other formats
            question = row.get('question', '')
            prompt_str = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}], 
                tokenize=False, 
                add_generation_prompt=True
            )
            raw_prompt = question
        
        prompts.append(prompt_str)
        prompts_raw.append(raw_prompt)
        
        # Ground truth is in reward_model column
        rm_data = row.get('reward_model', {})
        if isinstance(rm_data, dict):
            gt = rm_data.get('ground_truth', '')
        else:
            gt = ''
        ground_truths.append(gt)
    
    print(f"Loaded {len(prompts)} samples")
    print(f"Sample prompt:\n{prompts[0][:500]}...")
    print(f"Sample ground truth: {ground_truths[0]}")
    
    # Generate responses
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=n_samples,
    )
    
    outputs = llm.generate(prompts, sampling_params)
    
    # Collect all responses with their metadata
    results = []
    for i, output in enumerate(outputs):
        gt = ground_truths[i]
        raw_prompt = prompts_raw[i]
        for completion in output.outputs:
            results.append({
                "prompt": raw_prompt,
                "response": completion.text,
                "ground_truth": gt,
            })
    
    # Compute scores with BOTH methods
    disagreements = []  # Cases where strict succeeds but flexible fails
    
    for r in tqdm(results, desc="Scoring"):
        response = r["response"]
        gt = r["ground_truth"]
        
        score_strict = compute_score(response, gt, method="strict")
        score_flexible = compute_score(response, gt, method="flexible")
        
        r["extracted_strict"] = extract_solution(response, method="strict")
        r["extracted_flexible"] = extract_solution(response, method="flexible")
        r["correct_strict"] = bool(score_strict)
        r["correct_flexible"] = bool(score_flexible)
        
        # Track disagreements
        if score_strict > score_flexible:
            disagreements.append({
                "gt": gt,
                "strict_ans": r["extracted_strict"],
                "flexible_ans": r["extracted_flexible"],
                "response_tail": response[-200:] if len(response) > 200 else response
            })
    
    total = len(results)
    correct_strict = sum(1 for r in results if r["correct_strict"])
    correct_flexible = sum(1 for r in results if r["correct_flexible"])
    acc_strict = correct_strict / total if total > 0 else 0
    acc_flexible = correct_flexible / total if total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Model: {model_path}")
    print(f"Total samples: {total}")
    print(f"")
    print(f"  STRICT  (verl training):    {int(correct_strict):4d}/{total}  = {acc_strict*100:5.2f}%")
    print(f"  FLEXIBLE (lm-eval-harness): {int(correct_flexible):4d}/{total}  = {acc_flexible*100:5.2f}%")
    print(f"{'='*60}")
    
    # Show examples where strict succeeded but flexible failed
    if disagreements:
        print(f"\nFound {len(disagreements)} cases where STRICT ✓ but FLEXIBLE ✗:")
        for i, d in enumerate(disagreements[:3]):  # Show first 3
            print(f"\n--- Example {i+1} ---")
            print(f"Ground truth: {d['gt']}")
            print(f"Strict extracted: {d['strict_ans']}")
            print(f"Flexible extracted: {d['flexible_ans']}")
            print(f"Response tail: ...{d['response_tail']}")
    
    # Save results to file if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')
        print(f"\nResults saved to: {output_path}")
        print(f"  Load with: pd.read_json('{output_path}', lines=True)")
    
    return {"strict": acc_strict, "flexible": acc_flexible, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model or HuggingFace model ID")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test parquet file")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples per prompt")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file to save results")
    
    args = parser.parse_args()
    evaluate_gsm8k(args.model_path, args.data_path, args.n_samples, args.max_tokens, args.temperature, args.output)