"""Evaluate a base or fine-tuned model on GSM8K-format VERL data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

def row_to_messages(row: pd.Series) -> tuple[list[dict[str, str]], str]:
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        raw_prompt = prompt[0].get("content", "") if prompt else ""
        return prompt, raw_prompt
    if prompt is not None:
        prompt_text = str(prompt)
        return [{"role": "user", "content": prompt_text}], prompt_text

    question = str(row.get("question", ""))
    return [{"role": "user", "content": question}], question


def row_to_ground_truth(row: pd.Series) -> str:
    reward_model = row.get("reward_model", {})
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"])
    for key in ("ground_truth", "answer", "target"):
        if row.get(key) is not None:
            return str(row[key])
    return ""


def evaluate_gsm8k(
    model_path: str,
    data_path: str,
    n_samples: int = 1,
    max_tokens: int = 512,
    temperature: float = 0.0,
    output_path: str | None = None,
    limit: int | None = None,
    gpu_memory_utilization: float = 0.5,
):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from verl.utils.reward_score.gsm8k import compute_score, extract_solution

    data_file = Path(data_path).expanduser()
    if not data_file.exists():
        raise FileNotFoundError(f"Parquet file not found: {data_file}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    df = pd.read_parquet(data_file)
    if limit is not None:
        df = df.head(limit)
    if df.empty:
        raise ValueError("The evaluation dataset is empty.")

    prompts = []
    raw_prompts = []
    ground_truths = []
    for _, row in df.iterrows():
        messages, raw_prompt = row_to_messages(row)
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)
        raw_prompts.append(raw_prompt)
        ground_truths.append(row_to_ground_truth(row))

    print(f"Loaded {len(prompts)} samples from {data_file}")
    print(f"Sample prompt:\n{raw_prompts[0][:500]}...")
    print(f"Sample ground truth: {ground_truths[0]}")

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=n_samples,
    )
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for index, output in enumerate(outputs):
        for completion in output.outputs:
            results.append(
                {
                    "prompt": raw_prompts[index],
                    "response": completion.text,
                    "ground_truth": ground_truths[index],
                }
            )

    disagreements = []
    for result in tqdm(results, desc="Scoring"):
        response = result["response"]
        ground_truth = result["ground_truth"]

        score_strict = compute_score(response, ground_truth, method="strict")
        score_flexible = compute_score(response, ground_truth, method="flexible")

        result["extracted_strict"] = extract_solution(response, method="strict")
        result["extracted_flexible"] = extract_solution(response, method="flexible")
        result["correct_strict"] = bool(score_strict)
        result["correct_flexible"] = bool(score_flexible)

        if score_strict > score_flexible:
            disagreements.append(
                {
                    "gt": ground_truth,
                    "strict_ans": result["extracted_strict"],
                    "flexible_ans": result["extracted_flexible"],
                    "response_tail": response[-200:] if len(response) > 200 else response,
                }
            )

    total = len(results)
    correct_strict = sum(1 for result in results if result["correct_strict"])
    correct_flexible = sum(1 for result in results if result["correct_flexible"])
    acc_strict = correct_strict / total if total else 0.0
    acc_flexible = correct_flexible / total if total else 0.0

    print(f"\n{'=' * 60}")
    print(f"Model: {model_path}")
    print(f"Total samples: {total}")
    print(
        "  STRICT  (verl training):    "
        f"{correct_strict:4d}/{total}  = {acc_strict * 100:5.2f}%"
    )
    print(
        "  FLEXIBLE (lm-eval-harness): "
        f"{correct_flexible:4d}/{total}  = {acc_flexible * 100:5.2f}%"
    )
    print(f"{'=' * 60}")

    if disagreements:
        print(f"\nFound {len(disagreements)} cases where STRICT ✓ but FLEXIBLE ✗:")
        for index, example in enumerate(disagreements[:3], start=1):
            print(f"\n--- Example {index} ---")
            print(f"Ground truth: {example['gt']}")
            print(f"Strict extracted: {example['strict_ans']}")
            print(f"Flexible extracted: {example['flexible_ans']}")
            print(f"Response tail: ...{example['response_tail']}")

    if output_path:
        output_file = Path(output_path).expanduser()
        with output_file.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result) + "\n")
        print(f"\nResults saved to: {output_file}")

    return {"strict": acc_strict, "flexible": acc_flexible, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model or HuggingFace model ID")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test parquet file")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples per prompt")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file to save results")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of evaluation rows to load.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.5,
        help="Fraction of GPU memory to reserve for vLLM.",
    )

    args = parser.parse_args()
    evaluate_gsm8k(
        args.model_path,
        args.data_path,
        args.n_samples,
        args.max_tokens,
        args.temperature,
        args.output,
        args.limit,
        args.gpu_memory_utilization,
    )
