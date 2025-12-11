"""
Compare base model vs trained model results on GSM8K.
Find examples where base fails but trained succeeds.

Usage:
    python compare_results.py --base base_outputs.jsonl --trained trained_outputs.jsonl
    python compare_results.py --base base_outputs.jsonl --trained trained_outputs.jsonl --method strict
"""
import argparse
import pandas as pd


def main(base_path: str, trained_path: str, n_examples: int = 3, method: str = "flexible"):
    # Load results
    base = pd.read_json(base_path, lines=True)
    trained = pd.read_json(trained_path, lines=True)
    
    correct_col = f'correct_{method}'
    extracted_col = f'extracted_{method}'
    
    print(f"Using method: {method}")
    print(f"Base model:    {base[correct_col].sum()}/{len(base)} = {base[correct_col].mean()*100:.2f}%")
    print(f"Trained model: {trained[correct_col].sum()}/{len(trained)} = {trained[correct_col].mean()*100:.2f}%")
    
    # Find cases where base fails but trained succeeds
    improvements = []
    for i in range(len(base)):
        if not base.iloc[i][correct_col] and trained.iloc[i][correct_col]:
            improvements.append(i)
    
    print(f"\nFound {len(improvements)} cases where BASE ✗ but TRAINED ✓")
    
    # Print examples
    for idx in improvements[:n_examples]:
        b = base.iloc[idx]
        t = trained.iloc[idx]
        
        print(f"\n{'='*70}")
        print(f"QUESTION:")
        print(b['prompt'])
        print(f"\nGROUND TRUTH: {b['ground_truth']}")
        print(f"\n--- BASE MODEL (incorrect) ---")
        print(f"Extracted: {b[extracted_col]}")
        print(f"Response:\n{b['response']}")
        print(f"\n--- TRAINED MODEL (correct) ---")
        print(f"Extracted: {t[extracted_col]}")
        print(f"Response:\n{t['response']}")
        print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="base_outputs.jsonl", help="Base model results")
    parser.add_argument("--trained", type=str, default="trained_outputs.jsonl", help="Trained model results")
    parser.add_argument("-n", type=int, default=3, help="Number of examples to show")
    parser.add_argument("--method", type=str, default="flexible", choices=["strict", "flexible"],
                        help="Scoring method to use (default: flexible)")
    
    args = parser.parse_args()
    main(args.base, args.trained, args.n, args.method)

