"""Compare base-model and trained-model GSM8K generations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_results(path: str) -> pd.DataFrame:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Results file not found: {resolved}")
    return pd.read_json(resolved, lines=True)


def main(
    base_path: str,
    trained_path: str,
    n_examples: int = 3,
    method: str = "flexible",
):
    base = load_results(base_path)
    trained = load_results(trained_path)

    correct_col = f"correct_{method}"
    extracted_col = f"extracted_{method}"
    required_columns = {"prompt", "response", "ground_truth", correct_col, extracted_col}

    if len(base) != len(trained):
        raise ValueError(
            "Base and trained result files must contain the same number of rows."
        )
    missing_base = required_columns.difference(base.columns)
    missing_trained = required_columns.difference(trained.columns)
    if missing_base:
        raise ValueError(f"Base results missing columns: {sorted(missing_base)}")
    if missing_trained:
        raise ValueError(f"Trained results missing columns: {sorted(missing_trained)}")

    print(f"Using method: {method}")
    print(
        f"Base model:    {int(base[correct_col].sum())}/{len(base)} "
        f"= {base[correct_col].mean() * 100:.2f}%"
    )
    print(
        f"Trained model: {int(trained[correct_col].sum())}/{len(trained)} "
        f"= {trained[correct_col].mean() * 100:.2f}%"
    )

    improvement_mask = (~base[correct_col].astype(bool)) & trained[correct_col].astype(bool)
    improvements = trained.index[improvement_mask].tolist()

    print(f"\nFound {len(improvements)} cases where BASE ✗ but TRAINED ✓")

    for idx in improvements[:n_examples]:
        b = base.iloc[idx]
        t = trained.iloc[idx]

        print(f"\n{'=' * 70}")
        print(f"QUESTION:")
        print(b["prompt"])
        print(f"\nGROUND TRUTH: {b['ground_truth']}")
        print(f"\n--- BASE MODEL (incorrect) ---")
        print(f"Extracted: {b[extracted_col]}")
        print(f"Response:\n{b['response']}")
        print(f"\n--- TRAINED MODEL (correct) ---")
        print(f"Extracted: {t[extracted_col]}")
        print(f"Response:\n{t['response']}")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base", type=str, default="base_outputs.jsonl", help="Base model results"
    )
    parser.add_argument(
        "--trained",
        type=str,
        default="trained_outputs.jsonl",
        help="Trained model results",
    )
    parser.add_argument("-n", type=int, default=3, help="Number of examples to show")
    parser.add_argument(
        "--method",
        type=str,
        default="flexible",
        choices=["strict", "flexible"],
        help="Scoring method to use.",
    )

    args = parser.parse_args()
    main(args.base, args.trained, args.n, args.method)
