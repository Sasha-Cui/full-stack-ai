from __future__ import annotations

import argparse
import py_compile
import subprocess
import sys
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
PYTHON_FILES = [
    ROOT / "MWEs" / "Inference" / "GEPA_utils.py",
    ROOT / "MWEs" / "Inference" / "tools.py",
    ROOT / "MWEs" / "ray_train" / "model_par.py",
    ROOT / "MWEs" / "ray_train" / "train_cifar.py",
    ROOT / "MWEs" / "ray_train" / "zero_deepspeed.py",
    ROOT / "MWEs" / "verl" / "compare_results.py",
    ROOT / "MWEs" / "verl" / "evaluate_gsm8k.py",
]
DEFAULT_NOTEBOOKS = [
    ROOT / "MWEs" / "pytorch" / "pytorch_tutorial.ipynb",
    ROOT / "MWEs" / "Scaling_Laws" / "scaling_laws.ipynb",
    ROOT / "MWEs" / "agentic_rl_workshop.ipynb",
]
DEEP_NOTEBOOKS = [
    ROOT / "MWEs" / "LoRA_tutorials" / "lora_single_cell_demo_clean.ipynb",
    ROOT / "MWEs" / "LoRA_tutorials" / "pytorch_lightning_tutorial.ipynb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate full-stack-ai examples.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for notebook execution.",
    )
    parser.add_argument(
        "--execute-notebooks",
        action="store_true",
        help="Execute the default CPU-safe notebooks.",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Also execute the heavier LoRA notebooks.",
    )
    return parser.parse_args()


def compile_python_files() -> list[str]:
    errors: list[str] = []
    for path in PYTHON_FILES:
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(f"Python compile failed for {path}: {exc.msg}")
    return errors


def check_notebook_metadata() -> list[str]:
    errors: list[str] = []
    for path in ROOT.glob("MWEs/**/*.ipynb"):
        with path.open("r", encoding="utf-8") as handle:
            notebook = nbformat.read(handle, as_version=4)
        kernelspec = notebook.metadata.get("kernelspec", {})
        kernel_name = kernelspec.get("name")
        if kernel_name not in (None, "python3"):
            errors.append(
                f"{path} uses non-portable kernelspec '{kernel_name}'. "
                "Use 'python3' for repo portability."
            )
    return errors


def execute_notebooks(python_executable: str, notebooks: list[Path]) -> list[str]:
    errors: list[str] = []
    output_dir = ROOT / ".validation"
    output_dir.mkdir(exist_ok=True)

    for notebook in notebooks:
        output_path = output_dir / f"{notebook.stem}.executed.ipynb"
        cmd = [
            python_executable,
            "-m",
            "nbconvert",
            "--execute",
            "--to",
            "notebook",
            "--ExecutePreprocessor.kernel_name=python3",
            "--output",
            str(output_path),
            str(notebook),
        ]
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip()
            errors.append(f"Notebook execution failed for {notebook}:\n{stderr}")
    return errors


def main() -> None:
    args = parse_args()
    errors = []
    errors.extend(compile_python_files())
    errors.extend(check_notebook_metadata())

    if args.execute_notebooks:
        notebook_list = list(DEFAULT_NOTEBOOKS)
        if args.deep:
            notebook_list.extend(DEEP_NOTEBOOKS)
        errors.extend(execute_notebooks(args.python, notebook_list))

    if errors:
        print("Validation failed:\n")
        for error in errors:
            print(f"- {error}\n")
        raise SystemExit(1)

    print("Validation passed.")


if __name__ == "__main__":
    main()
