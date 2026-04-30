# VERL PPO GSM8K

This folder is a container-first walkthrough for PPO fine-tuning with VERL plus a pair of helper scripts for evaluation and qualitative comparison.

## Files

- `evaluate_gsm8k.py`: evaluate a base or trained model on GSM8K-format parquet data.
- `compare_results.py`: compare baseline and fine-tuned generations.
- `ppo_gsm8k.sbatch`: example SLURM submission file.
- `requirements.txt`: helper-script dependencies.

## Install The Helper Script Dependencies

```bash
pip install -r requirements.txt
```

The full PPO workflow itself is still based on the VERL container instructions, not a local pip install.

## Training Workflow

The intended path is:

1. Pull or build the VERL container.
2. Download GSM8K and warm the base model.
3. Launch PPO training inside the container.
4. Merge the saved checkpoint.
5. Run `evaluate_gsm8k.py`.
6. Run `compare_results.py`.

## Validation

- `evaluate_gsm8k.py` and `compare_results.py` compile cleanly.
- The helper scripts now have better path validation and clearer CLI behavior.
- The full VERL PPO workflow was not executed locally because it requires the containerized GPU stack.

## Important Notes

- The correct batch-submission file in this folder is `ppo_gsm8k.sbatch`.
- `evaluate_gsm8k.py` requires a parquet dataset and a vLLM-compatible model path.
- Full training is best treated as an advanced systems exercise, not a beginner entry point.

## References

- [VERL repository](https://github.com/volcengine/verl)
- [Qwen models](https://huggingface.co/Qwen)
