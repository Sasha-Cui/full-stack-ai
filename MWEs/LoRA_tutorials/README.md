# LoRA Tutorials

This folder turns LoRA from an abstract PEFT idea into two concrete workflows:

- a from-scratch single-cell biology notebook
- a PyTorch Lightning rewrite of a similar training pipeline

## Files

- `lora_single_cell_demo_clean.ipynb`: detailed LoRA notebook.
- `pytorch_lightning_tutorial.ipynb`: Lightning comparison notebook.
- `requirements.txt`: dependencies for both notebooks.

## What You Learn

- Why LoRA can match full fine-tuning
- When low-rank adaptation works well and when it does not
- Why layer coverage matters
- How to train on a real tabular/biological dataset instead of toy text prompts
- How PyTorch Lightning changes the training-loop ergonomics

## Install

```bash
pip install -r requirements.txt
```

The notebooks download the PBMC3k single-cell dataset on first run.

## Run

```bash
jupyter notebook lora_single_cell_demo_clean.ipynb
jupyter notebook pytorch_lightning_tutorial.ipynb
```

## Validation

Both notebooks were executed locally in a CPU-only Python 3.12 environment after normalizing the notebook kernel metadata and fixing sparse-matrix preprocessing for current `scanpy` / `scikit-learn` behavior.

## Runtime Notes

- CPU runs are fine, but slower.
- The notebooks are heavier than the PyTorch and scaling-law materials because they preprocess a real dataset.
- If you only want the main PEFT lesson, start with `lora_single_cell_demo_clean.ipynb`.

## Suggested Order

1. Run `lora_single_cell_demo_clean.ipynb`.
2. Compare the architectural choices and results.
3. Then open `pytorch_lightning_tutorial.ipynb` to see how the same workflow looks with more framework structure.

## References

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [Lightning docs](https://lightning.ai/docs/pytorch/stable/)
- [Scanpy docs](https://scanpy.readthedocs.io/)
