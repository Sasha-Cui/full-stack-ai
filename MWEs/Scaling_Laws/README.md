# Scaling Laws

This notebook is a conceptual bridge between basic deep learning and LLM systems work. It is useful before you read the serving, evaluation, or fine-tuning folders because it gives you a vocabulary for parameter count, token budget, and compute tradeoffs.

## Files

- `scaling_laws.ipynb`: main notebook.
- `requirements.txt`: lightweight dependency list.
- `*.png`: figures used by the notebook.

## Topics

- Power-law intuition
- Kaplan scaling laws
- Chinchilla-style compute-optimal tradeoffs
- Dataset-size versus parameter-size reasoning
- Practical implications for modern model development

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
jupyter notebook scaling_laws.ipynb
```

## Validation

This notebook was executed locally in a CPU-only Python 3.12 environment.

## Suggested Use

- Read it after `MWEs/pytorch/`.
- Treat it as intuition-building rather than a benchmark suite.
- Use it to sanity-check claims you hear about “bigger models” or “more data” elsewhere.

## References

- [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361)
- [Chinchilla / Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556)
