# LLM Evaluation And Alignment

This folder is about measuring models well, not just building them.

## Files

- `llm_evaluation_presentation.ipynb`: slide-style notebook.
- `requirements.txt`: dependency list.
- `figures/`: images used by the notebook.
- `results/`: example output artifacts.

## Topics

- Benchmark design
- Automated versus human evaluation
- `lm-eval`
- RLHF / DPO / alignment framing
- Common failure modes in evaluation

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
jupyter notebook llm_evaluation_presentation.ipynb
```

## Validation

This notebook is mostly presentation content. The notebook metadata was normalized and the folder now includes an explicit requirements file, but full `lm-eval` runs still depend on which local model or backend you choose.

## Suggested Use

- Read this after the inference notebook.
- Use it to pressure-test any “improvement” claim you get from prompting, LoRA, or RL.
- Treat the notebook as an overview and then move into official `lm-eval` docs when you want production-grade benchmarking.

## References

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [RLHF paper](https://arxiv.org/abs/2203.02155)
- [DPO paper](https://arxiv.org/abs/2305.18290)
