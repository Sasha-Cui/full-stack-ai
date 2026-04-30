# Full-Stack AI

Hands-on materials for learning modern AI systems end to end: PyTorch fundamentals, LLM inference, evaluation, LoRA, distributed training, serving, VERL, and introductory agent/robotics topics.

This repository started as a Yale working-group resource. It has been cleaned up so that the runnable pieces are easier to validate, the folder-level instructions are more honest, and the learning path is clearer for someone studying LLMs and AI agents on their own.

## What This Repo Contains

- Notebook-first tutorials for core concepts and visual explanations.
- Minimal Python scripts for distributed-training and evaluation workflows.
- Slide decks and an Overleaf tutorial draft for longer-form reading.
- A small validation script so you can check the portable examples quickly.

## Start Here

If you want the smoothest local experience, start with the CPU-safe environment and the locally validated notebooks.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt
python scripts/validate_examples.py --execute-notebooks --deep
```

That validates:

- `MWEs/pytorch/pytorch_tutorial.ipynb`
- `MWEs/Scaling_Laws/scaling_laws.ipynb`
- `MWEs/agentic_rl_workshop.ipynb`
- `MWEs/LoRA_tutorials/lora_single_cell_demo_clean.ipynb`
- `MWEs/LoRA_tutorials/pytorch_lightning_tutorial.ipynb`

## Recommended Learning Paths

### 1. Foundations First

1. `MWEs/pytorch/`
2. `MWEs/Scaling_Laws/`
3. `MWEs/Inference/`
4. `MWEs/LLM_Evaluation_Alignment/`
5. `MWEs/LoRA_tutorials/`
6. `MWEs/vllm+deepspeed/`
7. `MWEs/ray_train/`
8. `MWEs/verl/`
9. `MWEs/agentic_rl_workshop.ipynb`

### 2. Agent-Focused Path

1. `MWEs/Inference/`
2. `MWEs/agentic_rl_workshop.ipynb`
3. `MWEs/LLM_Evaluation_Alignment/`
4. `MWEs/vllm+deepspeed/`
5. `MWEs/Robotics/`

## Validation Status

| Area | Local status | Notes |
| --- | --- | --- |
| `MWEs/pytorch/` | Executed locally | CPU-safe. |
| `MWEs/Scaling_Laws/` | Executed locally | CPU-safe. |
| `MWEs/agentic_rl_workshop.ipynb` | Executed locally | CPU-safe. |
| `MWEs/LoRA_tutorials/` | Executed locally | CPU-safe but heavier; downloads PBMC3k data. |
| `MWEs/Inference/` | Support files compiled | Notebook needs API keys and network access. |
| `MWEs/LLM_Evaluation_Alignment/` | Reviewed + dependency file added | Full `lm-eval` runs depend on chosen model/backend. |
| `MWEs/ray_train/train_cifar.py` | Refactored for smoke testing | `--smoke-test --cpu` is the easiest check. |
| `MWEs/ray_train/model_par.py` | Compiles | Requires Linux + CUDA + multiple GPUs. |
| `MWEs/ray_train/zero_deepspeed.py` | Compiles | Requires Linux + CUDA + DeepSpeed. |
| `MWEs/vllm+deepspeed/` | Reviewed | Practical serving cells need Linux + NVIDIA GPU. |
| `MWEs/verl/` | Helper scripts compiled | Full PPO workflow requires the VERL container stack. |
| `MWEs/Robotics/` | Reviewed | Framework installs are intentionally optional and heavyweight. |

## Repository Map

```text
full-stack-ai/
├── MWEs/
│   ├── pytorch/
│   ├── Scaling_Laws/
│   ├── Inference/
│   ├── LLM_Evaluation_Alignment/
│   ├── LoRA_tutorials/
│   ├── vllm+deepspeed/
│   ├── ray_train/
│   ├── verl/
│   ├── Robotics/
│   └── agentic_rl_workshop.ipynb
├── overleaf/
├── slides/
├── requirements-cpu.txt
└── scripts/validate_examples.py
```

## Module Guide

### `MWEs/pytorch/`

PyTorch fundamentals: tensors, autograd, custom functions, debugging, AMP, and practical training utilities.

### `MWEs/Scaling_Laws/`

Scaling-law intuition for model size, dataset size, compute, and Chinchilla-style tradeoffs.

### `MWEs/Inference/`

API-based inference, tool use, MCP, prompt engineering, and GEPA prompt optimization. This folder is the best starting point for agent-oriented LLM workflows.

### `MWEs/LLM_Evaluation_Alignment/`

Evaluation principles, benchmark design, `lm-eval`, and alignment framing.

### `MWEs/LoRA_tutorials/`

Parameter-efficient fine-tuning with a concrete single-cell biology example and a companion PyTorch Lightning notebook.

### `MWEs/vllm+deepspeed/`

Serving and systems notebooks for PagedAttention, vLLM, and ZeRO-style training concepts.

### `MWEs/ray_train/`

Minimal scripts for data parallelism, ZeRO-3 training, and pipeline parallelism. Best treated as systems examples rather than beginner notebooks.

### `MWEs/verl/`

A container-first walkthrough for PPO on GSM8K with VERL, plus helper scripts for evaluation and result comparison.

### `MWEs/Robotics/`

Survey-style notebook for VLA/robotics frameworks such as Robosuite, RoboVerse, MetaSim, and LeRobot.

## Notes On Moving Targets

Model pricing, API availability, provider routing, and some framework install instructions change quickly. Where possible, this repo now points to official docs instead of freezing brittle numbers in place. If you are using the inference or serving materials for real work, check the provider documentation before assuming a model, price, or flag is still current.

## Authors And Contributors

Current named authors and senior contributors for the project:

- Sasha Cui
- Quan Le
- Alexander Mader
- Ping Luo
- John Lafferty

Module-level contributor attribution for the MWEs is summarized in [CONTRIBUTORS.md](CONTRIBUTORS.md).

## Sponsors And Advisers

The working group website credits the following organizations and advisers:

- **Sponsor**: [Requesty](https://www.requesty.ai), for API credits.
- **Faculty Adviser**: Prof. John Lafferty, Yale University, Wu Tsai Institute.
- **Faculty Adviser**: Prof. Jas Sekhon, Google DeepMind.
- **Technical Support Adviser**: Ping Luo, Yale University, Wu Tsai Institute.

The website also notes that the working group had access to an experimental GPU cluster hosted in The Matrix at the Wu Tsai Institute.

## Current Organizers

The current run of the event is being organized by:

- [Dongyu Gong](https://daniel-gong.github.io/) — `dongyu.gong@yale.edu`
- [Xiaowei Ou](https://www.linkedin.com/in/xiaowei-ou-a7a7951ba/) — `xiaowei.ou@yale.edu`

If you want more information about the current run, contact them directly at their Yale email addresses.

## Slides And Paper Draft

- `slides/` contains accompanying PDFs.
- `overleaf/` contains the tutorial paper draft and section files.

## Contributing

Before sending changes, run what you can locally:

```bash
python scripts/validate_examples.py
python scripts/validate_examples.py --execute-notebooks
python scripts/validate_examples.py --execute-notebooks --deep
```

For GPU-only folders, be explicit in documentation about what was reviewed versus what was executed.

## License

MIT. See [LICENSE](LICENSE).
