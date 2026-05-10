# Full-Stack AI

Hands-on material for learning modern AI systems from fundamentals through inference, evaluation, fine-tuning, serving, distributed training, and early agent/robotics topics.

This repo is intentionally mixed-format: some parts are polished, CPU-safe notebooks; some are API-key-dependent application walkthroughs; some are cluster- or container-oriented systems examples. The goal is not to pretend every folder is equally turnkey. The goal is to make it clear what each folder teaches, what it requires, and how a reader should progress through it.

## Who This Repo Is For

- readers who want to move from PyTorch basics into LLM systems work
- practitioners who understand model APIs but want more systems context
- working-group participants who want a structured self-study path
- students who need honest guidance on what is locally runnable versus what is only reviewed or partially validated

## Start Here

If you want the smoothest local path, use the CPU-friendly environment and validate the notebooks that were intentionally kept portable:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt
python scripts/validate_examples.py --execute-notebooks --deep
```

That command path checks Python helper scripts, notebook metadata portability, and executes the CPU-safe notebooks that were validated locally.

## Environment Tiers

| Tier | What it covers | Typical folders |
| --- | --- | --- |
| CPU-safe local | Notebooks and examples validated without GPUs or external services | `MWEs/pytorch/`, `MWEs/Scaling_Laws/`, `MWEs/agentic_rl_workshop.ipynb`, `MWEs/LoRA_tutorials/` |
| API-key + network | LLM application workflows that depend on providers and fast-moving APIs | `MWEs/Inference/` |
| Linux + NVIDIA GPU | Serving and distributed training examples | `MWEs/vllm+deepspeed/`, `MWEs/ray_train/` |
| Cluster / container heavy | More realistic systems workflows that are not beginner-local | `MWEs/verl/`, some robotics ecosystem exploration |

## Recommended Learning Paths

### 1. Foundations-first path

1. `MWEs/pytorch/`
2. `MWEs/Scaling_Laws/`
3. `MWEs/Inference/`
4. `MWEs/LLM_Evaluation_Alignment/`
5. `MWEs/LoRA_tutorials/`
6. `MWEs/vllm+deepspeed/`
7. `MWEs/ray_train/`
8. `MWEs/verl/`
9. `MWEs/agentic_rl_workshop.ipynb`
10. `MWEs/Robotics/`

### 2. Application / agent path

1. `MWEs/pytorch/`
2. `MWEs/Inference/`
3. `MWEs/agentic_rl_workshop.ipynb`
4. `MWEs/LLM_Evaluation_Alignment/`
5. `MWEs/LoRA_tutorials/`
6. `MWEs/Robotics/`

### 3. Systems path

1. `MWEs/pytorch/`
2. `MWEs/Scaling_Laws/`
3. `MWEs/vllm+deepspeed/`
4. `MWEs/ray_train/`
5. `MWEs/verl/`

## Module Index

See [MWEs/README.md](MWEs/README.md) for the detailed module-by-module index. At a high level:

- `MWEs/pytorch/`: PyTorch fundamentals and training mechanics
- `MWEs/Scaling_Laws/`: scaling-law intuition and compute/data tradeoffs
- `MWEs/Inference/`: API inference, tool use, MCP, prompting, and GEPA
- `MWEs/LLM_Evaluation_Alignment/`: evaluation framing and alignment basics
- `MWEs/LoRA_tutorials/`: parameter-efficient fine-tuning in practice
- `MWEs/vllm+deepspeed/`: serving and training-systems concepts
- `MWEs/ray_train/`: distributed-training scripts
- `MWEs/verl/`: PPO workflow with VERL helper scripts
- `MWEs/Robotics/`: survey notebook for robotics / VLA frameworks
- `MWEs/agentic_rl_workshop.ipynb`: workshop-style bridge from application LLMs into agentic and RL-flavored ideas

## Validation Status

| Area | Local status | Notes |
| --- | --- | --- |
| `MWEs/pytorch/` | Executed locally | CPU-safe. |
| `MWEs/Scaling_Laws/` | Executed locally | CPU-safe. |
| `MWEs/agentic_rl_workshop.ipynb` | Executed locally | CPU-safe. |
| `MWEs/LoRA_tutorials/` | Executed locally | CPU-safe but heavier; downloads PBMC3k data. |
| `MWEs/Inference/` | Helper files compile | Notebook requires API keys, network access, and current provider docs. |
| `MWEs/LLM_Evaluation_Alignment/` | Reviewed | Presentation-heavy; real `lm-eval` execution depends on your chosen backend. |
| `MWEs/ray_train/train_cifar.py` | Smoke-testable | `--smoke-test --cpu` is the easiest path. |
| `MWEs/ray_train/model_par.py` | Compiles | Linux + CUDA + multiple GPUs required. |
| `MWEs/ray_train/zero_deepspeed.py` | Compiles | Linux + CUDA + DeepSpeed required. |
| `MWEs/vllm+deepspeed/` | Reviewed | Practical cells need Linux and NVIDIA GPUs. |
| `MWEs/verl/` | Helper scripts compile | Full PPO workflow needs the VERL container stack. |
| `MWEs/Robotics/` | Reviewed | Framework installs are intentionally optional and heavyweight. |

## Working With The Repo

### Validate what you can first

```bash
python scripts/validate_examples.py
python scripts/validate_examples.py --execute-notebooks
python scripts/validate_examples.py --execute-notebooks --deep
```

### Install per-folder requirements when leaving the CPU-safe path

Many subfolders carry their own `requirements.txt`. Use the root `requirements-cpu.txt` for the validated local path, then install extra folder-specific dependencies as needed rather than treating the entire repo as one universal environment.

### Check official docs for moving targets

Inference APIs, provider pricing, model availability, vLLM install details, DeepSpeed flags, robotics frameworks, and RL tooling all change quickly. The repo tries to teach patterns and concepts without pretending every versioned instruction will stay current forever.

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
├── scripts/
│   └── validate_examples.py
├── overleaf/
├── slides/
└── requirements-cpu.txt
```

## Slides And Long-Form Material

- `slides/` contains slide exports used alongside the working group
- `overleaf/` contains the tutorial draft and section files

These are best treated as supporting reading rather than the primary hands-on path.

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

- Sponsor: [Requesty](https://www.requesty.ai) for API credits
- Faculty adviser: Prof. John Lafferty, Yale University, Wu Tsai Institute
- Faculty adviser: Prof. Jas Sekhon, Google DeepMind
- Technical support adviser: Ping Luo, Yale University, Wu Tsai Institute

The group also had access to an experimental GPU cluster hosted in The Matrix at the Wu Tsai Institute.

## Current Organizers

The current run of the event is being organized by:

- [Dongyu Gong](https://daniel-gong.github.io/) - `dongyu.gong@yale.edu`
- [Xiaowei Ou](https://www.linkedin.com/in/xiaowei-ou-a7a7951ba/) - `xiaowei.ou@yale.edu`

## Contributing

When you update materials in this repo:

- be explicit about whether something was executed, smoke-tested, or only reviewed
- avoid hard-coding brittle provider pricing or temporary install commands when official docs are better
- prefer folder-level README updates that explain prerequisites, expected outcomes, and validation status

## License

MIT. See [LICENSE](LICENSE).
