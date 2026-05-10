# MWEs

This directory holds the core learning materials for the repository. The modules are not all at the same level of portability or difficulty, so this index is meant to answer four questions quickly:

- what does this folder teach
- who should start there
- what environment does it need
- how much of it was actually validated locally

## At A Glance

| Path | Best for | Format | Environment | Local status |
| --- | --- | --- | --- | --- |
| `pytorch/` | PyTorch basics and training mechanics | Notebook | CPU-safe | Executed |
| `Scaling_Laws/` | Compute/data/model-size intuition | Notebook | CPU-safe | Executed |
| `Inference/` | LLM apps, tool use, MCP, prompting | Notebook + helper files | API keys + network | Helper files compiled |
| `LLM_Evaluation_Alignment/` | Evaluation framing and alignment basics | Notebook | Local + optional model backend | Reviewed |
| `LoRA_tutorials/` | Practical PEFT and training loops | Two notebooks | CPU-safe but heavier | Executed |
| `vllm+deepspeed/` | Serving and training-systems concepts | Notebooks | Linux + NVIDIA GPU | Reviewed |
| `ray_train/` | Distributed-training examples | Python scripts | CPU smoke test for one script; GPU for the rest | Smoke-testable / compiled |
| `verl/` | PPO workflow and evaluation helpers | Scripts + batch file | Containerized GPU stack | Helper files compiled |
| `Robotics/` | Robotics / VLA survey | Notebook + PDF | Optional heavyweight framework installs | Reviewed |
| `agentic_rl_workshop.ipynb` | Agent/RL-oriented workshop path | Notebook | CPU-safe | Executed |

## Where To Start

### If you are new to the repo

Start in this order:

1. `pytorch/`
2. `Scaling_Laws/`
3. `Inference/`
4. `LLM_Evaluation_Alignment/`

That path gives you the minimum conceptual stack needed before fine-tuning, serving, or distributed systems material starts making sense.

### If you want the most runnable path

Start with the CPU-safe set:

- `pytorch/`
- `Scaling_Laws/`
- `agentic_rl_workshop.ipynb`
- `LoRA_tutorials/`

These are the best folders to use with the root `requirements-cpu.txt` and `scripts/validate_examples.py`.

### If you care most about application-layer LLM work

Prioritize:

- `Inference/`
- `LLM_Evaluation_Alignment/`
- `agentic_rl_workshop.ipynb`
- `LoRA_tutorials/`

### If you care most about systems work

Prioritize:

- `vllm+deepspeed/`
- `ray_train/`
- `verl/`

Read these after `pytorch/` and `Scaling_Laws/`, not before.

## Module Notes

### `pytorch/`

Best first stop for readers who want to understand tensors, autograd, training loops, debugging, and memory behavior before touching LLM-specific tooling.

### `Scaling_Laws/`

Conceptual notebook for reasoning about parameter count, dataset size, compute budget, and why "bigger" is not the only relevant axis.

### `Inference/`

Most directly relevant module for people building LLM products. Covers API inference, tool calling, MCP ideas, prompt engineering, and GEPA-style prompt optimization. This one depends on live APIs and should be read with provider docs open.

### `LLM_Evaluation_Alignment/`

Good counterweight to the application material. Use it to think about how you would measure model improvements instead of just shipping the first prompt that looks better on a handful of examples.

### `LoRA_tutorials/`

Concrete PEFT workflow using a non-toy dataset. Useful bridge from conceptual fine-tuning discussion into an actual training notebook.

### `agentic_rl_workshop.ipynb`

Workshop-style notebook that sits between application prompting material and more advanced RL or agent-system discussion. Good follow-on once the inference basics feel comfortable.

### `vllm+deepspeed/`

Primarily systems intuition: KV cache pressure, serving behavior, PagedAttention, and ZeRO-style training concepts. Readable without a GPU, but only partly runnable without one.

### `ray_train/`

Contains scripts rather than teaching notebooks. `train_cifar.py --smoke-test --cpu` is the only deliberately lightweight validation path. The rest are examples for readers who already know why they want Ray or DeepSpeed.

### `verl/`

Advanced path for PPO / RLHF-style infrastructure. The included Python scripts are helper utilities; the full training workflow assumes the VERL container stack rather than a simple local pip install.

### `Robotics/`

Survey-style bridge into embodied AI and VLA frameworks. Best treated as a map of the ecosystem, not as a single reproducible environment.

## Practical Guidance

- Install the folder-specific `requirements.txt` when a folder provides one.
- Do not assume the whole directory shares one environment beyond the CPU-safe root path.
- For API, serving, RL, and robotics modules, check the official docs before copying version-sensitive commands into a fresh environment.
- When in doubt, prefer the validated CPU-safe materials first and treat the systems folders as second-pass study.
