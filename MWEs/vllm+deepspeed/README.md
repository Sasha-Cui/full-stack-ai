# vLLM And DeepSpeed

This folder mixes conceptual systems notebooks with a small amount of practical setup advice.

## Files

- `vllm_sections_1_4.ipynb`: PagedAttention and serving-oriented material.
- `deepspeed_tutorial_sections_1_4.ipynb`: ZeRO and training-systems material.

## Topics

- KV cache pressure
- PagedAttention
- Request scheduling
- vLLM runtime concepts
- ZeRO stages
- Memory sharding and distributed-training intuition

## Install

The practical cells are intended for Linux + NVIDIA GPU environments.

At minimum:

```bash
pip install torch torchvision
pip install vllm transformers accelerate
pip install jupyter ipykernel matplotlib numpy
```

Add `deepspeed` only if you plan to explore the DeepSpeed-specific workflow on a supported system.

## Run

```bash
jupyter notebook vllm_sections_1_4.ipynb
jupyter notebook deepspeed_tutorial_sections_1_4.ipynb
```

## Validation

- Notebook metadata was normalized so the notebooks are easier to open in a standard `python3` kernel.
- These notebooks were reviewed, but not executed end to end locally because the practical sections need Linux, CUDA, and NVIDIA GPUs.

## Suggested Use

- Read the conceptual sections even if you cannot run vLLM locally.
- Use these notebooks together with `MWEs/ray_train/` if you are studying systems scaling.
- Check the official vLLM and DeepSpeed docs before copying exact install flags into a fresh environment.

## References

- [vLLM docs](https://docs.vllm.ai/)
- [DeepSpeed docs](https://www.deepspeed.ai/)
- [PagedAttention paper](https://arxiv.org/abs/2309.06180)
