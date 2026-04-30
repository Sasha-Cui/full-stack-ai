# Distributed Training With Ray

This folder contains three minimal systems scripts rather than beginner notebooks.

## Files

- `train_cifar.py`: data-parallel training example with a smoke-test path.
- `zero_deepspeed.py`: ZeRO-3 example for Ray + DeepSpeed.
- `model_par.py`: pipeline-parallel example.
- `requirements.txt`: broad dependency list for this folder.

## Install

```bash
pip install -r requirements.txt
```

You still need a PyTorch install that matches your CUDA runtime if you plan to use GPUs.

## Easiest Local Check

```bash
python train_cifar.py --smoke-test --cpu
```

That path uses `FakeData`, a small CNN, and one Ray worker so you can validate the script without a GPU or a CIFAR download.

## What Each Script Is For

### `train_cifar.py`

Use this first. It now supports:

- `--smoke-test`
- `--use-fake-data`
- `--subset-size`
- `--cpu`
- `--model cnn|vit`

### `zero_deepspeed.py`

Use this only on Linux with CUDA and DeepSpeed available. It now has CLI flags for worker count, dataset slicing, and smoke testing, plus clearer hardware checks.

### `model_par.py`

Use this only if you specifically want to demonstrate pipeline parallelism across multiple GPUs.

## Validation

- All three scripts compile.
- `train_cifar.py` was refactored to support a realistic local smoke test.
- `zero_deepspeed.py` and `model_par.py` were reviewed and given clearer runtime guards, but not executed locally because they require CUDA/DeepSpeed environments.

## References

- [Ray Train docs](https://docs.ray.io/en/latest/train/train.html)
- [DeepSpeed docs](https://www.deepspeed.ai/)
