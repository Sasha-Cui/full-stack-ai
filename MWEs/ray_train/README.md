# Distributed Training with Ray

This folder contains three minimal examples of distributed training using the Ray ecosystem:

1. **Data-Parallel PyTorch**: `train_cifar.py`
2. **ZeRO-3 DeepSpeed**: `zero_deepspeed.py`
3. **Pipeline Parallelism**: `model_par.py`

## Installation

1. Create and activate an environment:
```bash
conda create -n raytrain python=3.10
conda activate raytrain
```

2. Install Ray:
```bash
pip install -U "ray[data,train]"
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install PyTorch for your CUDA version:
```bash
# Example for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Running the Examples

### 1. Data-Parallel CIFAR-10 (CPU or GPU)
```bash
python train_cifar.py
```
Notes:
- Adjust `num_workers` and `use_gpu` in `train_cifar.py` as needed.

### 2. ZeRO-3 DeepSpeed (GPU required)
```bash
python zero_deepspeed.py
```
Notes:
- Default config expects 4 GPUs. Update `num_workers` in `zero_deepspeed.py` if needed.

### 3. Pipeline Parallelism (2+ GPUs required)
```bash
python model_par.py
```
Notes:
- Requires at least 2 GPUs. The script checks this at runtime.

## Troubleshooting

- If you see CUDA errors, verify your CUDA toolkit and PyTorch install match.
- If you run on CPU-only, use the data-parallel example and set `use_gpu=False`.

## Slides

See `../../slides/ray_train.pdf` for accompanying material.
