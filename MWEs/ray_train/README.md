# Distrubuted Training with Ray

This folder lists 3 examples of distributed model training that utilizes the [Ray](https://docs.ray.io/en/latest/index.html) ecosystem.

## Examples

1. [Data-Parallel Pytorch](https://docs.ray.io/en/latest/train/examples/pytorch/distributing-pytorch/README.html#step-2-distribute-training-to-multiple-machines-with-ray-train): [train_cifar](./train_cifar.py)

2. [ZeRO-3 Deepspeed](https://docs.ray.io/en/latest/train/deepspeed.html): [zero_deepspeed](./zero_deepspeed.py)

3. Model Parallel: [model_par](./model_par.py)

## Installation

- Create and activate an isolated Python environment (e.g. using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)):

    ```bash
    conda create --name raytrain python=3
    conda activate raytrain
    ```

- Install Ray following the [official docs](https://docs.ray.io/en/latest/ray-overview/installation.html):

    ```bash
    pip install -U "ray[data,train,tune,serve]"
    ```

- Find CUDA compute capability:

   ```bash
   nvidia-smi --query-gpu=compute_cap --format=csv

- Find CUDA-toolkit version:

    ```bash
    nvcc --version
    // or
    nvidia-smi | grep "CUDA Version"
    
- Install proper torch version for cuda-toolkit via [pytorch.org](https://pytorch.org/get-started/locally/)

- Install missing dependencies from [requirements.txt](./requirements.txt):

    ```bash
    pip install -r requirements.txt
    ```

## Accompanying Slides

Find respective slides about Ray at [ray_train.pdf](../../slides/ray_train.pdf).
