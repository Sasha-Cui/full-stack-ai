# PyTorch Tutorial

## Overview
This tutorial provides a comprehensive introduction to PyTorch fundamentals for AI research and engineering. It covers essential concepts from basic tensor operations to advanced topics like custom autograd functions and optimization strategies.

## Topics Covered

### 1. **Autograd vs NumPy**
- Why automatic differentiation matters
- Performance comparison with NumPy
- GPU acceleration benefits

### 2. **Tensor Basics**
- Creation, shapes, dtypes, and devices
- Moving tensors between CPU and GPU
- Best practices for device management

### 3. **Autograd Tracking**
- Understanding `requires_grad`
- Using `torch.no_grad()` and `torch.inference_mode()`
- Breaking gradient flow with `.detach()`
- Difference between `.detach()` and `.item()`

### 4. **Custom Autograd Functions**
- Implementing custom forward and backward passes
- Example: Matérn kernel with Bessel functions
- Wrapping SciPy functions with PyTorch autograd

### 5. **Optimization Strategies**
- Parameter groups with different learning rates
- Early stopping and learning rate scheduling
- Gradient clipping
- Anomaly detection for debugging

### 6. **Mixed Precision Training (AMP)**
- Using float16/bfloat16 for faster training
- Automatic Mixed Precision with `torch.cuda.amp`
- Memory optimization techniques

### 7. **Debugging Tools**
- Anomaly detection mode
- NaN/Inf hooks
- Profiling with `torch.profiler`

### 8. **Training Utilities**
- Estimating training time (ETA)
- Progress tracking
- Moving averages for metrics

### 9. **Data Conversion**
- NumPy ↔ PyTorch conversion
- Pandas integration
- Zero-copy with DLPack

### 10. **Memory Profiling**
- CUDA memory tracking
- Using `torch.cuda.memory_allocated()`
- Simple experiment logging

## Prerequisites
- Python 3.8+
- Basic understanding of neural networks
- Familiarity with NumPy (helpful but not required)

## Installation

```bash
# Create a conda environment
conda create -n pytorch-tutorial python=3.10
conda activate pytorch-tutorial

# Install PyTorch (visit https://pytorch.org for latest instructions)
# For CPU only:
pip install torch torchvision torchaudio

# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install additional dependencies
pip install numpy pandas scipy matplotlib jupyter ipython memory_profiler
```

## Running the Tutorial

### Jupyter Notebook
```bash
jupyter notebook pytorch_tutorial.ipynb
```

### JupyterLab
```bash
jupyter lab pytorch_tutorial.ipynb
```

### VS Code
Open the notebook in VS Code with the Jupyter extension installed.

## GPU Requirements
- **Minimum**: No GPU required; CPU mode works for all examples
- **Recommended**: NVIDIA GPU with 4GB+ VRAM for AMP examples
- **Optimal**: NVIDIA GPU with 8GB+ VRAM for comfortable experimentation

## Expected Runtime
- **CPU**: ~10-15 minutes to run all cells
- **GPU**: ~5-10 minutes to run all cells

## Common Issues & Solutions

### Issue: `RuntimeError: CUDA out of memory`
**Solution**: Reduce batch sizes or use CPU mode. Clear cache with `torch.cuda.empty_cache()`.

### Issue: `ImportError: No module named 'scipy'`
**Solution**: Install SciPy: `pip install scipy`. The Bessel function example requires SciPy.

### Issue: Memory profiler not working
**Solution**: Install memory_profiler: `pip install memory_profiler`

### Issue: Slow NumPy operations
**Solution**: Ensure you have optimized BLAS libraries (MKL, OpenBLAS). Install via `conda install numpy` for optimized versions.

## Learning Path
1. Start with Sections 1-3 for fundamentals
2. Move to Sections 4-5 for custom operations and optimization
3. Study Sections 6-7 for performance and debugging
4. Review Sections 8-10 for practical utilities

## Additional Resources
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Forums](https://discuss.pytorch.org/)
- [Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html)

## Contributing
Found an issue or want to improve this tutorial? Please open an issue or pull request in the main repository.

## License
This tutorial is part of the Full-Stack AI working group materials at Yale University.

## Acknowledgments
Developed for the "Becoming Full-Stack AI Researchers" working group at Yale University, supported by the Wu Tsai Institute.

