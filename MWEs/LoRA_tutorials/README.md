# LoRA (Low-Rank Adaptation) Tutorials

## Overview
This folder contains comprehensive tutorials on LoRA (Low-Rank Adaptation), a parameter-efficient fine-tuning technique for large neural networks. We demonstrate LoRA's principles through a concrete biological application: single-cell RNA sequencing analysis.

## Tutorials Included

### 1. **LoRA Single-Cell Demo** (`lora_single_cell_demo_clean.ipynb`)
A complete end-to-end demonstration of LoRA applied to cell type classification using PBMC3k dataset.

### 2. **PyTorch Lightning Tutorial** (`pytorch_lightning_tutorial.ipynb`)
Training orchestration with PyTorch Lightning framework.

## Topics Covered

### Theoretical Foundations
1. **LoRA Motivation**
   - Why parameter-efficient fine-tuning (PEFT)?
   - The capacity mismatch: pretraining vs post-training
   - Low-rank hypothesis for updates

2. **Mathematical Foundation**
   - Core equation: W' = W + γBA
   - Rank, scaling factors, and adapter matrices
   - Parameter efficiency analysis

3. **LoRA Advantages**
   - Multi-tenant serving
   - Memory efficiency (training, storage, transfer)
   - Compute efficiency (FLOPs analysis)
   - Modular adapter management

4. **When LoRA Matches Full Fine-Tuning**
   - Dataset size considerations
   - Rank selection guidelines
   - Layer coverage importance (attention vs MLP)
   - Hyperparameter sensitivity

### Practical Implementation
1. **Custom LoRA Layer**
   - Implementing LoRALinear from scratch
   - Forward pass mechanics
   - Parameter initialization strategies

2. **Model Architectures**
   - Full Fine-Tuning baseline
   - LoRA Full Coverage (all layers)
   - LoRA Attention-Only (first layer)

3. **Training Strategies**
   - Freezing base weights
   - Optimizer configuration
   - Learning rate selection (10x higher for LoRA)
   - Batch size considerations

4. **Hyperparameter Experiments**
   - Rank sweep (r=2, 8, 16, 32)
   - Learning rate tuning
   - Performance vs parameter count analysis

### Biological Application
1. **PBMC3k Dataset**
   - 2,700 peripheral blood mononuclear cells
   - 2,000 highly variable genes
   - 8 cell type clusters

2. **Data Processing Pipeline**
   - Quality filtering
   - Normalization and log-transformation
   - Highly variable gene selection
   - Graph-based clustering (Leiden algorithm)

3. **Model Evaluation**
   - Classification accuracy
   - Confusion matrix analysis
   - Training curves and convergence
   - Computational efficiency metrics

## Prerequisites
- Python 3.8+
- Basic understanding of neural networks
- Familiarity with PyTorch
- Understanding of fine-tuning concepts
- (For biology examples) Basic knowledge of single-cell RNA-seq

## Installation

### 1. Create Environment
```bash
conda create -n lora-tutorial python=3.10
conda activate lora-tutorial
```

### 2. Install Core Dependencies
```bash
# PyTorch (visit https://pytorch.org for latest instructions)
pip install torch torchvision torchaudio

# Core ML libraries
pip install numpy pandas scikit-learn matplotlib seaborn
```

### 3. Install Single-Cell Analysis Tools
```bash
pip install scanpy leidenalg python-igraph
```

### 4. Install Jupyter
```bash
pip install jupyter jupyterlab ipython
```

### 5. (Optional) Install PyTorch Lightning
```bash
pip install pytorch-lightning
```

## Running the Tutorials

### Single-Cell Demo
```bash
jupyter notebook lora_single_cell_demo_clean.ipynb
```

### PyTorch Lightning Tutorial
```bash
jupyter notebook pytorch_lightning_tutorial.ipynb
```

## Hardware Requirements

### Minimum (CPU Mode)
- **CPU**: Modern multi-core processor
- **RAM**: 8GB+
- **Time**: ~5-10 minutes for single-cell demo

### Recommended (GPU)
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **RAM**: 16GB+
- **Time**: ~2-3 minutes for single-cell demo

## Single-Cell Demo Structure

### Part 1: Theory (Cells 0-1)
- **Time**: 5 minutes reading
- LoRA motivation and mathematical foundations
- When LoRA matches full fine-tuning

### Part 2: Setup & Data Loading (Cells 2-11)
- **Time**: 2 minutes
- Load PBMC3k dataset
- Quality filtering and preprocessing
- Cell type clustering

### Part 3: Model Implementation (Cells 12-22)
- **Time**: 3 minutes
- LoRALinear layer implementation
- Three model architectures
- Training utilities

### Part 4: Training Experiments (Cells 23-34)
- **Time**: ~10 minutes (CPU) or ~5 minutes (GPU)
- Train Full Fine-Tuning baseline
- Train LoRA Full Coverage
- Train LoRA Attention-Only
- Hyperparameter experiments (rank and learning rate sweeps)

### Part 5: Analysis & Results (Cells 35-47)
- **Time**: 5 minutes
- Training curves visualization
- Accuracy comparison
- Parameter efficiency analysis
- Computational cost comparison
- Biological interpretation (confusion matrix)

## Key Results from Single-Cell Demo

### Performance Comparison
- **Full Fine-Tuning**: 95.9% accuracy, 580,104 parameters
- **LoRA Full Coverage (r=8)**: 94.8% accuracy, 24,256 parameters (4.2% of FullFT)
- **LoRA Attention-Only**: 92.4% accuracy, 18,048 parameters

### Computational Efficiency
- **Parameters**: LoRA trains 24× fewer parameters
- **Speed**: LoRA is 1.9× faster than FullFT
- **Memory**: Significantly lower (no optimizer state for frozen weights)

### Key Findings
1. ✅ **LoRA Full Coverage matches FullFT** (within 1% accuracy)
2. ❌ **Attention-Only LoRA underperforms** (2-3% accuracy drop)
3. ⚡ **Higher ranks improve performance** up to r≈16, then plateau
4. 📊 **Learning rate matters**: LoRA needs ~10× higher LR than FullFT

## Common Issues & Solutions

### Issue: `ImportError: No module named 'scanpy'`
**Solution**: Install scanpy and dependencies:
```bash
pip install scanpy leidenalg python-igraph
```

### Issue: `RuntimeError: No module named 'louvain'`
**Solution**: Install leidenalg (better than louvain):
```bash
pip install leidenalg
```
The notebook already uses Leiden algorithm, not Louvain.

### Issue: Kernel crashes during clustering
**Solution**: This might be due to memory issues. Try:
1. Use CPU mode: `device = torch.device("cpu")`
2. Reduce dataset size
3. Close other programs

### Issue: Training is very slow on CPU
**Solution**: Expected behavior. Consider:
1. Use Google Colab with free GPU
2. Reduce number of epochs
3. Use smaller hidden dimensions

### Issue: Different results than shown in notebook
**Solution**: This is normal due to:
1. Different random seeds
2. Hardware differences (CPU vs GPU)
3. Library version differences
Results should be similar (within 1-2% accuracy)

### Issue: `ValueError: Unknown layer type`
**Solution**: Ensure you're using the correct PyTorch version:
```bash
pip install torch>=2.0.0
```

## Extending the Tutorials

### Try Different Datasets
```python
# Other single-cell datasets available in scanpy
import scanpy as sc
adata = sc.datasets.pbmc68k_reduced()  # Larger dataset
adata = sc.datasets.paul15()           # Differentiation data
```

### Experiment with Hyperparameters
```python
# Try different ranks
for r in [4, 8, 16, 32, 64]:
    model = MLP_LoRA_Full(..., r=r)
    
# Try different learning rates
for lr in [1e-4, 1e-3, 1e-2, 5e-2]:
    optimizer = torch.optim.Adam(..., lr=lr)
```

### Apply to Your Domain
The LoRA implementation is domain-agnostic:
- Replace PBMC3k with your dataset
- Adjust input_dim and num_classes
- Keep the LoRA layer implementation the same

## Learning Path

### Beginner
1. Read the theoretical introduction (Cells 0-1)
2. Run the complete notebook without modifications
3. Observe the results and understand the comparisons

### Intermediate
1. Modify rank values and observe effects
2. Try different learning rates
3. Implement LoRA on a different dataset
4. Experiment with applying LoRA to different layers

### Advanced
1. Implement QLoRA (quantized LoRA)
2. Combine multiple LoRA adapters
3. Implement LoRA merging strategies
4. Apply to larger language models
5. Explore rank selection algorithms

## Key Takeaways

### From Theory
1. **LoRA is not always worse than full fine-tuning**
2. **Layer coverage matters more than rank** (up to a point)
3. **Learning rate needs to be higher** for LoRA (~10× FullFT)
4. **Batch size sensitivity** is a LoRA characteristic

### From Practice
1. **Implementation is straightforward** (< 50 lines for LoRALinear)
2. **Computational savings are real** (24× fewer parameters, 2× faster)
3. **Biological interpretability is preserved** (similar confusion patterns)
4. **Rank=8-16 is a good default** for most applications

## Relation to "LoRA Without Regret" Paper

This tutorial validates key findings from the paper:
- LoRA matches FullFT with proper configuration
- Full layer coverage is critical
- Rank ≥ 8 generally sufficient for small datasets
- Learning rate scaling is important

## Additional Resources

### Papers
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) (Schulman & Thinking Machines Lab, 2025)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023)

### Libraries
- [PEFT (Hugging Face)](https://github.com/huggingface/peft) - Production LoRA implementation
- [LoRA-X](https://github.com/low-rank/LoRA-X) - Extended LoRA implementations
- [Scanpy](https://scanpy.readthedocs.io/) - Single-cell analysis

### Tutorials
- [Hugging Face PEFT Tutorial](https://huggingface.co/docs/peft/index)
- [Scanpy Tutorials](https://scanpy-tutorials.readthedocs.io/)
- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)

## Contributing
Found an issue or want to improve this tutorial? Please open an issue or pull request in the main repository.

## License
This tutorial is part of the Full-Stack AI working group materials at Yale University.

## Acknowledgments
Developed for the "Becoming Full-Stack AI Researchers" working group at Yale University, supported by the Wu Tsai Institute.

Thanks to the Hugging Face PEFT team and the original LoRA authors for their foundational work.

## Citation
If you use these materials in your research or teaching, please cite:
```bibtex
@misc{fullstackai2025lora,
  title={Becoming Full-Stack AI Researchers: LoRA Tutorial},
  author={Cui, Sasha and Le, Quan and Mader, Alexander and Sanok Dufallo, Will},
  year={2025},
  institution={Yale University}
}
```

