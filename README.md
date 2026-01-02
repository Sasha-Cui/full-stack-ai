# Becoming Full-Stack AI Researchers
# CURRENTLY UNDER CONSTRUCTION.  SOME OF THE CODE HAS NOT BEEN FULLY REVIEWED OR VERIFIED.  PLEASE COME BACK A LITTLE LATER.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## 🎯 Overview

Welcome to the **Becoming Full-Stack AI Researchers** working group at Yale University! This repository contains comprehensive tutorials, minimal working examples (MWEs), and educational materials covering the essential packages, frameworks, and tools for end-to-end AI development and research.

### Goals
- 🚀 Equip researchers with skills to go beyond narrow, single-aspect AI work toward holistic, end-to-end AI project capability
- 📚 Build reusable onboarding materials for Yale members interested in AI research
- 🤝 Create a community of Explorers and Builders in AI

### Deliverables
- **GitHub Repository**: Minimal working examples, demos, and slides
- **Tutorial Paper**: Comprehensive co-authored guide for all modules
- **Presentations**: In-depth framework introductions

## 📋 Table of Contents
- [Modules](#-modules)
- [Getting Started](#-getting-started)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Contributing](#-contributing)
- [Resources](#-resources)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)

---

## 🎓 Modules

Our curriculum is organized into six interconnected modules, each covering critical aspects of the AI research and engineering pipeline.

### Module 1: LLM as Black Boxes 1 – Datasets, Models, and Benchmarking
**Topics**: HuggingFace, Quantization (BitsAndBytes), Datasets (Parquet, PyArrow), Benchmarking (lm-eval, inspect-ai)

📁 **Materials**:
- `MWEs/LLM_Evaluation_Alignment/` - Evaluation and alignment presentation

🎯 **Learning Objectives**:
- Load and save sharded HuggingFace model checkpoints
- Quantize models for efficient deployment
- Store and load datasets in efficient formats
- Benchmark model performance

### Module 2: LLM as Black Boxes 2 – Inference, Evaluation, Deployment
**Topics**: OpenRouter, vLLM (PagedAttention), FastAPI, GEPA, Tools/MCP, TransformerLens

📁 **Materials**:
- `MWEs/Inference/` - Complete inference tutorial with API usage, tools, and GEPA
- `MWEs/vllm+deepspeed/` - vLLM tutorial with PagedAttention deep dive

🎯 **Learning Objectives**:
- Use APIs for LLM inference (OpenRouter, OpenAI)
- Understand model selection tradeoffs (cost, performance, latency)
- Implement tool calling and MCP integration
- Optimize prompts with GEPA
- Deploy models with vLLM for efficient serving

📄 **Tutorial Paper**: [`overleaf/sections/vllm.tex`](overleaf/sections/vllm.tex)

### Module 3: Post-Training LLMs 1 – Supervised Fine-Tuning (SFT)
**Topics**: LoRA/QLoRA with PEFT, PyTorch Lightning

📁 **Materials**:
- `MWEs/LoRA_tutorials/` - Comprehensive LoRA tutorial with single-cell biology demo
- `MWEs/pytorch/` - PyTorch fundamentals

🎯 **Learning Objectives**:
- Understand parameter-efficient fine-tuning (PEFT)
- Implement LoRA from scratch
- Compare LoRA with full fine-tuning
- Optimize rank selection and hyperparameters
- Orchestrate training with PyTorch Lightning

📄 **Tutorial Paper**: [`overleaf/sections/lora.tex`](overleaf/sections/lora.tex), [`overleaf/sections/sft.tex`](overleaf/sections/sft.tex)

### Module 4: Post-Training LLMs 2 – Reinforcement Learning (RL)
**Topics**: Docker/Apptainer, VERL, Ray, JAX, Weights & Biases

📁 **Materials**:
- `MWEs/verl/` - VERL tutorial for PPO training on GSM8K
- `MWEs/ray_train/` - Distributed training with Ray (data parallel, ZeRO, model parallel)
- `MWEs/vllm+deepspeed/` - DeepSpeed integration

🎯 **Learning Objectives**:
- Container workflows (Docker, Apptainer)
- Reinforcement learning with VERL (PPO)
- Distributed training strategies (Ray, DeepSpeed ZeRO)
- Experiment tracking (W&B)

📄 **Tutorial Paper**: [`overleaf/sections/ray.tex`](overleaf/sections/ray.tex), [`overleaf/sections/deepspeed.tex`](overleaf/sections/deepspeed.tex)

### Module 5: Agentic LLMs 1 – Software & Hardware Agents
**Topics**: LangChain, ReAct, MemGPT, OpenVLA

📁 **Materials**:
- `MWEs/agentic_rl_workshop.ipynb` - Agentic RL workshop
- `MWEs/Robotics/` - Vision-Language-Action frameworks

🎯 **Learning Objectives**:
- Build multi-step reasoning workflows
- Implement agent frameworks
- Vision-Language-Action models for robotics

### Module 6: Agentic LLMs 2 – End-to-End Project
**Topics**: Complete pipeline from data → training → deployment

🎯 **Learning Objectives**:
- Build complete AI pipelines
- Scale and debug on HPC clusters
- Deploy production systems

### Foundational Topics
**Topics**: PyTorch, JAX, TensorFlow, Scaling Laws

📁 **Materials**:
- `MWEs/pytorch/` - Comprehensive PyTorch tutorial (autograd, custom ops, optimization)
- `MWEs/Scaling_Laws/` - Scaling laws analysis (Kaplan, Chinchilla)

📄 **Tutorial Paper**: [`overleaf/sections/torch-jax-tf.tex`](overleaf/sections/torch-jax-tf.tex)

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.8+** (3.10 recommended)
- **Fluency in Python** (required)
- **Git** and **Conda** (or virtualenv)
- **CUDA-capable GPU** (optional but recommended for deep learning tasks)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/sashacui/full-stack-ai.git
cd full-stack-ai

# Choose a module to start with (e.g., PyTorch basics)
cd MWEs/pytorch

# Create environment and install dependencies
conda create -n pytorch-tutorial python=3.10
conda activate pytorch-tutorial
pip install torch numpy pandas jupyter

# Run the tutorial
jupyter notebook pytorch_tutorial.ipynb
```

### Recommended Learning Path

#### For Beginners
1. **Start with**: `MWEs/pytorch/` - Learn PyTorch fundamentals
2. **Move to**: `MWEs/Inference/` - Understand LLM APIs and inference
3. **Then try**: `MWEs/LoRA_tutorials/` - Learn parameter-efficient fine-tuning

#### For Intermediate Users
1. **Start with**: `MWEs/vllm+deepspeed/` - Efficient serving
2. **Move to**: `MWEs/ray_train/` - Distributed training
3. **Then try**: `MWEs/verl/` - RL fine-tuning

#### For Advanced Users
1. Explore all modules based on your research needs
2. Experiment with combinations (e.g., LoRA + VERL + vLLM)
3. Build end-to-end projects using multiple tools

---

## 📂 Repository Structure

```
full-stack-ai/
├── MWEs/                           # Minimal Working Examples
│   ├── pytorch/                    # PyTorch fundamentals
│   │   ├── pytorch_tutorial.ipynb
│   │   └── README.md
│   ├── Inference/                  # LLM inference, tools, GEPA
│   │   ├── inference.ipynb
│   │   ├── tools.py
│   │   ├── GEPA_utils.py
│   │   └── README.md
│   ├── LoRA_tutorials/             # LoRA/PEFT tutorials
│   │   ├── lora_single_cell_demo_clean.ipynb
│   │   ├── pytorch_lightning_tutorial.ipynb
│   │   └── README.md
│   ├── vllm+deepspeed/             # vLLM and DeepSpeed
│   │   ├── vllm_sections_1_4.ipynb
│   │   ├── deepspeed_tutorial_sections_1_4.ipynb
│   │   └── README.md
│   ├── ray_train/                  # Ray distributed training
│   │   ├── train_cifar.py
│   │   ├── zero_deepspeed.py
│   │   ├── model_par.py
│   │   └── README.md
│   ├── verl/                       # VERL RL training
│   │   ├── evaluate_gsm8k.py
│   │   ├── compare_results.py
│   │   └── README.md
│   ├── LLM_Evaluation_Alignment/   # Evaluation & alignment
│   │   └── llm_evaluation_presentation.ipynb
│   ├── Scaling_Laws/               # Scaling laws analysis
│   │   └── scaling_laws.ipynb
│   ├── Robotics/                   # VLA frameworks
│   │   └── frameworks.ipynb
│   └── agentic_rl_workshop.ipynb   # Agentic systems
│
├── overleaf/                       # Tutorial paper source
│   ├── tutorial.tex                # Main tutorial document
│   ├── syllabus.tex                # Course syllabus
│   └── sections/                   # Individual sections
│       ├── introduction.tex
│       ├── torch-jax-tf.tex
│       ├── ray.tex
│       ├── lora.tex
│       ├── vllm.tex
│       ├── deepspeed.tex
│       ├── sft.tex
│       └── conclusion.tex
│
├── slides/                         # Presentation materials
│   ├── ray_train.pdf
│   └── verl_tutorial.pdf
│
├── README.md                       # This file
└── CLEANUP_PLAN.md                 # Development roadmap
```

---

## 💻 Installation

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS (11+), or Windows (WSL2)
- **RAM**: 16GB+ (32GB recommended for large models)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- **Storage**: 50GB+ free space (for models and datasets)

### Environment Setup

We recommend using Conda for environment management:

```bash
# Install Miniconda (if not already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create a base environment
conda create -n fullstack-ai python=3.10
conda activate fullstack-ai

# Install common dependencies
pip install torch torchvision torchaudio
pip install transformers accelerate datasets
pip install jupyter jupyterlab ipython
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Module-Specific Installation

Each MWE folder contains its own `README.md` with specific installation instructions. For example:

```bash
# For vLLM tutorial
cd MWEs/vllm+deepspeed
pip install vllm
jupyter notebook vllm_sections_1_4.ipynb

# For LoRA tutorial
cd MWEs/LoRA_tutorials
pip install scanpy leidenalg
jupyter notebook lora_single_cell_demo_clean.ipynb

# For VERL tutorial
cd MWEs/verl
# Follow containerized setup in README.md
```

---

## 🎮 Usage

### Running Jupyter Notebooks
```bash
# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab

# Access via browser at http://localhost:8888
```

### Running Python Scripts
```bash
# Example: Ray training
cd MWEs/ray_train
python train_cifar.py

# Example: VERL evaluation
cd MWEs/verl
python evaluate_gsm8k.py --model_path <path> --data_path <path>
```

### Using HPC Clusters
```bash
# Example SLURM job submission
cd MWEs/verl
sbatch ppo_gsm8k.sbatch
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Ways to Contribute
1. **Report Issues**: Found a bug? Open an issue!
2. **Improve Documentation**: Fix typos, add examples, clarify instructions
3. **Add Examples**: Contribute new MWEs or extend existing ones
4. **Enhance Tutorials**: Improve the tutorial paper sections

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test thoroughly
5. Commit with clear messages (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style
- Follow PEP 8 for Python code
- Add docstrings to functions and classes
- Include comments for complex logic
- Update READMEs when adding new feature

---

## 📚 Resources

### Official Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Hugging Face Documentation](https://huggingface.co/docs)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Ray Documentation](https://docs.ray.io/)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)

### Course Website
- [Full-Stack AI Course Website](https://sashacui.com/full-stack.html)

### Papers
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)
- [ZeRO: DeepSpeed Optimization](https://arxiv.org/abs/1910.02054)
- [Scaling Laws for Neural LMs](https://arxiv.org/abs/2001.08361)

### Community
- [Discord](#) (Coming soon)
- [Mailing List](#) (Coming soon)
- Office Hours: Every other Friday @ 100 College Street, Yale

---

## 📖 Tutorial Paper

The complete tutorial paper is being developed in the `overleaf/` directory. Current sections include:

- ✅ Introduction
- ✅ PyTorch, JAX, and TensorFlow Fundamentals
- ✅ Ray: Distributed Training
- ✅ LoRA: Parameter-Efficient Fine-Tuning
- 🚧 vLLM: Efficient Inference
- 🚧 DeepSpeed: Memory-Efficient Training
- 🚧 SFT: Supervised Fine-Tuning
- 🚧 Evaluation and Benchmarking
- 🚧 Agentic Systems
- ✅ Conclusion

*Legend: ✅ = Complete, 🚧 = In Progress*

To compile the tutorial paper (requires LaTeX):
```bash
cd overleaf
pdflatex tutorial.tex
bibtex tutorial
pdflatex tutorial.tex
pdflatex tutorial.tex
```

---

## 📄 Citation

If you use these materials in your research or teaching, please cite:

```bibtex
@misc{fullstackai2025,
  title={Becoming Full-Stack AI Researchers: A Comprehensive Tutorial},
  author={Cui, Sasha and Le, Quan and Mader, Alexander and Sanok Dufallo, Will},
  year={2025},
  institution={Yale University},
  howpublished={\url{https://github.com/sashacui/full-stack-ai}},
  note={Fall 2025 Working Group}
}
```

---

## 🙏 Acknowledgments

### Institutions
- **Wu Tsai Institute at Yale University** - GPU resources and classroom space
- **Yale Department of Statistics \& Data Science**
- **Yale Department of Physics**
- **Yale Department of Philosophy**
- **Misha High Performance Computing Cluster**

### Contributors
- **Sasha Cui** (sasha.cui@yale.edu) - Project Lead, Statistics & Data Science
- **Quan Le** (quan.le@yale.edu) - Statistics & Data Science
- **Alexander Mader** (alexander.mader@yale.edu) - Physics
- **Will Sanok Dufallo** (will.sanokdufallo@yale.edu) - Philosophy

### Advisors & Supporters
We thank Ping Luo, John Lafferty, Linjun Zhang, Anurag Kashyap, Theo Saarinen, and Yuxuan Zhu for helpful comments and suggestions.

### Open Source Community
Special thanks to the developers and maintainers of:
- PyTorch, Hugging Face, vLLM, Ray, DeepSpeed, VERL
- Jupyter, NumPy, Pandas, Matplotlib, Scikit-learn
- And the entire open-source AI/ML community

---

## 📞 Contact

- **Email**: sasha.cui@yale.edu
- **Website**: [https://sashacui.com/full-stack.html](https://sashacui.com/full-stack.html)
- **Issues**: [GitHub Issues](https://github.com/sashacui/full-stack-ai/issues)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**[⬆ Back to Top](#becoming-full-stack-ai-researchers)**

Made with ❤️ at Yale University

</div>
