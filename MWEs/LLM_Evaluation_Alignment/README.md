# LLM Evaluation and Alignment

## Overview
This tutorial covers the evaluation and alignment of large language models, including benchmarking methodologies, evaluation frameworks (lm-eval-harness), and alignment techniques (RLHF).

## Topics Covered

### 1. **Evaluation Methodologies**
- Benchmark design principles
- Multi-task evaluation
- Domain-specific assessments
- Human evaluation vs automated metrics

### 2. **LM Evaluation Harness**
- Installation and setup
- Running standard benchmarks
- Custom evaluation tasks
- Result interpretation

### 3. **Alignment Techniques**
- Reinforcement Learning from Human Feedback (RLHF)
- Direct Preference Optimization (DPO)
- Constitutional AI
- Alignment tax and performance tradeoffs

### 4. **Common Evaluation Issues**
- Multi-issue problems in evaluation
- Benchmark contamination
- Gaming the metrics
- Evaluation reliability

## Materials
- `llm_evaluation_presentation.ipynb` - Comprehensive presentation notebook
- `figures/` - Visual aids (RLHF diagram, multi-issue analysis)
- `results/` - Example evaluation plots

## Installation
```bash
conda create -n eval-tutorial python=3.10
conda activate eval-tutorial
pip install lm-eval jupyter matplotlib pandas
pip install transformers datasets torch
```

## Running the Tutorial
```bash
jupyter notebook llm_evaluation_presentation.ipynb
```

## Key Resources
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [RLHF Paper](https://arxiv.org/abs/2203.02155)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)

## Learning Objectives
- Understand LLM evaluation principles
- Use lm-eval-harness for benchmarking
- Implement RLHF pipelines
- Recognize evaluation pitfalls

## Contributing
Part of the Full-Stack AI working group at Yale University.

