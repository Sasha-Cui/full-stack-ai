# vLLM + DeepSpeed Tutorial

## Overview
This tutorial covers two critical system-level optimizers for large-scale LLM deployment and training:
- **vLLM**: Efficient LLM inference engine with PagedAttention
- **DeepSpeed**: Distributed training framework with ZeRO optimizations

These tools form the backbone of production-grade LLM systems, enabling both fast generation (vLLM) and efficient training (DeepSpeed).

## Topics Covered

### vLLM (Sections 1-4)

#### 1. **The KV Cache Problem**
- Understanding attention mechanism and computational cost
- Why autoregressive generation is expensive
- Introduction to Key-Value caching
- Benefits and limitations of naive KV caching

#### 2. **PagedAttention: Memory Management**
- The memory fragmentation problem with concurrent requests
- How PagedAttention solves it with paging
- Page tables and indirection
- Benefits: no fragmentation, no copies, GPU efficiency

#### 3. **vLLM Runtime Architecture**
- Request queuing and scheduling
- Iteration-level batching (cellular batching)
- GPU workers and streaming
- How paging and scheduling work together

#### 4. **Practical vLLM Usage**
- Installation and setup
- Loading models (HuggingFace or local)
- Configuration options (GPU memory, max length, dtype)
- Sampling parameters (temperature, top_p, max_tokens)
- Running batch inference

### DeepSpeed (Tutorial in Development)
- ZeRO optimization stages
- Distributed training patterns
- Integration with PyTorch
- Memory-efficient training

## Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- Understanding of transformer architecture
- Familiarity with PyTorch

## Installation

### 1. Create Environment
```bash
conda create -n vllm-tutorial python=3.10
conda activate vllm-tutorial
```

### 2. Install PyTorch
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install vLLM
```bash
pip install vllm transformers accelerate
```

### 4. Install Additional Dependencies
```bash
pip install jupyter ipython matplotlib numpy
```

### 5. (Optional) Install DeepSpeed
```bash
pip install deepspeed
```

## Running the Tutorials

### vLLM Tutorial
```bash
jupyter notebook vllm_sections_1_4.ipynb
```

### DeepSpeed Tutorial
```bash
jupyter notebook deepspeed_tutorial_sections_1_4.ipynb
```

## Model Configuration

The vLLM tutorial uses configurable model paths. You have three options:

### Option 1: HuggingFace Model ID (Recommended)
```python
model = "Qwen/Qwen2.5-1.5B-Instruct"  # Auto-downloads on first run
```

**Recommended models by GPU memory:**
- **4-8 GB VRAM**: `Qwen/Qwen2.5-1.5B-Instruct` or `microsoft/phi-2`
- **8-16 GB VRAM**: `Qwen/Qwen2.5-7B-Instruct` or `mistralai/Mistral-7B-Instruct-v0.2`
- **16-24 GB VRAM**: `meta-llama/Llama-3.1-8B-Instruct`
- **24+ GB VRAM**: `Qwen/Qwen2.5-14B-Instruct` or larger models

### Option 2: Local Model Path
```python
model = os.path.expanduser("~/models/Qwen3-4B-Instruct-2507")
```

### Option 3: Environment Variable
```python
model = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-1.5B-Instruct")
```

Then set in your environment:
```bash
export MODEL_PATH="/path/to/your/model"
```

## GPU Requirements

### Minimum
- **GPU**: NVIDIA GPU with 6GB+ VRAM
- **CUDA**: 11.8 or higher
- **Model**: Qwen2.5-1.5B or Phi-2 (smaller models)

### Recommended
- **GPU**: NVIDIA GPU with 16GB+ VRAM (RTX 4080, A4000, A5000)
- **CUDA**: 12.1 or higher
- **Model**: 7B parameter models

### Optimal
- **GPU**: NVIDIA GPU with 24GB+ VRAM (RTX 4090, A100, H100)
- **CUDA**: 12.1 or higher
- **Model**: 14B+ parameter models

## vLLM Configuration Options

### GPU Memory Utilization
```python
gpu_memory_utilization=0.9  # Use 90% of available GPU memory
```
- **0.7-0.8**: Conservative (good for multi-process systems)
- **0.9**: Recommended (balances performance and safety)
- **0.95**: Aggressive (may cause OOM with some models)

### Max Model Length
```python
max_model_len=2048  # Maximum sequence length
```
- Smaller values save memory
- Must be ≤ model's trained context length
- Typical values: 512, 1024, 2048, 4096, 8192

### Data Type
```python
dtype="float16"  # or "bfloat16" or "float32"
```
- **float32**: Highest precision, most memory
- **float16**: Good balance, widely supported
- **bfloat16**: Better for large models, requires Ampere+ GPUs

### Sampling Parameters
```python
SamplingParams(
    temperature=0.7,      # Randomness (0=deterministic, 1=more random)
    top_p=0.9,           # Nucleus sampling
    max_tokens=200,      # Maximum generation length
    skip_special_tokens=False  # Whether to skip special tokens in output
)
```

## Expected Runtime
- **vLLM Sections 1-3** (Theory): ~15 minutes reading
- **vLLM Section 4** (Practical): ~5-10 minutes (including model download)
- **First run**: Additional time for model download (depends on internet speed)

## Common Issues & Solutions

### Issue: `RuntimeError: CUDA out of memory`
**Solutions**:
1. Reduce `gpu_memory_utilization` to 0.7 or 0.8
2. Reduce `max_model_len` to 1024 or 512
3. Use a smaller model
4. Close other GPU processes

### Issue: `ImportError: cannot import name 'LLM' from 'vllm'`
**Solution**: Update vLLM to the latest version:
```bash
pip install --upgrade vllm
```

### Issue: Model download is slow
**Solution**: 
1. Use HuggingFace mirror (if in China):
```python
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```
2. Pre-download models:
```bash
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct
```

### Issue: `ValueError: Model X is not supported`
**Solution**: Check vLLM's supported models list: https://docs.vllm.ai/en/latest/models/supported_models.html

### Issue: PagedAttention fails
**Solution**: Try setting `enforce_eager=True` to disable CUDA graph compilation

### Issue: Slow first inference
**Solution**: This is normal - vLLM compiles kernels on first run. Subsequent inferences will be much faster.

## Performance Benchmarks

On a typical NVIDIA A100 (40GB):

| Metric | vLLM | Naive PyTorch |
|--------|------|---------------|
| Throughput (tokens/s) | ~2000 | ~500 |
| Latency (ms/token) | ~5 | ~20 |
| Memory utilization | ~95% | ~60% |
| Batch size supported | ~100 | ~20 |

*Results vary by model size, sequence length, and hardware*

## Learning Path

### Beginner
1. Read Sections 1-2 to understand the problem
2. Skim Section 3 for architecture overview
3. Run Section 4 with a small model

### Intermediate
1. Complete all sections carefully
2. Experiment with different models and configurations
3. Try batch inference with multiple prompts
4. Profile memory usage

### Advanced
1. Implement custom sampling strategies
2. Integrate vLLM into a production application
3. Benchmark different models and configurations
4. Explore vLLM's distributed inference features

## Key Diagrams

### vLLM Architecture
```
User Requests → Queue → Scheduler → GPU Workers → PagedAttention → Output Stream
                   ↑                                      ↓
                   └────────── Page Table Management ─────┘
```

### Memory Layout Comparison
```
Without Paging:     [AAAAAAA BBB ... CCCCC D ...] ← fragmentation, wasted space
With PagedAttention: P1(A) P2(A) P3(B) P4(C) P5(A) ← no gaps, efficient reuse
```

## Deployment Scenarios

### 1. Single-User API
```python
from vllm import LLM
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")

def generate(prompt):
    return llm.generate(prompt, SamplingParams(temperature=0.7))
```

### 2. Multi-User Server
Use vLLM's OpenAI-compatible server:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

### 3. Production Deployment
Consider:
- Load balancing multiple vLLM instances
- Request queueing with Redis/RabbitMQ
- Monitoring with Prometheus/Grafana
- Logging and error handling
- Rate limiting and authentication

## Additional Resources

### vLLM
- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [vLLM Blog](https://blog.vllm.ai/)

### DeepSpeed
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed)
- [ZeRO Paper](https://arxiv.org/abs/1910.02054)
- [DeepSpeed Tutorials](https://www.deepspeed.ai/tutorials/)

## Contributing
Found an issue or want to improve this tutorial? Please open an issue or pull request in the main repository.

## License
This tutorial is part of the Full-Stack AI working group materials at Yale University.

## Acknowledgments
Developed for the "Becoming Full-Stack AI Researchers" working group at Yale University, supported by the Wu Tsai Institute.

Special thanks to the vLLM and DeepSpeed teams for their excellent open-source tools.

## Citation
If you use these materials in your research or teaching, please cite:
```bibtex
@misc{fullstackai2025vllm,
  title={Becoming Full-Stack AI Researchers: vLLM and DeepSpeed Tutorial},
  author={Cui, Sasha and Le, Quan and Mader, Alexander and Sanok Dufallo, Will},
  year={2025},
  institution={Yale University}
}
```

