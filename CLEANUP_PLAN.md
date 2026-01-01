# Full-Stack AI Repository Cleanup & Enhancement Plan

## Overview
This document tracks the comprehensive cleanup, debugging, and enhancement of the Full-Stack AI repository based on the vision outlined in `syllabus.tex`.

## Syllabus Structure (from syllabus.tex)

### Module 1: LLM as Black Boxes 1 - Datasets, Models, and Benchmarking
- **Topics**: HuggingFace, Quantization, Datasets, lm-eval, inspect-ai
- **Current MWEs**: LLM_Evaluation_Alignment/
- **Tutorial Section**: [TO BE WRITTEN]
- **Status**: Needs evaluation section

### Module 2: LLM as Black Boxes 2 - Inference, Evaluation, Deployment
- **Topics**: OpenRouter, vLLM, FastAPI, GEPA, Tools/MCP, TransformerLens
- **Current MWEs**: Inference/, vllm+deepspeed/
- **Tutorial Section**: vllm.tex (not in main tutorial.tex)
- **Status**: MWE exists, needs tutorial integration

### Module 3: Post Training LLMs 1 - Supervised Fine-Tuning (SFT)
- **Topics**: LoRA/QLoRA with PEFT, Lightning
- **Current MWEs**: LoRA_tutorials/
- **Tutorial Section**: lora.tex, sft.tex (sft not in main)
- **Status**: Good MWE, tutorial needs enhancement

### Module 4: Post Training LLMs 2 - Reinforcement Learning (RL)
- **Topics**: Docker/Apptainer, VERL, Ray, JAX, W&B
- **Current MWEs**: verl/, ray_train/, vllm+deepspeed/
- **Tutorial Section**: ray.tex, deepspeed.tex (deepspeed not in main)
- **Status**: Materials exist, need organization

### Module 5: Agentic LLMs 1 - Software & Hardware Agents
- **Topics**: LangChain, ReAct, MemGPT, OpenVLA
- **Current MWEs**: agentic_rl_workshop.ipynb, Robotics/
- **Tutorial Section**: [TO BE WRITTEN]
- **Status**: Exploratory stage

### Module 6: Agentic LLMs 2 - End-to-End Project
- **Topics**: Complete pipeline, scaling, debugging
- **Current MWEs**: Various
- **Tutorial Section**: [TO BE WRITTEN]
- **Status**: To be developed

## Foundational Topics
- **PyTorch fundamentals**: pytorch/ → torch-jax-tf.tex
- **Scaling Laws**: Scaling_Laws/ → [needs tutorial section]

---

## Phase 1: MWE Cleanup & Debugging

### 1.1 PyTorch Tutorial ✓ [IN PROGRESS]
**File**: `MWEs/pytorch/pytorch_tutorial.ipynb`
**Issues Found**:
- Line: `from pathlib import Path1` → should be `Path`
- Memory profiler extensions may need testing
**Actions**:
- [x] Fix import error
- [ ] Test all cells run properly
- [ ] Verify custom autograd example works
- [ ] Clean up output formatting

### 1.2 LoRA Tutorial
**File**: `MWEs/LoRA_tutorials/lora_single_cell_demo_clean.ipynb`
**Issues to Check**:
- Cell outputs present but may be from different environment
- Verify all dependencies (scanpy, leidenalg, etc.)
- Check CUDA/CPU compatibility
**Actions**:
- [ ] Review for errors
- [ ] Test end-to-end execution
- [ ] Ensure biological interpretability section is clear

### 1.3 Inference Tutorial
**File**: `MWEs/Inference/inference.ipynb`
**Issues to Check**:
- API key dependencies (OpenRouter, OpenAI, Notion, Google)
- Tools integration (tools.py, tools.json)
- GEPA_utils.py completeness
**Actions**:
- [ ] Add instructions for API key setup
- [ ] Test tool calling examples
- [ ] Verify GEPA optimization works
- [ ] Add error handling for missing keys

### 1.4 vLLM + DeepSpeed
**Files**: `MWEs/vllm+deepspeed/*.ipynb`
**Issues to Check**:
- Model path hardcoded: `/gpfs/radev/project/...`
- GPU memory requirements
- Incomplete tutorial sections
**Actions**:
- [ ] Parameterize model paths
- [ ] Complete missing sections
- [ ] Add resource requirement documentation
- [ ] Test on different GPU configurations

### 1.5 Ray Train
**Files**: `MWEs/ray_train/*.py`
**Issues to Check**:
- Dependencies in requirements.txt
- CUDA compatibility instructions
- Three separate examples need testing
**Actions**:
- [ ] Test train_cifar.py
- [ ] Test zero_deepspeed.py
- [ ] Test model_par.py
- [ ] Update README with clear instructions

### 1.6 VERL Tutorial
**Files**: `MWEs/verl/*`
**Issues to Check**:
- Container-based setup complexity
- Apptainer/Docker instructions
- File paths and environment variables
**Actions**:
- [ ] Simplify setup instructions
- [ ] Add troubleshooting section
- [ ] Test evaluation scripts
- [ ] Document expected results

### 1.7 LLM Evaluation & Alignment
**File**: `MWEs/LLM_Evaluation_Alignment/llm_evaluation_presentation.ipynb`
**Actions**:
- [ ] Review completeness
- [ ] Check figure quality
- [ ] Test lm-eval examples

### 1.8 Scaling Laws
**File**: `MWEs/Scaling_Laws/scaling_laws.ipynb`
**Actions**:
- [ ] Review notebook completeness
- [ ] Verify figures render properly
- [ ] Check mathematical derivations

### 1.9 Robotics
**File**: `MWEs/Robotics/frameworks.ipynb`
**Actions**:
- [ ] Review TODO.md
- [ ] Check media files
- [ ] Test examples

### 1.10 Agentic RL Workshop
**File**: `MWEs/agentic_rl_workshop.ipynb`
**Actions**:
- [ ] Review content
- [ ] Check if complete or placeholder

---

## Phase 2: Tutorial Paper Enhancement

### 2.1 Existing Sections Review

#### introduction.tex
- [ ] Review and enhance
- [ ] Ensure alignment with syllabus vision

#### torch-jax-tf.tex
- [ ] Review completeness
- [ ] Link to PyTorch MWE
- [ ] Add JAX and TensorFlow content if missing

#### ray.tex
- [ ] Review and enhance
- [ ] Link to ray_train MWEs
- [ ] Add distributed training concepts

#### lora.tex
- [ ] Review and enhance
- [ ] Link to LoRA MWE
- [ ] Ensure mathematical foundations are clear

#### conclusion.tex
- [ ] Review and enhance
- [ ] Update with project vision

### 2.2 Sections to Write/Integrate

#### vllm.tex (exists but not in main)
- [ ] Review content
- [ ] Add to tutorial.tex
- [ ] Link to vLLM MWE

#### deepspeed.tex (exists but not in main)
- [ ] Review content
- [ ] Add to tutorial.tex
- [ ] Link to DeepSpeed examples

#### sft.tex (exists but not in main)
- [ ] Review content
- [ ] Add to tutorial.tex
- [ ] Integrate with LoRA section

### 2.3 New Sections Needed

#### evaluation.tex
- [ ] Write section on benchmarking
- [ ] Cover lm-eval, inspect-ai
- [ ] Link to evaluation MWE

#### inference.tex (commented out in main)
- [ ] Write comprehensive inference section
- [ ] Cover API usage, deployment
- [ ] Link to Inference MWE

#### data.tex (commented out in main)
- [ ] Write section on datasets
- [ ] Cover HuggingFace datasets, formats
- [ ] Add examples

#### rl.tex (commented out in main)
- [ ] Write RL/RLHF section
- [ ] Cover VERL, PPO
- [ ] Link to VERL MWE

#### agents.tex [NEW]
- [ ] Write agentic systems section
- [ ] Cover tools, MCP, LangChain
- [ ] Link to relevant MWEs

#### scaling_laws.tex [NEW]
- [ ] Write scaling laws section
- [ ] Cover Kaplan, Chinchilla laws
- [ ] Link to Scaling_Laws MWE

---

## Phase 3: Repository Organization

### 3.1 Main README
- [ ] Add clear project description
- [ ] Create module-based organization
- [ ] Add prerequisites and setup instructions
- [ ] Link to tutorial paper
- [ ] Add contribution guidelines

### 3.2 MWE Organization
- [ ] Ensure each MWE has its own README
- [ ] Add requirements.txt where missing
- [ ] Standardize naming conventions
- [ ] Add expected outputs/results

### 3.3 Documentation
- [ ] Add CONTRIBUTING.md
- [ ] Add installation guides
- [ ] Add troubleshooting guides
- [ ] Add resources and references

---

## Phase 4: Testing & Validation

### 4.1 Environment Testing
- [ ] Test on CPU-only environments
- [ ] Test on single GPU
- [ ] Test on multi-GPU setups
- [ ] Test on HPC clusters

### 4.2 Dependency Management
- [ ] Create environment.yml files
- [ ] Test fresh installations
- [ ] Document version requirements
- [ ] Add compatibility matrices

### 4.3 Documentation Review
- [ ] Check all links work
- [ ] Verify figures display correctly
- [ ] Test code snippets
- [ ] Proofread text

---

## Priority Order

### High Priority (Immediate)
1. Fix PyTorch tutorial import error
2. Test and fix core MWEs (PyTorch, LoRA, Inference)
3. Update main README
4. Review existing tutorial sections

### Medium Priority (Next)
5. Complete vLLM, DeepSpeed, SFT tutorials
6. Integrate existing .tex files into main tutorial
7. Test Ray and VERL examples
8. Write evaluation and data sections

### Low Priority (Future)
9. Write new sections (agents, scaling laws)
10. Advanced testing on different platforms
11. Add advanced examples
12. Create video tutorials

---

## Metrics for Success
- [ ] All MWEs run without errors
- [ ] Tutorial paper covers all syllabus modules
- [ ] Clear installation instructions for all components
- [ ] Professional README and documentation
- [ ] All figures and media display correctly
- [ ] Consistent code style and formatting
- [ ] Comprehensive error handling and troubleshooting

