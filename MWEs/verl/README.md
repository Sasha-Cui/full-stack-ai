# VERL PPO GSM8K Tutorial

This walkthrough trains a small model with VERL PPO on GSM8K using the official vLLM container image.

## Prerequisites
- NVIDIA GPU
- Apptainer (or Docker, if you adapt the commands)
- Optional: Weights & Biases account for logging

## 1. Pull the Container and Create an Overlay
```bash
WORKDIR=$HOME/verl_tutorial
mkdir -p "$WORKDIR"
cd "$WORKDIR"

apptainer pull verl_vllm.sif docker://verlai/verl:vegalita-vllm
apptainer overlay create --fakeroot --size 5120 verl_overlay.img
```

## 2. Start the Container
```bash
apptainer shell \
  --nv \
  --fakeroot \
  --overlay "$WORKDIR/verl_overlay.img" \
  --bind "$WORKDIR:$WORKDIR" \
  --bind /tmp:/tmp \
  "$WORKDIR/verl_vllm.sif"
```

## 3. Install VERL and Configure Caches (inside the container)
```bash
cd "$WORKDIR"
git clone https://github.com/volcengine/verl
cd verl
pip3 install --no-deps -e .

# Optional: W&B logging
wandb login

# HuggingFace cache
export HF_HOME="$WORKDIR/hf_cache"
```

## 4. Download GSM8K and Warm the Model
```bash
python3 -B examples/data_preprocess/gsm8k.py --local_save_dir "$WORKDIR/data/gsm8k"
python3 -B -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-0.5B-Instruct')"
```

## 5. Train with PPO
```bash
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  data.train_files="$WORKDIR/data/gsm8k/train.parquet" \
  data.val_files="$WORKDIR/data/gsm8k/test.parquet" \
  data.train_batch_size=256 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  critic.optim.lr=1e-5 \
  critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  critic.ppo_micro_batch_size_per_gpu=4 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=verl_tutorial \
  trainer.experiment_name=qwen25_ppo_gsm8k \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.total_epochs=15 2>&1 | tee "$WORKDIR/verl_demo.log"
```

Notes:
- If you hit OOM, reduce `data.train_batch_size` and `actor_rollout_ref.rollout.gpu_memory_utilization`.
- For multi-GPU runs, increase `trainer.n_gpus_per_node` and update batch sizes accordingly.

## 6. Merge the Trained Checkpoint
```bash
python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir checkpoints/verl_tutorial/qwen25_ppo_gsm8k/global_step_435/actor \
  --target_dir checkpoints/verl_tutorial/qwen25_ppo_gsm8k/global_step_435/actor/huggingface
```

## 7. Evaluate Trained vs Baseline
Trained model:
```bash
python evaluate_gsm8k.py \
  --model_path ./checkpoints/verl_tutorial/qwen25_ppo_gsm8k/global_step_435/actor/huggingface \
  --data_path "$WORKDIR/data/gsm8k/test.parquet" \
  --output trained_outputs.jsonl
```

Baseline:
```bash
python evaluate_gsm8k.py \
  --model_path Qwen/Qwen2.5-0.5B-Instruct \
  --data_path "$WORKDIR/data/gsm8k/test.parquet" \
  --output base_outputs.jsonl
```

## 8. Compare Results
```bash
python compare_results.py --base base_outputs.jsonl --trained trained_outputs.jsonl -n 5
```

## 9. (Optional) Low-Rank Analysis
See https://github.com/xingzhis/lm_training_rank for a low-rank analysis workflow.
