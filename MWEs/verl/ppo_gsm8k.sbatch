#!/bin/bash
#SBATCH --job-name=verl_ppo
#SBATCH --partition=test
#SBATCH --nodelist=misha02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/diskarray/home/%u/verl_job_%j.log
#SBATCH --error=/diskarray/home/%u/verl_job_%j.err

# --- 1. Environment Setup ---
export HF_HOME="/diskarray/data/$USER/hf_cache"
export PYTHONDONTWRITEBYTECODE=1
export PROJECT_DIR="/diskarray/home/$USER/verl_tutorial"

# Define the variables used in your command
export verl_tutorial="verl_tutorial"
export qwen25_ppo_gsm8k="qwen25_ppo_gsm8k"

# --- 2. Run Exact Command ---
cd $PROJECT_DIR

apptainer exec \
    --nv \
    --fakeroot \
    --overlay /tmp/verl_overlay.img \
    --bind /diskarray:/diskarray \
    --bind /tmp:/tmp \
    --bind $HOME:$HOME \
    verl_vllm.sif \
    bash -c "
    unset ROCR_VISIBLE_DEVICES
    export HF_HOME=/diskarray/home/$USER/verl_tutorial
    PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=\$HOME/data/gsm8k/train.parquet \
    data.val_files=\$HOME/data/gsm8k/test.parquet \
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
    trainer.logger='[\"console\",\"wandb\"]' \
    trainer.project_name=\$verl_tutorial \
    trainer.experiment_name=\$qwen25_ppo_gsm8k \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 2>&1 | tee verl_demo.log
    "