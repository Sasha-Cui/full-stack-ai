# verl training tutorial
### 1. pull the container and set up overlay
```
apptainer pull verl_vllm.sif docker://verlai/verl:vegalita-vllm
apptainer shell --nv --bind /diskarray:/diskarray --bind $HOME:$HOME verl_vllm.sif
cd /tmp; apptainer overlay create --fakeroot --size 5120 verl_overlay.img; cd -
```
### 2. start the container
```
apptainer shell \
    --nv \
    --fakeroot \
    --overlay /tmp/verl_overlay.img \
    --bind /diskarray:/diskarray \
    --bind /tmp:/tmp \
    /diskarray/home/$USER/verl_tutorial/verl_vllm.sif
```
### 3. install necessary packages; set up wandb and huggingface
```
git clone https://github.com/volcengine/verl && cd verl
pip3 install --no-deps -e .
wandb login
export HF_HOME=/diskarray/home/$USER/verl_tutorial
```
### 4. Download dataset and model
```
python3 -B examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
python3 -B -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-0.5B-Instruct')"
```
### 5. Train with PPO
```
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=$HOME/data/gsm8k/train.parquet \
 data.val_files=$HOME/data/gsm8k/test.parquet \
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
 trainer.project_name=$verl_tutorial \
 trainer.experiment_name=$qwen25_ppo_gsm8k \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_epochs=15 2>&1 | tee verl_demo.
```
see training results on [wandb](https://api.wandb.ai/links/xingzhis/ece56sx3)
```
(TaskRunner pid=3915380) ("Final validation metrics: {'val-aux/openai/gsm8k/reward/mean@1': "
(TaskRunner pid=3915380)  "0.5595147839272175, 'val-core/openai/gsm8k/acc/mean@1': 0.5595147839272175, "
```
### 6. Merge trained checkpoint to huggingface model
```
python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/verl_tutorial/qwen25_ppo_gsm8k/global_step_435/actor \
    --target_dir checkpoints/verl_tutorial/qwen25_ppo_gsm8k/global_step_435/actor/huggingface
```
### 7. evaluate model; compare with baseline model
trained model:
```
python evaluate_gsm8k.py --model_path ./checkpoints/verl_tutorial/qwen25_ppo_gsm8k/global_step_435/actor/huggingface --data_path ~/data/gsm8k/test.parquet --output trained_outputs.jsonl
```
output:
```
============================================================
Model: ./checkpoints/verl_tutorial/qwen25_ppo_gsm8k/global_step_435/actor/huggingface
Total samples: 1319

  STRICT  (verl training):     704/1319  = 53.37%
  FLEXIBLE (lm-eval-harness):  704/1319  = 53.37%
============================================================
```
baseline:
```
python evaluate_gsm8k.py --model_path Qwen/Qwen2.5-0.5B-Instruct --data_path ~/data/gsm8k/test.parquet --output base_outputs.jsonl
```
output:
```
============================================================
Model: Qwen/Qwen2.5-0.5B-Instruct
Total samples: 1319

  STRICT  (verl training):       4/1319  =  0.30%
  FLEXIBLE (lm-eval-harness):  455/1319  = 34.50%
============================================================
```
### 9. case-study of the results
```
python compare_results.py --base base_outputs.jsonl --trained trained_outputs.jsonl -n 5
```
10. (optional) verify low-rank property of RL training.
see this [repo](https://github.com/xingzhis/lm_training_rank).
