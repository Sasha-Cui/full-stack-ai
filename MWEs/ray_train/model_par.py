from __future__ import annotations

import argparse
import itertools


def create_pipeline_model(num_stages: int = 2):
    import torch.nn as nn
    from deepspeed.pipe import LayerSpec, PipelineModule

    layers = [
        LayerSpec(nn.Linear, 128, 256),
        LayerSpec(nn.ReLU),
        LayerSpec(nn.Linear, 256, 512),
        LayerSpec(nn.ReLU),
        LayerSpec(nn.Linear, 512, 256),
        LayerSpec(nn.ReLU),
        LayerSpec(nn.Linear, 256, 128),
        LayerSpec(nn.ReLU),
        LayerSpec(nn.Linear, 128, 10),
    ]

    return PipelineModule(
        layers=layers,
        num_stages=num_stages,
        loss_fn=nn.CrossEntropyLoss(),
        partition_method="uniform",
        activation_checkpoint_interval=0,
    )


def train_func(config: dict) -> None:
    import deepspeed
    import deepspeed.comm as dist_comm
    import torch
    import torch.distributed as dist
    from ray import train
    from torch.utils.data import DataLoader, TensorDataset

    context = train.get_context()
    world_rank = context.get_world_rank()
    world_size = context.get_world_size()
    local_rank = context.get_local_rank()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            world_size=world_size,
            rank=world_rank,
        )

    if not dist_comm.is_initialized():
        dist_comm.init_distributed(
            dist_backend="nccl" if torch.cuda.is_available() else "gloo",
            auto_mpi_discovery=False,
            distributed_port=29500,
            verbose=False,
        )

    num_stages = config["num_stages"]
    model = create_pipeline_model(num_stages=num_stages)

    deepspeed_config = config["deepspeed_config"].copy()
    micro_batch_size = deepspeed_config["train_micro_batch_size_per_gpu"]
    gradient_accumulation_steps = deepspeed_config["gradient_accumulation_steps"]
    deepspeed_config["train_batch_size"] = (
        micro_batch_size * gradient_accumulation_steps
    )

    engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=deepspeed_config,
        dist_init_required=False,
    )

    inputs = torch.randn(320, 128)
    targets = torch.randint(0, 10, (320,))
    dataloader = DataLoader(
        TensorDataset(inputs, targets),
        batch_size=micro_batch_size,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(config["epochs"]):
        engine.train()
        total_loss = 0.0
        num_batches = 0
        data_cycle = itertools.cycle(dataloader)
        steps_per_epoch = max(1, len(dataloader) // gradient_accumulation_steps)

        for _ in range(steps_per_epoch):
            batch_inputs, batch_targets = next(data_cycle)

            def step_data_iter():
                for _ in range(gradient_accumulation_steps):
                    yield (batch_inputs.cpu(), batch_targets.cpu())

            loss = engine.train_batch(data_iter=step_data_iter())
            if loss is not None:
                total_loss += float(loss)
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        learning_rates = engine.get_lr()
        train.report(
            metrics={
                "epoch": epoch + 1,
                "loss": avg_loss,
                "learning_rate": learning_rates[0] if learning_rates else 0.0,
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ray Train + DeepSpeed pipeline-parallel example."
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    import ray
    import torch
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This example requires CUDA GPUs for pipeline parallelism.")
    if torch.cuda.device_count() < args.num_workers:
        raise SystemExit(
            f"Requested {args.num_workers} workers, but only "
            f"{torch.cuda.device_count()} CUDA devices are available."
        )

    deepspeed_config = {
        "train_micro_batch_size_per_gpu": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimizer": {"type": "Adam", "params": {"lr": 1e-3, "weight_decay": 0.01}},
        "fp16": {"enabled": False},
        "zero_optimization": {
            "stage": 1,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 2e8,
            "reduce_bucket_size": 2e8,
        },
        "steps_per_print": 5,
        "wall_clock_breakdown": False,
        "pipeline": {"pipe_partitioned": True, "grad_partitioned": True},
    }

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "epochs": args.epochs,
            "num_stages": args.num_workers,
            "deepspeed_config": deepspeed_config,
        },
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=True,
            resources_per_worker={"GPU": 1},
        ),
        run_config=RunConfig(),
    )

    ray.init(ignore_reinit_error=True, num_gpus=args.num_workers)
    try:
        result = trainer.fit()
        print(f"Pipeline-parallel training finished with metrics: {result.metrics}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
