import torch
import torch.nn as nn
import torch.distributed as dist
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
import ray
import ray.train as train
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer
from torch.utils.data import DataLoader, TensorDataset
import itertools


# -----------------------------
# 1. Define pipeline-parallel model
# -----------------------------
def create_pipeline_model(num_stages=2):
    """Create a pipeline model with proper layer specifications"""
    # Split the model into a sequence of layers
    layers = [
        LayerSpec(nn.Linear, 128, 256),  # Stage 0
        LayerSpec(nn.ReLU),  # Stage 0
        LayerSpec(nn.Linear, 256, 512),  # Stage 0
        LayerSpec(nn.ReLU),  # Stage 0
        LayerSpec(nn.Linear, 512, 256),  # Stage 1
        LayerSpec(nn.ReLU),  # Stage 1
        LayerSpec(nn.Linear, 256, 128),  # Stage 1
        LayerSpec(nn.ReLU),  # Stage 1
        LayerSpec(nn.Linear, 128, 10),  # Stage 1 (output layer)
    ]

    # Pipeline with specified number of stages
    model = PipelineModule(
        layers=layers,
        num_stages=num_stages,
        loss_fn=nn.CrossEntropyLoss(),
        partition_method="uniform",  # split layers evenly across stages
        activation_checkpoint_interval=0,
    )
    return model


# -----------------------------
# 2. Ray Train worker function
# -----------------------------
def train_func(config):
    """Training function for pipeline parallel model"""
    # Get Ray Train context
    context = train.get_context()
    world_rank = context.get_world_rank()
    world_size = context.get_world_size()
    local_rank = context.get_local_rank()

    print(f"Worker {world_rank}/{world_size}, Local rank: {local_rank}")

    # Set device
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(local_rank)

    # Initialize DeepSpeed distributed backend explicitly for PipelineModule
    if not dist.is_initialized():
        # Initialize with the same backend Ray Train uses
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            world_size=world_size,
            rank=world_rank,
        )
        print(f"Initialized distributed backend on rank {world_rank}")

    # Initialize DeepSpeed's distributed backend
    import deepspeed.comm as dist_comm

    if not dist_comm.is_initialized():
        dist_comm.init_distributed(
            dist_backend="nccl" if torch.cuda.is_available() else "gloo",
            auto_mpi_discovery=False,
            distributed_port=29500,
            verbose=False,
        )
        print(f"DeepSpeed distributed backend initialized on rank {world_rank}")

    # Create pipeline model
    num_stages = config.get("num_stages", 2)
    model = create_pipeline_model(num_stages=num_stages)

    print(f"Created pipeline model with {num_stages} stages")

    # Create proper DeepSpeed configuration with batch size constraints
    deepspeed_config = config["deepspeed_config"].copy()

    # Fix batch size configuration for pipeline parallelism
    micro_batch_size = deepspeed_config.get("train_micro_batch_size_per_gpu", 8)
    gradient_accumulation_steps = deepspeed_config.get("gradient_accumulation_steps", 2)

    # For pipeline parallelism, train_batch_size should be:
    # micro_batch_size * gradient_accumulation_steps * data_parallel_size
    train_batch_size = micro_batch_size * gradient_accumulation_steps

    deepspeed_config.update(
        {
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        }
    )

    # Initialize DeepSpeed with pipeline model
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=deepspeed_config,
        dist_init_required=False,
    )

    print(f"DeepSpeed initialized on rank {world_rank}")

    # Create dataset and dataloader
    num_samples = 320
    input_size = 128
    num_classes = 10

    # Generate dummy data
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(
        dataset, batch_size=micro_batch_size, shuffle=True, drop_last=True
    )

    # Training loop
    num_epochs = config.get("epochs", 3)

    for epoch in range(num_epochs):
        engine.train()
        total_loss = 0.0
        num_batches = 0

        # Create a cycle iterator to avoid StopIteration
        data_cycle = itertools.cycle(dataloader)

        steps_per_epoch = len(dataloader) // gradient_accumulation_steps

        if world_rank == 0:
            print(
                f"Starting epoch {epoch + 1}/{num_epochs} with {steps_per_epoch} steps"
            )

        for step in range(steps_per_epoch):
            # Get a batch from the cycled dataloader
            data, target = next(data_cycle)

            # Move data to CPU for pipeline
            if data.is_cuda:
                data = data.cpu()
            if target.is_cuda:
                target = target.cpu()

            # Create a simple data iterator for this step
            def step_data_iter():
                for _ in range(gradient_accumulation_steps):
                    yield (data, target)

            # Use train_batch for pipeline parallelism
            try:
                loss = engine.train_batch(data_iter=step_data_iter())

                # Accumulate loss
                if loss is not None:
                    total_loss += loss
                    if world_rank == 0 and step % 5 == 0:
                        print(f"Epoch {epoch+1}, Step {step}, Loss: {loss:.4f}")

                num_batches += 1

            except Exception as e:
                if world_rank == 0:
                    print(f"Error in step {step}: {e}")
                continue

        avg_loss = total_loss / max(num_batches, 1) if num_batches > 0 else 0.0

        if world_rank == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs} completed, Average Loss: {avg_loss:.4f}"
            )

        # Report metrics to Ray Train
        train.report(
            metrics={
                "loss": float(avg_loss) if avg_loss is not None else 0.0,
                "epoch": epoch,
                "learning_rate": engine.get_lr()[0] if engine.get_lr() else 0.0,
            }
        )

    print(f"Pipeline training completed on rank {world_rank}")


# -----------------------------
# 3. Main: Run with Ray
# -----------------------------
if __name__ == "__main__":
    ray.init(num_gpus=2)
    print("Ray initialized for pipeline-parallel training")

    # DeepSpeed config for pipeline parallelism + ZeRO stage 1
    deepspeed_config = {
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 2,
        "optimizer": {
            "type": "Adam",
            "params": {"lr": 1e-3, "weight_decay": 0.01},
        },
        "fp16": {"enabled": False},  # Keep FP32 for simplicity and stability
        "zero_optimization": {
            "stage": 1,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 2e8,
            "reduce_bucket_size": 2e8,
        },
        "steps_per_print": 5,
        "wall_clock_breakdown": False,
        # Pipeline-specific configurations
        "pipeline": {
            "pipe_partitioned": True,
            "grad_partitioned": True,
        },
    }

    # Training configuration
    train_config = {
        "epochs": 2,  # Reduced epochs for demo
        "num_stages": 2,  # Number of pipeline stages (should match num_workers)
        "deepspeed_config": deepspeed_config,
    }

    # Scaling configuration - one worker per pipeline stage
    scaling_config = ScalingConfig(
        num_workers=2,  # Must match num_stages for pipeline parallelism
        use_gpu=True,
        resources_per_worker={"GPU": 1},  # One GPU per worker/stage
    )

    # Run configuration
    run_config = RunConfig(
        # verbose=2,  # Enable for more detailed logging
    )

    # Create trainer
    trainer = TorchTrainer(
        train_func,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    print("Starting pipeline-parallel training with DeepSpeed...")
    print(f"Pipeline stages: {train_config['num_stages']}")
    print(f"Workers: {scaling_config.num_workers}")

    try:
        result = trainer.fit()
        print("\nPipeline-parallel training completed successfully!")
        print(f"Final metrics: {result.metrics}")

    except Exception as e:
        print(f"Training failed with error: {e}")
        print("Common issues:")
        print("1. Make sure you have at least 2 GPUs available")
        print("2. Ensure DeepSpeed is properly installed")
        print("3. Check that CUDA and NCCL are working correctly")

    finally:
        ray.shutdown()
        print("Ray shut down")
