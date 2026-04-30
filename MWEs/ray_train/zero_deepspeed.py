"""
Minimal Ray Train + DeepSpeed example adapted from:
https://github.com/huggingface/accelerate/blob/main/examples/nlp_example.py

Fine-tune a BERT model with DeepSpeed ZeRO-3 using Ray Train and Ray Data.
This script is intended for Linux + CUDA environments.
"""

from __future__ import annotations

import argparse
from tempfile import TemporaryDirectory


def train_func(config: dict) -> None:
    import deepspeed
    import ray.train
    import torch
    from deepspeed.accelerator import get_accelerator
    from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        set_seed,
    )

    set_seed(config["seed"])
    num_epochs = config["num_epochs"]
    train_batch_size = config["train_batch_size"]
    eval_batch_size = config["eval_batch_size"]
    deepspeed_config = config["deepspeed_config"]

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", return_dict=True
    )

    train_ds = ray.train.get_dataset_shard("train")
    eval_ds = ray.train.get_dataset_shard("validation")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def collate_fn(batch):
        outputs = tokenizer(
            list(batch["sentence1"]),
            list(batch["sentence2"]),
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )
        outputs["labels"] = torch.LongTensor(batch["label"])
        return outputs

    train_dataloader = train_ds.iter_torch_batches(
        batch_size=train_batch_size, collate_fn=collate_fn
    )
    eval_dataloader = eval_ds.iter_torch_batches(
        batch_size=eval_batch_size, collate_fn=collate_fn
    )

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=deepspeed_config,
    )
    device = get_accelerator().device_name(model_engine.local_rank)

    f1 = BinaryF1Score().to(device)
    accuracy = BinaryAccuracy().to(device)

    for epoch in range(num_epochs):
        model_engine.train()
        for batch in train_dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model_engine(**batch)
            loss = outputs.loss
            model_engine.backward(loss)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad()

        model_engine.eval()
        for batch in eval_dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                outputs = model_engine(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            f1.update(predictions, batch["labels"])
            accuracy.update(predictions, batch["labels"])

        eval_metric = {
            "f1": f1.compute().item(),
            "accuracy": accuracy.compute().item(),
            "epoch": epoch + 1,
        }
        f1.reset()
        accuracy.reset()

        if model_engine.global_rank == 0:
            print(f"Epoch {epoch + 1}: {eval_metric}")

        with TemporaryDirectory() as tmpdir:
            model_engine.save_checkpoint(tmpdir)
            torch.distributed.barrier()
            ray.train.report(
                metrics=eval_metric,
                checkpoint=ray.train.Checkpoint.from_directory(tmpdir),
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ray Train + DeepSpeed ZeRO-3 MRPC example."
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a short 1-worker check on a tiny dataset slice.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bfloat16 instead of fp16 (Ampere+ GPUs only).",
    )
    return parser.parse_args()


def limit_split(split, limit: int | None):
    if limit is None:
        return split
    return split.select(range(min(limit, len(split))))


def main() -> None:
    args = parse_args()

    import ray
    import torch
    from datasets import load_dataset
    from ray.train import DataConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    if args.smoke_test:
        args.num_workers = 1
        args.epochs = 1
        args.max_train_samples = args.max_train_samples or 128
        args.max_eval_samples = args.max_eval_samples or 64

    if not torch.cuda.is_available():
        raise SystemExit(
            "This example requires CUDA GPUs and a DeepSpeed-compatible Linux setup."
        )
    if torch.cuda.device_count() < args.num_workers:
        raise SystemExit(
            f"Requested {args.num_workers} GPU workers, but only "
            f"{torch.cuda.device_count()} CUDA devices are available."
        )

    deepspeed_config = {
        "optimizer": {"type": "AdamW", "params": {"lr": 2e-5}},
        "scheduler": {"type": "WarmupLR", "params": {"warmup_num_steps": 100}},
        "fp16": {"enabled": not args.bf16},
        "bf16": {"enabled": args.bf16},
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
        },
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
        "train_micro_batch_size_per_gpu": args.train_batch_size,
        "wall_clock_breakdown": False,
    }

    training_config = {
        "seed": 42,
        "num_epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "deepspeed_config": deepspeed_config,
    }

    hf_datasets = load_dataset("glue", "mrpc")
    train_split = limit_split(hf_datasets["train"], args.max_train_samples)
    eval_split = limit_split(hf_datasets["validation"], args.max_eval_samples)

    ray_datasets = {
        "train": ray.data.from_items([dict(item) for item in train_split]),
        "validation": ray.data.from_items([dict(item) for item in eval_split]),
    }

    trainer = TorchTrainer(
        train_func,
        train_loop_config=training_config,
        scaling_config=ScalingConfig(num_workers=args.num_workers, use_gpu=True),
        datasets=ray_datasets,
        dataset_config=DataConfig(datasets_to_split=["train", "validation"]),
    )

    ray.init(ignore_reinit_error=True)
    try:
        result = trainer.fit()
        print(f"Best metrics: {result.metrics}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
