from __future__ import annotations

import argparse
import os
import tempfile
import uuid
from pathlib import Path

import ray
import ray.train
import torch
from filelock import FileLock
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import VisionTransformer
from torchvision.transforms import Normalize, ToTensor
from tqdm import tqdm


class SmallCNN(nn.Module):
    """A lightweight CNN that trains quickly on CPU for smoke tests."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ray Train data-parallel CIFAR-10 example."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of Ray workers to launch.",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs."
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=128,
        help="Global batch size divided evenly across workers.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="~/data",
        help="Directory used for CIFAR-10 downloads.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Optional cap on the number of train and validation examples.",
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "vit"],
        default="cnn",
        help="Model architecture to train. Use 'cnn' for quick local runs.",
    )
    parser.add_argument(
        "--use-fake-data",
        action="store_true",
        help="Use torchvision FakeData instead of downloading CIFAR-10.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast local check with FakeData and a tiny subset.",
    )
    parser.add_argument(
        "--use-gpu",
        dest="use_gpu",
        action="store_true",
        help="Train on GPUs if available.",
    )
    parser.add_argument(
        "--cpu",
        dest="use_gpu",
        action="store_false",
        help="Force CPU training even if CUDA is available.",
    )
    parser.set_defaults(use_gpu=torch.cuda.is_available())
    return parser.parse_args()


def build_model(model_name: str) -> nn.Module:
    if model_name == "vit":
        return VisionTransformer(
            image_size=32,
            patch_size=4,
            num_layers=6,
            num_heads=4,
            hidden_dim=256,
            mlp_dim=512,
            num_classes=10,
        )
    return SmallCNN(num_classes=10)


def maybe_subset(dataset, subset_size: int | None):
    if subset_size is None:
        return dataset
    capped_size = min(subset_size, len(dataset))
    return Subset(dataset, range(capped_size))


def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    batch_size = config["batch_size_per_worker"]
    subset_size = config.get("subset_size")
    use_fake_data = config.get("use_fake_data", False)
    transform = transforms.Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    if use_fake_data:
        train_dataset = datasets.FakeData(
            size=subset_size or 512,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform,
        )
        valid_dataset = datasets.FakeData(
            size=max(128, min(subset_size or 256, 512)),
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform,
        )
    else:
        data_root = Path(config["data_root"]).expanduser()
        data_root.mkdir(parents=True, exist_ok=True)
        lock_path = data_root.with_suffix(".lock")
        with FileLock(str(lock_path)):
            train_dataset = datasets.CIFAR10(
                root=str(data_root),
                train=True,
                download=True,
                transform=transform,
            )
            valid_dataset = datasets.CIFAR10(
                root=str(data_root),
                train=False,
                download=True,
                transform=transform,
            )
        train_dataset = maybe_subset(train_dataset, subset_size)
        valid_dataset = maybe_subset(valid_dataset, subset_size)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
    return train_dataloader, valid_dataloader


def train_func_per_worker(config: dict) -> None:
    lr = config["lr"]
    epochs = config["epochs"]

    train_dataloader, valid_dataloader = get_dataloaders(config)
    train_dataloader = ray.train.torch.prepare_data_loader(train_dataloader)
    valid_dataloader = ray.train.torch.prepare_data_loader(valid_dataloader)

    model = build_model(config["model_name"])
    model = ray.train.torch.prepare_model(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    for epoch in range(epochs):
        if ray.train.get_context().get_world_size() > 1 and hasattr(
            train_dataloader.sampler, "set_epoch"
        ):
            train_dataloader.sampler.set_epoch(epoch)

        model.train()
        for inputs, targets in tqdm(train_dataloader, desc=f"Train epoch {epoch + 1}"):
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        valid_loss = 0.0
        num_correct = 0
        num_total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(
                valid_dataloader, desc=f"Validate epoch {epoch + 1}"
            ):
                predictions = model(inputs)
                loss = loss_fn(predictions, targets)

                valid_loss += loss.item()
                num_total += targets.shape[0]
                num_correct += (predictions.argmax(1) == targets).sum().item()

        valid_loss /= max(len(valid_dataloader), 1)
        accuracy = num_correct / max(num_total, 1)

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(
                model_to_save.state_dict(),
                os.path.join(temp_checkpoint_dir, "model.pt"),
            )
            ray.train.report(
                metrics={"loss": valid_loss, "accuracy": accuracy, "epoch": epoch + 1},
                checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
            )


def train_cifar_10(args: argparse.Namespace):
    if args.smoke_test:
        args.use_fake_data = True
        args.model = "cnn"
        args.subset_size = args.subset_size or 256
        args.num_workers = 1
        args.use_gpu = False

    if args.num_workers < 1:
        raise SystemExit("--num-workers must be at least 1.")
    if args.global_batch_size < args.num_workers:
        raise SystemExit("--global-batch-size must be >= --num-workers.")
    if args.use_gpu and torch.cuda.device_count() < args.num_workers:
        raise SystemExit(
            f"Requested {args.num_workers} GPU workers, but only "
            f"{torch.cuda.device_count()} CUDA devices are available."
        )

    train_config = {
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size_per_worker": max(1, args.global_batch_size // args.num_workers),
        "data_root": args.data_root,
        "subset_size": args.subset_size,
        "use_fake_data": args.use_fake_data,
        "model_name": args.model,
    }

    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=args.use_gpu,
        ),
        run_config=RunConfig(name=f"train-cifar-{uuid.uuid4().hex[:8]}"),
    )
    return trainer.fit()


def main() -> None:
    args = parse_args()
    ray.init(ignore_reinit_error=True)
    try:
        result = train_cifar_10(args)
        print(f"Training finished with metrics: {result.metrics}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
