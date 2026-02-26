"""Train a PCNetwork on MNIST with local learning rules."""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from pcdc.core.pc_network import PCNetwork
from pcdc.utils.config import PCConfig
from pcdc.utils.metrics_logger import EpochMetrics


def get_mnist_loaders(batch_size: int, data_dir: str = "./data") -> tuple[DataLoader, DataLoader]:
    """Load MNIST train and test sets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def to_onehot(labels: torch.Tensor, n_classes: int = 10) -> torch.Tensor:
    """Convert integer labels to one-hot representation."""
    return F.one_hot(labels, n_classes).float()


def train_epoch(
    net: PCNetwork,
    loader: DataLoader,
    device: str,
    epoch: int,
) -> EpochMetrics:
    """Train one epoch with PC local learning."""
    metrics = EpochMetrics(epoch=epoch)

    for images, labels in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        x_input = images.view(images.size(0), -1).to(device)
        y_target = to_onehot(labels, n_classes=net.config.layer_sizes[-1]).to(device)

        result = net.train_step(x_input, y_target)
        metrics.update(
            batch_energy=result["energy"],
            batch_correct=result["correct"],
            batch_total=result["total"],
            settle_steps=result["settle_steps"],
            oscillations=result.get("oscillations", 0),
            stability_failure=result.get("stability_failure", False),
        )

    return metrics


@torch.no_grad()
def evaluate(net: PCNetwork, loader: DataLoader, device: str) -> float:
    """Evaluate PCNetwork accuracy via inference settling."""
    correct = 0
    total = 0

    for images, labels in loader:
        x_input = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)

        pred, _ = net.infer(x_input)
        pred_labels = pred.argmax(dim=-1)
        correct += (pred_labels == labels).sum().item()
        total += images.size(0)

    return correct / total


def main():
    parser = argparse.ArgumentParser(description="Train PCNetwork on MNIST")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eta-x", type=float, default=0.1)
    parser.add_argument("--eta-w", type=float, default=0.001)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--layers", type=int, nargs="+", default=[784, 256, 256, 10])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    config = PCConfig(
        layer_sizes=args.layers,
        activation=args.activation,
        eta_x=args.eta_x,
        eta_w=args.eta_w,
        K=args.K,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
    )

    net = PCNetwork(config).to(args.device)
    train_loader, test_loader = get_mnist_loaders(args.batch_size)

    print(f"PCNetwork: {args.layers}, activation={args.activation}")
    print(f"η_x={args.eta_x}, η_w={args.eta_w}, K={args.K}")
    print(f"Device: {args.device}")
    print()

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        epoch_metrics = train_epoch(net, train_loader, args.device, epoch)
        val_acc = evaluate(net, test_loader, args.device)
        epoch_metrics.val_accuracy = val_acc
        best_val = max(best_val, val_acc)

        print(epoch_metrics.summary())

    print(f"\nBest validation accuracy: {best_val:.4f}")


if __name__ == "__main__":
    main()
