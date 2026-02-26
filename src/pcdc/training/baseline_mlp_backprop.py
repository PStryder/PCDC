"""Standard backprop MLP baseline for comparison with PCNetwork."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader


class BaselineMLP(nn.Module):
    """Simple feedforward MLP trained with backprop."""

    def __init__(self, layer_sizes: list[int], activation: str = "tanh"):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # no activation after last layer
                if activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "gelu":
                    layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def train_baseline_epoch(
    model: BaselineMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> dict:
    """Train one epoch of baseline MLP."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += images.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def eval_baseline(
    model: BaselineMLP,
    loader: DataLoader,
    device: str,
) -> dict:
    """Evaluate baseline MLP."""
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)

        logits = model(images)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += images.size(0)

    return {"accuracy": correct / total}
