"""Baseline models for continual learning comparison.

Linear probe and SGD MLP operating on the same frozen features
as the PC head, providing fair comparison targets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


class LinearProbe(nn.Module):
    """Single linear layer classifier on frozen features."""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class SGDBaseline(nn.Module):
    """Small MLP trained with SGD/Adam on frozen features."""

    def __init__(self, feature_dim: int, num_classes: int, hidden_sizes: list[int] | None = None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256]

        layers = []
        prev = feature_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ReplayBuffer:
    """Reservoir sampling replay buffer for continual learning."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.features: list[Tensor] = []
        self.labels: list[int] = []
        self._count = 0

    def add(self, features: Tensor, labels: Tensor):
        """Add examples using reservoir sampling."""
        for i in range(features.shape[0]):
            self._count += 1
            if len(self.features) < self.max_size:
                self.features.append(features[i].cpu())
                self.labels.append(labels[i].item())
            else:
                # Reservoir sampling: replace with probability max_size/count
                j = torch.randint(0, self._count, (1,)).item()
                if j < self.max_size:
                    self.features[j] = features[i].cpu()
                    self.labels[j] = labels[i].item()

    def sample(self, n: int) -> tuple[Tensor, Tensor] | None:
        """Sample n examples from the buffer."""
        if len(self.features) == 0:
            return None
        n = min(n, len(self.features))
        indices = torch.randperm(len(self.features))[:n]
        feats = torch.stack([self.features[i] for i in indices])
        labs = torch.tensor([self.labels[i] for i in indices], dtype=torch.long)
        return feats, labs

    def __len__(self) -> int:
        return len(self.features)


def train_baseline_on_features(
    model: nn.Module,
    features: Tensor,
    labels: Tensor,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 0.001,
    device: str = "cuda",
    replay_buffer: ReplayBuffer | None = None,
    replay_mix: float = 0.25,
    prev_weights: dict | None = None,
    l2_prev: float = 0.0,
) -> list[float]:
    """Train a baseline model on pre-extracted features.

    Args:
        model: LinearProbe or SGDBaseline.
        features: (N, D) feature tensor.
        labels: (N,) integer labels.
        epochs: Training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        device: Device.
        replay_buffer: Optional replay buffer for continual learning.
        replay_mix: Fraction of batch from replay.
        prev_weights: Previous task weights for L2 regularization.
        l2_prev: L2 penalty weight toward previous weights.

    Returns:
        List of per-epoch losses.
    """
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n = 0
        for batch_feats, batch_labels in loader:
            batch_feats = batch_feats.to(device)
            batch_labels = batch_labels.to(device)

            # Mix in replay examples
            if replay_buffer is not None and len(replay_buffer) > 0:
                replay_n = int(batch_size * replay_mix)
                replay_data = replay_buffer.sample(replay_n)
                if replay_data is not None:
                    r_feats, r_labels = replay_data
                    batch_feats = torch.cat([batch_feats, r_feats.to(device)])
                    batch_labels = torch.cat([batch_labels, r_labels.to(device)])

            logits = model(batch_feats)
            loss = F.cross_entropy(logits, batch_labels)

            # L2 regularization toward previous weights
            if prev_weights is not None and l2_prev > 0:
                reg = sum(
                    (p - prev_weights[name].to(device)).pow(2).sum()
                    for name, p in model.named_parameters()
                    if name in prev_weights
                )
                loss = loss + l2_prev * reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_feats.size(0)
            n += batch_feats.size(0)

        losses.append(epoch_loss / n)

    return losses


@torch.no_grad()
def eval_baseline_on_features(
    model: nn.Module,
    features: Tensor,
    labels: Tensor,
    device: str = "cuda",
    batch_size: int = 256,
) -> float:
    """Evaluate a baseline model on features."""
    model.eval()
    model = model.to(device)

    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    for feats, labs in loader:
        feats = feats.to(device)
        labs = labs.to(device)
        logits = model(feats)
        correct += (logits.argmax(dim=-1) == labs).sum().item()
        total += labs.size(0)

    return correct / total if total > 0 else 0.0
