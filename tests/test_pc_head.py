"""Tests for PCHead (thin wrapper over PCNetwork for frozen features)."""

import torch

from pcdc.gguf.pc_head import PCHead


def test_pchead_layer_sizes():
    head = PCHead(feature_dim=128, num_classes=10, hidden_sizes=[64, 32])
    assert head.config.layer_sizes == [128, 64, 32, 10]


def test_pchead_train_step():
    head = PCHead(feature_dim=32, num_classes=5, hidden_sizes=[16], eta_x=0.05, K=10)
    features = torch.randn(8, 32)
    targets = torch.zeros(8, 5)
    targets[range(8), torch.randint(0, 5, (8,))] = 1.0
    result = head.train_step(features, targets)
    assert result["total"] == 8


def test_pchead_infer():
    head = PCHead(feature_dim=32, num_classes=5, hidden_sizes=[16], eta_x=0.05, K=10)
    features = torch.randn(8, 32)
    pred, metrics = head.infer(features)
    assert pred.shape == (8, 5)
