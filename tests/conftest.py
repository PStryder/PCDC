"""Shared test fixtures for PCDC."""

import pytest
import torch

from pcdc.utils.config import PCConfig


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def small_config(device):
    """Small network config for fast tests."""
    return PCConfig(
        layer_sizes=[10, 8, 5],
        activation="tanh",
        eta_x=0.05,
        eta_w=0.001,
        eta_x_scale="uniform",
        state_norm="none",
        stability_mode="none",
        K=30,
        converge_tol=1e-5,
        device=device,
    )


@pytest.fixture
def mnist_config(device):
    """MNIST-shaped config."""
    return PCConfig(
        layer_sizes=[784, 256, 256, 10],
        activation="tanh",
        eta_x=0.1,
        eta_w=0.001,
        eta_x_scale="uniform",
        state_norm="none",
        stability_mode="none",
        K=20,
        device=device,
    )


@pytest.fixture(autouse=True)
def seed_everything():
    torch.manual_seed(42)
