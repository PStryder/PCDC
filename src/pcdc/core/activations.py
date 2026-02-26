"""Activation functions and their analytical derivatives for PC dynamics."""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

# Type alias: (forward_fn, derivative_fn)
ActivationPair = tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]


def _tanh(x: Tensor) -> Tensor:
    return torch.tanh(x)


def _tanh_prime(preact: Tensor) -> Tensor:
    """Derivative of tanh evaluated at pre-activation values."""
    t = torch.tanh(preact)
    return 1.0 - t * t


def _relu(x: Tensor) -> Tensor:
    return F.relu(x)


def _relu_prime(preact: Tensor) -> Tensor:
    return (preact > 0).float()


def _gelu(x: Tensor) -> Tensor:
    return F.gelu(x)


def _gelu_prime(preact: Tensor) -> Tensor:
    """Exact GELU derivative using the erf formulation (matches PyTorch's F.gelu)."""
    # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    # GELU'(x) = 0.5 * (1 + erf(x/sqrt(2))) + x * exp(-x^2/2) / sqrt(2*pi)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    inv_sqrt2pi = 1.0 / math.sqrt(2.0 * math.pi)
    erf_term = torch.erf(preact * inv_sqrt2)
    gauss_term = torch.exp(-0.5 * preact.pow(2))
    return 0.5 * (1.0 + erf_term) + preact * gauss_term * inv_sqrt2pi


def _linear(x: Tensor) -> Tensor:
    return x


def _linear_prime(preact: Tensor) -> Tensor:
    return torch.ones_like(preact)


_REGISTRY: dict[str, ActivationPair] = {
    "tanh": (_tanh, _tanh_prime),
    "relu": (_relu, _relu_prime),
    "gelu": (_gelu, _gelu_prime),
    "linear": (_linear, _linear_prime),
}


def get_activation(name: str) -> ActivationPair:
    """Return (f, f_prime) pair for the named activation.

    f(x) applies the activation.
    f_prime(preact) returns the derivative evaluated at pre-activation values.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {list(_REGISTRY)}")
    return _REGISTRY[name]
