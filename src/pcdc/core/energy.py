"""Energy computation and convergence checking for PC networks."""

from __future__ import annotations

import torch
from torch import Tensor


def compute_energy(errors: list[Tensor]) -> Tensor:
    """Compute total predictive coding energy (batch mean).

    E = 0.5 * Σ_l mean_over_batch( ||e[l]||² )

    Args:
        errors: List of error tensors e[l], each shape (B, D_l).

    Returns:
        Scalar energy tensor.
    """
    return 0.5 * sum(e.pow(2).sum(dim=-1).mean() for e in errors)


def compute_per_example_energy(errors: list[Tensor]) -> Tensor:
    """Compute energy per example in the batch.

    E_b = 0.5 * Σ_l ||e[l]_b||²

    Args:
        errors: List of error tensors e[l], each shape (B, D_l).

    Returns:
        Tensor of shape (B,) with per-example energy.
    """
    return 0.5 * sum(e.pow(2).sum(dim=-1) for e in errors)


def compute_layer_mse(errors: list[Tensor]) -> list[float]:
    """Compute mean squared error per layer.

    Returns:
        List of floats, one per error layer.
    """
    return [e.pow(2).mean().item() for e in errors]


def check_convergence(
    energy_prev: float,
    energy_curr: float,
    tol: float,
) -> bool:
    """Check if relative energy change is below tolerance.

    Returns True if |ΔE/E| < tol, with safe handling of near-zero energies.
    """
    abs_change = abs(energy_curr - energy_prev)
    if abs_change < 1e-12:
        return True
    return abs_change / (abs(energy_prev) + 1e-8) < tol
