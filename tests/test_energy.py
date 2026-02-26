"""Tests for energy computation and convergence checking."""

import torch

from pcdc.core.energy import (
    check_convergence,
    compute_energy,
    compute_layer_mse,
    compute_per_example_energy,
)


def test_compute_energy_zeros():
    errors = [torch.zeros(4, 8), torch.zeros(4, 5)]
    assert compute_energy(errors).item() == 0.0


def test_compute_energy_known():
    # Single layer, batch=1, e = [1, 1, 1, 1] -> ||e||^2 = 4, E = 0.5 * 4 = 2.0
    errors = [torch.ones(1, 4)]
    assert abs(compute_energy(errors).item() - 2.0) < 1e-6


def test_compute_energy_multi_layer():
    e0 = torch.ones(2, 3)  # ||e||^2 = 3 per sample, mean = 3
    e1 = torch.ones(2, 4) * 2  # ||e||^2 = 16 per sample, mean = 16
    # E = 0.5 * (3 + 16) = 9.5
    energy = compute_energy([e0, e1]).item()
    assert abs(energy - 9.5) < 1e-5


def test_per_example_energy_shape():
    """§5: per-example energy should return (B,) tensor."""
    e0 = torch.randn(4, 8)
    e1 = torch.randn(4, 5)
    per_ex = compute_per_example_energy([e0, e1])
    assert per_ex.shape == (4,)
    assert (per_ex >= 0).all()


def test_per_example_energy_matches_batch_mean():
    """Per-example energy averaged should equal batch energy."""
    e0 = torch.randn(8, 10)
    e1 = torch.randn(8, 5)
    batch_energy = compute_energy([e0, e1]).item()
    per_ex = compute_per_example_energy([e0, e1])
    # Batch energy = mean of per-example, since compute_energy uses mean
    assert abs(per_ex.mean().item() - batch_energy) < 1e-5


def test_layer_mse():
    e0 = torch.ones(2, 3)  # MSE = 1.0
    e1 = torch.ones(2, 4) * 2  # MSE = 4.0
    mse = compute_layer_mse([e0, e1])
    assert abs(mse[0] - 1.0) < 1e-6
    assert abs(mse[1] - 4.0) < 1e-6


def test_convergence_below_threshold():
    assert check_convergence(10.0, 10.0001, tol=0.001)


def test_convergence_above_threshold():
    assert not check_convergence(10.0, 12.0, tol=0.001)


def test_convergence_zero_energy():
    assert check_convergence(0.0, 0.0, tol=0.001)
