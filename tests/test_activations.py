"""Tests for activation functions and their derivatives."""

import pytest
import torch

from pcdc.core.activations import get_activation


@pytest.mark.parametrize("name", ["tanh", "relu", "gelu", "linear"])
def test_activation_output_shape(name):
    f, f_prime = get_activation(name)
    x = torch.randn(4, 8)
    assert f(x).shape == (4, 8)
    assert f_prime(x).shape == (4, 8)


@pytest.mark.parametrize("name", ["tanh", "gelu", "linear"])
def test_derivative_matches_autograd(name):
    """Verify analytical derivative matches autograd."""
    _, f_prime_analytic = get_activation(name)
    f, _ = get_activation(name)

    x = torch.randn(16, requires_grad=True)
    y = f(x).sum()
    y.backward()
    autograd_deriv = x.grad.clone()

    analytic_deriv = f_prime_analytic(x.detach())

    torch.testing.assert_close(analytic_deriv, autograd_deriv, atol=1e-5, rtol=1e-4)


def test_relu_derivative():
    _, f_prime = get_activation("relu")
    x = torch.tensor([-2.0, -0.5, 0.5, 2.0])
    expected = torch.tensor([0.0, 0.0, 1.0, 1.0])
    torch.testing.assert_close(f_prime(x), expected)


def test_relu_dead_units():
    """§1: Dead ReLU units must NOT propagate error feedback."""
    _, f_prime = get_activation("relu")
    # Negative preactivations should produce zero derivative
    preact = torch.tensor([-5.0, -0.1, 0.0, 0.1, 5.0])
    deriv = f_prime(preact)
    # Negative values must be exactly 0
    assert deriv[0].item() == 0.0
    assert deriv[1].item() == 0.0
    # Positive values must be exactly 1
    assert deriv[3].item() == 1.0
    assert deriv[4].item() == 1.0


def test_tanh_derivative_values():
    """§1: tanh derivative at known points."""
    _, f_prime = get_activation("tanh")
    # At 0: f'(0) = 1 - tanh(0)^2 = 1
    assert abs(f_prime(torch.tensor([0.0])).item() - 1.0) < 1e-7
    # At large values: f'(x) -> 0
    assert f_prime(torch.tensor([10.0])).item() < 1e-5


@pytest.mark.parametrize("name", ["tanh", "relu", "gelu", "linear"])
def test_derivative_shape_matches_input(name):
    """§1: f'(preact) must have exact same shape as preact for all activations."""
    _, f_prime = get_activation(name)
    for shape in [(4,), (4, 8), (2, 3, 5)]:
        preact = torch.randn(shape)
        deriv = f_prime(preact)
        assert deriv.shape == preact.shape, f"{name}: expected {preact.shape}, got {deriv.shape}"


def test_unknown_activation_raises():
    with pytest.raises(ValueError, match="Unknown activation"):
        get_activation("softmax")
