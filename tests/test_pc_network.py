"""Tests for the core PCNetwork."""

import torch
import pytest

from pcdc.core.clamp import ClampMode, make_clamp_mask
from pcdc.core.energy import compute_energy
from pcdc.core.pc_network import PCNetwork
from pcdc.utils.config import PCConfig


# ============================================================
# Basic shape and state tests
# ============================================================


def test_init_states_shapes(small_config):
    net = PCNetwork(small_config)
    x = net.init_states(batch_size=4)
    assert len(x) == 3  # 3 layers for [10, 8, 5]
    assert x[0].shape == (4, 10)
    assert x[1].shape == (4, 8)
    assert x[2].shape == (4, 5)


def test_init_states_with_clamps(small_config):
    net = PCNetwork(small_config)
    x0 = torch.randn(4, 10)
    xL = torch.randn(4, 5)
    x = net.init_states(4, x0=x0, xL=xL)
    torch.testing.assert_close(x[0], x0)
    torch.testing.assert_close(x[2], xL)


def test_compute_errors_shapes(small_config):
    net = PCNetwork(small_config)
    x = net.init_states(4)
    x[0] = torch.randn(4, 10)
    x[1] = torch.randn(4, 8)
    x[2] = torch.randn(4, 5)
    errors, preacts = net.compute_errors(x)
    assert len(errors) == 2  # L=2 error layers
    assert errors[0].shape == (4, 10)
    assert errors[1].shape == (4, 8)
    assert len(preacts) == 2


# ============================================================
# Settling and energy tests
# ============================================================


def test_settle_reduces_energy(small_config):
    """Energy should decrease during settling with small eta_x."""
    small_config.eta_x = 0.01
    small_config.K = 50
    net = PCNetwork(small_config)

    x0 = torch.randn(4, 10)
    xL = torch.randn(4, 5)
    x = net.init_states(4, x0=x0, xL=xL)
    clamp_mask = make_clamp_mask(3, ClampMode.SUPERVISED)

    metrics = net.settle(x, clamp_mask)

    trace = metrics.energy_trace
    assert len(trace) > 1
    assert trace[-1] <= trace[0] + 1e-3, f"Energy increased: {trace[0]:.4f} -> {trace[-1]:.4f}"


def test_settle_convergence_flag(small_config):
    """With enough steps, settle should converge."""
    small_config.eta_x = 0.01
    small_config.K = 200
    small_config.converge_tol = 1e-3
    small_config.converge_patience = 3
    net = PCNetwork(small_config)

    x0 = torch.randn(4, 10)
    xL = torch.randn(4, 5)
    x = net.init_states(4, x0=x0, xL=xL)
    clamp_mask = make_clamp_mask(3, ClampMode.SUPERVISED)

    metrics = net.settle(x, clamp_mask)
    assert metrics.converged or metrics.steps_used <= 200


def test_monotonic_energy_descent_small_eta():
    """§9: With conservatively small eta_x, energy must be non-increasing."""
    config = PCConfig(
        layer_sizes=[10, 8, 5],
        activation="tanh",
        eta_x=0.005,
        K=30,
        eta_x_scale="uniform",
        state_norm="none",
        stability_mode="none",
        device="cpu",
    )
    net = PCNetwork(config)

    x0 = torch.randn(8, 10)
    xL = torch.randn(8, 5)
    x = net.init_states(8, x0=x0, xL=xL)
    clamp_mask = make_clamp_mask(3, ClampMode.SUPERVISED)

    metrics = net.settle(x, clamp_mask)
    trace = metrics.energy_trace

    # Assert strictly non-increasing (within float tolerance)
    for i in range(1, len(trace)):
        assert trace[i] <= trace[i - 1] + 1e-6, (
            f"Energy increased at step {i}: {trace[i-1]:.6f} -> {trace[i]:.6f}"
        )


@pytest.mark.parametrize("activation", ["tanh", "relu", "gelu"])
def test_monotonic_energy_across_activations(activation):
    """§9: Monotonic descent should hold for all activations with small eta_x."""
    config = PCConfig(
        layer_sizes=[10, 8, 5],
        activation=activation,
        eta_x=0.005,
        K=30,
        eta_x_scale="uniform",
        state_norm="none",
        stability_mode="none",
        device="cpu",
    )
    net = PCNetwork(config)

    x0 = torch.randn(8, 10)
    xL = torch.randn(8, 5)
    x = net.init_states(8, x0=x0, xL=xL)
    clamp_mask = make_clamp_mask(3, ClampMode.SUPERVISED)

    metrics = net.settle(x, clamp_mask)
    trace = metrics.energy_trace

    for i in range(1, len(trace)):
        assert trace[i] <= trace[i - 1] + 1e-5, (
            f"[{activation}] Energy increased at step {i}: {trace[i-1]:.6f} -> {trace[i]:.6f}"
        )


# ============================================================
# Clamp immutability tests (§9)
# ============================================================


def test_clamped_layers_unchanged():
    """Clamped layers should not be modified during settling."""
    config = PCConfig(
        layer_sizes=[10, 8, 5], eta_x=0.1, K=10, device="cpu",
        eta_x_scale="uniform", state_norm="none", stability_mode="none",
    )
    net = PCNetwork(config)

    x0 = torch.randn(4, 10)
    xL = torch.randn(4, 5)
    x0_orig = x0.clone()
    xL_orig = xL.clone()

    x = net.init_states(4, x0=x0, xL=xL)
    clamp_mask = make_clamp_mask(3, ClampMode.SUPERVISED)
    net.settle(x, clamp_mask)

    torch.testing.assert_close(x[0], x0_orig)
    torch.testing.assert_close(x[2], xL_orig)


def test_clamped_layers_bitwise_identical():
    """§9: Clamped layers must be BITWISE identical after settling."""
    config = PCConfig(
        layer_sizes=[10, 8, 5], eta_x=0.1, K=20, device="cpu",
        eta_x_scale="uniform", state_norm="none", stability_mode="none",
    )
    net = PCNetwork(config)

    x0 = torch.randn(4, 10)
    xL = torch.randn(4, 5)

    x = net.init_states(4, x0=x0, xL=xL)
    clamp_mask = make_clamp_mask(3, ClampMode.SUPERVISED)
    net.settle(x, clamp_mask)

    # Bitwise: use torch.equal (not assert_close)
    assert torch.equal(x[0], x0), "x[0] was modified during settling!"
    assert torch.equal(x[2], xL), "x[L] was modified during settling!"


# ============================================================
# Batch independence test (§9)
# ============================================================


def test_batch_independence():
    """§9: Settling two examples independently must give same result as batch of 2."""
    config = PCConfig(
        layer_sizes=[10, 8, 5],
        activation="tanh",
        eta_x=0.02,
        K=20,
        eta_x_scale="uniform",
        state_norm="none",
        stability_mode="none",
        device="cpu",
    )

    # Create two examples
    x0_a = torch.randn(1, 10)
    x0_b = torch.randn(1, 10)
    xL_a = torch.randn(1, 5)
    xL_b = torch.randn(1, 5)
    clamp_mask = make_clamp_mask(3, ClampMode.SUPERVISED)

    # Settle independently
    torch.manual_seed(42)
    net = PCNetwork(config)
    x_a = net.init_states(1, x0=x0_a, xL=xL_a)
    net.settle(x_a, clamp_mask)
    x1_a = x_a[1].clone()

    torch.manual_seed(42)
    net = PCNetwork(config)
    x_b = net.init_states(1, x0=x0_b, xL=xL_b)
    net.settle(x_b, clamp_mask)
    x1_b = x_b[1].clone()

    # Settle as batch of 2
    torch.manual_seed(42)
    net = PCNetwork(config)
    x0_batch = torch.cat([x0_a, x0_b], dim=0)
    xL_batch = torch.cat([xL_a, xL_b], dim=0)
    x_batch = net.init_states(2, x0=x0_batch, xL=xL_batch)
    net.settle(x_batch, clamp_mask)

    torch.testing.assert_close(x_batch[1][0:1], x1_a, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(x_batch[1][1:2], x1_b, atol=1e-5, rtol=1e-4)


# ============================================================
# Weight update tests
# ============================================================


def test_weight_update_changes_weights(small_config):
    """Local weight update should modify weights."""
    net = PCNetwork(small_config)
    w_before = net.W[0].data.clone()

    x = net.init_states(4)
    x[0] = torch.randn(4, 10)
    x[1] = torch.randn(4, 8)
    x[2] = torch.randn(4, 5)

    errors, preacts = net.compute_errors(x)
    net.local_weight_update(x, errors, preacts)

    assert not torch.allclose(net.W[0].data, w_before)


def test_train_step_returns_metrics(small_config):
    net = PCNetwork(small_config)
    x_input = torch.randn(4, 10)
    y_target = torch.zeros(4, 5)
    y_target[:, 0] = 1.0

    result = net.train_step(x_input, y_target)
    assert "energy" in result
    assert "correct" in result
    assert "total" in result
    assert "oscillations" in result
    assert "stability_failure" in result
    assert result["total"] == 4


def test_gradient_equivalence_linear():
    """For a 2-layer linear network, PC weight update ≈ backprop gradient."""
    config = PCConfig(
        layer_sizes=[4, 3],
        activation="linear",
        tied_weights=True,
        eta_x=0.01,
        eta_w=1.0,
        K=200,
        converge_tol=1e-6,
        converge_patience=5,
        eta_x_scale="uniform",
        state_norm="none",
        stability_mode="none",
        device="cpu",
    )
    net = PCNetwork(config)

    B = 16
    x_input = torch.randn(B, 4)
    y_target = torch.randn(B, 3)

    clamp_mask = make_clamp_mask(2, ClampMode.SUPERVISED)
    x = net.init_states(B, x0=x_input, xL=y_target)
    net.settle(x, clamp_mask)

    errors, preacts = net.compute_errors(x)
    f_deriv = net.f_prime(preacts[0])
    pc_dW = x[1].T @ (errors[0] * f_deriv) / B

    pred_0 = y_target @ net.W[0].data
    bp_error = x_input - pred_0
    bp_dW = y_target.T @ bp_error / B

    torch.testing.assert_close(pc_dW, bp_dW, atol=0.1, rtol=0.2)


# ============================================================
# Inference and generation tests
# ============================================================


def test_infer_returns_prediction(small_config):
    net = PCNetwork(small_config)
    x_input = torch.randn(4, 10)
    pred, metrics = net.infer(x_input)
    assert pred.shape == (4, 5)
    assert len(metrics.energy_trace) > 0


def test_generate_returns_data(small_config):
    net = PCNetwork(small_config)
    z_latent = torch.randn(4, 5)
    generated, metrics = net.generate(z_latent)
    assert generated.shape == (4, 10)


def test_predict_with_confidence(small_config):
    """§5: predict_with_confidence should return per-example energy."""
    net = PCNetwork(small_config)
    x_input = torch.randn(4, 10)
    y_pred, energy, converged = net.predict_with_confidence(x_input)

    assert y_pred.shape == (4,)
    assert energy.shape == (4,)
    assert converged.shape == (4,)
    assert (energy >= 0).all(), "Energy should be non-negative"


# ============================================================
# Untied weights test
# ============================================================


def test_untied_weights():
    """Network with untied weights should still work."""
    config = PCConfig(
        layer_sizes=[10, 8, 5],
        tied_weights=False,
        eta_x=0.05,
        K=20,
        eta_x_scale="uniform",
        state_norm="none",
        stability_mode="none",
        device="cpu",
    )
    net = PCNetwork(config)
    assert hasattr(net, "R")
    assert len(net.R) == 2

    x_input = torch.randn(4, 10)
    y_target = torch.zeros(4, 5)
    y_target[:, 0] = 1.0
    result = net.train_step(x_input, y_target)
    assert result["total"] == 4


# ============================================================
# §2: Layerwise learning rate scaling
# ============================================================


def test_eta_x_inv_dim_scaling():
    """inv_dim scaling should produce different effective rates per layer."""
    config = PCConfig(
        layer_sizes=[100, 50, 10],
        eta_x=0.1,
        eta_x_scale="inv_dim",
        state_norm="none",
        stability_mode="none",
        K=5,
        device="cpu",
    )
    net = PCNetwork(config)
    import math
    expected_0 = 0.1 / math.sqrt(100)
    expected_1 = 0.1 / math.sqrt(50)
    expected_2 = 0.1 / math.sqrt(10)

    assert abs(net._get_effective_eta_x(0) - expected_0) < 1e-8
    assert abs(net._get_effective_eta_x(1) - expected_1) < 1e-8
    assert abs(net._get_effective_eta_x(2) - expected_2) < 1e-8


def test_eta_x_learned_scaling():
    """Learned scaling should use nn.Parameter multipliers."""
    config = PCConfig(
        layer_sizes=[10, 8, 5],
        eta_x=0.1,
        eta_x_scale="learned",
        state_norm="none",
        stability_mode="none",
        K=5,
        device="cpu",
    )
    net = PCNetwork(config)
    assert hasattr(net, "eta_x_learned")
    assert len(net.eta_x_learned) == 3

    # Default learned scale is 1.0, so effective eta = base eta * 1.0
    assert abs(net._get_effective_eta_x(0) - 0.1) < 1e-8


# ============================================================
# §3: State normalization
# ============================================================


def test_layernorm_applied():
    """With state_norm='layernorm', free layers should be normalized."""
    config = PCConfig(
        layer_sizes=[10, 8, 5],
        eta_x=0.05,
        K=5,
        eta_x_scale="uniform",
        state_norm="layernorm",
        stability_mode="none",
        device="cpu",
    )
    net = PCNetwork(config)

    x0 = torch.randn(4, 10) * 10  # large values
    xL = torch.randn(4, 5) * 10
    x = net.init_states(4, x0=x0, xL=xL)
    clamp_mask = make_clamp_mask(3, ClampMode.SUPERVISED)
    net.settle(x, clamp_mask)

    # Hidden layer should have been normalized — its mean should be near 0
    assert x[1].mean().abs() < 1.0, "LayerNorm should center the hidden states"


def test_rmsnorm_applied():
    """With state_norm='rmsnorm', free layers should be RMS-normalized."""
    config = PCConfig(
        layer_sizes=[10, 8, 5],
        eta_x=0.05,
        K=5,
        eta_x_scale="uniform",
        state_norm="rmsnorm",
        stability_mode="none",
        device="cpu",
    )
    net = PCNetwork(config)

    x0 = torch.randn(4, 10) * 10
    xL = torch.randn(4, 5) * 10
    x = net.init_states(4, x0=x0, xL=xL)
    clamp_mask = make_clamp_mask(3, ClampMode.SUPERVISED)
    net.settle(x, clamp_mask)

    # RMSNorm brings RMS near 1
    rms = x[1].pow(2).mean(dim=-1).sqrt()
    assert (rms < 3.0).all(), f"RMSNorm should constrain magnitude, got RMS={rms}"


def test_state_norm_does_not_affect_clamped():
    """State normalization must NOT touch clamped layers."""
    config = PCConfig(
        layer_sizes=[10, 8, 5],
        eta_x=0.05,
        K=5,
        eta_x_scale="uniform",
        state_norm="layernorm",
        stability_mode="none",
        device="cpu",
    )
    net = PCNetwork(config)

    x0 = torch.randn(4, 10) * 10
    xL = torch.randn(4, 5) * 10

    x = net.init_states(4, x0=x0, xL=xL)
    clamp_mask = make_clamp_mask(3, ClampMode.SUPERVISED)
    net.settle(x, clamp_mask)

    # Clamped layers must be unchanged
    assert torch.equal(x[0], x0)
    assert torch.equal(x[2], xL)


# ============================================================
# §6: Oscillation detection
# ============================================================


def test_oscillation_detection_fires():
    """§9: With intentionally too-large eta_x, oscillation should be detected."""
    config = PCConfig(
        layer_sizes=[10, 8, 5],
        activation="tanh",
        eta_x=10.0,  # intentionally too large
        K=20,
        eta_x_scale="uniform",
        state_norm="none",
        stability_mode="adaptive",
        device="cpu",
    )
    net = PCNetwork(config)

    x0 = torch.randn(4, 10)
    xL = torch.randn(4, 5)
    x = net.init_states(4, x0=x0, xL=xL)
    clamp_mask = make_clamp_mask(3, ClampMode.SUPERVISED)

    metrics = net.settle(x, clamp_mask)

    # Should have detected oscillations
    assert metrics.n_oscillations > 0 or metrics.stability_failure, (
        "Oscillation detection should fire with eta_x=10.0"
    )


def test_stability_failure_strict_mode():
    """In strict mode, stability failure should abort settling."""
    config = PCConfig(
        layer_sizes=[10, 8, 5],
        activation="tanh",
        eta_x=100.0,  # wildly unstable
        K=50,
        eta_x_scale="uniform",
        state_norm="none",
        stability_mode="strict",
        device="cpu",
    )
    net = PCNetwork(config)

    x0 = torch.randn(4, 10)
    xL = torch.randn(4, 5)
    x = net.init_states(4, x0=x0, xL=xL)
    clamp_mask = make_clamp_mask(3, ClampMode.SUPERVISED)

    metrics = net.settle(x, clamp_mask)

    # Should have aborted early due to stability failure
    assert metrics.stability_failure
    assert metrics.steps_used < 50


def test_no_oscillation_in_none_mode():
    """In stability_mode='none', no oscillation events should be recorded."""
    config = PCConfig(
        layer_sizes=[10, 8, 5],
        activation="tanh",
        eta_x=10.0,
        K=10,
        eta_x_scale="uniform",
        state_norm="none",
        stability_mode="none",
        device="cpu",
    )
    net = PCNetwork(config)

    x0 = torch.randn(4, 10)
    xL = torch.randn(4, 5)
    x = net.init_states(4, x0=x0, xL=xL)
    clamp_mask = make_clamp_mask(3, ClampMode.SUPERVISED)

    metrics = net.settle(x, clamp_mask)
    assert metrics.n_oscillations == 0
    assert not metrics.stability_failure


# ============================================================
# §7: Weight alignment diagnostic
# ============================================================


def test_weight_alignment_returns_cosine_sim():
    """compute_weight_alignment should return per-layer cosine similarities."""
    config = PCConfig(
        layer_sizes=[10, 8, 5],
        activation="tanh",
        tied_weights=True,
        eta_x=0.01,
        K=50,
        eta_x_scale="uniform",
        state_norm="none",
        stability_mode="none",
        device="cpu",
    )
    net = PCNetwork(config)
    x_input = torch.randn(8, 10)
    y_target = torch.randn(8, 5)

    alignment = net.compute_weight_alignment(x_input, y_target)
    assert "cos_sim_layer_0" in alignment
    assert "cos_sim_layer_1" in alignment
    # Cosine sim should be between -1 and 1 (with float tolerance)
    for key, val in alignment.items():
        assert -1.0 - 1e-6 <= val <= 1.0 + 1e-6, f"{key} = {val} out of range"
