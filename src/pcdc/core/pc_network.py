"""Core Predictive Coding Network implementation.

Implements a stateful dynamical system with:
- Top-down generative predictions
- Bottom-up error feedback
- Iterative settling (energy minimization)
- Local Hebbian-like weight updates (no backprop needed)
- Layerwise learning rate scaling
- Optional state normalization
- Oscillation detection and adaptive damping
"""

from __future__ import annotations

import math
import sys
import time

import torch
import torch.nn as nn
from torch import Tensor

from pcdc.core.activations import get_activation
from pcdc.core.clamp import ClampMode, make_clamp_mask
from pcdc.core.energy import (
    check_convergence,
    compute_energy,
    compute_layer_mse,
    compute_per_example_energy,
)
from pcdc.utils.config import PCConfig
from pcdc.utils.metrics_logger import OscillationEvent, SettleMetrics


class PCNetwork(nn.Module):
    """Predictive Coding Network.

    A network of L+1 layers where:
    - Layer 0: input (typically clamped to data)
    - Layer L: output/latent (clamped to labels in supervised mode)
    - Layers 1..L-1: hidden (free to settle)

    Generative weights W[l] predict layer l-1 from layer l.
    Error signals e[l] = x[l] - prediction drive state updates.
    """

    def __init__(self, config: PCConfig):
        super().__init__()
        self.config = config
        self.L = config.n_layers  # number of weight layers
        self.n_nodes = self.L + 1  # total number of layer nodes

        self.f, self.f_prime = get_activation(config.activation)

        # Generative weights: W[i] maps layer (i+1) -> prediction of layer i
        # W[i] shape: (D_{i+1}, D_i)
        # Index: W[0] connects layer 1->0, W[1] connects layer 2->1, etc.
        self.W = nn.ParameterList()
        for i in range(self.L):
            d_from = config.layer_sizes[i + 1]
            d_to = config.layer_sizes[i]
            w = torch.randn(d_from, d_to) * (1.0 / math.sqrt(d_from))
            self.W.append(nn.Parameter(w))

        # Optional untied feedback weights
        if not config.tied_weights:
            self.R = nn.ParameterList()
            for i in range(self.L):
                d_from = config.layer_sizes[i + 1]
                d_to = config.layer_sizes[i]
                r = torch.randn(d_to, d_from) * (1.0 / math.sqrt(d_to))
                self.R.append(nn.Parameter(r))

        # Layerwise eta_x scaling (§2)
        self._eta_x_scales: list[float] = []
        # Always register eta_x_learned so it exists regardless of mode
        self.eta_x_learned: nn.ParameterList | None = None
        if config.eta_x_scale == "learned":
            self.eta_x_learned = nn.ParameterList([
                nn.Parameter(torch.ones(1)) for _ in range(self.n_nodes)
            ])
        for l in range(self.n_nodes):
            if config.eta_x_scale == "inv_dim":
                self._eta_x_scales.append(1.0 / math.sqrt(config.layer_sizes[l]))
            else:
                self._eta_x_scales.append(1.0)

        # State normalization (§3)
        if config.state_norm == "layernorm":
            self._layer_norms = nn.ModuleList([
                nn.LayerNorm(config.layer_sizes[l], elementwise_affine=False)
                for l in range(self.n_nodes)
            ])

        # Pre-allocate dx buffer for inner loop (§10)
        self._dx_buffers: list[Tensor] | None = None

    def _ensure_dx_buffers(self, batch_size: int, device: torch.device) -> None:
        """Pre-allocate dx buffers to avoid per-iteration allocations."""
        if (
            self._dx_buffers is not None
            and self._dx_buffers[0].shape[0] == batch_size
            and self._dx_buffers[0].device == device
        ):
            return
        self._dx_buffers = [
            torch.zeros(batch_size, self.config.layer_sizes[l], device=device)
            for l in range(self.n_nodes)
        ]

    def _get_effective_eta_x(self, l: int) -> float:
        """Compute effective step size for layer l."""
        base = self.config.eta_x
        scale = self._eta_x_scales[l]
        if self.eta_x_learned is not None:
            scale *= self.eta_x_learned[l].item()
        return base * scale

    def _apply_state_norm(self, x: list[Tensor], clamp_mask: list[bool]) -> None:
        """Apply state normalization to free (unclamped) layers in-place."""
        mode = self.config.state_norm
        if mode == "none":
            return
        for l in range(self.n_nodes):
            if clamp_mask[l]:
                continue
            if mode == "layernorm":
                x[l] = self._layer_norms[l](x[l])
            elif mode == "rmsnorm":
                rms = x[l].pow(2).mean(dim=-1, keepdim=True).add(1e-8).sqrt()
                x[l] = x[l] / rms

    def init_states(
        self,
        batch_size: int,
        x0: Tensor | None = None,
        xL: Tensor | None = None,
    ) -> list[Tensor]:
        """Initialize representation states for all layers.

        Args:
            batch_size: Batch dimension.
            x0: Optional input to clamp at layer 0.
            xL: Optional target to clamp at layer L.

        Returns:
            List of tensors x[0..L], each shape (B, D_l).
        """
        device = next(self.parameters()).device
        x = []
        for l in range(self.n_nodes):
            d = self.config.layer_sizes[l]
            x.append(torch.zeros(batch_size, d, device=device))

        if x0 is not None:
            x[0] = x0.clone()
        if xL is not None:
            x[self.L] = xL.clone()

        return x

    def _predict_layer(self, x_above: Tensor, w_idx: int) -> tuple[Tensor, Tensor]:
        """Compute top-down prediction for layer w_idx from layer w_idx+1.

        Args:
            x_above: x[w_idx+1], shape (B, D_{w_idx+1}).
            w_idx: Weight index (predicts layer w_idx from w_idx+1).

        Returns:
            (prediction, preactivation) — both shape (B, D_{w_idx}).
        """
        preact = x_above @ self.W[w_idx]
        return self.f(preact), preact

    def compute_errors(
        self, x: list[Tensor]
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Compute prediction errors and cache preactivations.

        e[l] = x[l] - f(x[l+1] @ W[l]) for l = 0..L-1
        No error for layer L (or handled by prior term).

        Returns:
            (errors, preactivations) — errors[l] and preacts[l] for l=0..L-1.
        """
        errors = []
        preacts = []
        for l in range(self.L):
            pred, preact = self._predict_layer(x[l + 1], l)
            errors.append(x[l] - pred)
            preacts.append(preact)
        return errors, preacts

    def _feedback_weight(self, w_idx: int) -> Tensor:
        """Get the feedback weight matrix for weight index w_idx.

        In tied mode: W[w_idx]^T, shape (D_{w_idx}, D_{w_idx+1}).
        In untied mode: R[w_idx], shape (D_{w_idx}, D_{w_idx+1}).
        """
        if self.config.tied_weights:
            return self.W[w_idx].T
        else:
            return self.R[w_idx]

    def update_states(
        self,
        x: list[Tensor],
        errors: list[Tensor],
        preacts: list[Tensor],
        clamp_mask: list[bool],
        eta_x_override: float | None = None,
    ) -> tuple[list[float], list[float]]:
        """Update free (unclamped) states by descending energy.

        dx[l] = -e[l] + (e[l-1] @ R[l]) * f'(preact[l-1])

        The first term (-e[l]) pulls x[l] toward its top-down prediction.
        The second term pushes x[l] to better explain errors in the layer below.

        Args:
            x: Current states (modified in-place).
            errors: Prediction errors from compute_errors.
            preacts: Cached preactivations from compute_errors.
            clamp_mask: Boolean mask; True = skip update.
            eta_x_override: If set, overrides base eta_x (used by adaptive damping).

        Returns:
            (dx_norms, effective_etas) — per-layer diagnostics.
        """
        clip = self.config.dx_clip
        leak = self.config.leaky_rate
        dx_norms = []
        effective_etas = []

        assert len(errors) == self.L, f"Expected {self.L} errors, got {len(errors)}"
        assert len(preacts) == self.L, f"Expected {self.L} preacts, got {len(preacts)}"

        self._ensure_dx_buffers(x[0].shape[0], x[0].device)

        for l in range(self.n_nodes):
            if clamp_mask[l]:
                dx_norms.append(0.0)
                effective_etas.append(0.0)
                continue

            # Compute effective eta for this layer (§2)
            if eta_x_override is not None:
                eta = eta_x_override * self._eta_x_scales[l]
                if self.eta_x_learned is not None:
                    eta *= self.eta_x_learned[l].item()
            else:
                eta = self._get_effective_eta_x(l)
            effective_etas.append(eta)

            # Reuse pre-allocated buffer
            dx = self._dx_buffers[l]
            dx.zero_()

            # Term 1: self-prediction error (only if l < L, i.e., l has a prediction)
            if l < self.L:
                dx.sub_(errors[l])

            # Term 2: feedback from layer below (only if l > 0)
            if l > 0:
                f_deriv = self.f_prime(preacts[l - 1])
                feedback = (errors[l - 1] * f_deriv) @ self._feedback_weight(l - 1)
                dx.add_(feedback)

            # Clip
            if clip > 0:
                dx.clamp_(-clip, clip)

            # Apply update with optional leaky integration
            # Leaky: x_new = (1-λ)*x_old + λ*(x_old + η*dx) = x_old + λ*η*dx
            # This smooths the update by mixing old and new states.
            if leak > 0:
                x[l].add_(dx, alpha=leak * eta)
            else:
                x[l].add_(dx, alpha=eta)

            dx_norms.append(dx.norm().item())

        return dx_norms, effective_etas

    def settle(
        self,
        x: list[Tensor],
        clamp_mask: list[bool],
        K: int | None = None,
        record_metrics: bool = False,
    ) -> SettleMetrics:
        """Run iterative settling to minimize energy.

        Includes oscillation detection (§6) and optional adaptive damping.

        Args:
            x: Initial states (modified in-place).
            clamp_mask: Boolean mask for clamped layers.
            K: Max iterations (defaults to config.K).
            record_metrics: If True, record per-step diagnostics.

        Returns:
            SettleMetrics with energy trace, convergence, and stability info.
        """
        if K is None:
            K = self.config.K

        metrics = SettleMetrics()
        tol = self.config.converge_tol
        patience = self.config.converge_patience
        patience_counter = 0
        stability = self.config.stability_mode
        verbose = self.config.verbose

        # Adaptive damping state (§6)
        current_eta_x = self.config.eta_x
        consecutive_increases = 0

        t0 = time.perf_counter()
        with torch.no_grad():
            for step in range(K):
                errors, preacts = self.compute_errors(x)
                energy = compute_energy(errors).item()
                metrics.energy_trace.append(energy)

                if record_metrics:
                    metrics.layer_mse.append(compute_layer_mse(errors))

                if verbose:
                    print(f"  settle step {step:3d} | energy={energy:.6f}", file=sys.stderr)

                # --- Oscillation detection (§6) ---
                if step > 0:
                    prev_energy = metrics.energy_trace[-2]

                    if energy > prev_energy:
                        consecutive_increases += 1
                    else:
                        consecutive_increases = 0

                    # Oscillation: energy increased for 2 consecutive steps
                    if consecutive_increases >= 2 and stability != "none":
                        old_eta = current_eta_x
                        if stability == "adaptive":
                            current_eta_x *= 0.5
                            metrics.oscillation_events.append(OscillationEvent(
                                step=step,
                                energy_before=prev_energy,
                                energy_after=energy,
                                eta_x_before=old_eta,
                                eta_x_after=current_eta_x,
                            ))
                            if verbose:
                                print(
                                    f"  OSCILLATION at step {step}: "
                                    f"eta_x {old_eta:.6f} -> {current_eta_x:.6f}",
                                    file=sys.stderr,
                                )
                            consecutive_increases = 0
                        elif stability == "strict":
                            metrics.oscillation_events.append(OscillationEvent(
                                step=step,
                                energy_before=prev_energy,
                                energy_after=energy,
                                eta_x_before=old_eta,
                                eta_x_after=old_eta,
                            ))

                    # Hard stability check: energy divergence
                    if (
                        stability != "none"
                        and metrics.energy_trace[0] > 0
                        and energy > 10.0 * metrics.energy_trace[0]
                    ):
                        metrics.stability_failure = True
                        metrics.steps_used = step + 1
                        if verbose:
                            print(
                                f"  STABILITY FAILURE at step {step}: "
                                f"energy {energy:.4f} > 10x initial {metrics.energy_trace[0]:.4f}",
                                file=sys.stderr,
                            )
                        if stability == "strict":
                            metrics.wall_clock_s = time.perf_counter() - t0
                            return metrics
                        # In adaptive mode, halve eta and continue
                        current_eta_x *= 0.5

                    # Check convergence
                    if check_convergence(prev_energy, energy, tol):
                        patience_counter += 1
                        if patience_counter >= patience:
                            metrics.converged = True
                            metrics.steps_used = step + 1
                            metrics.wall_clock_s = time.perf_counter() - t0
                            return metrics
                    else:
                        patience_counter = 0

                # Update states (pass override eta if adaptive damping changed it)
                eta_override = current_eta_x if current_eta_x != self.config.eta_x else None
                dx_norms, effective_etas = self.update_states(
                    x, errors, preacts, clamp_mask, eta_x_override=eta_override
                )
                if record_metrics:
                    metrics.dx_norms.append(dx_norms)
                    metrics.effective_eta_x.append(effective_etas)

                # Apply state normalization (§3)
                self._apply_state_norm(x, clamp_mask)

        metrics.steps_used = K
        metrics.wall_clock_s = time.perf_counter() - t0
        return metrics

    def local_weight_update(
        self, x: list[Tensor], errors: list[Tensor], preacts: list[Tensor]
    ) -> None:
        """Apply local Hebbian-like weight updates.

        ΔW[l] ∝ x[l+1]^T @ (e[l] * f'(preact[l]))

        Weight updates use only locally available information:
        pre-synaptic activity (x above) times post-synaptic error.
        """
        eta_w = self.config.eta_w
        wd = self.config.weight_decay
        B = x[0].shape[0]

        with torch.no_grad():
            for l in range(self.L):
                f_deriv = self.f_prime(preacts[l])
                modulated_error = errors[l] * f_deriv  # (B, D_l)

                # ΔW = x[l+1]^T @ modulated_error / B
                dW = x[l + 1].T @ modulated_error / B  # (D_{l+1}, D_l)

                self.W[l].data.add_(dW, alpha=eta_w)

                if wd > 0:
                    self.W[l].data.mul_(1.0 - wd * eta_w)

                if not self.config.tied_weights:
                    dR = modulated_error.T @ x[l + 1] / B  # (D_l, D_{l+1})
                    self.R[l].data.add_(dR, alpha=eta_w)

    def train_step(self, x_input: Tensor, y_target: Tensor) -> dict:
        """Full supervised training step.

        1. Clamp x[0] = input, x[L] = target one-hot
        2. Settle to equilibrium
        3. Apply local weight updates

        Args:
            x_input: Input data, shape (B, D_0).
            y_target: Target representation, shape (B, D_L).

        Returns:
            Dict with energy, accuracy, settle_steps, stability info.
        """
        clamp_mask = make_clamp_mask(self.n_nodes, ClampMode.SUPERVISED)
        x = self.init_states(x_input.shape[0], x0=x_input, xL=y_target)
        metrics = self.settle(x, clamp_mask)

        # Compute errors at equilibrium for weight update
        errors, preacts = self.compute_errors(x)
        self.local_weight_update(x, errors, preacts)

        # Compute accuracy (argmax of x[L] vs target)
        pred = x[self.L].argmax(dim=-1)
        target_idx = y_target.argmax(dim=-1)
        correct = (pred == target_idx).sum().item()

        return {
            "energy": metrics.final_energy,
            "correct": correct,
            "total": x_input.shape[0],
            "settle_steps": metrics.steps_used,
            "converged": metrics.converged,
            "oscillations": metrics.n_oscillations,
            "stability_failure": metrics.stability_failure,
        }

    def infer(self, x_input: Tensor, K: int | None = None) -> tuple[Tensor, SettleMetrics]:
        """Run inference: clamp input, settle, read output.

        Args:
            x_input: Input data, shape (B, D_0).
            K: Max settle steps (defaults to config.K).

        Returns:
            (prediction tensor shape (B, D_L), settle metrics).
        """
        clamp_mask = make_clamp_mask(self.n_nodes, ClampMode.INFERENCE)
        x = self.init_states(x_input.shape[0], x0=x_input)
        metrics = self.settle(x, clamp_mask, K=K)
        return x[self.L], metrics

    def predict_with_confidence(
        self, x_input: Tensor, K: int | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run inference with per-example energy as confidence signal (§5).

        Args:
            x_input: Input data, shape (B, D_0).
            K: Max settle steps.

        Returns:
            (y_pred, energy, converged) where:
            - y_pred: (B,) predicted class indices
            - energy: (B,) per-example convergence energy (lower = more confident)
            - converged: (B,) bool — whether each example hit convergence criterion
        """
        clamp_mask = make_clamp_mask(self.n_nodes, ClampMode.INFERENCE)
        x = self.init_states(x_input.shape[0], x0=x_input)
        metrics = self.settle(x, clamp_mask, K=K)

        # Compute per-example energy at equilibrium
        with torch.no_grad():
            errors, _ = self.compute_errors(x)
            per_example_e = compute_per_example_energy(errors)

        y_pred = x[self.L].argmax(dim=-1)

        # Convergence is batch-level in the current settle loop;
        # report it uniformly (all examples in a batch converge together)
        converged = torch.full_like(y_pred, metrics.converged, dtype=torch.bool)

        metrics.per_example_energy = per_example_e
        return y_pred, per_example_e, converged

    def generate(self, z_latent: Tensor, K: int | None = None) -> tuple[Tensor, SettleMetrics]:
        """Run generative mode: clamp latent, settle, read output at layer 0.

        Args:
            z_latent: Latent representation, shape (B, D_L).
            K: Max settle steps.

        Returns:
            (generated data shape (B, D_0), settle metrics).
        """
        clamp_mask = make_clamp_mask(self.n_nodes, ClampMode.GENERATIVE)
        x = self.init_states(z_latent.shape[0], xL=z_latent)
        metrics = self.settle(x, clamp_mask, K=K)
        return x[0], metrics

    def compute_weight_alignment(
        self, x_input: Tensor, y_target: Tensor
    ) -> dict[str, float]:
        """Compare PC weight update direction to backprop gradient (§7).

        For diagnostic use: measures how well the local PC learning rule
        aligns with the global backprop gradient. Only meaningful for
        tied weights.

        Args:
            x_input: Input batch, shape (B, D_0).
            y_target: Target batch, shape (B, D_L).

        Returns:
            Dict mapping f"cos_sim_layer_{l}" to cosine similarity values.
        """
        B = x_input.shape[0]
        device = x_input.device

        # --- PC gradient direction ---
        clamp_mask = make_clamp_mask(self.n_nodes, ClampMode.SUPERVISED)
        x = self.init_states(B, x0=x_input, xL=y_target)
        self.settle(x, clamp_mask)
        errors, preacts = self.compute_errors(x)

        result = {}
        for l in range(self.L):
            # PC gradient (no autograd needed)
            with torch.no_grad():
                f_deriv = self.f_prime(preacts[l])
                pc_dW = x[l + 1].T @ (errors[l] * f_deriv) / B

            # Backprop gradient via autograd (needs grad tracking)
            x_above = x[l + 1].detach()
            x_target = x[l].detach()
            W_copy = self.W[l].data.clone().requires_grad_(True)
            pred = self.f(x_above @ W_copy)
            loss = 0.5 * (x_target - pred).pow(2).sum() / B
            loss.backward()
            bp_dW = W_copy.grad

            # Cosine similarity
            with torch.no_grad():
                cos_sim = torch.nn.functional.cosine_similarity(
                    pc_dW.flatten().unsqueeze(0),
                    bp_dW.flatten().unsqueeze(0),
                ).item()
            result[f"cos_sim_layer_{l}"] = cos_sim

        return result
