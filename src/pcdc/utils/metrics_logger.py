"""Metrics logging for PC network training and settling."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class OscillationEvent:
    """Record of an oscillation event during settling."""

    step: int
    energy_before: float
    energy_after: float
    eta_x_before: float
    eta_x_after: float


@dataclass
class SettleMetrics:
    """Metrics from a single settle() call."""

    energy_trace: list[float] = field(default_factory=list)
    layer_mse: list[list[float]] = field(default_factory=list)  # per-step, per-layer
    dx_norms: list[list[float]] = field(default_factory=list)  # per-step, per-layer
    effective_eta_x: list[list[float]] = field(default_factory=list)  # per-step, per-layer
    converged: bool = False
    steps_used: int = 0

    # Stability tracking
    oscillation_events: list[OscillationEvent] = field(default_factory=list)
    stability_failure: bool = False

    # Per-example energy (populated by predict_with_confidence)
    per_example_energy: Tensor | None = None

    @property
    def final_energy(self) -> float:
        return self.energy_trace[-1] if self.energy_trace else float("inf")

    @property
    def energy_reduction(self) -> float:
        if len(self.energy_trace) < 2:
            return 0.0
        return self.energy_trace[0] - self.energy_trace[-1]

    @property
    def n_oscillations(self) -> int:
        return len(self.oscillation_events)


@dataclass
class EpochMetrics:
    """Aggregated metrics over one training epoch."""

    epoch: int = 0
    train_energy: float = 0.0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    avg_settle_steps: float = 0.0
    n_batches: int = 0
    n_oscillations: int = 0
    n_stability_failures: int = 0

    def update(
        self,
        batch_energy: float,
        batch_correct: int,
        batch_total: int,
        settle_steps: int,
        oscillations: int = 0,
        stability_failure: bool = False,
    ):
        self.n_batches += 1
        self.train_energy += (batch_energy - self.train_energy) / self.n_batches
        self.train_accuracy += (batch_correct / batch_total - self.train_accuracy) / self.n_batches
        self.avg_settle_steps += (settle_steps - self.avg_settle_steps) / self.n_batches
        self.n_oscillations += oscillations
        if stability_failure:
            self.n_stability_failures += 1

    def summary(self) -> str:
        s = (
            f"Epoch {self.epoch:3d} | "
            f"energy={self.train_energy:.4f} | "
            f"train_acc={self.train_accuracy:.4f} | "
            f"val_acc={self.val_accuracy:.4f} | "
            f"settle_steps={self.avg_settle_steps:.1f}"
        )
        if self.n_oscillations > 0:
            s += f" | osc={self.n_oscillations}"
        if self.n_stability_failures > 0:
            s += f" | UNSTABLE={self.n_stability_failures}"
        return s
