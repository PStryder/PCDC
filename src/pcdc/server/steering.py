"""SteeringEngine — PCHead-based temperature modulation for LLM generation."""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass, field

import torch
from torch import Tensor

from pcdc.gguf.gguf_backend import GGUFBackend
from pcdc.gguf.pc_head import PCHead

logger = logging.getLogger(__name__)


@dataclass
class SteeringResult:
    adjusted_temp: float
    energy: float
    converged: bool
    settle_steps: int
    steering_vector: Tensor | None = None


@dataclass
class SteeringStats:
    total_requests: int = 0
    energy_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    energy_median: float = 0.0


class SteeringEngine:
    """Owns GGUFBackend + PCHead, provides steering for chat generation.

    Energy-to-temperature mapping:
        temp = base * (1 + alpha * tanh((energy - median) / scale))
        Clamped to [0.1, 2.0]
    """

    def __init__(
        self,
        backend: GGUFBackend,
        pc_head: PCHead,
        alpha: float = 0.5,
        energy_scale: float = 1.0,
        temp_min: float = 0.1,
        temp_max: float = 2.0,
    ):
        self.backend = backend
        self.pc_head = pc_head
        self.alpha = alpha
        self.energy_scale = energy_scale
        self.temp_min = temp_min
        self.temp_max = temp_max

        self._lock = threading.Lock()
        self.stats = SteeringStats()

        # Replay buffer for future online learning
        self._replay: list[Tensor] = []
        self._replay_max = 10_000

    def steer(self, prompt_text: str, base_temp: float = 1.0) -> SteeringResult:
        """Run PCHead settling on prompt embedding, return steering result."""
        with self._lock:
            embedding = self.backend.embed_chat(prompt_text)

        # PCHead expects (B, feature_dim)
        x_input = embedding.unsqueeze(0).to(next(self.pc_head.parameters()).device)

        with torch.no_grad():
            steering_vector, metrics = self.pc_head.infer(x_input)

        energy = metrics.final_energy

        # Update running stats
        self.stats.total_requests += 1
        self.stats.energy_history.append(energy)
        self.stats.energy_median = self._compute_median()

        # Energy → temperature
        median = self.stats.energy_median
        scale = self.energy_scale if self.energy_scale > 0 else 1.0
        adjusted = base_temp * (1.0 + self.alpha * math.tanh((energy - median) / scale))
        adjusted = max(self.temp_min, min(self.temp_max, adjusted))

        logger.info(
            "steer: energy=%.4f median=%.4f base_temp=%.2f adjusted_temp=%.3f converged=%s steps=%d",
            energy, median, base_temp, adjusted, metrics.converged, metrics.steps_used,
        )

        return SteeringResult(
            adjusted_temp=adjusted,
            energy=energy,
            converged=metrics.converged,
            settle_steps=metrics.steps_used,
            steering_vector=steering_vector.squeeze(0),
        )

    def record(self, text: str) -> None:
        """Buffer an embedding in the replay buffer for future training."""
        with self._lock:
            embedding = self.backend.embed_chat(text)
        if len(self._replay) < self._replay_max:
            self._replay.append(embedding)

    def get_stats(self) -> dict:
        return {
            "total_requests": self.stats.total_requests,
            "energy_median": self.stats.energy_median,
            "replay_buffer_size": len(self._replay),
            "energy_history_len": len(self.stats.energy_history),
        }

    def _compute_median(self) -> float:
        if not self.stats.energy_history:
            return 0.0
        vals = sorted(self.stats.energy_history)
        n = len(vals)
        if n % 2 == 1:
            return vals[n // 2]
        return (vals[n // 2 - 1] + vals[n // 2]) / 2.0
