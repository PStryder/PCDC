"""SteeringEngine — PCHead-based temperature modulation for LLM generation."""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Generator

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

    def steer_and_generate(
        self,
        prompt_text: str,
        base_temp: float = 1.0,
        **completion_kwargs: Any,
    ) -> tuple[SteeringResult, Any]:
        """Atomic embed → steer → generate with KV-cache reuse.

        Holds ``self._lock`` for the entire sequence so the warm KV cache
        from ``embed_and_warm`` is still intact when ``create_completion``
        runs, giving it a prefix hit that skips prompt re-processing.

        For streaming (``stream=True`` in *completion_kwargs*), the returned
        iterator wrapper keeps the lock held until the stream is exhausted.

        Args:
            prompt_text: Fully-formatted prompt (Llama-3 template already
                applied).
            base_temp: Base temperature before energy modulation.
            **completion_kwargs: Forwarded to
                ``backend.llm.create_completion()`` — include ``top_p``,
                ``max_tokens``, ``stop``, ``stream``, etc.

        Returns:
            ``(SteeringResult, completion_result_or_stream)``
        """
        is_stream = completion_kwargs.get("stream", False)
        self._lock.acquire()
        released = False
        try:
            embedding, tokens = self.backend.embed_and_warm(prompt_text)

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
                "steer_and_generate: energy=%.4f median=%.4f base=%.2f adj=%.3f converged=%s steps=%d",
                energy, median, base_temp, adjusted, metrics.converged, metrics.steps_used,
            )

            steering = SteeringResult(
                adjusted_temp=adjusted,
                energy=energy,
                converged=metrics.converged,
                settle_steps=metrics.steps_used,
                steering_vector=steering_vector.squeeze(0),
            )

            # Generate — tokens prefix-match the warm KV cache
            result = self.backend.llm.create_completion(
                prompt=tokens,
                temperature=adjusted,
                **completion_kwargs,
            )

            if is_stream:
                released = True
                return steering, self._locked_iter(result)
            else:
                return steering, result
        finally:
            if not released:
                self._lock.release()

    def _locked_iter(self, it: Any) -> Generator:
        """Wrap an iterator so ``self._lock`` is released when exhausted."""
        try:
            yield from it
        finally:
            self._lock.release()

    def record(self, text: str) -> None:
        """Buffer an embedding in the replay buffer for future training."""
        with self._lock:
            embedding = self.backend.embed_chat(text)
        if len(self._replay) < self._replay_max:
            self._replay.append(embedding)

    def save_checkpoint(self, path: str) -> None:
        """Save PCHead weights and steering stats to disk."""
        import torch
        data = {
            "pc_head_state_dict": self.pc_head.state_dict(),
            "stats_total_requests": self.stats.total_requests,
            "stats_energy_history": list(self.stats.energy_history),
            "stats_energy_median": self.stats.energy_median,
        }
        torch.save(data, path)
        logger.info("Checkpoint saved to %s (%d requests)", path, self.stats.total_requests)

    def load_checkpoint(self, path: str) -> None:
        """Load PCHead weights and steering stats from disk."""
        import torch
        data = torch.load(path, weights_only=False)
        self.pc_head.load_state_dict(data["pc_head_state_dict"])
        self.pc_head.eval()
        self.stats.total_requests = data.get("stats_total_requests", 0)
        history = data.get("stats_energy_history", [])
        self.stats.energy_history = deque(history, maxlen=1000)
        self.stats.energy_median = data.get("stats_energy_median", 0.0)
        logger.info("Checkpoint loaded from %s (%d prior requests)", path, self.stats.total_requests)

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
