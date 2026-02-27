"""SteeringEngine — PCHead-based temperature modulation for LLM generation."""

from __future__ import annotations

import logging
import math
import random
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Generator

import torch
from torch import Tensor

from pcdc.gguf.gguf_backend import GGUFBackend
from pcdc.gguf.pc_head import PCHead

logger = logging.getLogger(__name__)


@dataclass
class SteeringResult:
    adjusted_temp: float
    energy: float                           # blended
    reconstruction_energy: float            # Phase 1
    predictive_energy: float                # Phase 2
    converged: bool
    settle_steps: int
    cosine_distance: float | None = None    # distance to previous embedding
    steering_vector: Tensor | None = None
    retrieval_triggered: bool = False
    retrieval_count: int = 0
    retrieval_query: str | None = None
    deviation_match_score: float | None = None  # best cosine sim to replay buffer
    deviation_match_idx: int | None = None      # index in replay buffer


@dataclass
class SteeringStats:
    total_requests: int = 0
    energy_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    energy_median: float = 0.0
    recon_energy_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    predict_energy_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    cosine_distance_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    retrieval_triggered_total: int = 0
    deviation_routing_total: int = 0
    deviation_match_history: deque = field(default_factory=lambda: deque(maxlen=1000))


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
        energy_scale: float = 0.0,
        temp_min: float = 0.1,
        temp_max: float = 2.0,
        beta: float = 0.5,
        online_eta_w: float = 0.0001,
        replay_k: int = 4,
        settle_k: int | None = None,
        # MemoryGate retrieval
        memory_client: Any = None,
        retrieval_recon_threshold: float = float("inf"),
        retrieval_predict_threshold: float = float("inf"),
        retrieval_predict_percentile: float | None = None,
        retrieval_limit: int = 3,
        retrieval_min_confidence: float = 0.5,
        format_prompt_fn: Callable | None = None,
        # Deviation routing
        deviation_routing_enabled: bool = False,
        deviation_routing_threshold: float = 0.6,
        # Telemetry
        telemetry_db: Any = None,
    ):
        self.backend = backend
        self.pc_head = pc_head
        self.alpha = alpha
        self.energy_scale = energy_scale
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.beta = beta
        self.online_eta_w = online_eta_w
        self.replay_k = replay_k
        self.settle_k = settle_k

        # MemoryGate retrieval
        self._memory_client = memory_client
        self._retrieval_recon_threshold = retrieval_recon_threshold
        self._retrieval_predict_threshold = retrieval_predict_threshold
        self._retrieval_predict_percentile = retrieval_predict_percentile
        self._retrieval_limit = retrieval_limit
        self._retrieval_min_confidence = retrieval_min_confidence
        self._format_prompt_fn = format_prompt_fn

        # Deviation routing
        self._deviation_routing_enabled = deviation_routing_enabled
        self._deviation_routing_threshold = deviation_routing_threshold

        # Telemetry
        self._telemetry_db = telemetry_db

        self._lock = threading.Lock()
        self.stats = SteeringStats()

        # Replay buffer for record() (completion embeddings)
        self._replay: list[Tensor] = []
        self._replay_max = 10_000

        # Two-phase replay buffers (prompt embeddings)
        self._prev_embedding: Tensor | None = None
        self._prompt_replay: list[Tensor] = []
        self._prompt_pair_replay: list[tuple[Tensor, Tensor]] = []
        self._prompt_deviation_replay: list[Tensor] = []  # deviation vectors parallel to _prompt_replay

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
        scale = self._effective_scale(self.stats.energy_history)
        adjusted = base_temp * (1.0 + self.alpha * math.tanh((energy - median) / scale))
        adjusted = max(self.temp_min, min(self.temp_max, adjusted))

        logger.info(
            "steer: energy=%.4f median=%.4f scale=%.1f base_temp=%.2f adjusted_temp=%.3f converged=%s steps=%d",
            energy, median, scale, base_temp, adjusted, metrics.converged, metrics.steps_used,
        )

        return SteeringResult(
            adjusted_temp=adjusted,
            energy=energy,
            reconstruction_energy=energy,
            predictive_energy=energy,
            converged=metrics.converged,
            settle_steps=metrics.steps_used,
            steering_vector=steering_vector.squeeze(0),
        )

    def steer_and_generate(
        self,
        prompt_text: str,
        base_temp: float = 1.0,
        messages: list[dict] | None = None,
        **completion_kwargs: Any,
    ) -> tuple[SteeringResult, Any]:
        """Atomic embed → two-phase settle → retrieval gate → infer → generate.

        Phase 1 (Reconstruction): train_step(embedding, embedding)
            → How novel is this content? (self-reconstruction energy)

        Phase 2 (Predictive): train_step(prev_embedding, current_embedding)
            → How surprising is this transition? (predictive energy)

        Retrieval Gate: if either energy exceeds its threshold, query
        MemoryGate for relevant context, inject it into the prompt, and
        re-embed to get new generation tokens (KV cache refreshed).

        Both training phases use a conservative online learning rate and
        mix in replay samples to prevent catastrophic forgetting.

        Holds ``self._lock`` for the entire sequence so the warm KV cache
        is still intact when ``create_completion`` runs.

        For streaming (``stream=True`` in *completion_kwargs*), the returned
        iterator wrapper keeps the lock held until the stream is exhausted.

        Args:
            prompt_text: Fully-formatted prompt (Llama-3 template already
                applied).
            base_temp: Base temperature before energy modulation.
            messages: Raw message dicts (``[{"role":..,"content":..}]``)
                for prompt reconstruction on retrieval. Optional — if not
                provided, retrieval gate is skipped.
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
            device = next(self.pc_head.parameters()).device
            x_input = embedding.unsqueeze(0).to(device)

            # --- Cosine distance to previous embedding ---
            cosine_dist = None
            if self._prev_embedding is not None:
                prev_flat = self._prev_embedding.to(device).float()
                curr_flat = embedding.to(device).float()
                cos_sim = torch.nn.functional.cosine_similarity(
                    prev_flat.unsqueeze(0), curr_flat.unsqueeze(0),
                ).item()
                cosine_dist = 1.0 - cos_sim

            # --- Save and swap learning rate + settle steps ---
            original_eta_w = self.pc_head.config.eta_w
            original_K = self.pc_head.config.K
            self.pc_head.config.eta_w = self.online_eta_w
            if self.settle_k is not None:
                self.pc_head.config.K = self.settle_k

            # --- Phase 1: Reconstruction settle ---
            recon_batch = [x_input]
            if self._prompt_replay:
                k = min(self.replay_k, len(self._prompt_replay))
                samples = random.sample(self._prompt_replay, k)
                recon_batch.extend(s.unsqueeze(0).to(device) for s in samples)
            recon_input = torch.cat(recon_batch, dim=0)
            recon_metrics = self.pc_head.train_step(recon_input, recon_input)
            e_recon = recon_metrics["energy"]

            # --- Phase 2: Predictive settle ---
            if self._prev_embedding is not None:
                prev = self._prev_embedding.unsqueeze(0).to(device)
                pred_prev_batch = [prev]
                pred_curr_batch = [x_input]
                if self._prompt_pair_replay:
                    k = min(self.replay_k, len(self._prompt_pair_replay))
                    pairs = random.sample(self._prompt_pair_replay, k)
                    for p, c in pairs:
                        pred_prev_batch.append(p.unsqueeze(0).to(device))
                        pred_curr_batch.append(c.unsqueeze(0).to(device))
                pred_input = torch.cat(pred_prev_batch, dim=0)
                pred_target = torch.cat(pred_curr_batch, dim=0)
                pred_metrics = self.pc_head.train_step(pred_input, pred_target)
                e_predict = pred_metrics["energy"]
                total_steps = recon_metrics["settle_steps"] + pred_metrics["settle_steps"]
                converged = recon_metrics["converged"] and pred_metrics["converged"]
            else:
                e_predict = e_recon
                total_steps = recon_metrics["settle_steps"]
                converged = recon_metrics["converged"]

            # --- Restore learning rate + settle steps ---
            self.pc_head.config.eta_w = original_eta_w
            self.pc_head.config.K = original_K

            # --- Blend energies ---
            energy = self.beta * e_recon + (1.0 - self.beta) * e_predict

            # --- Retrieval gate ---
            retrieval_triggered = False
            retrieval_count = 0
            retrieval_query = None

            # Compute dynamic predict threshold from percentile if configured
            predict_threshold = self._retrieval_predict_threshold
            if (
                self._retrieval_predict_percentile is not None
                and len(self.stats.predict_energy_history) >= 3
            ):
                predict_threshold = self._compute_percentile(
                    self.stats.predict_energy_history,
                    self._retrieval_predict_percentile,
                )

            if (
                self._memory_client is not None
                and self._format_prompt_fn is not None
                and messages is not None
                and (
                    e_recon > self._retrieval_recon_threshold
                    or e_predict > predict_threshold
                )
            ):
                retrieval_query = _extract_last_user_content(messages)
                if retrieval_query:
                    results = self._memory_client.search(
                        query=retrieval_query,
                        limit=self._retrieval_limit,
                        min_confidence=self._retrieval_min_confidence,
                    )
                    if results:
                        retrieval_triggered = True
                        retrieval_count = len(results)
                        self.stats.retrieval_triggered_total += 1
                        # Rebuild prompt with injected context, re-embed
                        augmented_prompt = self._format_prompt_fn(messages, results)
                        _, tokens = self.backend.embed_and_warm(augmented_prompt)
                        logger.info(
                            "retrieval: triggered (e_recon=%.4f e_predict=%.4f), "
                            "injected %d memory items for query=%r",
                            e_recon, e_predict, retrieval_count, retrieval_query[:80],
                        )

            # --- Infer for steering vector (on updated weights, original embedding) ---
            with torch.no_grad():
                steering_vector, infer_metrics = self.pc_head.infer(x_input)

            # --- Compute and store deviation ---
            deviation_match_score = None
            deviation_match_idx = None

            with torch.no_grad():
                deviation = steering_vector.squeeze(0) - embedding.to(device)
                current_deviation = deviation.detach().cpu()

            # --- Deviation routing (deviation-to-deviation comparison) ---
            if (
                self._deviation_routing_enabled
                and len(self._prompt_deviation_replay) >= 10
            ):
                dev_norm = current_deviation.norm()
                if dev_norm > 1e-8:
                    dev_unit = current_deviation / dev_norm
                    # Batch cosine similarity against stored deviations (all CPU)
                    dev_replay_stack = torch.stack(self._prompt_deviation_replay)  # (N, D)
                    cos_scores = torch.nn.functional.cosine_similarity(
                        dev_unit.unsqueeze(0), dev_replay_stack,
                    )  # (N,)
                    best_score_t, best_idx_t = cos_scores.max(dim=0)
                    deviation_match_score = best_score_t.item()
                    deviation_match_idx = best_idx_t.item()

                    self.stats.deviation_match_history.append(deviation_match_score)

                    # Gate retrieval on deviation match
                    if (
                        deviation_match_score > self._deviation_routing_threshold
                        and self._memory_client is not None
                        and self._format_prompt_fn is not None
                        and messages is not None
                        and not retrieval_triggered  # don't double-trigger
                    ):
                        retrieval_query = _extract_last_user_content(messages)
                        if retrieval_query:
                            results = self._memory_client.search(
                                query=retrieval_query,
                                limit=self._retrieval_limit,
                                min_confidence=self._retrieval_min_confidence,
                            )
                            if results:
                                retrieval_triggered = True
                                retrieval_count = len(results)
                                self.stats.retrieval_triggered_total += 1
                                self.stats.deviation_routing_total += 1
                                augmented_prompt = self._format_prompt_fn(messages, results)
                                _, tokens = self.backend.embed_and_warm(augmented_prompt)
                                logger.info(
                                    "deviation_routing: triggered (score=%.4f idx=%d), "
                                    "injected %d memory items",
                                    deviation_match_score, deviation_match_idx,
                                    retrieval_count,
                                )

            # Update running stats
            self.stats.total_requests += 1
            self.stats.energy_history.append(energy)
            self.stats.recon_energy_history.append(e_recon)
            self.stats.predict_energy_history.append(e_predict)
            if cosine_dist is not None:
                self.stats.cosine_distance_history.append(cosine_dist)
            self.stats.energy_median = self._compute_median()

            # Energy → temperature
            median = self.stats.energy_median
            scale = self._effective_scale(self.stats.energy_history)
            adjusted = base_temp * (1.0 + self.alpha * math.tanh((energy - median) / scale))
            adjusted = max(self.temp_min, min(self.temp_max, adjusted))

            logger.info(
                "steer_and_generate: e_recon=%.4f e_predict=%.4f blended=%.4f "
                "median=%.4f scale=%.1f base=%.2f adj=%.3f converged=%s steps=%d "
                "cos_dist=%s retrieval=%s",
                e_recon, e_predict, energy, median, scale, base_temp, adjusted,
                converged, total_steps,
                f"{cosine_dist:.6f}" if cosine_dist is not None else "N/A",
                retrieval_triggered,
            )

            steering = SteeringResult(
                adjusted_temp=adjusted,
                energy=energy,
                reconstruction_energy=e_recon,
                predictive_energy=e_predict,
                converged=converged,
                settle_steps=total_steps,
                cosine_distance=cosine_dist,
                steering_vector=steering_vector.squeeze(0),
                retrieval_triggered=retrieval_triggered,
                retrieval_count=retrieval_count,
                retrieval_query=retrieval_query,
                deviation_match_score=deviation_match_score,
                deviation_match_idx=deviation_match_idx,
            )

            # --- Telemetry ---
            if self._telemetry_db is not None:
                try:
                    dev_np = current_deviation.numpy().astype("float32")
                    self._telemetry_db.record_turn(
                        energy_recon=e_recon,
                        energy_predict=e_predict,
                        energy_blended=energy,
                        cosine_distance=cosine_dist,
                        deviation_norm=current_deviation.norm().item(),
                        deviation_vector=dev_np.tobytes(),
                        top_match_score=deviation_match_score,
                        top_match_idx=deviation_match_idx,
                        adjusted_temp=adjusted,
                        converged=converged,
                        settle_steps=total_steps,
                        retrieval_triggered=retrieval_triggered,
                    )
                except Exception:
                    logger.exception("Telemetry recording failed")

            # --- Update replay buffers ---
            if len(self._prompt_replay) < self._replay_max:
                self._prompt_replay.append(embedding.detach().cpu())
                self._prompt_deviation_replay.append(current_deviation)
            if self._prev_embedding is not None and len(self._prompt_pair_replay) < self._replay_max:
                self._prompt_pair_replay.append(
                    (self._prev_embedding.detach().cpu(), embedding.detach().cpu())
                )
            self._prev_embedding = embedding.detach().cpu()

            # Generate — tokens may be original or augmented (retrieval gate)
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
        """Save PCHead weights, steering stats, and replay state to disk."""
        import torch
        data = {
            "pc_head_state_dict": self.pc_head.state_dict(),
            "stats_total_requests": self.stats.total_requests,
            "stats_energy_history": list(self.stats.energy_history),
            "stats_energy_median": self.stats.energy_median,
            "stats_recon_energy_history": list(self.stats.recon_energy_history),
            "stats_predict_energy_history": list(self.stats.predict_energy_history),
            "stats_cosine_distance_history": list(self.stats.cosine_distance_history),
            "prev_embedding": self._prev_embedding,
            "prompt_replay": self._prompt_replay,
            "prompt_pair_replay": self._prompt_pair_replay,
            "prompt_deviation_replay": self._prompt_deviation_replay,
            "stats_deviation_routing_total": self.stats.deviation_routing_total,
            "stats_deviation_match_history": list(self.stats.deviation_match_history),
        }
        torch.save(data, path)
        logger.info("Checkpoint saved to %s (%d requests)", path, self.stats.total_requests)

    def load_checkpoint(self, path: str) -> None:
        """Load PCHead weights, steering stats, and replay state from disk."""
        import torch
        data = torch.load(path, weights_only=False)
        self.pc_head.load_state_dict(data["pc_head_state_dict"])
        self.pc_head.eval()
        self.stats.total_requests = data.get("stats_total_requests", 0)
        history = data.get("stats_energy_history", [])
        self.stats.energy_history = deque(history, maxlen=1000)
        self.stats.energy_median = data.get("stats_energy_median", 0.0)
        recon_history = data.get("stats_recon_energy_history", [])
        self.stats.recon_energy_history = deque(recon_history, maxlen=1000)
        predict_history = data.get("stats_predict_energy_history", [])
        self.stats.predict_energy_history = deque(predict_history, maxlen=1000)
        cosine_history = data.get("stats_cosine_distance_history", [])
        self.stats.cosine_distance_history = deque(cosine_history, maxlen=1000)
        self._prev_embedding = data.get("prev_embedding", None)
        self._prompt_replay = data.get("prompt_replay", [])
        self._prompt_pair_replay = data.get("prompt_pair_replay", [])
        self._prompt_deviation_replay = data.get("prompt_deviation_replay", [])
        self.stats.deviation_routing_total = data.get("stats_deviation_routing_total", 0)
        dev_match_history = data.get("stats_deviation_match_history", [])
        self.stats.deviation_match_history = deque(dev_match_history, maxlen=1000)
        logger.info("Checkpoint loaded from %s (%d prior requests)", path, self.stats.total_requests)

    def get_stats(self) -> dict:
        return {
            "total_requests": self.stats.total_requests,
            "energy_median": self.stats.energy_median,
            "recon_energy_median": self._compute_median_from(self.stats.recon_energy_history),
            "predict_energy_median": self._compute_median_from(self.stats.predict_energy_history),
            "replay_buffer_size": len(self._replay),
            "prompt_replay_size": len(self._prompt_replay),
            "deviation_replay_size": len(self._prompt_deviation_replay),
            "pair_replay_size": len(self._prompt_pair_replay),
            "energy_history_len": len(self.stats.energy_history),
            "retrieval_triggered_total": self.stats.retrieval_triggered_total,
            "deviation_routing_total": self.stats.deviation_routing_total,
            # Full histories for plotting
            "recon_energy_history": list(self.stats.recon_energy_history),
            "predict_energy_history": list(self.stats.predict_energy_history),
            "cosine_distance_history": list(self.stats.cosine_distance_history),
            "deviation_match_history": list(self.stats.deviation_match_history),
        }

    def _effective_scale(self, history: deque) -> float:
        """Return the energy scale for the tanh temperature mapping.

        If ``energy_scale`` was set to a positive value, use it directly.
        Otherwise compute an adaptive scale from the IQR of the energy
        history so the tanh input is normalised to roughly [-2, 2] and
        the temperature uses the full [temp_min, temp_max] range.

        Falls back to the median itself (or 1.0) when there aren't
        enough samples yet.
        """
        if self.energy_scale > 0:
            return self.energy_scale
        if len(history) < 4:
            # Not enough data for IQR — use median as rough scale
            m = self._compute_median_from(history)
            return m if m > 0 else 1.0
        q25 = self._compute_percentile(history, 25)
        q75 = self._compute_percentile(history, 75)
        iqr = q75 - q25
        # Scale so that +/- 1 IQR from median maps to tanh(+/- 1) ≈ +/- 0.76
        # giving meaningful gradation across the middle of the distribution
        return max(iqr, 1.0)

    def _compute_median(self) -> float:
        return self._compute_median_from(self.stats.energy_history)

    @staticmethod
    def _compute_median_from(history: deque) -> float:
        if not history:
            return 0.0
        vals = sorted(history)
        n = len(vals)
        if n % 2 == 1:
            return vals[n // 2]
        return (vals[n // 2 - 1] + vals[n // 2]) / 2.0

    @staticmethod
    def _compute_percentile(history: deque, percentile: float) -> float:
        """Compute a percentile value from a deque of floats."""
        if not history:
            return float("inf")
        vals = sorted(history)
        n = len(vals)
        idx = (percentile / 100.0) * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return vals[lo]
        frac = idx - lo
        return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _extract_last_user_content(messages: list[dict]) -> str | None:
    """Return content of the last user message, or None."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content")
    return None
