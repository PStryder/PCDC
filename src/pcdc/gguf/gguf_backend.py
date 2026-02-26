"""GGUF model backend for feature extraction via llama-cpp-python."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm


class GGUFBackend:
    """Loads a GGUF model and extracts hidden-state features for the PC head.

    Features are the embedding-space representation of each prompt,
    extracted from the model's final layer (default) or a specified layer.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 8,
        n_gpu_layers: int = 0,
    ):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for GGUF backend. "
                "Install with: uv sync --extra gguf"
            )

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            embedding=True,
            verbose=False,
        )
        self.hidden_dim = self.llm.n_embd()
        self._n_layers = self.llm.n_layer() if hasattr(self.llm, 'n_layer') else None

    def embed_and_warm(self, text: str) -> tuple[Tensor, list[int]]:
        """Embed text and leave KV cache warm for generation prefix reuse.

        Tokenizes and evals the prompt, extracts the last-token embedding,
        and leaves the KV cache populated so a subsequent
        ``create_completion(prompt=tokens)`` gets a prefix hit and skips
        prompt re-processing.

        Returns:
            (embedding, tokens) — embedding is ``(hidden_dim,)``; tokens
            can be passed directly to ``create_completion()`` for cache reuse.
        """
        tokens = self.llm.tokenize(text.encode("utf-8"), special=True)

        self.llm.reset()
        self.llm.eval(tokens)

        # After eval with default logits_all=False, only the last token has
        # its embedding stored.  Index -1 → last output position.
        n_embd = self.llm.n_embd()
        emb_ptr = self.llm._ctx.get_embeddings_ith(-1)
        emb = [emb_ptr[i] for i in range(n_embd)]

        return torch.tensor(emb, dtype=torch.float32), tokens

    def embed_chat(self, text: str) -> Tensor:
        """Embed a single text and return a (hidden_dim,) tensor.

        Convenience wrapper for chat serving — avoids the batch dimension
        and progress-bar overhead of extract_features.  Mean-pools across
        tokens when the backend returns per-token embeddings.
        """
        emb = self.llm.embed(text)
        t = torch.tensor(emb, dtype=torch.float32)
        if t.dim() == 2:
            # Per-token embeddings (n_tokens, hidden_dim) → mean pool
            t = t.mean(dim=0)
        return t

    def extract_features(
        self,
        texts: list[str],
        feature_layer: str | int = "last",
        show_progress: bool = False,
    ) -> Tensor:
        """Extract embedding features for a batch of texts.

        Args:
            texts: List of input prompts.
            feature_layer: Which layer to extract from (§8):
                - "last" (default): final layer embedding
                - "middle": layer at n_layers // 2
                - int: specific layer index
                NOTE: Per-layer hidden state extraction depends on
                llama-cpp-python exposing per-layer access. If unavailable,
                falls back to final-layer embeddings with a warning.
            show_progress: Show progress bar.

        Returns:
            Tensor of shape (len(texts), hidden_dim).
        """
        # Resolve layer index
        if feature_layer == "last":
            use_per_layer = False
        elif feature_layer == "middle":
            if self._n_layers is not None:
                target_layer = self._n_layers // 2
                use_per_layer = True
            else:
                print(
                    "WARNING: Cannot determine model layer count. "
                    "Falling back to last-layer embeddings."
                )
                use_per_layer = False
        elif isinstance(feature_layer, int):
            target_layer = feature_layer
            use_per_layer = True
        else:
            raise ValueError(f"Invalid feature_layer: {feature_layer}")

        # Check if per-layer extraction is available
        if use_per_layer and not hasattr(self.llm, 'embed_layer'):
            print(
                f"WARNING: llama-cpp-python does not expose per-layer hidden states. "
                f"Requested layer {target_layer}, falling back to last-layer embeddings. "
                f"To use middle-layer features, upgrade llama-cpp-python or use the "
                f"logits+projection fallback."
            )
            use_per_layer = False

        features = []
        iterator = tqdm(texts, desc="Extracting features") if show_progress else texts

        for text in iterator:
            if use_per_layer:
                emb = self.llm.embed_layer(text, layer=target_layer)
            else:
                emb = self.llm.embed(text)
            features.append(torch.tensor(emb, dtype=torch.float32))

        return torch.stack(features)

    def precompute_dataset(
        self,
        texts: list[str],
        labels: list[int],
        cache_path: str,
        feature_layer: str | int = "last",
        show_progress: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Extract features for an entire dataset and cache to disk.

        If cache exists, loads from disk instead of re-extracting.

        Args:
            texts: All input texts.
            labels: Corresponding integer labels.
            cache_path: Path to save/load the cached tensors.
            feature_layer: Which layer to extract from.

        Returns:
            (features tensor, labels tensor).
        """
        cache = Path(cache_path)

        if cache.exists():
            data = torch.load(cache, weights_only=True)
            print(f"Loaded cached features from {cache_path} "
                  f"(shape: {data['features'].shape})")
            return data["features"], data["labels"]

        cache.parent.mkdir(parents=True, exist_ok=True)
        features = self.extract_features(
            texts, feature_layer=feature_layer, show_progress=show_progress
        )
        label_tensor = torch.tensor(labels, dtype=torch.long)

        torch.save({
            "features": features,
            "labels": label_tensor,
            "feature_layer": str(feature_layer),
        }, cache)
        print(f"Cached features to {cache_path} (shape: {features.shape})")

        return features, label_tensor
