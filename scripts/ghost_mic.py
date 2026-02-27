#!/usr/bin/env python3
"""Ghost Mic — Decode what the PCHead 'wants to say' about a prompt.

Two modes:

  Mode A (lm_head): Project settled state / deviation through the LLM's
      unembedding matrix. Requires manifold alignment (currently fails —
      PCHead output space is disjoint from LLM residual stream).

  Mode B (replay retrieval): Use the deviation vector as a retrieval key
      against the checkpoint's replay buffer. Stays in the PCHead's own
      learned space — no manifold alignment needed. Reports which past
      embeddings produced similar PCHead responses and what kind of
      transitions they came from.

Usage:
    python scripts/ghost_mic.py \
        --model F:/models/Llama-3-8B-Lexi-Uncensored.Q8_0.gguf \
        --n-gpu-layers 33 \
        --checkpoint pchead.ckpt \
        --prompt "Hello! How are you doing today?" \
        --mode both
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Q8_0 dequantization (for lm_head mode)
# ---------------------------------------------------------------------------

def dequantize_q8_0(raw_data: np.ndarray, shape: tuple) -> Tensor:
    """Dequantize Q8_0 GGUF tensor to float32.

    Q8_0 format: blocks of 32 values.
    Each block = 2 bytes (float16 scale) + 32 bytes (int8 quantized values) = 34 bytes.
    """
    n_rows = raw_data.shape[0]
    block_size = 32
    n_blocks_per_row = shape[0] // block_size
    bytes_per_block = 34

    raw = raw_data.reshape(n_rows, n_blocks_per_row, bytes_per_block)
    scale_bytes = raw[:, :, :2].copy()
    scales = scale_bytes.view(np.float16).squeeze(-1).astype(np.float32)
    quant_values = raw[:, :, 2:].astype(np.int8).astype(np.float32)
    scales_expanded = scales[:, :, np.newaxis]
    dequant = quant_values * scales_expanded
    result = dequant.reshape(n_rows, -1)
    return torch.from_numpy(result).float()


def load_lm_head_and_norm(model_path: str) -> tuple[Tensor, Tensor]:
    """Extract output.weight and output_norm.weight from GGUF file."""
    from gguf import GGUFReader

    print(f"Reading GGUF tensors from {model_path}...")
    t0 = time.time()
    reader = GGUFReader(model_path)

    lm_head = None
    output_norm = None

    for tensor in reader.tensors:
        if tensor.name == "output.weight":
            lm_head = dequantize_q8_0(tensor.data, tuple(tensor.shape))
            print(f"  output.weight: dequantized to {lm_head.shape}")
        elif tensor.name == "output_norm.weight":
            output_norm = torch.from_numpy(tensor.data.copy()).float()
            print(f"  output_norm.weight: {output_norm.shape}")

        if lm_head is not None and output_norm is not None:
            break

    elapsed = time.time() - t0
    print(f"  Loaded in {elapsed:.1f}s")
    return lm_head, output_norm


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    """Apply RMSNorm (Llama-3 style)."""
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight


# ---------------------------------------------------------------------------
# Mode A: lm_head projection (manifold hypothesis test)
# ---------------------------------------------------------------------------

def mode_a_lm_head(
    settled: Tensor,
    original: Tensor,
    deviation: Tensor,
    dev_norm: float,
    lm_head: Tensor,
    output_norm: Tensor,
    tokenizer,
    top_k: int,
):
    """Project through LLM's unembedding matrix."""
    print(f"\n{'='*60}")
    print("MODE A: lm_head PROJECTION (manifold hypothesis)")
    print(f"{'='*60}")

    normed_settled = rms_norm(settled, output_norm)
    logits_settled = torch.matmul(lm_head, normed_settled)
    top_settled = logits_settled.topk(top_k)

    print(f"\n--- A1: Raw Settled State ---")
    print(f"  Top {top_k} tokens:")
    for i, (score, idx) in enumerate(zip(top_settled.values, top_settled.indices)):
        token_bytes = tokenizer.detokenize([idx.item()])
        token_text = token_bytes.decode("utf-8", errors="replace")
        print(f"    [{i+1}] {score:.2f}  id={idx.item():>6d}  '{token_text}'")

    normed_dev = rms_norm(deviation, output_norm)
    logits_dev = torch.matmul(lm_head, normed_dev)
    top_dev = logits_dev.topk(top_k)

    print(f"\n--- A2: Deviation Vector ---")
    print(f"  Top {top_k} tokens:")
    for i, (score, idx) in enumerate(zip(top_dev.values, top_dev.indices)):
        token_bytes = tokenizer.detokenize([idx.item()])
        token_text = token_bytes.decode("utf-8", errors="replace")
        print(f"    [{i+1}] {score:.2f}  id={idx.item():>6d}  '{token_text}'")

    alpha = 5.0
    blended = original + alpha * deviation
    normed_blended = rms_norm(blended, output_norm)
    logits_blended = torch.matmul(lm_head, normed_blended)
    top_blended = logits_blended.topk(top_k)

    print(f"\n--- A3: Original + {alpha}x Deviation ---")
    print(f"  Top {top_k} tokens:")
    for i, (score, idx) in enumerate(zip(top_blended.values, top_blended.indices)):
        token_bytes = tokenizer.detokenize([idx.item()])
        token_text = token_bytes.decode("utf-8", errors="replace")
        print(f"    [{i+1}] {score:.2f}  id={idx.item():>6d}  '{token_text}'")

    print(f"\n--- A: Logit Entropy (lower = more confident) ---")
    for name, logits in [
        ("Settled state", logits_settled),
        ("Deviation", logits_dev),
        ("Amplified blend", logits_blended),
    ]:
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        print(f"  {name:20s}: entropy={entropy:.2f}, max_prob={probs.max():.4f}")


# ---------------------------------------------------------------------------
# Mode B: Replay buffer retrieval (deviation as retrieval key)
# ---------------------------------------------------------------------------

def mode_b_replay_retrieval(
    settled: Tensor,
    original: Tensor,
    deviation: Tensor,
    checkpoint_data: dict,
    backend,
    pc_head,
    top_k: int,
):
    """Use deviation as retrieval key against replay buffer embeddings.

    For each stored embedding, we compute what the PCHead's deviation would
    have been (settle and subtract), then compare deviation directions via
    cosine similarity. Close matches = 'the PCHead reacted similarly'.

    We also do a simpler version: cosine similarity between the deviation
    and the raw stored embeddings, to see which regions of embedding space
    the deviation points toward.
    """
    prompt_replay = checkpoint_data.get("prompt_replay", [])
    pair_replay = checkpoint_data.get("prompt_pair_replay", [])

    print(f"\n{'='*60}")
    print("MODE B: REPLAY BUFFER RETRIEVAL (PCHead's own space)")
    print(f"{'='*60}")
    print(f"  Replay buffer: {len(prompt_replay)} embeddings, {len(pair_replay)} transition pairs")

    if not prompt_replay:
        print("  No replay buffer in checkpoint -- nothing to search.")
        return

    device = deviation.device

    # --- B1: Cosine similarity between deviation and stored embeddings ---
    # "Which past inputs does the deviation point toward?"
    replay_stack = torch.stack(prompt_replay).to(device)  # (N, 4096)
    dev_unit = deviation / (deviation.norm() + 1e-8)
    cos_to_replay = F.cosine_similarity(
        dev_unit.unsqueeze(0),  # (1, 4096)
        replay_stack,           # (N, 4096)
    )  # (N,)

    print(f"\n--- B1: Deviation vs Stored Embeddings (cosine similarity) ---")
    print(f"  'Which past inputs does the deviation direction point toward?'")
    top_b1 = cos_to_replay.topk(min(top_k, len(cos_to_replay)))
    for i, (score, idx) in enumerate(zip(top_b1.values, top_b1.indices)):
        emb = prompt_replay[idx.item()]
        emb_norm = emb.norm().item()
        print(f"    [{i+1}] cos={score:.4f}  buffer_idx={idx.item():>3d}  emb_norm={emb_norm:.1f}")

    # Also show bottom matches (most opposite direction)
    bot_b1 = (-cos_to_replay).topk(min(3, len(cos_to_replay)))
    print(f"  Most opposite:")
    for i, (neg_score, idx) in enumerate(zip(bot_b1.values, bot_b1.indices)):
        print(f"    [{i+1}] cos={-neg_score.item():.4f}  buffer_idx={idx.item():>3d}")

    # --- B2: Settle each stored embedding, compute its deviation, compare ---
    # "Which past inputs made the PCHead react the same way?"
    print(f"\n--- B2: Deviation Fingerprint Matching ---")
    print(f"  Settling all {len(prompt_replay)} replay embeddings (this may take a moment)...")

    t0 = time.time()
    deviations = []
    with torch.no_grad():
        for emb in prompt_replay:
            x = emb.unsqueeze(0).to(device)
            settled_i, _ = pc_head.infer(x)
            dev_i = settled_i.squeeze(0) - emb.to(device)
            deviations.append(dev_i)

    dev_stack = torch.stack(deviations)  # (N, 4096)
    elapsed = time.time() - t0
    print(f"  Settled {len(prompt_replay)} embeddings in {elapsed:.1f}s")

    # Cosine similarity between current deviation and all stored deviations
    dev_norms = dev_stack.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    dev_units = dev_stack / dev_norms
    cos_dev_to_dev = F.cosine_similarity(
        dev_unit.unsqueeze(0),  # (1, 4096)
        dev_units,              # (N, 4096)
    )  # (N,)

    print(f"\n  'Which past inputs produced similar PCHead deviations?'")
    top_b2 = cos_dev_to_dev.topk(min(top_k, len(cos_dev_to_dev)))
    for i, (score, idx) in enumerate(zip(top_b2.values, top_b2.indices)):
        stored_dev = deviations[idx.item()]
        stored_dev_norm = stored_dev.norm().item()
        stored_emb_norm = prompt_replay[idx.item()].norm().item()
        dev_ratio = stored_dev_norm / (stored_emb_norm + 1e-8)
        print(f"    [{i+1}] cos={score:.4f}  buffer_idx={idx.item():>3d}  "
              f"dev_norm={stored_dev_norm:.1f}  dev_ratio={dev_ratio:.3f}")

    # --- B3: Deviation magnitude distribution ---
    all_dev_norms = dev_norms.squeeze(-1)
    current_dev_norm = deviation.norm().item()
    mean_dev = all_dev_norms.mean().item()
    std_dev = all_dev_norms.std().item()
    percentile_rank = (all_dev_norms < current_dev_norm).float().mean().item() * 100

    print(f"\n--- B3: Deviation Magnitude Context ---")
    print(f"  Current deviation norm: {current_dev_norm:.2f}")
    print(f"  Replay deviations: mean={mean_dev:.2f}, std={std_dev:.2f}")
    print(f"  Current percentile: {percentile_rank:.0f}th")
    print(f"  Range: [{all_dev_norms.min().item():.2f}, {all_dev_norms.max().item():.2f}]")

    # --- B4: Transition pair analysis ---
    if pair_replay:
        print(f"\n--- B4: Transition Pair Analysis ---")
        print(f"  {len(pair_replay)} stored (prev, curr) pairs")

        # For each pair, compute deviation of curr, compare to our deviation
        pair_devs = []
        with torch.no_grad():
            for prev_emb, curr_emb in pair_replay:
                x = curr_emb.unsqueeze(0).to(device)
                settled_i, _ = pc_head.infer(x)
                dev_i = settled_i.squeeze(0) - curr_emb.to(device)
                pair_devs.append(dev_i)

        pair_dev_stack = torch.stack(pair_devs)
        pair_dev_units = pair_dev_stack / pair_dev_stack.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        cos_pair = F.cosine_similarity(
            dev_unit.unsqueeze(0),
            pair_dev_units,
        )

        top_b4 = cos_pair.topk(min(top_k, len(cos_pair)))
        print(f"  'Which past transitions produced similar deviations?'")
        for i, (score, idx) in enumerate(zip(top_b4.values, top_b4.indices)):
            prev_emb, curr_emb = pair_replay[idx.item()]
            transition_cos = F.cosine_similarity(
                prev_emb.unsqueeze(0),
                curr_emb.unsqueeze(0),
            ).item()
            print(f"    [{i+1}] cos={score:.4f}  pair_idx={idx.item():>3d}  "
                  f"transition_distance={1 - transition_cos:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def ghost_mic(
    model_path: str,
    prompt: str,
    n_gpu_layers: int = 0,
    checkpoint_path: str | None = None,
    settle_k: int = 100,
    top_k: int = 10,
    mode: str = "both",
):
    """Run the Ghost Mic experiment."""
    from pcdc.gguf.gguf_backend import GGUFBackend
    from pcdc.gguf.pc_head import PCHead

    # --- Load LLM ---
    print(f"\nLoading LLM...")
    backend = GGUFBackend(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
    )
    hidden_dim = backend.hidden_dim
    print(f"  hidden_dim={hidden_dim}")

    tokenizer = backend.llm

    # --- Load PCHead ---
    pc_head = PCHead(
        feature_dim=hidden_dim,
        num_classes=hidden_dim,
        hidden_sizes=[1024, 256],
        K=settle_k,
    )

    checkpoint_data = {}
    if checkpoint_path:
        import os
        if os.path.exists(checkpoint_path):
            checkpoint_data = torch.load(checkpoint_path, weights_only=False)
            pc_head.load_state_dict(checkpoint_data["pc_head_state_dict"])
            print(f"  PCHead loaded from {checkpoint_path}")
            n_replay = len(checkpoint_data.get("prompt_replay", []))
            n_pairs = len(checkpoint_data.get("prompt_pair_replay", []))
            print(f"  Replay buffer: {n_replay} embeddings, {n_pairs} pairs")
        else:
            print(f"  No checkpoint at {checkpoint_path}, using fresh weights")
    pc_head.eval()

    # --- Embed the prompt ---
    print(f"\nPrompt: {prompt!r}")
    embedding = backend.embed_chat(prompt)
    print(f"  Embedding shape: {embedding.shape}, norm: {embedding.norm():.2f}")

    # --- Run PCHead inference (settle) ---
    device = next(pc_head.parameters()).device
    x_input = embedding.unsqueeze(0).to(device)

    with torch.no_grad():
        settled_state, metrics = pc_head.infer(x_input)

    settled = settled_state.squeeze(0)
    print(f"  Settled state norm: {settled.norm():.2f}, energy: {metrics.final_energy:.2f}, "
          f"converged: {metrics.converged}, steps: {metrics.steps_used}")

    # --- Compute deviation ---
    original = embedding.to(device)
    deviation = settled - original
    dev_norm = deviation.norm().item()
    print(f"  Deviation norm: {dev_norm:.4f} (ratio to original: {dev_norm / original.norm().item():.4f})")

    # --- Mode A: lm_head projection ---
    if mode in ("a", "both", "lm_head"):
        lm_head, output_norm = load_lm_head_and_norm(model_path)
        lm_head = lm_head.to(device)
        output_norm = output_norm.to(device)
        mode_a_lm_head(settled, original, deviation, dev_norm, lm_head, output_norm, tokenizer, top_k)

    # --- Mode B: Replay buffer retrieval ---
    if mode in ("b", "both", "replay"):
        mode_b_replay_retrieval(settled, original, deviation, checkpoint_data, backend, pc_head, top_k)


def main():
    parser = argparse.ArgumentParser(description="Ghost Mic -- PCHead deviation analysis")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="GPU layers")
    parser.add_argument("--checkpoint", default=None, help="PCHead checkpoint path")
    parser.add_argument("--settle-k", type=int, default=100, help="Settle steps")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k tokens to show")
    parser.add_argument("--prompt", required=True, help="Prompt to test")
    parser.add_argument("--mode", default="both", choices=["a", "b", "both", "lm_head", "replay"],
                        help="Mode: a/lm_head (unembedding), b/replay (buffer retrieval), both")
    args = parser.parse_args()

    ghost_mic(
        model_path=args.model,
        prompt=args.prompt,
        n_gpu_layers=args.n_gpu_layers,
        checkpoint_path=args.checkpoint,
        settle_k=args.settle_k,
        top_k=args.top_k,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
