#!/usr/bin/env python3
"""Pre-train the PCHead offline to bootstrap attractor basins.

Reads a text corpus, chunks it, embeds each chunk through the GGUF backend
using the same Llama-3 chat template as the server, and runs train_step
for multiple epochs (both reconstruction and predictive phases).

This is necessary before the deviation-based routing can produce meaningful
cosine similarities — without pre-training, the PCHead's deviation vectors
are dominated by random initialization and all point in near-orthogonal
directions.

Usage:
    python scripts/pretrain_pchead.py \
        --model F:/models/Llama-3-8B-Lexi-Uncensored.Q8_0.gguf \
        --n-gpu-layers 33 \
        --corpus corpus.txt \
        --epochs 5 \
        --batch-size 8 \
        --output pchead_pretrained.ckpt
"""

from __future__ import annotations

import argparse
import random
import sys
import time

import torch
from torch import Tensor


def chunk_corpus(text: str, min_chars: int = 100) -> list[str]:
    """Split corpus into paragraph-level chunks.

    Splits on double newlines (paragraph breaks). Chunks shorter than
    min_chars are merged with the next chunk. Returns non-empty chunks.
    """
    raw_chunks = text.split("\n\n")
    chunks = []
    buffer = ""

    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if buffer:
            buffer += "\n\n" + chunk
        else:
            buffer = chunk
        if len(buffer) >= min_chars:
            chunks.append(buffer)
            buffer = ""

    # Don't lose the trailing buffer
    if buffer:
        if chunks:
            chunks[-1] += "\n\n" + buffer
        else:
            chunks.append(buffer)

    return chunks


def format_as_chat(text: str) -> str:
    """Wrap a text chunk in the Llama-3 chat template.

    Matches the format used by the server's format_llama3_prompt so
    embedding norms and geometry are consistent.
    """
    from pcdc.server.app import format_llama3_prompt
    from pcdc.server.schemas import ChatMessage

    messages = [
        ChatMessage(role="user", content=text),
    ]
    return format_llama3_prompt(messages)


def embed_corpus(
    backend,
    chunks: list[str],
) -> list[Tensor]:
    """Embed all chunks through the GGUF backend."""
    from tqdm import tqdm

    embeddings = []
    for chunk in tqdm(chunks, desc="Embedding corpus"):
        formatted = format_as_chat(chunk)
        emb = backend.embed_chat(formatted)
        embeddings.append(emb)

    return embeddings


def pretrain(
    model_path: str,
    corpus_path: str,
    n_gpu_layers: int = 0,
    epochs: int = 5,
    batch_size: int = 8,
    output_path: str = "pchead_pretrained.ckpt",
    settle_k: int = 50,
    eta_w: float = 0.0001,
    min_chunk_chars: int = 100,
):
    """Run the pre-training loop."""
    from pcdc.gguf.gguf_backend import GGUFBackend
    from pcdc.gguf.pc_head import PCHead

    # --- Load model ---
    print(f"Loading GGUF model...")
    backend = GGUFBackend(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
    )
    hidden_dim = backend.hidden_dim
    print(f"  hidden_dim={hidden_dim}")

    # --- Load and chunk corpus ---
    print(f"Reading corpus from {corpus_path}...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_corpus(text, min_chars=min_chunk_chars)
    print(f"  {len(chunks)} chunks (min {min_chunk_chars} chars)")

    if len(chunks) < 2:
        print("ERROR: Need at least 2 chunks for predictive training.")
        sys.exit(1)

    # --- Embed all chunks ---
    print(f"\nEmbedding {len(chunks)} chunks...")
    t0 = time.time()
    embeddings = embed_corpus(backend, chunks)
    embed_time = time.time() - t0
    print(f"  Embedded in {embed_time:.1f}s ({len(embeddings)/embed_time:.1f} chunks/s)")

    norms = torch.tensor([e.norm().item() for e in embeddings])
    print(f"  Embedding norms: mean={norms.mean():.1f}, std={norms.std():.1f}, "
          f"range=[{norms.min():.1f}, {norms.max():.1f}]")

    # --- Create PCHead ---
    pc_head = PCHead(
        feature_dim=hidden_dim,
        num_classes=hidden_dim,
        hidden_sizes=[1024, 256],
        K=settle_k,
    )
    device = next(pc_head.parameters()).device
    print(f"\nPCHead created: [4096, 1024, 256, 4096], K={settle_k}, device={device}")

    # --- Set learning rate ---
    original_eta_w = pc_head.config.eta_w
    pc_head.config.eta_w = eta_w
    print(f"  eta_w: {original_eta_w} -> {eta_w}")

    # --- Training loop ---
    print(f"\nPre-training for {epochs} epochs, batch_size={batch_size}")
    print(f"  Reconstruction batches per epoch: {len(embeddings) // batch_size}")
    print(f"  Predictive pairs: {len(embeddings) - 1}")
    print()

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()

        # Shuffle indices for reconstruction
        indices = list(range(len(embeddings)))
        random.shuffle(indices)

        recon_energies = []
        pred_energies = []

        # --- Phase 1: Reconstruction (shuffled batches) ---
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            batch = torch.stack([embeddings[j] for j in batch_idx]).to(device)
            result = pc_head.train_step(batch, batch)
            recon_energies.append(result["energy"])

        # --- Phase 2: Predictive (sequential pairs) ---
        pair_indices = list(range(len(embeddings) - 1))
        random.shuffle(pair_indices)

        for i in range(0, len(pair_indices), batch_size):
            batch_idx = pair_indices[i:i + batch_size]
            prev_batch = torch.stack([embeddings[j] for j in batch_idx]).to(device)
            curr_batch = torch.stack([embeddings[j + 1] for j in batch_idx]).to(device)
            result = pc_head.train_step(prev_batch, curr_batch)
            pred_energies.append(result["energy"])

        elapsed = time.time() - t_epoch
        avg_recon = sum(recon_energies) / len(recon_energies)
        avg_pred = sum(pred_energies) / len(pred_energies) if pred_energies else 0

        print(f"  Epoch {epoch}/{epochs}: "
              f"recon_energy={avg_recon:.2f}  pred_energy={avg_pred:.2f}  "
              f"({elapsed:.1f}s)")

    # --- Restore eta_w ---
    pc_head.config.eta_w = original_eta_w

    # --- Save checkpoint ---
    print(f"\nSaving checkpoint to {output_path}...")
    data = {
        "pc_head_state_dict": pc_head.state_dict(),
        "prompt_replay": [e.detach().cpu() for e in embeddings],
        "prompt_pair_replay": [
            (embeddings[i].detach().cpu(), embeddings[i + 1].detach().cpu())
            for i in range(len(embeddings) - 1)
        ],
        "pretrain_info": {
            "corpus": corpus_path,
            "n_chunks": len(chunks),
            "epochs": epochs,
            "batch_size": batch_size,
            "settle_k": settle_k,
            "eta_w": eta_w,
        },
    }
    torch.save(data, output_path)
    print(f"  Saved ({len(embeddings)} replay embeddings, "
          f"{len(embeddings) - 1} transition pairs)")


def main():
    parser = argparse.ArgumentParser(description="Pre-train PCHead for attractor basin bootstrapping")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="GPU layers")
    parser.add_argument("--corpus", required=True, help="Path to text corpus file")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--output", "-o", default="pchead_pretrained.ckpt", help="Output checkpoint path")
    parser.add_argument("--settle-k", type=int, default=50, help="Settle steps per phase")
    parser.add_argument("--eta-w", type=float, default=0.0001, help="Weight learning rate")
    parser.add_argument("--min-chunk-chars", type=int, default=100, help="Minimum chunk size in characters")
    args = parser.parse_args()

    pretrain(
        model_path=args.model,
        corpus_path=args.corpus,
        n_gpu_layers=args.n_gpu_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_path=args.output,
        settle_k=args.settle_k,
        eta_w=args.eta_w,
        min_chunk_chars=args.min_chunk_chars,
    )


if __name__ == "__main__":
    main()
