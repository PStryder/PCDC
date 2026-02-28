# PCDC System Architecture

A detailed explanation of how the Predictive Coding Digital Circuit works, from first principles through to the full serving stack.

## Table of Contents

1. [Overview](#overview)
2. [Predictive Coding Theory](#predictive-coding-theory)
3. [PCNetwork: The Core Engine](#pcnetwork-the-core-engine)
4. [PCHead: LLM Feature Extraction](#pchead-llm-feature-extraction)
5. [The Chat Server Pipeline](#the-chat-server-pipeline)
6. [Two-Phase Online Learning](#two-phase-online-learning)
7. [Deviation Vectors and Routing](#deviation-vectors-and-routing)
8. [Pre-Training: Bootstrapping Attractor Basins](#pre-training-bootstrapping-attractor-basins)
9. [Energy-Triggered Memory Retrieval](#energy-triggered-memory-retrieval)
10. [Replay Buffers and Checkpointing](#replay-buffers-and-checkpointing)
11. [Ghost Mic: Deviation Analysis](#ghost-mic-deviation-analysis)
12. [Signal Summary](#signal-summary)
13. [Data Flow Diagram](#data-flow-diagram)

---

## Overview

PCDC wraps a frozen LLM (loaded via GGUF) with a lightweight predictive coding network (PCHead) that learns online during conversation. The PCHead doesn't generate text — the LLM does that. Instead, the PCHead acts as an executive controller that observes the conversation's embedding stream and produces signals that modulate the LLM's generation behavior.

The key insight: a predictive coding network's settling energy and deviation vectors carry information about how familiar or surprising the current input is relative to everything the network has seen before. PCDC uses this information to:

- **Adjust temperature** — novel content gets higher temperature (more creative), familiar content gets lower temperature (more precise)
- **Trigger memory retrieval** — when energy spikes or deviation patterns match past interactions, the system autonomously fetches relevant context from MemoryGate
- **Track conversation dynamics** — cosine distance between consecutive embeddings, energy trends, and deviation fingerprints build a continuous picture of the conversation's trajectory

## Predictive Coding Theory

Predictive coding is a theory from computational neuroscience proposing that the brain constantly generates top-down predictions about incoming sensory data and learns by minimizing prediction errors.

In a predictive coding network:

1. Each layer `l` has a state vector `x[l]`
2. Generative weights `W[l]` project from higher layers to lower layers, producing predictions
3. Prediction errors are computed: `e[l] = x[l] - f(x[l+1] @ W[l])`
4. The total energy is `E = 0.5 * sum(||e[l]||^2)` across all layers
5. **Settling**: internal states are iteratively updated to minimize E (inference)
6. **Learning**: weights are updated using local Hebbian-like rules derived from the energy gradient

No backpropagation. No computation graph. No global loss function. Each weight sees only the activity above it and the error below it.

### Why This Matters for LLM Steering

The settling energy E tells you how well the network's internal model matches the current input:

- **Low energy**: the input is well-predicted — the network has seen similar patterns before
- **High energy**: the input is surprising — it doesn't match the network's learned structure

The deviation vector (settled state minus original input) tells you *what* the network wanted to change — its prediction error fingerprint. Two inputs that produce similar deviations are being processed similarly by the network, even if the raw inputs look different in embedding space.

## PCNetwork: The Core Engine

`src/pcdc/core/pc_network.py` (~590 lines)

The PCNetwork is an MLP-shaped predictive coding network with configurable:

- **Architecture**: arbitrary layer sizes, activation functions (tanh, relu, gelu, linear)
- **Settling**: K iterations of state updates with early stopping on convergence
- **Stability**: gradient clipping, oscillation detection, adaptive damping, state normalization
- **Weight learning**: local Hebbian updates with configurable learning rate and decay

### Settling Loop

```
for step in range(K):
    compute prediction errors e[l] for each layer
    compute total energy E = 0.5 * sum(||e[l]||^2)

    if oscillation detected (consecutive energy increases):
        halve learning rate (adaptive mode)
        or abort (strict mode)

    if converged (|deltaE/E| < tolerance for patience steps):
        break early

    for each free (unclamped) layer:
        dx[l] = -e[l] + (e[l-1] @ W[l]^T) * f'(preact[l])
        x[l] += eta_x * dx[l]
        apply normalization (LayerNorm/RMSNorm)
```

### Three Operating Modes

- **Supervised** (`train_step`): clamp input + target, settle, update weights. The target acts as a teaching signal — the network learns to predict the target from the input.
- **Inference** (`infer`): clamp input only, settle. The output layer's settled state is the network's prediction.
- **Generative**: clamp latent representation only, settle. The input layer reconstructs what the latent represents. (Not currently used in the server.)

## PCHead: LLM Feature Extraction

`src/pcdc/gguf/pc_head.py`

PCHead is a thin subclass of PCNetwork with architecture `[feature_dim, *hidden_sizes, num_classes]`. In the server configuration, this is `[4096, 1024, 256, 4096]` — the input and output dimensions match the LLM's hidden dimension.

The frozen LLM acts as a feature extractor:

```
User prompt --> Llama-3 chat template --> GGUF model (frozen) --> 4096-dim embedding
```

The PCHead then operates on these embeddings. It never touches the LLM's weights — it learns its own internal model of the embedding distribution.

### Why 4096 -> 1024 -> 256 -> 4096?

The bottleneck architecture (4096 -> 1024 -> 256) forces the PCHead to learn a compressed representation of the embedding space. The expansion back to 4096 means the output has the same dimensionality as the input, enabling:

- **Reconstruction**: `train_step(embedding, embedding)` — can the network reconstruct the input through the bottleneck?
- **Prediction**: `train_step(prev_embedding, curr_embedding)` — can the network predict the next turn from the current one?
- **Deviation computation**: `infer(embedding)` -> settled output, then `deviation = settled - input` is a 4096-dim vector in the same space as the embeddings

## The Chat Server Pipeline

`src/pcdc/server/app.py` + `src/pcdc/server/steering.py`

When a chat request arrives at `/v1/chat/completions`:

```
1. Extract user message text
2. Format as Llama-3 chat template
3. Embed through frozen LLM -> 4096-dim tensor
4. Pass to SteeringEngine.steer_and_generate()
   a. Phase 1: reconstruction training
   b. Phase 2: predictive training
   c. Infer: settle on current embedding
   d. Compute deviation vector
   e. Deviation routing check
   f. Energy-based retrieval check
   g. Compute adjusted temperature
5. Generate LLM response with adjusted temperature
6. Return response + pcdc metadata
```

The LLM generates text normally — it has no idea the PCHead exists. The PCHead's influence is indirect: it modulates temperature and decides when to inject memory context into the prompt.

## Two-Phase Online Learning

Each turn triggers two `train_step()` calls on the PCHead with a conservative learning rate (`eta_w` ~1e-5):

### Phase 1: Reconstruction

```python
result = pc_head.train_step(embedding, embedding)  # input = target
E_recon = result["energy"]
```

Measures self-reconstruction difficulty. A low E_recon means the PCHead can easily compress and reconstruct this embedding — the content is familiar. A high E_recon means the embedding doesn't fit the network's learned manifold — novel content.

### Phase 2: Predictive (Sequential Transitions)

```python
result = pc_head.train_step(prev_embedding, current_embedding)
E_predict = result["energy"]
```

Measures transition surprise. The PCHead learns to predict the next turn from the previous one. High E_predict means this transition was unexpected — a topic shift, a surprising follow-up, a domain change.

### Replay Buffer Mixing

Both phases mix in replay samples from stored embeddings to prevent catastrophic forgetting. Without replay, the PCHead would rapidly overfit to the most recent turns and lose sensitivity to earlier patterns.

```
training_batch = [current_example] + random.sample(replay_buffer, replay_k)
```

### Why Two Phases?

Reconstruction energy (Phase 1) answers: "Is this content novel?"
Predictive energy (Phase 2) answers: "Is this transition surprising?"

These are independent signals. A question about Rust following a question about Rust has low predictive energy (expected transition) but might have high reconstruction energy (if the specific Rust topic is new). A question about cooking following a question about Rust has high predictive energy (unexpected topic shift) regardless of whether cooking is familiar content.

The blended energy combines both:

```
energy = beta * E_recon + (1 - beta) * E_predict
```

### Temperature Mapping

```
adjusted_temp = base_temp * (1 + alpha * tanh((energy - median) / scale))
```

- `median` is the running median of the energy history
- `alpha` controls coupling strength (0 = no modulation, 1 = strong modulation)
- `scale` normalizes the energy range

**Effect**: novel or surprising content gets higher temperature (more creative responses), familiar or expected content gets lower temperature (more precise responses). The temperature adapts automatically to the conversation's dynamics.

When `energy_scale=0` (the default), scale is computed adaptively from the IQR of the blended energy history. This normalises the tanh input so +/- 1 IQR from median maps to tanh(+/- 1). Session 5 validated that this adaptive scale converges by turn ~50 and remains stable through 200+ turns (IQR/Median drops from 0.49 to 0.24 without collapsing or diverging).

### Learning Rate Sensitivity

The online learning rate `eta_w` is critical:

- **eta_w = 1e-4**: too fast. Energy drops monotonically as the network adapts. No sensitivity to topic shifts. Temperature stays flat.
- **eta_w = 1e-5**: the sweet spot. Energy declines slowly (~2.5% over 12 turns), topic shifts register as energy spikes, temperature actually varies.
- **eta_w = 1e-6**: too slow. The PCHead barely learns, energy is dominated by initialization.

## Deviation Vectors and Routing

After both training phases, `infer()` runs on the current embedding:

```python
settled_state, metrics = pc_head.infer(embedding)
deviation = settled_state.squeeze(0) - embedding
```

The deviation vector is the difference between what the PCHead settled to and what the input actually was. It encodes the network's "opinion" about the input — what it wanted to change.

### Why Deviation-to-Deviation, Not Deviation-to-Embedding

The initial implementation compared the current deviation against stored embeddings. This was changed to deviation-to-deviation comparison because:

1. **Same space, same semantics**: comparing deviations compares "how the PCHead reacted" to different inputs, not "what the inputs looked like"
2. **Higher signal**: two inputs from the same domain (e.g., Rust programming) produce similar deviations even if their raw embeddings are quite different
3. **Validated experimentally**: Ghost Mic B2 tests show deviation-to-deviation cosine similarities of 0.49-0.65 for same-domain content, versus near-zero for cross-domain

### Routing Logic

```python
if deviation_routing_enabled and len(deviation_replay) >= 10:
    # Normalize current deviation
    dev_unit = current_deviation / (current_deviation.norm() + 1e-8)

    # Compare against all stored deviations
    cos_scores = cosine_similarity(dev_unit, stacked_deviations)
    best_score, best_idx = cos_scores.max()

    if best_score > threshold:
        # Trigger MemoryGate retrieval
        results = memory_client.search(user_message)
        if results:
            augment prompt with retrieved context
            re-embed augmented prompt
```

**Performance**: the cosine similarity computation is a single batched matmul — O(N * 4096) where N is the replay buffer size. For N=10,000, this takes <1ms on CPU.

### The Dual Replay Buffer

The steering engine maintains two parallel replay buffers:

- `_prompt_replay`: stores embeddings (4096-dim tensors, detached + CPU)
- `_prompt_deviation_replay`: stores corresponding deviation vectors (4096-dim tensors, detached + CPU)

Both are capped at `replay_max` (default 10,000) entries and persisted in checkpoints. Pre-training seeds both buffers from the corpus.

**VRAM safety**: all stored tensors use `.detach().cpu()` to prevent computation graph accumulation. Without this, the server would OOM within ~50 turns as gradients accumulate.

## Pre-Training: Bootstrapping Attractor Basins

`scripts/pretrain_pchead.py`

Without pre-training, the PCHead's attractor basins haven't formed. Deviation vectors are dominated by random initialization — all near-orthogonal with cosine similarities of 0.05-0.09 between any pair. Deviation routing produces no meaningful matches.

Pre-training pushes thousands of embeddings through the PCHead before the server starts serving:

### Process

1. **Load corpus**: read `data/bootstrap_corpus.txt` (52 diverse paragraphs covering tech, philosophy, cooking, science, code, medicine, history, art)
2. **Chunk**: split on paragraph breaks, merge small chunks
3. **Format**: wrap each chunk in the Llama-3 chat template (critical — must match the server's format so embedding geometry is consistent)
4. **Embed**: run each formatted chunk through the GGUF backend -> list of 4096-dim tensors
5. **Train** (multiple epochs):
   - **Reconstruction**: shuffled batches of `train_step(embedding, embedding)`
   - **Predictive**: sequential pairs of `train_step(emb[i], emb[i+1])`
   - Energy drops across epochs (e.g., 2924 -> 1688 over 5 epochs)
6. **Compute deviations**: after training, run `infer()` on each embedding and store `settled - original`
7. **Save checkpoint**: PCHead state dict + all embeddings + all deviations + transition pairs

### Checkpoint Contents

```python
{
    "pc_head_state_dict": ...,           # trained weights
    "prompt_replay": [...],              # list of 4096-dim embedding tensors
    "prompt_deviation_replay": [...],    # list of 4096-dim deviation tensors
    "prompt_pair_replay": [...],         # list of (prev, curr) embedding pairs
    "pretrain_info": {
        "corpus": "...",
        "n_chunks": 52,
        "epochs": 5,
        "batch_size": 8,
        "settle_k": 50,
        "eta_w": 0.0001,
    },
}
```

### Pre-Training vs Online Learning Rate

Pre-training uses `eta_w = 1e-4` (10x the online rate) because:

- It processes many more examples (52 chunks * 5 epochs = 260 training steps)
- There's no risk of catastrophic forgetting during pre-training (replay buffer is the entire corpus)
- The goal is to form attractor basins quickly, not preserve sensitivity

The server uses `eta_w = 1e-5` for online learning to maintain novelty sensitivity.

### Corpus Design

The bootstrap corpus (`data/bootstrap_corpus.txt`) contains 52 diverse paragraphs deliberately spanning many domains:

- Computer science (transformers, Rust, binary trees, hash tables, Docker, REST APIs, TCP, MapReduce)
- Mathematics and physics (quantum mechanics, general relativity, Bayesian statistics, elliptic curves)
- Philosophy (Kant, Sartre, consciousness, negative capability)
- Biology and medicine (immune system, photosynthesis, sepsis, amygdala)
- Cooking (French onion soup, kimchi jjigae, sourdough bread, winemaking)
- History and culture (Westphalia, Renaissance, 2008 crisis, ukiyo-e)
- Literature (sonnet form, Keats)

Domain diversity ensures the PCHead develops distinct attractor basins for different types of content, enabling deviation routing to discriminate between domains.

## Energy-Triggered Memory Retrieval

Two independent pathways trigger MemoryGate retrieval:

### Pathway 1: Energy Thresholds

```
if E_recon > recon_threshold or E_predict > predict_threshold:
    search MemoryGate
```

Thresholds can be absolute values or dynamic percentiles (e.g., 95th percentile of the session's energy history). Energy-based retrieval fires on novelty — when the PCHead encounters unfamiliar content.

### Pathway 2: Deviation Routing

```
if max(cosine_similarity(current_deviation, stored_deviations)) > threshold:
    search MemoryGate
```

Deviation routing fires on structural similarity — when the PCHead's reaction to the current input matches its reaction to a past input. This can trigger even when energy is normal, if the deviation pattern is familiar.

### Retrieval Flow

When either pathway triggers:

1. Extract last user message as search query
2. Call MemoryGate MCP endpoint via JSON-RPC over HTTP
3. If results returned:
   - Format retrieved memories as system context
   - Prepend to the conversation
   - Re-embed the augmented prompt through the LLM (refreshes KV cache)
4. Generate response with augmented context

The LLM sees a longer prompt with relevant memories prepended. It doesn't know why the context appeared — the PCHead made the retrieval decision, the LLM just responds to what it sees.

**Failure handling**: all MemoryGate calls have a timeout cap (default 2s). If retrieval fails or times out, generation proceeds with the original prompt. Retrieval never blocks generation.

### Inverted RAG

This inverts the typical RAG (Retrieval-Augmented Generation) architecture:

- **Traditional RAG**: the LLM decides what to retrieve (via generated queries or user prompts)
- **PCDC**: the PCHead's settling dynamics decide when and what to retrieve. The LLM is purely downstream — it generates text, the PCHead governs context.

## Replay Buffers and Checkpointing

### Buffer Structure

The steering engine maintains several buffers:

| Buffer | Contents | Max Size | Purpose |
|--------|----------|----------|---------|
| `_prompt_replay` | Embeddings (4096-dim) | 10,000 | Reconstruction training replay |
| `_prompt_deviation_replay` | Deviation vectors (4096-dim) | 10,000 | Deviation routing comparisons |
| `_prompt_pair_replay` | (prev, curr) embedding pairs | 10,000 | Predictive training replay |
| `_recon_energy_history` | Energy scalars | 1,000 | Energy percentile computation |
| `_predict_energy_history` | Energy scalars | 1,000 | Energy percentile computation |
| `_deviation_match_history` | Cosine similarity scalars | 1,000 | Deviation routing statistics |

### Checkpoint Persistence

`POST /v1/pcdc/checkpoint` or automatic save on shutdown:

```python
{
    "pc_head_state_dict": ...,
    "prompt_replay": [...],
    "prompt_deviation_replay": [...],
    "prompt_pair_replay": [...],
    "prev_embedding": ...,
    "stats_turn_count": ...,
    "stats_total_retrieval_count": ...,
    "stats_deviation_routing_total": ...,
    "stats_recon_energy_history": [...],
    "stats_predict_energy_history": [...],
}
```

Checkpoints include everything needed to resume a session: trained weights, replay buffers, energy history, turn count, and the previous embedding (for predictive training on the next turn).

## Ghost Mic: Deviation Analysis

`scripts/ghost_mic.py`

Ghost Mic is an experimental script that analyzes PCHead deviation vectors in isolation, outside the server loop. It was used to validate the deviation routing concept.

### Mode A: LM Head Projection (Experimental)

Projects the settled state through the LLM's unembedding matrix (`output.weight`) to produce token probabilities. This tests whether the PCHead's output space aligns with the LLM's residual stream.

**Result**: it doesn't. All outputs are noise tokens (`addCriterion`, `/stdc`, `baise`). The PCHead's output manifold is disjoint from the LLM's — the settled states don't lie on the LLM's residual stream where unembedding would produce meaningful tokens.

This approach would require either a trained adapter matrix or a much more mature PCHead with explicit manifold alignment training.

### Mode B: Replay Buffer Retrieval (Working)

Uses deviation vectors as retrieval keys against the replay buffer. Four sub-tests:

- **B1**: deviation vs stored embeddings (cosine similarity)
- **B2**: deviation fingerprint matching (deviation vs stored deviations) -- the approach adopted in the server
- **B3**: magnitude context (deviation norm distribution)
- **B4**: transition pair analysis (next-turn prediction from pairs)

**Results with pre-trained checkpoint** (5 epochs on 52-paragraph corpus):

| Prompt | Best Match | Cosine Similarity |
|--------|------------|-------------------|
| "Explain Rust's ownership model" | Rust paragraph | 0.654 |
| "What is the categorical imperative?" | Kant paragraph | 0.608 |
| "Tell me about French onion soup" | Cooking paragraph | ~0.55 |

**Results without pre-training** (25 online turns only):

All cosine similarities 0.05-0.09 — near-orthogonal. No semantic discrimination. This confirmed that pre-training is essential.

## Long-Session Test Suite

`scripts/long_session_test.py` + `scripts/analyse_session.py` + `data/long_session_prompts.json`

The long-session test suite validates system behaviour over 100-200+ turns — well beyond the 20-25 turn sessions used during initial development.

### Test Runner

`long_session_test.py` follows the same `urllib.request` pattern as `gather_telemetry.py` but is designed for extended runs:

```
python scripts/long_session_test.py \
    --scenario full --delay 2.0 --max-tokens 80 \
    --output results/session_YYYYMMDD.json --label "long-session-v1"
```

**Prompt corpus** (`data/long_session_prompts.json`): 210 prompts organised as 6 structured scenarios (105 prompts) probing specific dynamics, plus a general corpus (105 prompts) across 21 domains. Each entry carries `domain` and `complexity` metadata for post-hoc analysis.

**Structured scenarios**:

| Scenario | Turns | Tests |
|----------|-------|-------|
| `sustained_domain` | 20 | Within-domain energy stabilisation |
| `rapid_switching` | 20 | E_predict under constant domain pivots |
| `gradual_drift` | 15 | Gradual vs abrupt transition dynamics |
| `return_to_origin` | 25 | Deviation routing across a domain gap |
| `complexity_ladder` | 15 | E_recon response to complexity within a domain |
| `repetition_probe` | 10 | Energy floor on repeated content + novelty spike |

**Running modes**: `--scenario full` (all 210), `--scenario all` (6 scenarios only), `--scenario general` (general corpus only), or any individual scenario name. Error handling: log and continue (or `--fail-fast` to abort). Fetches final `/v1/pcdc/stats` snapshot.

### Analysis Pipeline

`analyse_session.py` reads the runner's JSON output and produces 7 diagnostic plots plus a summary markdown:

1. **Energy Trajectory** — E_recon + E_predict with crossing markers, blended energy with median/IQR bands, cosine distance bars. Scenario boundaries annotated.
2. **Temperature Distribution** — Histogram + trajectory over turns, color-coded by domain.
3. **IQR/Median Evolution** — Rolling IQR scale at each turn. Key plot for validating adaptive scaling at scale.
4. **Deviation Match Score** — Trajectory with threshold line and domain annotations.
5. **Signal Crossing Analysis** — E_recon vs E_predict scatter, colored by domain.
6. **Per-Domain Statistics** — Grouped bars for mean energy, cosine distance, deviation match, temperature per domain.
7. **Deviation Norm Stability** — Norm trajectory over turns.

The auto-generated summary includes: run metadata, global statistics table, IQR stability at checkpoints (turn 25/50/100/200), signal crossing rates, per-domain summary, and auto-detected observations.

### Key Finding: System Stability at 210 Turns

Session 5 confirmed long-session stability across all subsystems:
- IQR/Median converges from 0.49 (turn 25) to 0.24 (turn 200)
- Energy signals remain non-monotonic (50% of transitions show E_recon increases)
- Temperature spans [0.62, 1.49] without floor/ceiling domination
- Deviation match stable at mean 0.82 (drift 0.007)

## Signal Summary

PCDC produces five signals per turn:

| Signal | Source | Range | Meaning |
|--------|--------|-------|---------|
| `reconstruction_energy` | Phase 1 train_step | [0, inf) | Content novelty — how hard is self-reconstruction? |
| `predictive_energy` | Phase 2 train_step | [0, inf) | Transition surprise — how unexpected is this turn? |
| `cosine_distance` | Embedding comparison | [0, 2] | Raw embedding geometry — how far is this turn from the last? |
| `deviation_match_score` | Deviation routing | [-1, 1] | Structural similarity — does the PCHead react to this like something it's seen before? |
| `adjusted_temperature` | Energy mapping | (0, 2) | LLM generation temperature — higher for novel, lower for familiar |

The first four are informational (reported in the `pcdc` metadata of every response). The fifth directly affects generation.

## Data Flow Diagram

```
                          User Prompt
                              |
                              v
                   +--------------------+
                   | Llama-3 Chat       |
                   | Template Formatting|
                   +--------------------+
                              |
                              v
                   +--------------------+
                   | GGUF LLM (frozen)  |
                   | embed_chat()       |
                   +--------------------+
                              |
                         embedding (4096-dim)
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
     +----------------+ +----------------+ +---------+
     | Phase 1:       | | Phase 2:       | | Cosine  |
     | Reconstruction | | Predictive     | | Distance|
     | train_step     | | train_step     | | vs prev |
     | (emb, emb)     | | (prev, emb)    | |         |
     +----------------+ +----------------+ +---------+
              |               |               |
          E_recon         E_predict       cos_dist
              |               |               |
              v               v               |
     +-----------------------------+          |
     | Blended Energy              |          |
     | E = beta*E_r + (1-b)*E_p    |          |
     +-----------------------------+          |
              |                               |
              v                               |
     +-----------------------------+          |
     | Temperature Mapping         |          |
     | T = T0 * (1 + a*tanh(...))  |          |
     +-----------------------------+          |
              |                               |
              v                               |
     +--------------------+                   |
     | infer(embedding)   |                   |
     | -> settled state   |                   |
     +--------------------+                   |
              |                               |
              v                               |
     +--------------------+                   |
     | Deviation Vector   |                   |
     | = settled - emb    |                   |
     +--------------------+                   |
              |                               |
              v                               |
     +----------------------------+           |
     | Deviation Routing          |           |
     | cos_sim vs stored devs     |           |
     +----------------------------+           |
              |                               |
         match_score                          |
              |                               |
              v                               v
     +------------------------------------------+
     | Retrieval Decision                        |
     | if E > threshold OR match > threshold     |
     |   -> query MemoryGate                     |
     |   -> augment prompt with retrieved context|
     +------------------------------------------+
              |
              v
     +--------------------+
     | LLM Generation     |
     | (adjusted temp)    |
     +--------------------+
              |
              v
         Response + pcdc metadata
```
