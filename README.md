# PCDC — Predictive Coding Digital Circuit

A predictive coding network implemented as a stateful dynamical system in PyTorch. No backprop in the inner loop — learning emerges from local Hebbian-like weight updates driven by prediction error minimization.

Three tiers:

1. **Core PCNetwork** — MLP-shaped predictive coding network benchmarked on MNIST
2. **GGUF PCHead** — Frozen LLM (via GGUF) as feature extractor + lightweight PC classifier for continual learning
3. **Chat API Server** — OpenAI-compatible API where PCHead settling dynamics steer LLM generation: two-phase online learning produces reconstruction energy, predictive energy, and cosine distance signals, which drive temperature modulation and autonomous memory retrieval via MemoryGate

## How It Works

### Predictive Coding in 30 Seconds

A predictive coding network has layers connected by **generative weights** `W[l]` that make top-down predictions. Each layer computes a prediction error:

```
e[l] = x[l] - f(x[l+1] @ W[l])
```

The network **settles** by iteratively updating its internal states to minimize total energy `E = 0.5 * Σ ||e[l]||²`. Once settled, weights update using only local information:

```
ΔW[l] ∝ x[l+1]ᵀ @ (e[l] * f'(preact[l]))
```

No loss function. No backward pass. No computation graph. Each weight sees only the activity above it and the error below it.

### Three Operating Modes

| Mode | Clamped | Free | Use case |
|------|---------|------|----------|
| **Supervised** | Input + Label | Hidden layers | Training — settle with answer pinned |
| **Inference** | Input only | Hidden + Output | Classification — read output after settling |
| **Generative** | Latent only | Hidden + Input | Generation — read input layer after settling |

## Setup

Requires Python 3.11+.

```bash
# Clone and install
git clone https://github.com/PStryder/PCDC.git
cd PCDC
uv sync                        # core deps (torch, torchvision, etc.)

# Optional: GGUF backend for frozen LLM features
uv sync --extra gguf           # adds llama-cpp-python

# Optional: Chat API server
uv sync --extra server         # adds fastapi, uvicorn, sse-starlette, httpx

# Dev tools
uv sync --group dev            # adds pytest, pytest-cov
```

## Quick Start

### MNIST Benchmark

Train a PCNetwork on MNIST alongside a backprop MLP baseline:

```bash
uv run pcdc-mnist --epochs 20 --K 20 --eta-x 0.1 --eta-w 0.001
```

Or use the Python API directly:

```python
from pcdc.core.pc_network import PCNetwork
from pcdc.utils.config import PCConfig

config = PCConfig(
    layer_sizes=[784, 256, 256, 10],
    activation="tanh",
    eta_x=0.1,
    eta_w=0.001,
    K=20,
)
net = PCNetwork(config).to("cuda")

# Training: clamp input + label, settle, update weights
result = net.train_step(x_input, y_onehot)
# result: {"energy": ..., "correct": ..., "total": ..., "settle_steps": ...}

# Inference: clamp input, settle, read output
prediction, metrics = net.infer(x_input)
labels = prediction.argmax(dim=-1)
```

### Continual Learning

Sequential task training with replay buffers and forgetting metrics:

```bash
uv run pcdc-continual --n-tasks 5 --replay-mode joint --device cuda
```

This runs PCHead, a linear probe, and an SGD MLP on the same task sequence, then reports per-task accuracy and forgetting.

## Architecture

```
src/pcdc/
├── core/                          # The predictive coding engine
│   ├── pc_network.py              # PCNetwork — settle, update, learn (~590 lines)
│   ├── activations.py             # f(x) + f'(x) pairs: tanh, relu, gelu, linear
│   ├── energy.py                  # Energy computation + convergence checking
│   └── clamp.py                   # ClampMode enum + mask generation
│
├── gguf/                          # Frozen LLM feature extraction tier
│   ├── gguf_backend.py            # GGUF loading + embed_chat() for serving
│   ├── pc_head.py                 # PCHead — thin PCNetwork subclass
│   ├── datasets.py                # Split-class + template-shift task sequences
│   └── baselines.py               # Linear probe, SGD MLP, replay buffer
│
├── server/                        # OpenAI-compatible chat API
│   ├── app.py                     # FastAPI app, routes, CLI entry point
│   ├── steering.py                # SteeringEngine — two-phase settling, retrieval gate, temperature mapping
│   ├── memory_client.py           # Sync httpx client for MemoryGate MCP JSON-RPC
│   ├── telemetry_db.py            # SQLite per-turn telemetry recorder (WAL mode)
│   └── schemas.py                 # Pydantic models (request/response/SSE)
│
├── data/
│   ├── bootstrap_corpus.txt       # Diverse text corpus for PCHead pre-training
│   └── long_session_prompts.json  # 210 prompts with domain/complexity metadata for long-session testing
│
scripts/                               # Standalone experiment and utility scripts
├── pretrain_pchead.py                 # Offline PCHead pre-training for attractor basins
├── ghost_mic.py                       # Deviation vector analysis experiments
├── gather_telemetry.py                # On-corpus telemetry collection (20 prompts, 8 domains)
├── gather_telemetry_offcorpus.py      # Off-corpus generalization test (20 prompts, novel domains)
├── long_session_test.py               # 210-turn long-session test runner with full transcript capture
├── analyse_session.py                 # Post-hoc analysis: 7 diagnostic plots + summary markdown
└── plot_session.py                    # Session energy visualization
│
├── training/                      # Training loops and evaluation
│   ├── train_mnist_pc.py          # PC training loop on MNIST
│   ├── baseline_mlp_backprop.py   # Standard backprop MLP for comparison
│   ├── eval_mnist_pc.py           # Energy curves + convergence visualization
│   └── continual_runner.py        # Sequential task orchestration + forgetting
│
└── utils/
    ├── config.py                  # PCConfig + ContinualConfig dataclasses
    └── metrics_logger.py          # SettleMetrics, EpochMetrics, OscillationEvent
```

## PCNetwork Internals

### Settling Dynamics

The `settle()` loop runs K iterations (default 20) inside `torch.no_grad()`:

1. Compute prediction errors `e[l]` and cache preactivations
2. Check for oscillations (consecutive energy increases)
3. Check convergence (relative energy change below tolerance)
4. Update free states: `dx[l] = -e[l] + (e[l-1] @ Rᵀ) * f'(pre)`
5. Apply state normalization (LayerNorm/RMSNorm on free layers)

Early stopping when `|ΔE/E| < tol` for `converge_patience` consecutive steps.

### Stability Controls

| Control | Config key | Default | Effect |
|---------|-----------|---------|--------|
| Gradient clipping | `dx_clip` | 50.0 | Clamps state update magnitude |
| Leaky integration | `leaky_rate` | 0.0 | Smooths updates: `x += λ * η * dx` |
| Oscillation damping | `stability_mode` | `"adaptive"` | Halves η_x on consecutive energy increases |
| State normalization | `state_norm` | `"layernorm"` | Normalizes free layer activations each step |
| Convergence patience | `converge_patience` | 3 | Steps below tol before early stop |

**Stability modes:**
- `"adaptive"` — halves η_x when oscillations detected, continues settling
- `"strict"` — aborts settling immediately on energy divergence (>10x initial)
- `"none"` — no oscillation detection or intervention

### Layerwise Learning Rate Scaling

The base `eta_x` is scaled per layer:

- `"uniform"` — same η_x everywhere
- `"inv_dim"` — `η_x / sqrt(D_l)`, normalizes for layer width (default)
- `"learned"` — trainable per-layer scalar multipliers

### Weight Alignment Diagnostic

`compute_weight_alignment()` measures cosine similarity between the local PC weight update and the equivalent backprop gradient. For tied-weight linear networks, these should be identical (known theoretical result). Useful for verifying the implementation and understanding how nonlinearity affects alignment.

### Energy-Based Confidence

`predict_with_confidence()` returns per-example settling energy alongside predictions. Lower energy = the network settled cleanly = higher confidence. Examples that don't settle well (high residual energy) are uncertain.

```python
y_pred, energy, converged = net.predict_with_confidence(x_input)
uncertain = energy > energy.median()  # flag high-energy examples
```

## GGUF Tier

### Pipeline

```
Prompt text → GGUF LLM (frozen) → hidden-state features → PCHead → prediction
```

The LLM is a frozen feature extractor. All learning happens in the PCHead, which is a standard PCNetwork operating on the extracted features.

### Feature Extraction

```python
from pcdc.gguf.gguf_backend import GGUFBackend

backend = GGUFBackend("model.gguf", n_gpu_layers=35)

# Extract and cache features
features, labels = backend.precompute_dataset(
    texts, labels, cache_path="features.pt",
    feature_layer="last",  # or "middle" or int
)
```

Feature layer options:
- `"last"` — final layer embedding (default)
- `"middle"` — layer at `n_layers // 2` (requires llama-cpp-python per-layer support)
- `int` — specific layer index

If the binding can't return hidden states, the fallback is last-layer embeddings. A logits + projection approach is also viable but not yet implemented — it trades feature quality for broader binding compatibility.

### Continual Learning Setup

Tasks are presented sequentially. After training on each task, all models are evaluated on all tasks seen so far, building an accuracy matrix.

**Anti-forgetting mechanisms:**
- **Replay buffer** — reservoir sampling, mixed into training batches (default 25%)
- **Joint settling** — replay examples are concatenated with current-task examples before the settle loop, acting as implicit regularization
- **L2 regularization** — penalty toward previous-task weights (baselines only)

**Baselines** (same features, same replay protocol):
- Linear probe (single `nn.Linear`)
- SGD MLP (same hidden architecture as PCHead, trained with Adam + cross-entropy)

**Metrics:**
- Per-task accuracy matrix `A[after_task][eval_task]`
- Forgetting: `max(A[t..T][t]) - A[T][t]` for each task t
- Average final accuracy across all tasks

## Chat API Server

An OpenAI-compatible chat API where the PCHead acts as the conversation's executive controller. Before each completion, the engine runs two online learning phases that produce distinct energy signals, then uses those signals to both modulate temperature and autonomously retrieve memory context.

### Two-Phase Settling

Each request triggers two `train_step()` calls with a conservative online learning rate (recommended `eta_w=0.00001`):

1. **Phase 1 — Reconstruction**: `train_step(embedding, embedding)` → reconstruction energy `E_recon`. Measures how novel this content is (self-reconstruction difficulty).
2. **Phase 2 — Predictive**: `train_step(prev_embedding, current_embedding)` → predictive energy `E_predict`. Measures how surprising this transition is from the previous turn.
3. **Cosine Distance**: `1 - cos_sim(prev_embedding, current_embedding)` — raw embedding-space distance between consecutive turns, independent of PCHead learning state.

Both phases mix in replay samples (configurable `replay_k`) from prompt embedding buffers to prevent catastrophic forgetting. Settle steps per phase are configurable via `--pc-settle-k` (default: PCHead config K=20, recommended: 50-100). On the first turn, Phase 2 is skipped and `E_predict = E_recon`.

The blended energy drives temperature:

```
energy = β * E_recon + (1-β) * E_predict
temp = base_temp * (1 + α * tanh((energy - median) / scale))
```

By default, `scale` is adaptive (IQR-based): computed from the interquartile range of the blended energy history so that `±1 IQR` from median maps to `tanh(±1) ≈ ±0.76`. This gives smooth temperature gradation across the full `[base*(1-α), base*(1+α)]` range regardless of the model's energy regime. A fixed scale can be set with `--pc-energy-scale`.

After both training phases, `infer()` runs on the updated weights to produce a steering vector and a deviation vector (`settled - original`), which is used by deviation routing (see below).

**Tuning insight:** At `eta_w=1e-4`, energy declines monotonically (~17% over 8 turns) regardless of topic shifts — the network adapts faster than novelty can register. At `eta_w=1e-5`, decline is only ~2.5% over 12 turns, energy responds to domain shifts, and temperature actually varies (0.35–1.05 vs stuck at 0.35). See `docs/session2_energy_plot.png` for the comparison.

### Deviation Routing

After settling, the PCHead produces a deviation vector: `deviation = settled_state - original_embedding`. This vector encodes what the network "wanted to change" about the input — its prediction error fingerprint.

Deviation routing compares the current turn's deviation against stored deviations from all previous turns (deviation-to-deviation cosine similarity). When a match exceeds the threshold, it triggers MemoryGate retrieval — the PCHead's prediction errors drive memory recall:

```
deviation = settled - embedding
cos_scores = cosine_similarity(deviation, stored_deviations)
if max(cos_scores) > threshold:
    trigger MemoryGate retrieval
```

This creates a second retrieval pathway independent of energy thresholds. Energy-based retrieval fires on novelty (high energy = unfamiliar content). Deviation routing fires on structural similarity (similar prediction error pattern = similar type of content, even if energy is normal).

**Pre-training is required** for deviation routing to work. Without it, all deviations are near-orthogonal (cosine ~0.05-0.09) because the attractor basins haven't formed. After pre-training on the bootstrap corpus, cosine similarities reach 0.49-0.65 with semantic domain matching (e.g., a Rust question matches the Rust corpus paragraph at cos=0.654).

Enable with `--pc-deviation-routing --pc-deviation-threshold 0.4`.

### Energy-Triggered Memory Retrieval

When either energy exceeds its threshold, the steering engine autonomously queries MemoryGate for relevant context — the PCHead decides when to retrieve, not the LLM:

```
if E_recon > recon_threshold or E_predict > predict_threshold:
    results = memorygate.search(last_user_message)
    inject results as system context
    re-embed augmented prompt (refreshes KV cache)
```

Thresholds can be absolute values or **dynamic percentiles** — `--mg-predict-percentile 95` triggers retrieval when predictive energy exceeds the 95th percentile of the session's history, adapting to the model's energy distribution automatically.

The LLM just sees a longer prompt with relevant context prepended. It doesn't know why the context appeared. This inverts the typical RAG architecture: settling dynamics drive retrieval, the LLM is purely downstream.

**Common case** (below threshold): zero overhead, original warm KV cache used.
**Retrieval case**: one extra `embed_and_warm` + MemoryGate HTTP call (~2s timeout cap).

The MemoryGate client (`memory_client.py`) calls via MCP JSON-RPC over HTTP. All failures are caught — retrieval never blocks generation.

### Quick Start

```bash
uv sync --extra server

# Step 1: Pre-train PCHead (recommended before first use)
python scripts/pretrain_pchead.py \
    --model path/to/model.gguf \
    --n-gpu-layers 33 \
    --corpus data/bootstrap_corpus.txt \
    --epochs 5 \
    --batch-size 8 \
    --output pchead_pretrained.ckpt

# Step 2: Serve (basic — no retrieval)
uv run pcdc-serve \
    --model path/to/model.gguf \
    --n-gpu-layers 33 \
    --pc-checkpoint pchead_pretrained.ckpt

# Full stack: pre-trained checkpoint + deviation routing + telemetry + MemoryGate
uv run pcdc-serve \
    --model path/to/model.gguf \
    --n-gpu-layers 33 \
    --pc-checkpoint pchead_pretrained.ckpt \
    --pc-online-eta-w 0.00001 \
    --pc-settle-k 100 \
    --pc-deviation-routing \
    --pc-deviation-threshold 0.4 \
    --telemetry-db pcdc_telemetry.db \
    --mg-url http://localhost:8080/mcp \
    --mg-predict-percentile 95

# Chat via curl
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

Any OpenAI-compatible frontend can connect to `http://localhost:8000/v1`.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (streaming + non-streaming) |
| `/v1/models` | GET | List available models |
| `/v1/pcdc/stats` | GET | Steering statistics (energy medians, replay sizes, retrieval count) |
| `/v1/pcdc/checkpoint` | POST | Save PCHead weights, stats, and replay buffers to disk |
| `/v1/pcdc/train` | POST | Online training trigger (placeholder, returns 501) |

Responses include a `pcdc` metadata field:

```json
{
  "settling_energy": 10090.5,
  "reconstruction_energy": 10172.3,
  "predictive_energy": 10008.7,
  "converged": false,
  "adjusted_temperature": 0.35,
  "settle_steps": 200,
  "cosine_distance": 0.280,
  "retrieval_triggered": false,
  "retrieval_count": 0,
  "deviation_match_score": 0.654
}
```

Standard OpenAI clients ignore this field.

### Checkpointing

Pass `--pc-checkpoint path/to/pchead.ckpt` to persist PCHead weights, steering stats, replay buffers, and previous-turn embedding across restarts. The checkpoint is loaded on startup (if the file exists) and saved on shutdown. Use `POST /v1/pcdc/checkpoint` for explicit saves.

### Server Options

```bash
uv run pcdc-serve \
    --model MODEL_PATH              # Required: path to GGUF model
    --host 0.0.0.0                  # Bind address (default: 0.0.0.0)
    --port 8000                     # Port (default: 8000)
    --n-ctx 4096                    # Context window size
    --n-threads 8                   # CPU threads
    --n-gpu-layers 0                # GPU offload layers (0 = CPU only)
    --pc-alpha 0.5                  # Energy-temperature coupling strength
    --pc-energy-scale 0.0           # Energy normalization (0 = adaptive IQR, default)
    --pc-checkpoint PATH            # Save/load PCHead state + replay buffers
    --pc-beta 0.5                   # Blend ratio: energy = β*E_recon + (1-β)*E_predict
    --pc-online-eta-w 0.00001       # Online learning rate (recommended: 1e-5)
    --pc-replay-k 4                 # Replay samples per training phase
    --pc-settle-k 100               # Settle steps per phase (default: PCHead K)
    --mg-url URL                    # MemoryGate MCP endpoint (enables retrieval)
    --mg-timeout 2.0                # MemoryGate request timeout in seconds
    --mg-bearer-token TOKEN         # Bearer token for MemoryGate auth
    --mg-recon-threshold INF        # Reconstruction energy retrieval threshold
    --mg-predict-threshold INF      # Predictive energy retrieval threshold
    --mg-predict-percentile 95      # Dynamic predict threshold as percentile of history
    --mg-retrieval-limit 3          # Max memory items to retrieve
    --mg-retrieval-min-confidence 0.5  # Min confidence for retrieved memories
    --pc-deviation-routing          # Enable deviation-based retrieval routing
    --pc-deviation-threshold 0.6    # Cosine similarity threshold for deviation routing
    --telemetry-db PATH             # SQLite DB for per-turn telemetry (deviation vectors, energies, match scores)
    --log-level INFO                # Log level
```

## Configuration Reference

### PCConfig

```python
PCConfig(
    # Architecture
    layer_sizes=[784, 256, 256, 10],  # [input, *hidden, output]
    activation="tanh",                 # tanh | relu | gelu | linear
    tied_weights=True,                 # R[l] = W[l]ᵀ (untied adds separate feedback weights)

    # Settling
    eta_x=0.1,                         # base state learning rate
    eta_x_scale="inv_dim",             # uniform | inv_dim | learned
    K=20,                              # max settle iterations
    converge_tol=1e-4,                 # relative energy change threshold
    converge_patience=3,               # consecutive below-tol steps to stop

    # Stability
    state_norm="layernorm",            # none | layernorm | rmsnorm
    leaky_rate=0.0,                    # leaky integration (0 = off)
    dx_clip=50.0,                      # gradient clipping on dx
    stability_mode="adaptive",         # adaptive | strict | none

    # Weight learning
    eta_w=0.001,                       # weight update rate
    weight_decay=0.0,                  # L2 on weights

    # Diagnostics
    verbose=False,                     # print per-step energy to stderr
)
```

### ContinualConfig

```python
ContinualConfig(
    hidden_sizes=[1024, 256],          # PCHead hidden layers
    num_classes=10,
    replay_size=1000,                  # replay buffer capacity
    replay_mix=0.25,                   # fraction of batch from replay
    replay_mode="joint",               # joint | separate
    l2_prev_weight=0.01,              # L2 toward prev weights (baselines)
    feature_layer="last",              # GGUF extraction layer
    n_tasks=5,
    classes_per_task=2,
)
```

## Tests

```bash
uv run pytest tests/ -v              # 59 tests
uv run pytest tests/ -v --cov=pcdc   # with coverage
```

Test coverage includes:
- Activation derivatives match autograd (tanh, gelu, linear)
- Energy computation against known values
- Monotonic energy descent during settling (all activations)
- Clamped layers are bitwise unchanged after settling
- Batch independence (examples don't leak into each other)
- Local weight updates move weights in correct direction
- Gradient equivalence: PC update matches backprop for linear networks
- Oscillation detection fires and adaptive damping responds
- Strict mode aborts on energy divergence
- State normalization applied only to free layers
- PCHead layer construction and train/infer round-trip

## CLI Reference

```bash
# MNIST benchmark (PC network + backprop baseline)
uv run pcdc-mnist \
    --epochs 20 \
    --batch-size 64 \
    --eta-x 0.1 \
    --eta-w 0.001 \
    --K 20 \
    --activation tanh \
    --layers 784 256 256 10 \
    --device cuda \
    --seed 42 \
    --skip-baseline \
    --save-dir ./experiments

# Continual learning
uv run pcdc-continual \
    --n-tasks 5 \
    --replay-mode joint \
    --device cuda \
    --seed 42 \
    --output-dir ./experiments

# Pre-train PCHead offline
python scripts/pretrain_pchead.py \
    --model path/to/model.gguf \
    --n-gpu-layers 33 \
    --corpus data/bootstrap_corpus.txt \
    --epochs 5 \
    --batch-size 8 \
    --settle-k 50 \
    --eta-w 0.0001 \
    --output pchead_pretrained.ckpt

# Ghost Mic experiment (deviation analysis)
python scripts/ghost_mic.py \
    --model path/to/model.gguf \
    --n-gpu-layers 33 \
    --checkpoint pchead_pretrained.ckpt \
    --prompt "Explain Rust's ownership model" \
    --mode b

# Chat API server (basic)
uv run pcdc-serve \
    --model path/to/model.gguf \
    --n-gpu-layers 33 \
    --pc-checkpoint pchead_pretrained.ckpt \
    --port 8000

# Chat API server (full stack)
uv run pcdc-serve \
    --model path/to/model.gguf \
    --n-gpu-layers 33 \
    --pc-checkpoint pchead_pretrained.ckpt \
    --pc-online-eta-w 0.00001 \
    --pc-settle-k 100 \
    --pc-deviation-routing \
    --pc-deviation-threshold 0.4 \
    --telemetry-db pcdc_telemetry.db \
    --mg-url http://localhost:8080/mcp \
    --mg-predict-percentile 95
```

## Status

This is a research prototype. The core PC dynamics and local learning rules are implemented and tested. MNIST training runs end-to-end. The GGUF pipeline is wired up but requires a compatible GGUF model file to run.

**What works:**
- Settling reduces energy (verified across tanh/relu/gelu)
- Local Hebbian updates train the network without backprop
- Oscillation detection + adaptive damping stabilize settling
- Continual learning pipeline with replay and forgetting metrics
- Weight alignment diagnostic confirms PC ≈ backprop for linear case
- Two-phase online learning (reconstruction + predictive) during generation
- Three complementary signals: E_recon (content novelty), E_predict (transition surprise), cosine distance (embedding geometry)
- Energy-triggered autonomous memory retrieval via MemoryGate with dynamic percentile thresholds
- Configurable settle steps per phase (K override for online learning vs offline)
- Replay buffer mixing in both training phases to prevent catastrophic forgetting
- PCHead checkpoint save/load with full replay state across server restarts
- Plotting script for energy and cosine distance visualization (`scripts/plot_session.py`)
- Validated across five live sessions: eta_w=1e-5 preserves novelty sensitivity over 210+ turns
- Offline pre-training bootstraps attractor basins from a text corpus (`scripts/pretrain_pchead.py`)
- Deviation routing: deviation-to-deviation cosine similarity triggers memory retrieval based on prediction error fingerprints
- Pre-trained deviations show semantic domain matching (Rust prompt matches Rust paragraph at cos=0.654, philosophy matches philosophy at cos=0.608)
- Ghost Mic experiment validates deviation routing end-to-end (`scripts/ghost_mic.py`)
- Dual replay buffers (embeddings + deviations) persist through checkpoints and survive server restarts
- Adaptive IQR temperature scale — auto-calibrates to the model's energy distribution, replaces fixed scale that caused saturation (18 unique temps vs 3)
- SQLite telemetry database records per-turn deviation vectors, energies, match scores, and temperature (`--telemetry-db`)
- **Off-corpus generalization confirmed**: pre-trained deviation matches generalize to novel domains not in the bootstrap corpus (avg 0.728 off-corpus vs 0.731 on-corpus vs ~0.07 untrained baseline). Tested on sports, law, music, agriculture, film, psychology, linguistics, and truly alien domains (dog grooming, cricket, knitting)
- **210-turn long-session stability confirmed**: IQR adaptive scale converges by turn ~50 (IQR/Median: 0.494→0.615→0.380→0.238 at turns 25/50/100/200); E_recon rises on 50% of transitions (definitively not monotonically declining at scale); signal crossing rate 50.2% (105/209); temperature stable across full session (mean 1.16, range 0.62–1.49, no floor/ceiling hits); deviation match stable with CV=8.5% and drift of only 0.007 between first and second half. Tested with 6 structured scenarios (sustained domain, rapid switching, gradual drift, return to origin, complexity ladder, repetition probe) plus 105 general prompts across 21 domains

**What's next:**
- Store text alongside deviation replay entries so deviation-matched text can be used as retrieval query directly (currently the user's prompt is used)
- Cosine distance as third retrieval gate signal (high cos_dist = domain shift)
- v2 steering: project steering vector into logit bias via trained adapter (lm_head projection produces noise without manifold alignment)
- Run MNIST to convergence and tune hyperparameters for >90% accuracy
- End-to-end GGUF continual learning with a real model
- Benchmark forgetting: PCHead vs baselines across task sequences

## Session 5 — Long-Session Characterisation (210 Turns)

The longest test: 210 turns across 6 structured scenarios and 21 domains, probing system stability at scale. Full transcripts, telemetry, and analysis in `results/`.

### Test Structure

| Scenario | Turns | What it probes |
|----------|-------|----------------|
| `sustained_domain` | 20 | 10 CS → 10 philosophy. Within-domain E_recon stabilisation |
| `rapid_switching` | 20 | Alternating 6 domains, 2 full cycles. E_predict under constant pivots |
| `gradual_drift` | 15 | CS → math → physics → chemistry → biology. Gradual vs abrupt transitions |
| `return_to_origin` | 25 | 5 CS, 15 philosophy, 5 CS (same sub-topics). Deviation routing across a gap |
| `complexity_ladder` | 15 | CS only: 5 simple → 5 moderate → 5 dense. E_recon vs complexity |
| `repetition_probe` | 10 | Same prompt 5×, novel, repeat, 3 novel. Energy floor + spike |
| General corpus | 105 | 21 domains, mixed complexity, edge cases ("Why?", code, German) |

### Key Findings

**IQR adaptive scale converges:**

| Checkpoint | Median | IQR | IQR/Median |
|------------|--------|-----|------------|
| Turn 25 | 3438 | 1699 | 0.494 |
| Turn 50 | 4910 | 3021 | 0.615 |
| Turn 100 | 5899 | 2240 | 0.380 |
| Turn 200 | 6452 | 1538 | 0.238 |

The ΔIQR plot shows wild swings in the first ~40 turns, then flatlines near zero. By turn 50-60 the adaptive scale has found its footing and holds steady through 210.

**Energy is NOT monotonically declining:** E_recon rises on 50% of transitions (104/209). The PCHead responds to novel content at every turn, even at turn 210. This definitively answers the session 2 question — the 7/25 rises were not noise.

**Signal crossing rate: 50.2%** (105/209 transitions). E_recon and E_predict are genuinely independent signals — the scatter plot confirms broad spread around the diagonal with no domain clustering.

**Temperature stays responsive:** Mean 1.16, range 0.62–1.49, no floor or ceiling hits. No drift over time. The adaptive IQR scale prevents the saturation seen in session 3's fixed scale.

**Deviation routing is stable at scale:** Mean deviation match 0.82, CV=8.5%, drift of only 0.007 between first and second halves. The repetition probe shows deviation match pinning to 1.000 during repeated prompts (turns 96-105), dropping on novel input. Edge cases ("Why?", German text) correctly produce the lowest scores (~0.63-0.66).

**No performance degradation:** Response time stable at ~18s per turn across all 210 turns.

### Running the Test Suite

```bash
# Quick validation (10 turns)
python scripts/long_session_test.py --scenario repetition_probe --delay 1.0

# All structured scenarios (105 turns)
python scripts/long_session_test.py --scenario all

# Full run (210 turns, ~70 min)
python scripts/long_session_test.py --scenario full --label "my-session"

# Analyse results (7 plots + summary markdown)
python scripts/analyse_session.py --input results/session_YYYYMMDD_HHMMSS.json
```

### Analysis Outputs

The analysis script produces 7 diagnostic plots and a summary markdown:

1. **Energy Trajectory** — E_recon/E_predict with crossing markers, blended energy with IQR bands, cosine distance
2. **Temperature Distribution** — trajectory over turns (color-coded by domain) + histogram
3. **IQR/Median Evolution** — cumulative median, IQR, and ΔIQR convergence
4. **Deviation Match Score** — trajectory with threshold line and domain annotations
5. **Signal Crossing Analysis** — E_recon vs E_predict scatter, colored by domain
6. **Per-Domain Statistics** — grouped bars with error bars for all metrics
7. **Deviation Norm Stability** — trajectory with mean/σ bands and CV

## Documentation

- [System Architecture](docs/system_architecture.md) — detailed explanation of how all components work together
- [Session 3 — Pre-Trained PCHead + Deviation Routing](docs/session_transcript_2026-02-27_session3.md) — first live session with pre-trained checkpoint, deviation routing, and adaptive IQR temperature fix
- [Session 4 — Off-Corpus Generalization](docs/session_transcript_2026-02-27_session4.md) — test of deviation matching on 17 domains not in the bootstrap corpus
- Session 5 — Long-Session Characterisation: `results/session_20260228_transcript_analysis/` (7 plots + summary)
