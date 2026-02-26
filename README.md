# PCDC — Predictive Coding Digital Circuit

A predictive coding network implemented as a stateful dynamical system in PyTorch. No backprop in the inner loop — learning emerges from local Hebbian-like weight updates driven by prediction error minimization.

Three tiers:

1. **Core PCNetwork** — MLP-shaped predictive coding network benchmarked on MNIST
2. **GGUF PCHead** — Frozen LLM (via GGUF) as feature extractor + lightweight PC classifier for continual learning
3. **Chat API Server** — OpenAI-compatible API where PCHead settling energy steers LLM generation temperature

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
uv sync --extra server         # adds fastapi, uvicorn, sse-starlette

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
│   ├── steering.py                # SteeringEngine — energy→temperature mapping
│   └── schemas.py                 # Pydantic models (request/response/SSE)
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

An OpenAI-compatible chat API that uses PCHead settling dynamics to steer LLM generation. Before each completion, the server embeds the prompt, runs PCHead settling, and uses the resulting energy to modulate temperature:

```
temp = base_temp * (1 + α * tanh((energy - median) / scale))
```

High energy (uncertain settling) → higher temperature → more exploratory output. Low energy (confident settling) → lower temperature → more focused output.

### Quick Start

```bash
uv sync --extra server

# Start the server
uv run pcdc-serve \
    --model path/to/model.gguf \
    --n-gpu-layers 33 \
    --pc-checkpoint pchead.ckpt

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
| `/v1/pcdc/stats` | GET | Steering statistics (energy median, request count, replay buffer size) |
| `/v1/pcdc/checkpoint` | POST | Save PCHead weights + stats to disk |
| `/v1/pcdc/train` | POST | Online training trigger (placeholder, returns 501) |

Responses include a `pcdc` metadata field with settling energy, convergence status, adjusted temperature, and settle steps. Standard OpenAI clients ignore this field.

### Checkpointing

Pass `--pc-checkpoint path/to/pchead.ckpt` to persist PCHead weights and steering stats across restarts. The checkpoint is loaded on startup (if the file exists) and saved on shutdown. Use `POST /v1/pcdc/checkpoint` for explicit saves.

### Server Options

```bash
uv run pcdc-serve \
    --model MODEL_PATH          # Required: path to GGUF model
    --host 0.0.0.0              # Bind address (default: 0.0.0.0)
    --port 8000                 # Port (default: 8000)
    --n-ctx 4096                # Context window size
    --n-threads 8               # CPU threads
    --n-gpu-layers 0            # GPU offload layers (0 = CPU only)
    --pc-alpha 0.5              # Energy-temperature coupling strength
    --pc-energy-scale 1.0       # Energy normalization scale
    --pc-checkpoint PATH        # Save/load PCHead state
    --log-level INFO            # Log level
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

# Chat API server
uv run pcdc-serve \
    --model path/to/model.gguf \
    --n-gpu-layers 33 \
    --pc-checkpoint pchead.ckpt \
    --port 8000
```

## Status

This is a research prototype. The core PC dynamics and local learning rules are implemented and tested. MNIST training runs end-to-end. The GGUF pipeline is wired up but requires a compatible GGUF model file to run.

**What works:**
- Settling reduces energy (verified across tanh/relu/gelu)
- Local Hebbian updates train the network without backprop
- Oscillation detection + adaptive damping stabilize settling
- Continual learning pipeline with replay and forgetting metrics
- Weight alignment diagnostic confirms PC ≈ backprop for linear case
- Chat API server with energy-based temperature steering
- PCHead checkpoint save/load across server restarts

**What's next:**
- PCHead training on conversation embeddings (needs quality signal design)
- v2 steering: project steering vector → logit bias via LLM embedding matrix
- Run MNIST to convergence and tune hyperparameters for >90% accuracy
- End-to-end GGUF continual learning with a real model
- Benchmark forgetting: PCHead vs baselines across task sequences
