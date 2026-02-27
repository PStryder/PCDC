# PCDC Session 3 — Pre-Trained PCHead with Deviation Routing and Telemetry

**Date:** 2026-02-27
**Server:** pcdc-serve on localhost:8000
**Model:** Llama-3-8B-Lexi-Uncensored.Q8_0.gguf (33 GPU layers)
**PCHead:** [4096, 1024, 256, 4096], **pre-trained** checkpoint (5 epochs on 52-paragraph corpus)
**Config changes from Session 2:**
- Pre-trained checkpoint: `pchead_pretrained.ckpt` (energy dropped 2924 -> 1688 during pre-training)
- `online_eta_w`: **0.00001** (same as session 2)
- `settle_k`: 100 -> **50** (reduced, 100 total for two-phase)
- **Deviation routing: enabled** (threshold=0.4, deviation-to-deviation cosine similarity)
- **Telemetry DB: enabled** (`pcdc_telemetry.db`, SQLite, full per-turn recording)
- MemoryGate retrieval: **not connected** this session (no --mg-url)

**Session ID:** `101ff7d3-af39-4220-9c59-2c4955b1aece`

**Key differences from prior sessions:**
1. First session with a **pre-trained** PCHead — attractor basins bootstrapped offline
2. First session with **deviation routing** active — deviation-to-deviation cosine similarity computed per turn
3. First session with **SQLite telemetry** — every turn recorded with full metrics + deviation vectors
4. Prompts sent programmatically via `scripts/gather_telemetry.py` (20 diverse prompts across 8 domain clusters)

---

## Prompt Sequence

Prompts were organized into domain clusters with deliberate cross-domain transitions to test energy and deviation routing responses:

| # | Cluster | Prompt |
|---|---------|--------|
| 1 | CS/Programming | Explain how Rust's borrow checker prevents data races at compile time. |
| 2 | CS/Programming | What is the difference between a mutex and a semaphore? |
| 3 | CS/Programming | How does garbage collection work in the JVM? |
| 4 | Philosophy | What did Kant mean by the categorical imperative? |
| 5 | Philosophy | Explain Sartre's concept of radical freedom and bad faith. |
| 6 | Philosophy | What is the hard problem of consciousness? |
| 7 | Cooking | How do you make authentic French onion soup? |
| 8 | Cooking | What is the difference between fermentation in sourdough vs commercial yeast bread? |
| 9 | Science | Explain the double-slit experiment in quantum mechanics. |
| 10 | Science | How does CRISPR-Cas9 gene editing work? |
| 11 | Science | What causes the greenhouse effect? |
| 12 | CS (return) | How does TCP ensure reliable data delivery over an unreliable network? |
| 13 | CS (return) | Explain how a hash table handles collisions. |
| 14 | History | What were the main causes of the 2008 financial crisis? |
| 15 | History | How did the Treaty of Westphalia shape modern international relations? |
| 16 | Art/Mixed | Compare the aesthetics of Japanese ukiyo-e prints with European Renaissance painting. |
| 17 | AI/Mixed | If predictive coding is how the brain works, what does that imply for artificial intelligence? |
| 18 | CS (return) | Explain ownership and lifetimes in Rust. |
| 19 | Philosophy (return) | What is Kant's distinction between phenomena and noumena? |
| 20 | Cooking (return) | Describe the process of making kimchi jjigae. |

---

## Full Energy & Deviation Trajectory (20 turns)

| Turn | Domain | E_recon | E_predict | Blended | Temp | Cos Dist | Dev Match | Match Idx | Signal |
|------|--------|---------|-----------|---------|------|----------|-----------|-----------|--------|
| 1 | CS/Rust | 3272.9 | 3272.9 | 3272.9 | 1.000 | N/A | 0.3023 | 3 | First turn, no prior |
| 2 | CS/Mutex | 3068.5 | 3231.5 | 3150.0 | 0.500 | 0.269 | **0.7459** | 52 | Same domain, strong match |
| 3 | CS/GC | 3325.9 | 3170.3 | 3248.1 | 1.000 | 0.314 | **0.7873** | 52 | Corpus CS match |
| 4 | Phil/Kant | 5132.7 | 3277.5 | 4205.1 | **1.500** | **0.449** | 0.5977 | 52 | **E_recon spike** on domain shift |
| 5 | Phil/Sartre | 3333.9 | 3371.9 | 3352.9 | 1.500 | 0.232 | **0.7709** | 55 | Within-domain, low cos |
| 6 | Phil/Consc. | 3437.3 | 3351.7 | 3394.5 | 1.500 | 0.370 | **0.7373** | 55 | Philosophy cluster match |
| 7 | Cook/Soup | 3446.6 | **5053.1** | 4249.9 | 1.500 | **0.433** | 0.5715 | 57 | **E_predict spike** on domain shift |
| 8 | Cook/Bread | 4878.9 | 4944.6 | 4911.8 | 1.500 | 0.362 | 0.6716 | 57 | Both energies elevated |
| 9 | Sci/Quantum | 4886.5 | 3334.4 | 4110.5 | 1.500 | 0.303 | **0.8557** | 57 | Highest match so far |
| 10 | Sci/CRISPR | 4924.1 | 3307.0 | 4115.5 | 1.500 | 0.322 | **0.7117** | 52 | Science cluster |
| 11 | Sci/Climate | 6611.6 | 3200.5 | 4906.1 | 1.500 | 0.293 | **0.8556** | 57 | **E_recon spike** (6612!) |
| 12 | CS/TCP | 4857.8 | 3381.7 | 4119.8 | 1.500 | **0.437** | **0.7041** | 52 | Return to CS domain |
| 13 | CS/Hash | 8237.4 | 6561.1 | **7399.3** | 1.500 | 0.306 | **0.7877** | 52 | **Session max energy** |
| 14 | Hist/2008 | 3466.5 | **6577.8** | 5022.2 | 1.500 | **0.404** | **0.8495** | 62 | **E_predict spike** on domain shift |
| 15 | Hist/Westph. | 3287.7 | **6713.8** | 5000.7 | 1.500 | 0.160 | **0.8463** | 65 | Highest E_predict |
| 16 | Art/Mixed | 3280.5 | 3313.2 | 3296.8 | **0.500** | 0.326 | 0.6960 | 59 | Energy drops, temp drops |
| 17 | AI/Mixed | 3240.7 | 3249.4 | 3245.1 | **0.500** | 0.320 | 0.7322 | 61 | **Session min energy** |
| 18 | CS/Rust | 4899.9 | 3280.5 | 4090.2 | 0.500 | 0.371 | **0.8468** | 52 | Return to Rust |
| 19 | Phil/Kant | 3322.3 | **6553.9** | 4938.1 | 1.500 | **0.491** | **0.9559** | 55 | **Session max match!** |
| 20 | Cook/Kimchi | 4890.3 | 3387.2 | 4138.7 | 1.500 | **0.473** | **0.7429** | 58 | Return to cooking |

---

## Key Observations

### Deviation Routing Works

The pre-trained PCHead produces semantically meaningful deviation matches:

- **Average match score: 0.74** (range 0.30 - 0.96)
- **Turn 19 (Kant/noumena) matched at 0.9559** — the highest in the session — against corpus index 55 (Sartre/existentialism paragraph), showing the PCHead clusters philosophy topics tightly
- **CS prompts consistently match corpus index 52** (lock-free concurrent queue paragraph) — the PCHead recognizes CS/programming as a coherent domain
- **Cooking prompts match corpus indices 57-58** (fermentation/sourdough paragraphs)
- **History prompts match corpus indices 62, 65** (2008 crisis, Westphalia paragraphs from the corpus)

Match indices map to the pre-trained corpus entries (indices 0-51 = corpus, 52+ = corpus entries in the replay buffer after pre-training).

### Energy Responds to Domain Shifts

| Transition | E_predict | Cos Dist | Signal |
|------------|-----------|----------|--------|
| CS -> Philosophy (turn 3->4) | 3277 (normal) | **0.449** | Cos dist detects shift |
| Philosophy -> Cooking (turn 6->7) | **5053** | **0.433** | E_predict spikes |
| Cooking -> Science (turn 8->9) | 3334 (normal) | 0.303 | Moderate shift |
| Science -> CS (turn 11->12) | 3382 (normal) | **0.437** | Cos dist detects |
| CS -> History (turn 13->14) | **6578** | **0.404** | E_predict spikes |
| History -> Art (turn 15->16) | 3313 (normal) | 0.326 | Gentle transition |
| CS -> Philosophy (turn 18->19) | **6554** | **0.491** | E_predict spikes |
| Philosophy -> Cooking (turn 19->20) | 3387 (normal) | **0.473** | Cos dist detects |

**Pattern**: E_predict spikes reliably on major domain shifts (philosophy->cooking, CS->history, CS->philosophy). Cosine distance is elevated (>0.4) on most domain transitions but also on some within-domain transitions.

### Temperature Modulation

- **Temp = 0.500**: turns 2, 16, 17, 18 — familiar content (CS continuation, moderate-energy mixed topics, return to Rust)
- **Temp = 1.000**: turns 1, 3 — near median energy
- **Temp = 1.500**: turns 4-15, 19-20 — above-median energy (domain shifts, novel content, high deviation matches)

Temperature is responding to energy dynamics but is somewhat coarse — most turns land at the 1.5 cap because energy is generally above the running median. The `alpha=0.5` coupling may be too aggressive for this session.

### Comparison with Session 2 (Untrained PCHead)

| Metric | Session 2 (untrained) | Session 3 (pre-trained) |
|--------|----------------------|------------------------|
| E_recon range | 9478 - 10172 | 3069 - 8237 |
| E_predict range | 9635 - 10136 | 3200 - 6714 |
| Energy variance | Low (~7% range) | **High (~168% range)** |
| Temp range | 0.35 - 1.05 | 0.50 - 1.50 |
| Dev match scores | N/A (not computed) | 0.30 - **0.96** |
| Signal crossings | 8 | Many (E_predict > E_recon on most domain shifts) |

The pre-trained PCHead produces **much wider energy variance** — the network has learned to discriminate between domains, so novel content genuinely registers as high-energy. Session 2's untrained PCHead had narrow energy bands because it hadn't learned anything yet.

### Deviation Norm Stability

Deviation norms are remarkably stable: mean=162.6, range=[157.3, 167.4]. This suggests the PCHead's attractor basins are well-formed — it consistently "pulls" inputs by a similar magnitude, with the direction (not magnitude) carrying the semantic information.

---

## Raw Data

Full telemetry data (excluding deviation vectors) is available in `docs/session3_telemetry.json`.

The SQLite database `pcdc_telemetry.db` contains the complete dataset including the full 4096-dim deviation vectors (16,384 bytes each as float32 BLOBs).

---

## Session 3b — Adaptive IQR Temperature Scale

Session 3a revealed that the temperature mapping was saturating: 14/20 turns hit the 1.5 cap because `energy_scale=1.0` was absurdly small relative to energies in the thousands. `tanh((4200 - 3300) / 1.0)` saturates instantly.

**Fix:** Changed `energy_scale` default from 1.0 (fixed) to 0 (adaptive). When set to 0, the scale is computed from the IQR (interquartile range) of the blended energy history. This normalizes the tanh input so `+/- 1 IQR` from median maps to `tanh(+/- 1) ~ +/- 0.76`, giving meaningful gradation.

Same 20 prompts, same pre-trained checkpoint, same config except adaptive scale.

### Temperature Comparison (Session 3a vs 3b)

| Turn | Domain | Blended | Temp (3a) | Temp (3b) | Change |
|------|--------|---------|-----------|-----------|--------|
| 1 | CS/Rust | 3235 | 1.000 | 1.000 | -- |
| 2 | CS/Mutex | 3304 | 0.500 | 1.005 | +0.505 |
| 3 | CS/GC | 4044 | 1.000 | 1.110 | +0.110 |
| 4 | Phil/Kant | 3401 | 1.500 | 1.088 | -0.412 |
| 5 | Phil/Sartre | 3392 | 1.500 | 1.000 | -0.500 |
| 6 | Phil/Consc. | 4968 | 1.500 | 1.496 | -0.004 |
| 7 | Cook/Soup | 3413 | 1.500 | 1.015 | -0.485 |
| 8 | Cook/Bread | 5887 | 1.500 | 1.496 | -0.004 |
| 9 | Sci/Quantum | 4191 | 1.500 | 1.375 | -0.125 |
| 10 | Sci/CRISPR | 3270 | 1.500 | 0.918 | -0.582 |
| 11 | Sci/Climate | 4121 | 1.500 | 1.352 | -0.148 |
| 12 | CS/TCP | 4235 | 1.500 | 1.272 | -0.228 |
| 13 | CS/Hash | 5657 | 1.500 | 1.479 | -0.021 |
| 14 | Hist/2008 | 4151 | 1.500 | 1.041 | -0.459 |
| 15 | Hist/Westph. | 4961 | 1.500 | 1.302 | -0.198 |
| 16 | Art/Mixed | 5737 | 0.500 | 1.386 | +0.886 |
| 17 | AI/Mixed | 4059 | 0.500 | 0.980 | +0.480 |
| 18 | CS/Rust | 4038 | 0.500 | 0.981 | +0.481 |
| 19 | Phil/Kant | 4139 | 1.500 | 1.008 | -0.492 |
| 20 | Cook/Kimchi | 5002 | 1.500 | 1.255 | -0.245 |

### Distribution

| Bucket | Session 3a | Session 3b |
|--------|-----------|-----------|
| <0.8 | 4 | 0 |
| 0.8-1.0 | 0 | 3 |
| 1.0-1.2 | 2 | 8 |
| 1.2-1.4 | 0 | 6 |
| >1.4 | 14 | 3 |

**Session 3a:** 3 unique temperatures, binary saturation (0.5 or 1.5)
**Session 3b:** 18 unique temperatures, smooth distribution across [0.918, 1.496]

### Raw Data

- Session 3a: `docs/session3_telemetry.json`
- Session 3b: `docs/session3b_telemetry.json`
- Full dataset (including deviation vectors): `pcdc_telemetry.db`

---

## Configuration Used

### Session 3a (fixed scale)

```bash
.venv/Scripts/python -m pcdc.server.app \
    --model F:/models/Llama-3-8B-Lexi-Uncensored.Q8_0.gguf \
    --n-gpu-layers 33 \
    --pc-checkpoint pchead_pretrained.ckpt \
    --pc-online-eta-w 0.00001 \
    --pc-settle-k 50 \
    --pc-deviation-routing \
    --pc-deviation-threshold 0.4 \
    --pc-energy-scale 1.0 \
    --telemetry-db pcdc_telemetry.db \
    --log-level INFO
```

### Session 3b (adaptive IQR scale)

```bash
# Same config, but energy_scale default changed to 0 (adaptive)
.venv/Scripts/python -m pcdc.server.app \
    --model F:/models/Llama-3-8B-Lexi-Uncensored.Q8_0.gguf \
    --n-gpu-layers 33 \
    --pc-checkpoint pchead_pretrained.ckpt \
    --pc-online-eta-w 0.00001 \
    --pc-settle-k 50 \
    --pc-deviation-routing \
    --pc-deviation-threshold 0.4 \
    --telemetry-db pcdc_telemetry.db \
    --log-level INFO
```
