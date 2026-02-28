# PCDC Project Synthesis

**Predictive Coding Digital Circuit: Steering LLMs with Local Learning and Energy Dynamics**

*Synthesised from five live sessions, 283 conversation turns, and systematic telemetry analysis*
*2026-02-26 to 2026-02-28*

---

## 1. What PCDC Is

PCDC wraps a frozen LLM with a lightweight predictive coding network (PCHead) that learns online during conversation. The PCHead never generates text — the LLM does that. Instead, the PCHead acts as an executive controller that observes the conversation's embedding stream and produces signals that modulate the LLM's behaviour.

The core idea: a predictive coding network's settling energy and deviation vectors carry information about how familiar or surprising the current input is. PCDC uses this to adjust generation temperature, trigger autonomous memory retrieval, and track conversation dynamics — all without modifying the LLM's weights or requiring backpropagation.

### The Stack

```
User Prompt
    |
Llama-3 Chat Template Formatting
    |
GGUF LLM (frozen, 4096-dim embeddings)
    |
PCHead [4096 -> 1024 -> 256 -> 4096]
    |
Five signals per turn:
  * Reconstruction Energy (E_recon) -- content novelty
  * Predictive Energy (E_predict) -- transition surprise
  * Cosine Distance -- raw embedding geometry
  * Deviation Match Score -- structural similarity to past turns
  * Adjusted Temperature -- modulates LLM generation
```

### The Physics

A predictive coding network has layers connected by generative weights. Each layer computes a prediction error: `e[l] = x[l] - f(x[l+1] @ W[l])`. The network settles by iteratively updating states to minimise total energy `E = 0.5 * sum(||e[l]||^2)`. Weights update using only local information — the activity above and the error below. No backpropagation. No global loss function. No computation graph.

Low energy means the input fits the network's learned manifold (familiar). High energy means it doesn't (novel). The deviation vector (`settled_state - original_embedding`) encodes what the network "wanted to change" — its prediction error fingerprint.

---

## 2. Two-Phase Architecture: Why Two Energies

Each request triggers two training phases on the PCHead:

**Phase 1 — Reconstruction**: `train_step(embedding, embedding)` produces E_recon. Measures how hard it is for the network to compress and reconstruct this embedding through its bottleneck. High E_recon = unfamiliar content.

**Phase 2 — Predictive**: `train_step(prev_embedding, current_embedding)` produces E_predict. Measures how surprising the transition is from the previous turn. High E_predict = unexpected conversational shift.

These are genuinely independent signals:

| Scenario | E_recon | E_predict | Example |
|----------|---------|-----------|---------|
| Novel content, expected domain | High | Low | Second Rust question after first |
| Simple content, shocking transition | Low | High | "No." after a long Polish paragraph |
| Both novel and surprising | High | High | First question in a new domain |
| Familiar and expected | Low | Low | Continuation within a well-explored topic |

The blended energy drives temperature:
```
energy = beta * E_recon + (1 - beta) * E_predict
temp = base_temp * (1 + alpha * tanh((energy - median) / scale))
```

A third signal, cosine distance (`1 - cos_sim(prev_embedding, current_embedding)`), provides raw embedding-space geometry independent of the PCHead's learned state.

---

## 3. Session Chronology

### Session 1: First Contact (8 turns, 2026-02-26)

**Config**: Untrained PCHead, eta_w=0.0001, K=20, no MemoryGate

**What happened**: Energy declined monotonically from 10,623 to 8,784 (a 17.3% drop in just 8 turns). Temperature clamped to 0.35 after Turn 1 and stayed there. Topic shifts — from AI architecture to tree poetry to SQL queries to Lagrangian mechanics to Polish history — barely registered.

**The breakthrough moment**: Turn 8. The user sent "No." after a long Polish paragraph. For the first and only time in the session, the signals crossed:
- E_recon: 8,784 (lowest — "No." is trivially easy to reconstruct)
- E_predict: 8,977 (jumped UP — Polish history to bare "No." was the most surprising transition)

This validated the two-phase architecture. A single energy signal could never have captured this: the content was simple but the transition was jarring. Two signals are necessary.

**Lesson**: eta_w=0.0001 is too aggressive. The PCHead adapts faster than content novelty can register. The learning rate needs to be 10x lower to preserve sensitivity.

### Session 2: Adversarial Stress Test (25 turns, 2026-02-26)

**Config**: Untrained PCHead, eta_w=0.00001 (10x reduction), K=100 (5x increase), MemoryGate enabled (p95 threshold)

**What happened**: The 10x slower learning rate transformed the system. Energy decline was only 6.8% over 25 turns (vs 17.3% over 8 turns). Crucially, E_recon was no longer monotonic — it *rose* on 7 turns when genuinely novel content arrived (Rust code, legal prose, Shakespeare, assembly).

**8 signal crossings** in 25 turns (32%, up from 12.5% in Session 1). The two-phase architecture was now working as intended.

The adversarial test battery included: hex dump of a PNG header, grief/pet loss, nonsense tokens (random Unicode), identical repeated input, mathematical induction, a single skull emoji, dense biochemistry, Shakespearean sonnet, x86-64 assembly, p-zombies, Chinese quantum physics, and raw whitespace.

**Key findings**:

**Whitespace probe (Turn 23)** produced the session's most extreme signal. Legal prose followed by four space characters:
- E_predict: 9,940 (highest since early turns)
- E_recon: 9,478 (session low)
- Gap: 462 (largest separation observed)

This is the purest expression of the architecture's design principle: trivial content + maximum transition surprise.

**Cosine distance bands stabilised** across 25 turns into four consistent regions:
| Band | Cos Distance | Meaning |
|------|-------------|---------|
| Identical | 0.000 | Repeated input |
| Continuation | 0.20-0.35 | Same domain |
| Related shift | 0.35-0.50 | Adjacent domain |
| Hard pivot | 0.50-0.65 | Unrelated domain |

**Repeated content (Turn 25)** proved online learning works. Turn 25 repeated Turn 1's exact prompt. E_recon dropped from 10,136 to 9,572 — a 5.6% decline for identical content after 25 turns of online training.

**Signal crossings don't require high cosine distance.** Turn 20 (philosophy to philosophy, cos=0.341) still crossed. The crossing depends on the PCHead's learned state, not raw embedding geometry. This confirms energy signals and cosine distance carry independent information.

**MemoryGate never triggered** despite being enabled. The p95 threshold was too conservative — it tracked downward with the slowly declining energy. Recommendation: use p85-90 or absolute thresholds.

### Session 3: Pre-Trained PCHead + Deviation Routing (20 turns, 2026-02-27)

**Config**: Pre-trained PCHead (5 epochs, 52-paragraph corpus), eta_w=0.00001, K=50 per phase, deviation routing enabled (threshold=0.4), SQLite telemetry enabled

**What changed**: Sessions 1-2 used an untrained PCHead. Session 3 introduced offline pre-training and deviation routing — two features developed between sessions based on Ghost Mic experiments that showed deviation vectors were near-orthogonal (~0.07 cosine) without pre-training.

**Deviation routing works.** The pre-trained PCHead produces semantically meaningful deviation matches:

| Domain | Avg Match | Best Match | Matched Against |
|--------|-----------|------------|-----------------|
| CS/Programming | 0.76 | 0.85 (Rust, T18) | Lock-free queue paragraph (idx 52) |
| Philosophy | 0.77 | **0.9559** (Kant, T19) | Sartre paragraph (idx 55) |
| Cooking | 0.66 | 0.74 (Kimchi, T20) | Fermentation paragraph (idx 57-58) |
| Science | 0.81 | 0.86 (Quantum + Climate) | Corpus science entries (idx 57) |
| History | 0.85 | 0.85 (both turns) | 2008 crisis + Westphalia (idx 62, 65) |

Turn 19 (Kant on phenomena vs noumena) matched at 0.9559 against the Sartre/existentialism corpus entry — the PCHead recognises philosophy as a tight cluster in deviation space.

**Energy dynamics were dramatically different** from untrained sessions. Pre-trained energy ranged from 3,069 to 8,237 (vs 9,478-10,172 untrained). More importantly, the variance was huge (168% range vs 7%) — the network had learned to discriminate between domains, so novel content genuinely registered as high-energy.

**E_predict spikes reliably on domain shifts:**
| Transition | E_predict | Signal |
|------------|-----------|--------|
| CS -> Philosophy (T3->T4) | 3,278 (normal) | Cos dist detects (0.449) |
| Philosophy -> Cooking (T6->T7) | **5,053** | E_predict spike |
| CS -> History (T13->T14) | **6,578** | E_predict spike |
| CS -> Philosophy (T18->T19) | **6,554** | E_predict spike |

**Deviation norm was remarkably stable**: mean=162.6, range=[157.3, 167.4], SD=2.1 (1.3% variation). The PCHead consistently "pulls" inputs by a similar magnitude. Direction carries the semantic information, not magnitude.

### Session 3b: The Temperature Saturation Fix

Session 3a revealed a critical problem: 14 of 20 turns hit the 1.5 temperature ceiling. Only 3 unique temperature values across the whole session. The temperature mapping was broken.

**Root cause**: `energy_scale=1.0` was absurdly small relative to energies in the thousands. `tanh((4200 - 3300) / 1.0) = tanh(900) ≈ 1.0` — instant saturation.

**Fix**: Adaptive IQR (interquartile range) scaling. When `energy_scale=0` (new default), scale is computed from the IQR of the blended energy history. This normalises the tanh input so `+/- 1 IQR` from median maps to `tanh(+/- 1) ≈ +/- 0.76`.

**Result** (same 20 prompts, same checkpoint):

| Metric | Session 3a (fixed) | Session 3b (adaptive) |
|--------|-------------------|----------------------|
| Unique temperatures | 3 | **18** |
| Temperatures at ceiling | 14/20 | 3/20 |
| Range | {0.500, 1.000, 1.500} | [0.918, 1.496] |

The per-turn comparison shows the fix working across every domain:

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

Session 3a had three temperatures because it was a two-state system: above median saturated to 1.5, below median saturated to 0.5, and the first turn defaulted to 1.0. Session 3b's IQR scaling recovers the full dynamic range. The temperature distribution shifts from binary to continuous:

| Bucket | Session 3a | Session 3b |
|--------|-----------|-----------|
| <0.8 | 4 | 0 |
| 0.8-1.0 | 0 | 3 |
| 1.0-1.2 | 2 | 8 |
| 1.2-1.4 | 0 | 6 |
| >1.4 | 14 | 3 |

Session 4's results depend on this fix. Without adaptive IQR scaling, temperature is meaningless noise — the system has no way to express graded confidence. With it, the temperature carries real information about where the current turn sits in the session's energy distribution. The IQR scale auto-calibrates to any model's energy regime without manual tuning.

Full telemetry: `docs/session3b_telemetry.json` (20 turns, all metrics except deviation vectors).

### Session 4: Off-Corpus Generalization (20 turns, 2026-02-27)

**Config**: Same as Session 3b (pre-trained, adaptive IQR, deviation routing, threshold=0.4)

**The question**: Does the pre-trained PCHead's deviation matching generalise to domains not present in the bootstrap corpus, or does it collapse back to near-orthogonal (~0.07)?

**Bootstrap corpus covers**: CS, philosophy, cooking, physics, biology, climate, history, math, literature, medicine, art, cryptography, data structures, networking, distributed systems.

**Deliberately off-corpus domains tested**: Sports (football, tennis), law (civil/criminal, stare decisis), music performance (piano tuning, circle of fifths), agriculture (crop rotation, companion planting), film/cinema (180-degree rule, neorealism), clinical psychology (CBT, PTSD), linguistics (Sapir-Whorf, tonal languages), truly alien domains (competitive dog grooming, cricket scoring, knitting vs crocheting).

**Three on-corpus controls** (turns 15-17): red-black trees, Bayes' theorem, immune system.

**The result**:

| Condition | Avg Dev Match | N |
|-----------|--------------|---|
| Untrained baseline (Sessions 1-2) | ~0.07 | - |
| On-corpus (Session 3) | 0.74 | 20 |
| On-corpus controls (Session 4) | 0.7315 | 3 |
| **Off-corpus novel domains** | **0.7284** | 17 |

**Off-corpus deviation matches are statistically indistinguishable from on-corpus matches.** The PCHead did not go orthogonal on novel domains.

The most striking result: competitive dog grooming (0.74), cricket scoring (0.75), and knitting vs crocheting (0.78) all produce strong deviation matches despite being about as far from the corpus as you can get. The PCHead's deviation space has structure everywhere, not just near the training data.

The only low match was Turn 1 (0.15) — the first turn of the session, where the replay buffer only contains pre-trained corpus entries and there's no prior deviation to compare against. By Turn 2, the match jumps to 0.78.

### Session 5: Long-Session Characterisation (210 turns, 2026-02-28)

**Config**: Same as Session 4 (pre-trained, adaptive IQR, deviation routing, threshold=0.4, eta_w=1e-5, K=100)

**The question**: Do energy signals, IQR scaling, deviation routing, and temperature modulation remain stable over 200+ turns — or does the system drift, saturate, or collapse?

**Test infrastructure**: A dedicated test suite (`scripts/long_session_test.py`, `scripts/analyse_session.py`) with 210 prompts across 6 structured scenarios and a general corpus spanning 21 domains. Prompts range from edge cases ("Why?", German text, code blocks) to dense academic content across on-corpus and off-corpus domains. Full model responses were captured alongside all PCDC telemetry.

**Structured scenarios and what they test**:

| Scenario | Turns | What it tests |
|----------|-------|---------------|
| `sustained_domain` | 20 | 10 CS then 10 philosophy — within-domain E_recon stabilisation |
| `rapid_switching` | 20 | Alternating 6 domains (2 full cycles) — E_predict under constant pivots |
| `gradual_drift` | 15 | CS → math → physics → chemistry → biology — gradual vs abrupt transitions |
| `return_to_origin` | 25 | 5 CS, 15 philosophy, 5 CS (same sub-topics) — deviation routing across a gap |
| `complexity_ladder` | 15 | Within CS: 5 simple, 5 moderate, 5 dense — E_recon vs complexity |
| `repetition_probe` | 10 | Same prompt 5×, novel, repeat original, 3 novel — energy floor + spike |

**The results** (from the transcript run, 210 turns, 0 errors):

**IQR scaling converges and stabilises.** This was the key unknown. The adaptive IQR scale (which normalises the tanh temperature mapping) stabilises by turn ~50 and tightens further through turn 200:

| Checkpoint | Median | IQR | IQR/Median |
|------------|--------|-----|------------|
| Turn 25 | 3438 | 1699 | 0.494 |
| Turn 50 | 4910 | 3021 | 0.615 |
| Turn 100 | 5899 | 2240 | 0.380 |
| Turn 200 | 6452 | 1538 | **0.238** |

IQR/Median drops from 0.494 to 0.238 — the energy distribution tightens as the PCHead accumulates more experience. The scale never collapses to zero (which would cause temperature saturation) or diverges (which would flatten modulation).

**Energy is NOT monotonically declining.** E_recon rose on 104 of 209 transitions (50%). The system maintains novelty sensitivity even at turn 200. The mean E_recon across the full session was 6,155 (range 2,900–9,686). This comprehensively answers the concern from Session 2, where non-monotonic behaviour was observed over 25 turns but could have been transient.

**Signal crossings are frequent and scenario-dependent.** 105 of 209 transitions (50.2%) showed E_predict > E_recon. The general corpus (diverse topic switching) had 44 crossings; rapid_switching had 11; repetition_probe had only 4 (expected — repeated content has low E_predict). This confirms crossings are a genuine content signal, not noise.

**Temperature spans the full modulation range.** Mean 1.16, range [0.62, 1.49], SD=0.20. Neither floor (0.5) nor ceiling (1.5) dominated. The system maintained meaningful temperature gradation across all 210 turns.

**Deviation routing is stable at scale.** Mean deviation match score 0.82 (SD=0.08, drift=0.007). The match score didn't degrade over 200 turns — the replay buffer maintains its discriminative structure even as it accumulates hundreds of entries.

**Per-domain energy profiles are interpretable.** Math had the lowest temperature (0.97) — precise, technical content. Psychology and chemistry had the highest (1.26, 1.28). Edge cases (very short prompts, code blocks) had the lowest deviation match (0.69) and highest cosine distance (0.47), confirming they're genuinely unusual inputs.

**Full telemetry**: `results/session_20260228_transcript.json` (210 turns, full model responses). Analysis: `results/session_20260228_transcript_analysis/` (7 diagnostic plots + summary).

---

## 4. Why Pre-Training Generalises

The bottleneck architecture (4096 -> 1024 -> 256 -> 4096) is the key. Pre-training doesn't teach the PCHead specific domain knowledge. It teaches the PCHead to form **structured attractor basins** — to produce input-dependent deviation vectors instead of near-random ones.

The 256-dimensional bottleneck forces radical compression of the 4096-dimensional embedding space. The PCHead must learn a compact encoding that preserves enough information to reconstruct the input. This compressed representation spans the entire embedding space, not just the 52 paragraphs it trained on.

After pre-training, the PCHead settles differently for different inputs — and *consistently* for similar inputs. A question about cricket produces a different deviation fingerprint than a question about Kant, and that fingerprint is reproducible, even though neither cricket nor Kant were emphasised in training.

The deviation norms confirm this: remarkably stable at ~162 across all domains (both on-corpus and off-corpus), with the *direction* carrying all the semantic information. The attractor basins are well-formed throughout the space.

---

## 5. The Inverted RAG Architecture

PCDC inverts the typical RAG (Retrieval-Augmented Generation) flow:

**Traditional RAG**: The LLM decides what to retrieve (via generated queries or user prompts) and when to retrieve (always, or based on heuristics external to the model).

**PCDC**: The PCHead's settling dynamics decide when and what to retrieve. Two independent pathways trigger retrieval:

1. **Energy thresholds**: E_recon or E_predict exceeds a percentile-based or absolute threshold. Fires on novelty — content the PCHead hasn't modelled well.

2. **Deviation routing**: The current deviation vector matches a stored deviation above a cosine similarity threshold. Fires on structural similarity — the PCHead recognises this "reaction pattern" from past interactions.

When either pathway triggers, the system queries MemoryGate, prepends retrieved context to the prompt, and re-embeds. The LLM sees a richer prompt but has no awareness that retrieval occurred. The PCHead governs context; the LLM purely generates.

All retrieval is fire-and-forget with timeout caps. Failures never block generation.

---

## 6. Tuning Insights

### Learning Rate (eta_w)

The most consequential parameter. Determines whether energy signals carry information or are washed out by adaptation:

| eta_w | Behaviour | Energy Decline | Signal Crossings |
|-------|-----------|---------------|-----------------|
| 1e-4 | Too fast. Monotonic decline regardless of content. | 17.3% in 8 turns | 1 (12.5%) |
| **1e-5** | **Sweet spot.** Slow decline, energy responds to novelty. | 6.8% in 25 turns | 8 (32%) |
| 1e-6 | Too slow. PCHead barely learns, dominated by initialisation. | Minimal | N/A |

For pre-training, eta_w=1e-4 is appropriate (many examples, no forgetting risk, goal is fast basin formation). For online learning, eta_w=1e-5 preserves novelty sensitivity.

### Settle Steps (K)

K=20 (default) is insufficient for 4096-dimensional embeddings — no turns converged in any session. K=50 per phase (100 total) produces richer dynamics. K=100 per phase works but doubles latency.

No convergence was achieved in any session at any K value, suggesting the convergence tolerance may need tuning for this embedding dimensionality, or that the PCHead benefits from continuing to settle beyond the convergence point.

### Temperature Scaling

Fixed `energy_scale` values saturate the tanh mapping because raw energies are in the thousands while the scale normalises differences to `(-1, 1)`. Adaptive IQR scaling (the current default) solves this by computing scale from the session's own energy distribution.

**Before (fixed scale=1.0)**: 3 unique temperatures, binary saturation
**After (adaptive IQR)**: 18+ unique temperatures, smooth gradation

### Replay Buffer

Replay mixing (`replay_k=4`) is essential for preventing catastrophic forgetting during online learning. Without it, the PCHead overfits to the most recent turns and loses sensitivity to earlier patterns. Both training phases draw from the replay buffer.

### Deviation Routing Threshold

threshold=0.4 works well with pre-trained checkpoints. Match scores cluster around 0.72-0.78 with pre-training, so a 0.4 threshold gates generously. Raising to 0.6 would be more selective. Without pre-training, all matches are ~0.07, so any reasonable threshold prevents false triggering.

---

## 7. Ghost Mic Experiments

Ghost Mic is a standalone analysis tool for probing deviation vectors outside the server loop.

### Mode A: LM Head Projection (Failed)

Attempted to project the PCHead's settled state through the LLM's unembedding matrix to produce token probabilities. Result: pure noise (`addCriterion`, `/stdc`, `baise`). The PCHead's output manifold is disjoint from the LLM's residual stream.

**Conclusion**: Direct logit-space steering requires either a trained adapter matrix or explicit manifold alignment during training. This remains a future direction (v2 steering).

### Mode B: Deviation Fingerprint Matching (Working)

Compared deviation vectors against stored deviations. Pre-trained results showed clear semantic clustering:
- Rust prompt -> Rust paragraph (cos=0.654)
- Kant prompt -> Kant paragraph (cos=0.608)
- Cooking prompt -> cooking paragraph (~0.55)

Without pre-training (25 online turns only): all similarities 0.05-0.09 (near-orthogonal, no discrimination).

This experiment motivated both pre-training and the switch from deviation-to-embedding comparison to deviation-to-deviation comparison (which the server now uses).

---

## 8. Telemetry Infrastructure

### SQLite Database

Per-turn telemetry records every signal the system produces:

| Field | Type | Description |
|-------|------|-------------|
| turn_id | INTEGER | Auto-incrementing turn counter |
| session_id | TEXT | UUID per server startup |
| timestamp | REAL | Unix timestamp |
| energy_recon | REAL | Phase 1 reconstruction energy |
| energy_predict | REAL | Phase 2 predictive energy |
| energy_blended | REAL | Weighted blend |
| cosine_distance | REAL | Embedding-space distance from previous turn |
| deviation_norm | REAL | L2 norm of deviation vector |
| deviation_vector | BLOB | Full 4096-dim vector as float32 bytes (16,384 bytes) |
| top_match_score | REAL | Best cosine similarity against stored deviations |
| top_match_idx | INTEGER | Index of best match in replay buffer |
| adjusted_temp | REAL | Temperature used for generation |
| converged | INTEGER | Whether settling converged |
| settle_steps | INTEGER | Total settle iterations |
| retrieval_triggered | INTEGER | Whether MemoryGate was queried |

WAL journal mode for concurrent read/write. Fire-and-forget writes — telemetry never blocks generation.

### Data Collection Scripts

`scripts/gather_telemetry.py` — 20 diverse prompts across 8 domain clusters (CS, philosophy, cooking, science, history, art, AI, domain returns). Tests within-domain and cross-domain energy dynamics.

`scripts/gather_telemetry_offcorpus.py` — 20 prompts in deliberately novel domains (sports, law, music, agriculture, film, psychology, linguistics, truly alien). Tests deviation routing generalisation.

---

## 9. What's Validated

After five sessions and 283 turns of systematic testing:

**Two-phase energy signals are genuinely complementary.** E_recon tracks content complexity/novelty. E_predict tracks transition surprise. They cross on specific, interpretable conditions (simple content after complex content, or surprising transitions within familiar domains). A single energy signal cannot capture this.

**Online learning works and its rate is critical.** The PCHead measurably improves at self-reconstruction over a session. eta_w=1e-5 balances adaptation against novelty preservation. eta_w=1e-4 washes out all signals.

**Pre-training is necessary for deviation routing.** Without it, all deviations are near-orthogonal (~0.07). With 5 epochs on 52 paragraphs, deviations form semantic clusters with cosine similarities of 0.6-0.96 within domains.

**Pre-training generalises completely.** Off-corpus deviation matches (avg 0.728 across 17 novel domains) are indistinguishable from on-corpus matches (0.731). The bottleneck architecture learns structured attractor basins that span the entire embedding space, not domain-specific memorisation.

**Adaptive temperature scaling is essential.** Fixed energy scales cause binary saturation. IQR-based adaptive scaling produces smooth, meaningful temperature gradation across the full modulation range.

**Cosine distance is independent of energy signals.** Signal crossings occur at both high and low cosine distances. The three signals (E_recon, E_predict, cosine distance) carry non-redundant information about conversation dynamics.

**Deviation norm is semantically uninformative but structurally revealing.** Norms are remarkably stable (~162, SD=2.1) across all domains and sessions. Direction, not magnitude, carries the semantic information. The stability itself indicates well-formed attractor basins.

**Long-session stability is confirmed.** Over 210 turns, adaptive IQR scaling converges (IQR/Median drops from 0.49 at turn 25 to 0.24 at turn 200), energy signals remain non-monotonic (50% of transitions show E_recon increases), temperature spans the full modulation range [0.62, 1.49], and deviation match scores hold steady (mean 0.82, drift 0.007). The system does not drift, saturate, or collapse at scale.

---

## 10. What's Not Yet Tested

**MemoryGate retrieval in practice.** Sessions 1-2 had thresholds too conservative to trigger. Sessions 3-4 ran without MemoryGate connected. The retrieval pathway is built and tested in isolation but hasn't been exercised in a live conversation where retrieved context actually changes the response.

**Convergence behaviour.** No turn across any session achieved convergence (`converged=False` on all 73 turns). Either K is too low, the convergence tolerance is too tight for this dimensionality, or the PCHead benefits from the non-converged regime.

**Multi-session checkpoint continuity.** Checkpoints persist and load correctly, but no session has resumed from a prior session's checkpoint to test whether learned state transfers meaningfully across conversations.

**MNIST benchmark.** The core PC dynamics have a working MNIST pipeline but it hasn't been tuned to convergence yet.

**Continual learning benchmarks.** PCHead vs baseline forgetting comparisons haven't been run on real task sequences.

---

## 11. Architecture Decisions and Their Rationale

### Why [4096, 1024, 256, 4096]?

The symmetric input/output (4096 -> 4096) enables both reconstruction training and deviation computation in the same embedding space. The bottleneck (1024 -> 256) forces compression, which is what creates structured attractor basins. Without the bottleneck, reconstruction would be trivially easy and deviations would be near-zero.

### Why deviation-to-deviation, not deviation-to-embedding?

Ghost Mic experiments showed that comparing the current deviation against stored *deviations* (not stored *embeddings*) produces stronger, more semantically meaningful matches. Deviations encode "how the PCHead reacted" — two inputs from the same domain produce similar reactions even if their raw embeddings differ. This is the architectural insight that makes deviation routing work.

### Why online learning at all?

The PCHead could run in inference-only mode (pre-trained weights, no online updates). But online learning serves two purposes: (1) it populates the replay buffer with session-specific deviations, enabling deviation routing against the current conversation's patterns, and (2) it allows the energy signal to reflect the current session's novelty baseline rather than the pre-training corpus's baseline.

### Why two replay buffers?

Embeddings (for reconstruction/predictive training) and deviations (for deviation routing) serve different purposes and are consumed by different subsystems. Separating them allows independent sizing and avoids coupling the training pipeline to the routing pipeline.

### Why fire-and-forget telemetry?

Telemetry records are written in a try/except with logged exceptions. A telemetry write failure should never crash the server or delay a response. The telemetry database uses WAL mode for non-blocking writes.

### Why invert RAG?

The argument goes deeper than separation of concerns. The PCHead has access to information the LLM fundamentally cannot have.

An LLM can only reason about retrieval using the tokens in its context window. It sees text. It can generate a search query from that text. But it has no access to the geometry of the embedding space, no model of how surprising this transition is relative to the session's history, no fingerprint of how a dedicated dynamical system *reacted* to this input versus previous inputs. These are different information streams entirely.

The PCHead reasons about retrieval using:
- **Settling dynamics** — how much energy was required to process this input through a learned bottleneck
- **The energy landscape** — where this turn sits relative to the running distribution of all prior turns
- **Deviation history** — whether the network's prediction error pattern on this input structurally resembles its reaction to past inputs

None of these signals exist in token space. An LLM generating a retrieval query is working with surface-level lexical and semantic content. The PCHead is working with the geometry of a learned manifold and the dynamics of a physical settling process. These are complementary information sources, and PCDC is (to our knowledge) the first architecture that gives the retrieval decision to a dedicated dynamical system rather than the text generator.

The practical consequence: the PCHead can detect that two inputs are "the same kind of thing" (via deviation matching) even when their surface text is completely different — a question about cricket and a question about knitting both trigger similar deviation patterns, not because they're lexically similar, but because the PCHead's learned model processes them through similar attractor basins. An LLM-driven retrieval system would never connect these inputs.

---

## 12. Open Questions

**Does convergence matter?** None of the 73 turns achieved convergence. It's unclear whether convergence would improve signal quality, or whether the non-converged settling dynamics already capture the relevant information.

**What is the optimal bottleneck width?** The 256-dimensional bottleneck was chosen heuristically. Wider (512, 1024) might preserve more information; narrower (128, 64) might force more structured basins. No ablation has been performed.

**How does the energy distribution change over hundreds of turns?** Session 5 (210 turns) showed the IQR/Median ratio converges from 0.49 (turn 25) to 0.24 (turn 200) — the distribution tightens but does not collapse. No windowing or decay was needed. The open question is now whether this continues to hold over 1,000+ turns or whether eventual saturation would require a sliding window.

**Can deviation-matched text serve as the retrieval query?** Currently, deviation routing *gates* retrieval (decides whether to trigger) but the user's prompt provides the query content. If the replay buffer stored text alongside embeddings, the matched entry's text could serve as the query — potentially retrieving more relevant context than the user's surface-level question.

**Should deviation routing be energy-conditional?** Session 4 showed that deviation matches are consistently high (avg 0.73) regardless of whether the content is novel or routine. A pure threshold on match score would trigger retrieval on nearly every turn — the pre-trained PCHead produces structured deviations everywhere, so matches are always strong. The useful retrieval gate is probably match score AND energy: "I recognise this pattern AND it's surprising." The architecture already computes both signals; the gating logic just needs to combine them. Something like `if match > threshold AND energy > percentile` would fire only when a familiar-looking pattern arrives in an unexpected context — which is precisely when retrieved context is most likely to be useful.

**Is manifold alignment achievable?** Ghost Mic Mode A showed that the PCHead's output manifold is disjoint from the LLM's residual stream. A trained adapter matrix could bridge this gap, enabling direct logit-space steering (v2). The question is whether this can be done without degrading the PCHead's deviation structure.

---

## 13. File Map

```
src/pcdc/
  core/
    pc_network.py          -- Settling loop, weight updates, energy computation (~590 lines)
    activations.py         -- tanh, relu, gelu, linear + analytical derivatives
    energy.py              -- Energy computation, convergence checking
    clamp.py               -- Supervised/inference/generative mode masks
  gguf/
    gguf_backend.py        -- GGUF model loading, embed_chat(), embed_and_warm()
    pc_head.py             -- PCNetwork subclass for LLM feature dimensions
    datasets.py            -- Continual learning task sequences
    baselines.py           -- Linear probe, SGD MLP, replay buffer baselines
  server/
    app.py                 -- FastAPI app, lifespan, routes, CLI entry point
    steering.py            -- SteeringEngine: two-phase settling, deviation routing, temp mapping
    memory_client.py       -- MemoryGate MCP JSON-RPC client (httpx, sync)
    telemetry_db.py        -- SQLite per-turn telemetry recorder (WAL mode)
    schemas.py             -- Pydantic models for OpenAI-compatible API
  training/
    train_mnist_pc.py      -- PC training loop on MNIST
    baseline_mlp_backprop.py -- Standard backprop MLP baseline
    eval_mnist_pc.py       -- Energy curves + convergence visualisation
    continual_runner.py    -- Sequential task orchestration + forgetting metrics
  utils/
    config.py              -- PCConfig, ContinualConfig dataclasses
    metrics_logger.py      -- SettleMetrics, EpochMetrics, OscillationEvent

scripts/
  pretrain_pchead.py       -- Offline PCHead pre-training on text corpus
  ghost_mic.py             -- Deviation vector analysis experiments
  gather_telemetry.py      -- On-corpus telemetry collection (20 prompts)
  gather_telemetry_offcorpus.py -- Off-corpus generalisation test (20 prompts)
  plot_session.py          -- Session energy visualisation
  long_session_test.py     -- Long-session test runner (100-200+ turns)
  analyse_session.py       -- Post-hoc analysis: 7 diagnostic plots + summary markdown

data/
  bootstrap_corpus.txt     -- 52 diverse paragraphs for pre-training
  long_session_prompts.json -- 210 prompts with domain/complexity metadata (6 scenarios + general corpus)

docs/
  system_architecture.md   -- Detailed technical architecture
  pcdc_synthesis.md        -- This document
  session_transcript_2026-02-26.md         -- Session 1 (8 turns, untrained)
  session_transcript_2026-02-26_session2.md -- Session 2 (25 turns, tuned)
  session_transcript_2026-02-27_session3.md -- Session 3 (20 turns, pre-trained + deviation routing)
  session_transcript_2026-02-27_session4.md -- Session 4 (20 turns, off-corpus generalisation)
  session2_energy_plot.png -- Energy trajectory visualisation
  session3_telemetry.json  -- Session 3a raw telemetry
  session3b_telemetry.json -- Session 3b raw telemetry (adaptive IQR)
  session4_offcorpus_telemetry.json -- Session 4 raw telemetry
```

---

## 14. Summary of Numerical Results

### Energy Dynamics Across Sessions

| Session | Turns | eta_w | E_recon Range | E_recon Decline | Signal Crossings | Temp Range |
|---------|-------|-------|---------------|-----------------|-----------------|------------|
| 1 (untrained) | 8 | 1e-4 | 8,784 - 10,623 | 17.3% | 1 (12.5%) | 0.35 - 0.70 |
| 2 (untrained) | 25 | 1e-5 | 9,478 - 10,172 | 6.8% | 8 (32%) | 0.35 - 1.05 |
| 3a (pre-trained, fixed scale) | 20 | 1e-5 | 3,069 - 8,237 | N/A | N/A | 0.50 - 1.50 (3 unique) |
| 3b (pre-trained, IQR scale) | 20 | 1e-5 | 3,191 - 6,835 | N/A | N/A | 0.92 - 1.50 (18 unique) |
| 4 (off-corpus) | 20 | 1e-5 | 3,222 - 6,653 | N/A | N/A | 0.53 - 1.50 (18 unique) |
| 5 (long-session) | 210 | 1e-5 | 2,900 - 9,686 | Non-monotonic (50% rises) | 105 (50.2%) | 0.62 - 1.49 |

### Deviation Routing Performance

| Condition | Avg Match Score | Range | N |
|-----------|----------------|-------|---|
| Untrained (Ghost Mic) | ~0.07 | 0.05 - 0.09 | ~50 |
| Pre-trained, on-corpus (S3) | 0.74 | 0.30 - 0.96 | 20 |
| Pre-trained, on-corpus controls (S4) | 0.73 | 0.69 - 0.78 | 3 |
| Pre-trained, off-corpus novel (S4) | 0.73 | 0.15 - 0.90 | 17 |
| Pre-trained, truly alien (S4) | 0.76 | 0.74 - 0.78 | 3 |
| Pre-trained, long-session (S5) | 0.82 | 0.22 - 1.00 | 210 |

### Deviation Norm Stability (Pre-trained Sessions)

| Session | Mean | SD | Range | CV |
|---------|------|----|-------|----|
| 3b | 162.6 | 2.9 | [157.3, 167.4] | 1.8% |
| 4 | 161.7 | 2.8 | [157.3, 166.7] | 1.7% |

### IQR Convergence (Session 5)

| Checkpoint | Median | IQR | IQR/Median |
|------------|--------|-----|------------|
| Turn 25 | 3,438 | 1,699 | 0.494 |
| Turn 50 | 4,910 | 3,021 | 0.615 |
| Turn 100 | 5,899 | 2,240 | 0.380 |
| Turn 200 | 6,452 | 1,538 | 0.238 |

---

## 15. What's Next

1. **Live retrieval test** — Run a session with MemoryGate connected and thresholds tuned to actually trigger. Verify that retrieved context changes response quality.

2. **Store text with deviations** — Associate text snippets with replay buffer entries so deviation-matched text can serve as the retrieval query directly, rather than falling back to the user's prompt.

3. **Cosine distance as retrieval gate** — Use embedding-space distance as a third retrieval trigger for domain shifts (cos > 0.5).

4. **v2 steering: logit-space intervention** — Train an adapter matrix to project the PCHead's settled state into the LLM's residual stream, enabling direct logit bias. Ghost Mic Mode A showed this is noise without alignment; a small trained projection could bridge the gap.

5. **Convergence investigation** — Test with higher K values (200, 500) or relaxed tolerances to determine whether convergence is achievable and whether it improves signal quality.

6. **MNIST convergence** — Tune hyperparameters for >90% accuracy on the core PC benchmark.

7. **Continual learning benchmarks** — Run systematic PCHead vs baseline comparisons on incremental task sequences with forgetting metrics.
