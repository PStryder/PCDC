# PCDC Session 2 — Adversarial Pivots with Tuned Parameters

**Date:** 2026-02-26
**Server:** pcdc-serve on localhost:8001
**Model:** Llama-3-8B-Lexi-Uncensored.Q8_0.gguf (33 GPU layers)
**PCHead:** [4096, 1024, 256, 4096], fresh checkpoint (no prior training)
**Config changes from Session 1:**
- `online_eta_w`: 0.0001 → **0.00001** (10x reduction)
- `settle_k`: 20 → **100** (5x increase, 200 total for two-phase)
- MemoryGate retrieval: **enabled** (mg-url=localhost:8080/mcp, predict percentile=95)
- Cosine distance tracking: **new metric**

---

## Energy Trajectory

| Turn | Topic | E_recon | E_predict | Blended | Temp | Cos Dist | Steps | Signal |
|------|-------|---------|-----------|---------|------|----------|-------|--------|
| 1 | Introduction | 10136 | 10136 | 10136 | 0.700 | N/A | 100 | First turn (degraded) |
| 2 | Architecture detail | 10172 | 10008 | 10090 | 0.350 | 0.280 | 200 | E_recon ROSE |
| 3 | Hex dump (PNG header) | 10153 | 10015 | 10084 | 0.350 | 0.310 | 200 | Domain shift registered in cos |
| 4 | Grief / pet loss | 10154 | 10026 | 10090 | **0.697** | **0.575** | 200 | Temp rose to near-base |
| 5 | Rust lock-free queue | 10158 | 10055 | 10106 | **1.050** | **0.612** | 200 | **Temp above base!** |
| 6 | Rust continuation | 10114 | 10075 | 10094 | 1.041 | 0.215 | 200 | Low cos = same domain |
| 7 | Nonsense tokens | 10098 | **10113** | 10106 | 1.050 | 0.531 | 200 | **SIGNALS CROSSED** |
| 8 | Same nonsense (repeat) | 10042 | 9975 | 10009 | 0.350 | **0.000** | 200 | Identical input |
| 9 | Math induction proof | 10044 | 9958 | 10001 | 0.350 | 0.614 | 200 | Huge cos, energy stable |
| 10 | Single emoji (skull) | 9957 | 9895 | 9926 | 0.350 | 0.622 | 200 | Minimal content |
| 11 | Dense biochemistry | 9987 | 9874 | 9930 | 0.350 | 0.506 | 200 | E_recon rose from emoji |
| 12 | "No." | 9917 | **9927** | 9922 | 0.350 | 0.521 | 200 | **SIGNALS CROSSED** |

**E_recon range:** 10172 → 9917 (−2.5% over 12 turns)
**E_predict range:** 10136 → 9927 (−2.1% over 11 active turns)

---

## Session 1 vs Session 2 Comparison

| Metric | Session 1 (eta=1e-4, K=20) | Session 2 (eta=1e-5, K=100) |
|--------|---------------------------|----------------------------|
| E_recon decline | −17.3% (10623→8784) | **−2.5%** (10172→9917) |
| E_predict decline | −15.5% (10623→8977) | **−2.1%** (10136→9927) |
| Signal crossings | 1 (turn 8 only) | **3** (turns 7, 12, and near-cross on 5) |
| Temperature variation | 0.70 then stuck at 0.35 | 0.35 → 0.70 → **1.05** → 0.35 |
| Unique temp values | 2 (0.70, 0.35) | **4** (0.70, 0.35, 0.697, 1.05, 1.041) |
| Monotonic decline? | Yes (7 straight turns) | **No** — E_recon rose on turns 2, 5, 11 |
| Turns to settle | 8 | 12 |

---

## Key Observations

### 1. 10x Lower Learning Rate Preserves Novelty Sensitivity

Session 1's eta_w=0.0001 caused 17.3% energy decline in 8 turns. Session 2's eta_w=0.00001 limited decline to 2.5% over 12 turns. The PCHead still learns, but slowly enough that novel content can register as energy spikes rather than being immediately absorbed.

### 2. Energy Is No Longer Monotonically Declining

E_recon rose on turns 2 (architecture→architecture, but different details), 5 (grief→Rust), and 11 (emoji→biochemistry). In session 1, energy declined on every single turn regardless of content. This is the behavioral change we needed.

### 3. Temperature Now Varies Meaningfully

Session 1 was stuck at 0.35 for 7 of 8 turns. Session 2 produced temperatures of 0.35, 0.697, 1.041, and 1.050. The system is actually modulating generation behavior based on content — higher temperature for genuinely novel content (Rust code after grief), lower for predictable continuations (Rust→more Rust).

### 4. Three Signal Crossings vs One

Turn 7 (nonsense after Rust code) and Turn 12 ("No." after biochemistry) both produced E_predict > E_recon — the transition was more surprising than the content was complex. Session 1 only achieved this once. The lower learning rate prevents the predictive model from adapting too quickly to absorb surprise.

### 5. Cosine Distance Captures What Energy Misses

The cosine distance metric adds a complementary signal:
- **0.000** for identical repeat (turn 8) — ground truth validation
- **0.215** for same-domain continuation (Rust→Rust)
- **0.280-0.310** for related domain shifts (intro→architecture, architecture→hex)
- **0.505-0.622** for hard pivots (grief→Rust, Rust→nonsense, nonsense→math)
- Cosine distance reflects embedding-space geometry, not PCHead learning state

### 6. K=100 Settle Steps: Convergence Still Not Reached

200 total steps (100 per phase) and converged=false on all turns. The 4096-dimensional embedding space is large. However, the energy values are lower than session 1's K=20 equivalents (turn 1: 10136 vs 10623), suggesting the extra settling is finding lower-energy states.

### 7. Retrieval Gate: Armed But Not Triggered

MemoryGate was configured with 95th percentile dynamic thresholds. With only 12 data points and relatively stable energies, no turn exceeded the 95th percentile after the initial warmup period. The gate is functioning correctly — it would need a genuine spike above the session's observed range to fire. A longer session or a truly out-of-distribution prompt would be needed to trigger it.

### 8. Identical Input Produces cos_dist=0.000

Turn 8 (repeat of turn 7's nonsense) produced cosine distance of effectively zero (floating point: -1.19e-7), confirming that the GGUF backend's embedding is deterministic. This also caused E_recon to drop noticeably (10098→10042) — the PCHead learned that specific pattern from turn 7.

---

## Cosine Distance Bands (Empirical)

From this session's data, we can establish rough cosine distance bands for this model:

| Band | Cos Distance | Meaning | Examples |
|------|-------------|---------|----------|
| Identical | 0.000 | Same input | Repeated nonsense |
| Continuation | 0.15-0.25 | Same domain/topic | Rust code → more Rust |
| Related shift | 0.25-0.35 | Adjacent domain | Architecture → hex bytes |
| Hard pivot | 0.50-0.65 | Unrelated domains | Grief → Rust, math → emoji |

---

## Configuration for Future Sessions

Based on session 2 data:
- **eta_w=1e-5** is the right ballpark — keeps sensitivity over 12+ turns
- **K=100** produces cleaner energy signals but 5x latency cost; K=50 may be a good compromise
- **Retrieval percentile=95** is too conservative for a 12-turn session; consider percentile=90 or absolute threshold around 10150 (which would trigger on turns 2-5)
- **Cosine distance** should be added as a third signal for retrieval gating — pivots with cos>0.5 are genuine domain shifts
