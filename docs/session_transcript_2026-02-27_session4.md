# PCDC Session 4 — Off-Corpus Generalization Test

**Date:** 2026-02-27
**Server:** pcdc-serve on localhost:8000
**Model:** Llama-3-8B-Lexi-Uncensored.Q8_0.gguf (33 GPU layers)
**PCHead:** [4096, 1024, 256, 4096], pre-trained checkpoint (5 epochs on 52-paragraph corpus)
**Config:** Same as Session 3b (adaptive IQR scale, deviation routing threshold=0.4)
**Session ID:** `43ea3630-f5ae-4b92-9f13-52283d3a939b`

**Purpose:** Test whether the pre-trained PCHead's deviation matching generalizes to domains not present in the bootstrap corpus, or collapses back to near-orthogonal (~0.07) like the untrained model.

---

## Corpus Coverage vs Test Domains

**Bootstrap corpus covers:** CS/programming, philosophy, cooking, physics, biology, climate, history, math, literature, medicine, art, cryptography, data structures, networking, distributed systems

**Deliberately off-corpus domains tested:**
- Sports (football, tennis)
- Law (civil/criminal, stare decisis)
- Music performance (piano tuning, circle of fifths)
- Agriculture (crop rotation, companion planting)
- Film/Cinema (180-degree rule, neorealism)
- Clinical psychology (CBT, PTSD)
- Linguistics (Sapir-Whorf, tonal languages)
- Truly alien domains (competitive dog grooming, cricket scoring, knitting vs crocheting)

**On-corpus controls** (turns 15-17): red-black trees, Bayes' theorem, immune system

---

## Full Trajectory (20 turns)

| Turn | Domain | In Corpus? | E_recon | E_blend | Temp | Dev Match | Signal |
|------|--------|-----------|---------|---------|------|-----------|--------|
| 1 | Sports/Football | No | 3458 | 3458 | 1.000 | **0.1508** | First turn, low match |
| 2 | Sports/Tennis | No | 3427 | 3405 | 0.996 | 0.7784 | Strong match after 1 turn |
| 3 | Law/Civil-Criminal | No | 3334 | 3366 | 0.994 | 0.7474 | Off-corpus, still strong |
| 4 | Law/Stare Decisis | No | 3222 | 3246 | 0.532 | 0.7583 | Within-law continuation |
| 5 | Music/Piano Tuning | No | 4953 | 4102 | 1.500 | 0.6857 | Domain shift + novel |
| 6 | Music/Circle of 5ths | No | 5017 | 4130 | 1.422 | **0.8202** | Strong within-music |
| 7 | Agri/Crop Rotation | No | 3249 | 3354 | 0.939 | 0.7299 | Off-corpus, solid |
| 8 | Agri/Companion Plant | No | 4923 | 4054 | 1.355 | 0.7819 | Within-domain |
| 9 | Film/180-degree Rule | No | 3428 | 4277 | 1.403 | **0.8977** | Highest off-corpus match |
| 10 | Film/Neorealism | No | 3242 | 3297 | 0.910 | 0.6763 | Within-film |
| 11 | Psych/CBT | No | 3289 | 4094 | 1.349 | 0.7874 | Off-corpus, strong |
| 12 | Psych/PTSD | No | 4890 | 4112 | 1.223 | 0.7057 | Within-psychology |
| 13 | Ling/Sapir-Whorf | No | 6460 | 4907 | 1.408 | **0.8832** | High recon + high match |
| 14 | Ling/Tonal Languages | No | 6653 | 4973 | 1.417 | 0.7084 | Highest E_recon |
| 15 | CS/Red-Black Tree | **Yes** | 3232 | 4067 | 1.000 | 0.6855 | Corpus control |
| 16 | Math/Bayes' Theorem | **Yes** | 4876 | 4115 | 1.024 | 0.7824 | Corpus control |
| 17 | Bio/Immune System | **Yes** | 4994 | 4111 | 1.012 | 0.7265 | Corpus control |
| 18 | Dog Grooming | No | 6381 | 6467 | 1.499 | 0.7358 | Truly alien, still works |
| 19 | Cricket Scoring | No | 4932 | 5723 | 1.485 | 0.7530 | Truly alien, still works |
| 20 | Knitting vs Crochet | No | 4902 | 4881 | 1.328 | 0.7826 | Truly alien, still works |

---

## Key Finding: The Training Generalized

| Condition | Avg Dev Match | N |
|-----------|--------------|---|
| Untrained baseline (Ghost Mic) | ~0.07 | - |
| On-corpus (session 3) | 0.74 | 20 |
| **On-corpus controls (session 4)** | **0.7315** | 3 |
| **Off-corpus novel domains** | **0.7284** | 17 |

**Off-corpus deviation matches are statistically indistinguishable from on-corpus matches.** The PCHead did not go orthogonal on novel domains.

The only low match was turn 1 (0.1508) — the first turn of the session, where the replay buffer only contains pre-trained corpus entries and there's no prior deviation to compare against. By turn 2 (same sports domain, different topic), the match jumps to 0.78.

### Why Does It Generalize?

The pre-training doesn't teach the PCHead specific domain knowledge. It teaches the PCHead to form **structured attractor basins** — to produce input-dependent deviation vectors instead of near-random ones. The bottleneck architecture (4096 -> 1024 -> 256 -> 4096) forces the PCHead to learn a compressed representation of the embedding space that generalizes to any content the LLM can embed.

The deviation vectors are structured because the PCHead has learned to settle differently for different inputs, not because it has memorized the corpus domains. A question about cricket produces a different deviation fingerprint than a question about Kant, and that fingerprint is consistent across similar inputs — even though neither cricket nor Kant's noumena were emphasized in training.

### Truly Alien Domains Still Match

The most striking result: competitive dog grooming (0.74), cricket scoring (0.75), and knitting (0.78) all produce strong deviation matches despite being about as far from the corpus as you can get. The PCHead's deviation space has structure everywhere, not just near the training data.

### Energy Responds Appropriately

- Highest E_recon: tonal languages (6653) and Sapir-Whorf (6460) — linguistics is genuinely novel territory
- Highest E_blend: dog grooming (6467) — the most alien topic produced the highest blended energy
- Lowest E_recon: stare decisis (3222) — legal concepts embed similarly to the philosophical content in the corpus

---

## Raw Data

- Telemetry: `docs/session4_offcorpus_telemetry.json`
- Script: `scripts/gather_telemetry_offcorpus.py`
