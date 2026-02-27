#!/usr/bin/env python3
"""Send off-corpus prompts to test deviation routing on novel domains.

The bootstrap corpus covers: CS, philosophy, cooking, physics, biology,
climate, history, math, literature, medicine, art, cryptography.

These prompts deliberately target domains NOT in the corpus to test
whether the pre-trained PCHead generalizes or goes orthogonal.
"""

import json
import time
import urllib.request

BASE = "http://localhost:8000/v1/chat/completions"

PROMPTS = [
    # --- Cluster 1: Sports (not in corpus) ---
    "How does the offside rule work in football?",
    "What makes a perfect tennis serve?",
    # --- Cluster 2: Law (not in corpus) ---
    "Explain the difference between civil and criminal law.",
    "What is the doctrine of stare decisis?",
    # --- Cluster 3: Music performance (not in corpus) ---
    "How do you tune a piano using equal temperament?",
    "What is the circle of fifths and why does it matter?",
    # --- Cluster 4: Agriculture / Gardening (not in corpus) ---
    "How does crop rotation prevent soil depletion?",
    "What is companion planting and does it actually work?",
    # --- Cluster 5: Film / Cinema (not in corpus) ---
    "What is the 180-degree rule in cinematography?",
    "How did Italian neorealism change filmmaking?",
    # --- Cluster 6: Psychology (clinical, not philosophical) ---
    "What is cognitive behavioral therapy and how does it work?",
    "Explain the difference between PTSD and complex PTSD.",
    # --- Cluster 7: Linguistics (not in corpus) ---
    "What is the Sapir-Whorf hypothesis?",
    "How do tonal languages like Mandarin encode meaning in pitch?",
    # --- Cluster 8: Return to corpus domains for comparison ---
    "Explain how a red-black tree maintains balance.",  # CS - in corpus
    "What is Bayes' theorem?",                          # Math - in corpus
    "How does the immune system fight a viral infection?",  # Biology - in corpus
    # --- Cluster 9: Truly alien domains ---
    "What are the rules of competitive dog grooming?",
    "How do you score a game of cricket?",
    "Explain the difference between knitting and crocheting.",
]

for i, prompt in enumerate(PROMPTS, 1):
    print(f"\n{'='*70}")
    print(f"[{i}/{len(PROMPTS)}] {prompt[:60]}...")
    t0 = time.time()

    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 80,
        "temperature": 1.0,
    }).encode()
    req = urllib.request.Request(
        BASE, data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())

    elapsed = time.time() - t0
    pcdc = data.get("pcdc", {})
    reply = data["choices"][0]["message"]["content"][:120]

    cos_dist = pcdc.get("cosine_distance")
    dev_match = pcdc.get("deviation_match_score")

    print(f"  Time: {elapsed:.1f}s")
    print(f"  E_recon:  {pcdc.get('reconstruction_energy', 0):.2f}")
    print(f"  E_pred:   {pcdc.get('predictive_energy', 0):.2f}")
    print(f"  E_blend:  {pcdc.get('settling_energy', 0):.2f}")
    print(f"  cos_dist: {f'{cos_dist:.4f}' if cos_dist is not None else 'N/A (first turn)'}")
    print(f"  dev_match:{f'{dev_match:.4f}' if dev_match is not None else 'N/A'}")
    print(f"  adj_temp: {pcdc.get('adjusted_temperature', 0):.3f}")
    print(f"  converged:{pcdc.get('converged', '?')}")
    print(f"  Reply: {reply}...")

print(f"\n{'='*70}")
print(f"Done. {len(PROMPTS)} prompts sent.")
