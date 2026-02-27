#!/usr/bin/env python3
"""Send a diverse set of prompts to the PCDC server and print pcdc metadata."""

import json
import sys
import time
import urllib.request
import urllib.error

BASE = "http://localhost:8000/v1/chat/completions"

PROMPTS = [
    # --- Cluster 1: Programming / CS ---
    "Explain how Rust's borrow checker prevents data races at compile time.",
    "What is the difference between a mutex and a semaphore?",
    "How does garbage collection work in the JVM?",
    # --- Cluster 2: Philosophy ---
    "What did Kant mean by the categorical imperative?",
    "Explain Sartre's concept of radical freedom and bad faith.",
    "What is the hard problem of consciousness?",
    # --- Cluster 3: Cooking ---
    "How do you make authentic French onion soup?",
    "What is the difference between fermentation in sourdough vs commercial yeast bread?",
    # --- Cluster 4: Science ---
    "Explain the double-slit experiment in quantum mechanics.",
    "How does CRISPR-Cas9 gene editing work?",
    "What causes the greenhouse effect?",
    # --- Cluster 5: Back to CS (test deviation matching) ---
    "How does TCP ensure reliable data delivery over an unreliable network?",
    "Explain how a hash table handles collisions.",
    # --- Cluster 6: History ---
    "What were the main causes of the 2008 financial crisis?",
    "How did the Treaty of Westphalia shape modern international relations?",
    # --- Cluster 7: Mixed / novel ---
    "Compare the aesthetics of Japanese ukiyo-e prints with European Renaissance painting.",
    "If predictive coding is how the brain works, what does that imply for artificial intelligence?",
    # --- Cluster 8: Return to earlier domains ---
    "Explain ownership and lifetimes in Rust.",
    "What is Kant's distinction between phenomena and noumena?",
    "Describe the process of making kimchi jjigae.",
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
    print(f"  retrieval:{pcdc.get('retrieval_triggered', False)}")
    print(f"  Reply: {reply}...")

print(f"\n{'='*70}")
print(f"Done. {len(PROMPTS)} prompts sent.")
