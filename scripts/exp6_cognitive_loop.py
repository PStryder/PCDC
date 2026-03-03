#!/usr/bin/env python3
"""Experiment 6: Cognitive Loop Test Battery.

Tests the four-stage cognitive loop (Perceive → Recall → Comprehend → Respond)
against baseline LLM and standard PCDC across five conversational scenarios.

Usage:
    python scripts/exp6_cognitive_loop.py
    python scripts/exp6_cognitive_loop.py --n-gpu-layers 33
    python scripts/exp6_cognitive_loop.py --scenario 3  # run only scenario 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUT_DIR = PROJECT_DIR / "experiments" / "cognitive_loop"
PLOTS_DIR = OUT_DIR / "plots"

sys.path.insert(0, str(PROJECT_DIR / "src"))

from pcdc.gguf.gguf_backend import GGUFBackend
from pcdc.gguf.pc_head import PCHead
from pcdc.server.steering import SteeringEngine, CognitiveLoopMetadata


# ---------------------------------------------------------------------------
# Llama-3 chat template (standalone, mirrors app.py)
# ---------------------------------------------------------------------------

LLAMA3_BOS = "<|begin_of_text|>"
LLAMA3_HEADER = "<|start_header_id|>{role}<|end_header_id|>\n\n"
LLAMA3_EOT = "<|eot_id|>"


def format_llama3_prompt(messages: list[dict]) -> str:
    parts = [LLAMA3_BOS]
    for msg in messages:
        parts.append(LLAMA3_HEADER.format(role=msg["role"]))
        parts.append(msg["content"])
        parts.append(LLAMA3_EOT)
    parts.append(LLAMA3_HEADER.format(role="assistant"))
    return "".join(parts)


def _sanitize(text: str) -> str:
    for token in ("<|begin_of_text|>", "<|end_of_text|>",
                  "<|start_header_id|>", "<|end_header_id|>",
                  "<|eot_id|>"):
        text = text.replace(token, "")
    return text


def build_comprehension_prompt(user_text: str, recalled_texts: list[str]) -> str:
    recalled_block = "\n".join(
        f"[{i + 1}] {_sanitize(t[:500])}"
        for i, t in enumerate(recalled_texts)
    )
    messages = [
        {"role": "system", "content": "You are analyzing patterns across a conversation. Be concise."},
        {"role": "user", "content": (
            f'Current input: "{_sanitize(user_text[:500])}"\n\n'
            f"Related past interactions:\n{recalled_block}\n\n"
            "In 1-2 sentences, what connects these experiences? "
            "What pattern is emerging?"
        )},
    ]
    return format_llama3_prompt(messages)


def format_comprehension_response_prompt(
    messages: list[dict], comprehension_summary: str,
) -> str:
    safe_summary = _sanitize(comprehension_summary)
    context_msg = {"role": "system", "content": f"Contextual insight from experiential pattern matching:\n{safe_summary}"}
    return format_llama3_prompt([context_msg] + messages)


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS = {
    1: {
        "name": "domain_continuity",
        "description": "10 turns within computer science — tests coherence within sustained topic",
        "prompts": [
            "Explain how hash tables handle collisions using chaining.",
            "What are the tradeoffs between open addressing and separate chaining?",
            "How does the load factor affect hash table performance?",
            "Describe how dynamic resizing works in hash maps.",
            "What makes a good hash function for string keys?",
            "Compare the time complexity of hash tables vs balanced BSTs.",
            "How do concurrent hash maps handle thread safety?",
            "Explain consistent hashing and its use in distributed systems.",
            "What is the birthday paradox and how does it relate to hashing?",
            "Describe bloom filters and their relationship to hash functions.",
        ],
    },
    2: {
        "name": "domain_switching",
        "description": "10 turns alternating domains — tests cross-domain pattern recognition",
        "prompts": [
            "Explain the transformer self-attention mechanism.",
            "How do you properly caramelize onions for French cooking?",
            "What is Kant's categorical imperative?",
            "Describe how TCP establishes a connection.",
            "What were the main causes of the 2008 financial crisis?",
            "How does quantum entanglement work?",
            "What are the five stages of grief?",
            "Explain Rust's ownership and borrowing system.",
            "How does climate change affect ocean currents?",
            "Describe the role of prediction errors in predictive coding theory.",
        ],
    },
    3: {
        "name": "callbacks",
        "description": "15 turns: 5 topic A, 5 topic B, 5 topic A — tests deviation recall on return",
        "prompts": [
            # Topic A: Neural networks (turns 1-5)
            "What is backpropagation and how does it compute gradients?",
            "Explain the vanishing gradient problem in deep networks.",
            "How do residual connections in ResNets help training?",
            "What role does batch normalization play in training stability?",
            "Compare SGD, Adam, and AdaGrad optimizers.",
            # Topic B: Cooking (turns 6-10)
            "How do you make authentic kimchi jjigae?",
            "What's the difference between fermented and fresh kimchi in cooking?",
            "Explain the Maillard reaction in browning meat.",
            "How does slow cooking affect collagen in tough cuts of meat?",
            "What role does acid play in balancing flavors in soups?",
            # Topic A return: Neural networks (turns 11-15)
            "How does dropout regularization prevent overfitting?",
            "Explain the difference between L1 and L2 regularization.",
            "What is transfer learning and when should you use it?",
            "How do attention mechanisms differ from convolutions?",
            "Describe how learning rate scheduling affects convergence.",
        ],
    },
    4: {
        "name": "ambiguous",
        "description": "10 turns with ambiguous/multi-domain prompts — tests comprehension quality",
        "prompts": [
            "Tell me about layers.",
            "How do you handle overflow?",
            "What makes a good model?",
            "Explain the concept of attention.",
            "How does memory work?",
            "What is the role of temperature?",
            "Describe the process of training.",
            "How do you deal with noise?",
            "What is convergence?",
            "Explain the concept of energy.",
        ],
    },
    5: {
        "name": "long_session",
        "description": "50 turns with mixed domains and gradual drift — tests scaling and coherence",
        "prompts": [
            # Block 1: CS fundamentals (1-8)
            "What is a binary search tree and how does insertion work?",
            "Explain the difference between depth-first and breadth-first search.",
            "How does dynamic programming differ from greedy algorithms?",
            "Describe the concept of amortized time complexity.",
            "What is a lock-free data structure?",
            "How do garbage collectors decide what to free?",
            "Explain the CAP theorem in distributed systems.",
            "What are consensus algorithms like Raft and Paxos?",
            # Block 2: Drift to physics (9-16)
            "How do distributed systems relate to statistical mechanics?",
            "What is entropy in thermodynamics?",
            "Explain the second law of thermodynamics.",
            "How does quantum computing use superposition?",
            "What is quantum decoherence?",
            "Describe the double-slit experiment.",
            "How does the observer effect manifest in quantum mechanics?",
            "What is the measurement problem in quantum physics?",
            # Block 3: Drift to philosophy (17-24)
            "How does the measurement problem connect to philosophy of mind?",
            "What is the hard problem of consciousness?",
            "Explain philosophical zombies as a thought experiment.",
            "How does functionalism address the mind-body problem?",
            "What is intentionality in philosophy of mind?",
            "Describe the Chinese Room argument.",
            "How does emergence relate to consciousness?",
            "What is panpsychism and what motivates it?",
            # Block 4: Drift to neuroscience (25-32)
            "How does neuroscience approach the consciousness question?",
            "What is the neural correlate of consciousness?",
            "Explain predictive processing in the brain.",
            "How do prediction errors drive learning in neural circuits?",
            "What is the free energy principle?",
            "How does active inference differ from reinforcement learning?",
            "Describe the hierarchical nature of cortical processing.",
            "What role does the thalamus play in consciousness?",
            # Block 5: Loop back to CS/AI (33-40)
            "How does the free energy principle connect to machine learning?",
            "What is variational inference?",
            "How do variational autoencoders work?",
            "Explain the reparameterization trick in VAEs.",
            "How do diffusion models generate images?",
            "What is the relationship between diffusion and score matching?",
            "How do transformers handle long-range dependencies?",
            "What is the connection between attention and memory retrieval?",
            # Block 6: Mixed callbacks (41-50)
            "Thinking about our earlier discussion of hash tables — how do they relate to memory in the brain?",
            "How is the Maillard reaction similar to protein folding?",
            "What would Kant say about artificial intelligence?",
            "Does the Chinese Room argument apply to large language models?",
            "How does the observer effect in quantum mechanics relate to predictive coding?",
            "Can lock-free algorithms teach us about neural synchronization?",
            "How does the concept of entropy connect thermodynamics, information theory, and cooking?",
            "What are the philosophical implications of transfer learning?",
            "How does the free energy principle unify the topics we've discussed?",
            "Summarize the key themes that emerged across our conversation.",
        ],
    },
}


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------

MAX_HISTORY_TURNS = 10  # keep last N turns to avoid context overflow


def _trim_history(messages: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    """Keep only the last max_turns user/assistant pairs."""
    if len(messages) <= max_turns * 2:
        return messages
    return messages[-(max_turns * 2):]


def run_baseline(backend: GGUFBackend, prompts: list[str], max_tokens: int = 128) -> list[dict]:
    """Condition A: Direct LLM generation, no PCDC."""
    results = []
    messages_history: list[dict] = []

    for i, prompt_text in enumerate(prompts):
        messages_history.append({"role": "user", "content": prompt_text})
        formatted = format_llama3_prompt(_trim_history(messages_history))

        t0 = time.perf_counter()
        embedding, tokens = backend.embed_and_warm(formatted)
        result = backend.llm.create_completion(
            prompt=tokens,
            temperature=1.0,
            max_tokens=max_tokens,
            stop=["<|eot_id|>"],
        )
        latency = (time.perf_counter() - t0) * 1000

        content = result["choices"][0]["text"].strip()
        messages_history.append({"role": "assistant", "content": content})

        # Embed the response for coherence analysis
        resp_emb = backend.embed_chat(content)

        results.append({
            "turn": i + 1,
            "prompt": prompt_text,
            "response": content[:500],
            "latency_ms": latency,
            "response_embedding": resp_emb.numpy().tolist(),
        })
        print(f"  Baseline turn {i + 1}/{len(prompts)}: {latency:.0f}ms")

    return results


def run_pcdc(engine: SteeringEngine, backend: GGUFBackend,
             prompts: list[str], max_tokens: int = 128) -> list[dict]:
    """Condition B: Standard PCDC (steer_and_generate)."""
    results = []
    messages_history: list[dict] = []

    for i, prompt_text in enumerate(prompts):
        messages_history.append({"role": "user", "content": prompt_text})
        trimmed = _trim_history(messages_history)
        formatted = format_llama3_prompt(trimmed)
        messages_raw = list(trimmed)

        t0 = time.perf_counter()
        steering, result = engine.steer_and_generate(
            formatted,
            base_temp=1.0,
            messages=messages_raw,
            max_tokens=max_tokens,
            stop=["<|eot_id|>"],
        )
        latency = (time.perf_counter() - t0) * 1000

        content = result["choices"][0]["text"].strip()
        messages_history.append({"role": "assistant", "content": content})

        resp_emb = backend.embed_chat(content)

        results.append({
            "turn": i + 1,
            "prompt": prompt_text,
            "response": content[:500],
            "latency_ms": latency,
            "response_embedding": resp_emb.numpy().tolist(),
            "pcdc": {
                "energy": steering.energy,
                "e_recon": steering.reconstruction_energy,
                "e_predict": steering.predictive_energy,
                "adjusted_temp": steering.adjusted_temp,
                "converged": steering.converged,
                "cosine_distance": steering.cosine_distance,
                "deviation_match_score": steering.deviation_match_score,
            },
        })
        print(f"  PCDC turn {i + 1}/{len(prompts)}: {latency:.0f}ms  "
              f"E={steering.energy:.1f} T={steering.adjusted_temp:.3f}")

    return results


def run_cognitive(engine: SteeringEngine, backend: GGUFBackend,
                  prompts: list[str], max_tokens: int = 128) -> list[dict]:
    """Condition C: Cognitive loop (steer_comprehend_and_generate)."""
    results = []
    messages_history: list[dict] = []

    for i, prompt_text in enumerate(prompts):
        messages_history.append({"role": "user", "content": prompt_text})
        trimmed = _trim_history(messages_history)
        formatted = format_llama3_prompt(trimmed)
        messages_raw = list(trimmed)

        t0 = time.perf_counter()
        steering, result, loop_meta = engine.steer_comprehend_and_generate(
            formatted,
            prompt_text,
            base_temp=1.0,
            messages=messages_raw,
            build_comprehension_prompt_fn=build_comprehension_prompt,
            format_response_prompt_fn=format_comprehension_response_prompt,
            max_tokens=max_tokens,
            stop=["<|eot_id|>"],
        )
        latency = (time.perf_counter() - t0) * 1000

        content = result["choices"][0]["text"].strip()
        messages_history.append({"role": "assistant", "content": content})

        resp_emb = backend.embed_chat(content)

        results.append({
            "turn": i + 1,
            "prompt": prompt_text,
            "response": content[:500],
            "latency_ms": latency,
            "response_embedding": resp_emb.numpy().tolist(),
            "pcdc": {
                "energy": steering.energy,
                "e_recon": steering.reconstruction_energy,
                "e_predict": steering.predictive_energy,
                "adjusted_temp": steering.adjusted_temp,
                "converged": steering.converged,
                "cosine_distance": steering.cosine_distance,
                "deviation_match_score": steering.deviation_match_score,
            },
            "cognitive": {
                "recall_count": loop_meta.recall_count,
                "recall_scores": loop_meta.recall_scores,
                "recall_texts_preview": loop_meta.recall_texts_preview,
                "comprehension_summary": loop_meta.comprehension_summary,
                "comprehension_tokens": loop_meta.comprehension_tokens,
                "comprehension_latency_ms": loop_meta.comprehension_latency_ms,
                "skipped_comprehension": loop_meta.skipped_comprehension,
                "total_latency_ms": loop_meta.total_latency_ms,
            },
        })
        recall_info = f"recall={loop_meta.recall_count}" if not loop_meta.skipped_comprehension else "no-recall"
        print(f"  Cognitive turn {i + 1}/{len(prompts)}: {latency:.0f}ms  "
              f"E={steering.energy:.1f} T={steering.adjusted_temp:.3f} {recall_info}")

    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_session_coherence(turns: list[dict]) -> list[float]:
    """Consecutive cosine similarity between response embeddings."""
    coherences = []
    for i in range(1, len(turns)):
        emb_prev = torch.tensor(turns[i - 1]["response_embedding"])
        emb_curr = torch.tensor(turns[i]["response_embedding"])
        cos_sim = F.cosine_similarity(emb_prev.unsqueeze(0), emb_curr.unsqueeze(0)).item()
        coherences.append(cos_sim)
    return coherences


def analyze_scenario(scenario_data: dict) -> dict:
    """Compute aggregate metrics for a scenario."""
    analysis = {}

    for cond_name, turns in scenario_data["conditions"].items():
        latencies = [t["latency_ms"] for t in turns]
        coherences = compute_session_coherence(turns)

        stats = {
            "latency_mean_ms": float(np.mean(latencies)),
            "latency_std_ms": float(np.std(latencies)),
            "latency_median_ms": float(np.median(latencies)),
            "coherence_mean": float(np.mean(coherences)) if coherences else 0.0,
            "coherence_std": float(np.std(coherences)) if coherences else 0.0,
            "n_turns": len(turns),
        }

        # Condition-specific metrics
        if cond_name == "cognitive":
            recall_counts = [t["cognitive"]["recall_count"] for t in turns]
            comp_latencies = [t["cognitive"]["comprehension_latency_ms"]
                              for t in turns if not t["cognitive"]["skipped_comprehension"]]
            stats["recall_count_mean"] = float(np.mean(recall_counts))
            stats["comprehension_turns"] = len(comp_latencies)
            stats["comprehension_latency_mean_ms"] = (
                float(np.mean(comp_latencies)) if comp_latencies else 0.0
            )
            # Collect all recall scores across turns
            all_scores = []
            for t in turns:
                all_scores.extend(t["cognitive"]["recall_scores"])
            stats["recall_score_mean"] = float(np.mean(all_scores)) if all_scores else 0.0

        analysis[cond_name] = stats

    return analysis


def generate_plots(all_results: dict[int, dict]):
    """Generate comparison plots across scenarios."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Latency comparison ---
    fig, axes = plt.subplots(1, len(all_results), figsize=(4 * len(all_results), 5), squeeze=False)
    for idx, (scen_id, data) in enumerate(sorted(all_results.items())):
        ax = axes[0, idx]
        cond_names = []
        cond_latencies = []
        for cond in ["baseline", "pcdc", "cognitive"]:
            if cond in data["conditions"]:
                cond_names.append(cond)
                cond_latencies.append([t["latency_ms"] for t in data["conditions"][cond]])
        ax.boxplot(cond_latencies, labels=cond_names)
        ax.set_title(f"S{scen_id}: {data['scenario_name']}", fontsize=9)
        ax.set_ylabel("Latency (ms)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "latency_comparison.png", dpi=150)
    plt.close()

    # --- Plot 2: Session coherence ---
    fig, axes = plt.subplots(1, len(all_results), figsize=(4 * len(all_results), 5), squeeze=False)
    for idx, (scen_id, data) in enumerate(sorted(all_results.items())):
        ax = axes[0, idx]
        for cond in ["baseline", "pcdc", "cognitive"]:
            if cond in data["conditions"]:
                coherences = compute_session_coherence(data["conditions"][cond])
                ax.plot(range(2, len(coherences) + 2), coherences, label=cond, alpha=0.8)
        ax.set_title(f"S{scen_id}: {data['scenario_name']}", fontsize=9)
        ax.set_ylabel("Cosine similarity")
        ax.set_xlabel("Turn")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "session_coherence.png", dpi=150)
    plt.close()

    # --- Plot 3: Recall scores over turns (cognitive only) ---
    fig, axes = plt.subplots(1, len(all_results), figsize=(4 * len(all_results), 5), squeeze=False)
    for idx, (scen_id, data) in enumerate(sorted(all_results.items())):
        ax = axes[0, idx]
        if "cognitive" in data["conditions"]:
            turns = data["conditions"]["cognitive"]
            turn_nums = []
            best_scores = []
            for t in turns:
                scores = t["cognitive"]["recall_scores"]
                if scores:
                    turn_nums.append(t["turn"])
                    best_scores.append(max(scores))
            if turn_nums:
                ax.plot(turn_nums, best_scores, "o-", markersize=3)
                ax.axhline(y=0.6, color="r", linestyle="--", alpha=0.5, label="threshold=0.6")
        ax.set_title(f"S{scen_id}: {data['scenario_name']}", fontsize=9)
        ax.set_ylabel("Best recall score")
        ax.set_xlabel("Turn")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "recall_scores.png", dpi=150)
    plt.close()

    print(f"Plots saved to {PLOTS_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Experiment 6: Cognitive Loop Test Battery")
    parser.add_argument("--model", default=str(PROJECT_DIR / "Llama-3-8B-Lexi-Uncensored.Q8_0.gguf"))
    parser.add_argument("--checkpoint", default=str(PROJECT_DIR / "pchead_pretrained.ckpt"))
    parser.add_argument("--n-gpu-layers", type=int, default=33)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--scenario", type=int, default=0, help="Run only this scenario (0=all)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--settle-k", type=int, default=100)
    parser.add_argument("--online-eta-w", type=float, default=1e-5)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    print(f"Loading GGUF model...")
    backend = GGUFBackend(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_threads=8,
        n_gpu_layers=args.n_gpu_layers,
    )
    hidden_dim = backend.hidden_dim
    print(f"  Model loaded: hidden_dim={hidden_dim}")

    # --- Build PCHead ---
    pc_head = PCHead(
        feature_dim=hidden_dim,
        num_classes=hidden_dim,
        hidden_sizes=[1024, 256],
    )

    # --- Create engine (fresh for each scenario to reset replay) ---
    def make_engine():
        engine = SteeringEngine(
            backend=backend,
            pc_head=pc_head,
            alpha=0.5,
            beta=0.5,
            online_eta_w=args.online_eta_w,
            replay_k=4,
            settle_k=args.settle_k,
        )
        # Load pretrained weights
        if Path(args.checkpoint).exists():
            engine.load_checkpoint(args.checkpoint)
            print(f"  Checkpoint loaded: {args.checkpoint}")
        return engine

    # --- Run scenarios ---
    scenarios_to_run = (
        {args.scenario: SCENARIOS[args.scenario]} if args.scenario > 0
        else SCENARIOS
    )

    all_results: dict[int, dict] = {}

    for scen_id, scenario in scenarios_to_run.items():
        print(f"\n{'='*60}")
        print(f"Scenario {scen_id}: {scenario['name']}")
        print(f"  {scenario['description']}")
        print(f"  {len(scenario['prompts'])} turns")
        print(f"{'='*60}")

        prompts = scenario["prompts"]
        scenario_data = {
            "scenario_id": scen_id,
            "scenario_name": scenario["name"],
            "description": scenario["description"],
            "n_turns": len(prompts),
            "conditions": {},
        }

        # --- Condition A: Baseline ---
        print("\n[A] Baseline (no PCDC)...")
        baseline_results = run_baseline(backend, prompts, max_tokens=args.max_tokens)
        scenario_data["conditions"]["baseline"] = baseline_results

        # --- Condition B: Standard PCDC ---
        print("\n[B] Standard PCDC...")
        engine_b = make_engine()
        pcdc_results = run_pcdc(engine_b, backend, prompts, max_tokens=args.max_tokens)
        scenario_data["conditions"]["pcdc"] = pcdc_results

        # --- Condition C: Cognitive Loop ---
        print("\n[C] Cognitive Loop...")
        engine_c = make_engine()
        cognitive_results = run_cognitive(engine_c, backend, prompts, max_tokens=args.max_tokens)
        scenario_data["conditions"]["cognitive"] = cognitive_results

        # --- Analyze ---
        scenario_data["analysis"] = analyze_scenario(scenario_data)

        all_results[scen_id] = scenario_data

        # Save individual scenario
        out_path = OUT_DIR / f"scenario_{scen_id}_{scenario['name']}.json"
        with open(out_path, "w") as f:
            json.dump(scenario_data, f, indent=2, default=str)
        print(f"\nScenario {scen_id} saved to {out_path}")

    # --- Summary ---
    summary = {
        "experiment": "exp6_cognitive_loop",
        "config": {
            "model": args.model,
            "checkpoint": args.checkpoint,
            "settle_k": args.settle_k,
            "online_eta_w": args.online_eta_w,
            "max_tokens": args.max_tokens,
            "n_gpu_layers": args.n_gpu_layers,
        },
        "scenarios": {},
    }

    for scen_id, data in sorted(all_results.items()):
        summary["scenarios"][scen_id] = data["analysis"]

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # --- FINDINGS ---
    write_findings(all_results, summary)

    # --- Plots ---
    generate_plots(all_results)

    print(f"\nAll results saved to {OUT_DIR}")


def write_findings(all_results: dict, summary: dict):
    """Write FINDINGS.txt with key observations."""
    lines = ["EXPERIMENT 6: COGNITIVE LOOP TEST BATTERY", "=" * 50, ""]

    for scen_id, data in sorted(all_results.items()):
        lines.append(f"Scenario {scen_id}: {data['scenario_name']}")
        lines.append(f"  {data['description']}")
        lines.append(f"  Turns: {data['n_turns']}")
        lines.append("")

        analysis = data["analysis"]
        for cond in ["baseline", "pcdc", "cognitive"]:
            if cond in analysis:
                a = analysis[cond]
                line = f"  [{cond:>10}] latency={a['latency_mean_ms']:.0f}ms±{a['latency_std_ms']:.0f} "
                line += f"coherence={a['coherence_mean']:.3f}±{a['coherence_std']:.3f}"
                if cond == "cognitive":
                    line += f" recall_mean={a.get('recall_count_mean', 0):.1f}"
                    line += f" comp_turns={a.get('comprehension_turns', 0)}"
                    line += f" recall_score={a.get('recall_score_mean', 0):.3f}"
                lines.append(line)
        lines.append("")

    # Overall summary
    lines.append("")
    lines.append("KEY FINDINGS")
    lines.append("-" * 40)

    # Compute averages across scenarios
    latency_ratios = []
    coherence_diffs = []
    for scen_id, data in all_results.items():
        a = data["analysis"]
        if "baseline" in a and "cognitive" in a:
            latency_ratios.append(a["cognitive"]["latency_mean_ms"] / max(a["baseline"]["latency_mean_ms"], 1))
            coherence_diffs.append(a["cognitive"]["coherence_mean"] - a["baseline"]["coherence_mean"])

    if latency_ratios:
        lines.append(f"  Cognitive/Baseline latency ratio: {np.mean(latency_ratios):.2f}x")
    if coherence_diffs:
        lines.append(f"  Cognitive-Baseline coherence delta: {np.mean(coherence_diffs):+.3f}")

    # Recall activation
    for scen_id, data in sorted(all_results.items()):
        if "cognitive" in data["conditions"]:
            turns = data["conditions"]["cognitive"]
            comp_turns = sum(1 for t in turns if not t["cognitive"]["skipped_comprehension"])
            lines.append(f"  Scenario {scen_id}: comprehension active on {comp_turns}/{len(turns)} turns")

    lines.append("")
    lines.append("Configuration:")
    cfg = summary["config"]
    lines.append(f"  settle_k={cfg['settle_k']}, online_eta_w={cfg['online_eta_w']}")
    lines.append(f"  max_tokens={cfg['max_tokens']}")
    lines.append("")

    findings_path = OUT_DIR / "FINDINGS.txt"
    with open(findings_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Findings written to {findings_path}")


if __name__ == "__main__":
    main()
