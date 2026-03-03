"""Consolidate exp6 scenario JSONs into summary.json and FINDINGS.txt."""

import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "experiments" / "cognitive_loop"


def load_scenarios():
    """Load all scenario_*.json files."""
    scenarios = {}
    for p in sorted(OUT_DIR.glob("scenario_*.json")):
        with open(p) as f:
            data = json.load(f)
        sid = data["scenario_id"]
        scenarios[sid] = data
    return scenarios


def main():
    scenarios = load_scenarios()
    print(f"Found {len(scenarios)} scenario files: {sorted(scenarios.keys())}")

    # --- Build summary.json ---
    summary = {
        "experiment": "exp6_cognitive_loop",
        "config": {
            "settle_k": 100,
            "online_eta_w": 1e-5,
            "max_tokens": 64,
            "n_gpu_layers": 33,
        },
        "scenarios": {},
    }

    for sid, data in sorted(scenarios.items()):
        summary["scenarios"][str(sid)] = data["analysis"]

    # Aggregate across all scenarios
    all_baseline_lat = []
    all_pcdc_lat = []
    all_cognitive_lat = []
    all_baseline_coh = []
    all_pcdc_coh = []
    all_cognitive_coh = []
    all_recall_scores = []
    total_comp_turns = 0
    total_turns = 0

    for sid, data in scenarios.items():
        a = data["analysis"]
        n = a["baseline"]["n_turns"]
        total_turns += n
        all_baseline_lat.append(a["baseline"]["latency_mean_ms"])
        all_pcdc_lat.append(a["pcdc"]["latency_mean_ms"])
        all_cognitive_lat.append(a["cognitive"]["latency_mean_ms"])
        all_baseline_coh.append(a["baseline"]["coherence_mean"])
        all_pcdc_coh.append(a["pcdc"]["coherence_mean"])
        all_cognitive_coh.append(a["cognitive"]["coherence_mean"])
        if "recall_score_mean" in a["cognitive"]:
            all_recall_scores.append(a["cognitive"]["recall_score_mean"])
        if "comprehension_turns" in a["cognitive"]:
            total_comp_turns += a["cognitive"]["comprehension_turns"]

    n_scen = len(scenarios)
    summary["aggregate"] = {
        "n_scenarios": n_scen,
        "total_turns": total_turns,
        "baseline_latency_mean": sum(all_baseline_lat) / n_scen,
        "pcdc_latency_mean": sum(all_pcdc_lat) / n_scen,
        "cognitive_latency_mean": sum(all_cognitive_lat) / n_scen,
        "cognitive_baseline_latency_ratio": (
            sum(all_cognitive_lat) / sum(all_baseline_lat)
        ),
        "baseline_coherence_mean": sum(all_baseline_coh) / n_scen,
        "pcdc_coherence_mean": sum(all_pcdc_coh) / n_scen,
        "cognitive_coherence_mean": sum(all_cognitive_coh) / n_scen,
        "cognitive_baseline_coherence_delta": (
            sum(all_cognitive_coh) / n_scen - sum(all_baseline_coh) / n_scen
        ),
        "recall_score_grand_mean": (
            sum(all_recall_scores) / len(all_recall_scores) if all_recall_scores else None
        ),
        "comprehension_active_turns": total_comp_turns,
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary.json ({n_scen} scenarios)")

    # --- Build FINDINGS.txt ---
    lines = [
        "EXPERIMENT 6: COGNITIVE LOOP TEST BATTERY",
        "=" * 50,
        "",
    ]

    scenario_names = {
        1: ("domain_continuity", "10 turns within computer science - tests coherence within sustained topic"),
        2: ("domain_switching", "10 turns alternating between domains - tests adaptation to topic shifts"),
        3: ("callbacks", "15 turns: 5 topic A -> 5 topic B -> 5 topic A - tests recall of earlier context"),
        4: ("ambiguous", "10 turns with multi-domain prompts - tests handling of ambiguous inputs"),
        5: ("long_session", "50 turns with gradual drift - tests long-term stability and context management"),
    }

    for sid, data in sorted(scenarios.items()):
        a = data["analysis"]
        name, desc = scenario_names.get(sid, (data.get("scenario_name", "?"), ""))
        lines.append(f"Scenario {sid}: {name}")
        lines.append(f"  {desc}")
        lines.append(f"  Turns: {a['baseline']['n_turns']}")
        lines.append("")

        for cond in ["baseline", "pcdc", "cognitive"]:
            c = a[cond]
            s = f"  [{cond:>9s}] latency={c['latency_mean_ms']:.0f}ms +/- {c['latency_std_ms']:.0f} coherence={c['coherence_mean']:.3f} +/- {c['coherence_std']:.3f}"
            if cond == "cognitive":
                s += f" recall_mean={c.get('recall_count_mean', 0):.1f}"
                s += f" comp_turns={c.get('comprehension_turns', 0)}"
                s += f" recall_score={c.get('recall_score_mean', 0):.3f}"
            lines.append(s)
        lines.append("")
        lines.append("")

    # Aggregate section
    agg = summary["aggregate"]
    lines.append("AGGREGATE RESULTS")
    lines.append("-" * 40)
    lines.append(f"  Scenarios: {agg['n_scenarios']}, Total turns: {agg['total_turns']}")
    lines.append(f"  Cognitive/Baseline latency ratio: {agg['cognitive_baseline_latency_ratio']:.2f}x")
    lines.append(f"  Mean coherence: baseline={agg['baseline_coherence_mean']:.3f}, pcdc={agg['pcdc_coherence_mean']:.3f}, cognitive={agg['cognitive_coherence_mean']:.3f}")
    lines.append(f"  Cognitive-Baseline coherence delta: {agg['cognitive_baseline_coherence_delta']:+.3f}")
    if agg["recall_score_grand_mean"] is not None:
        lines.append(f"  Recall score grand mean: {agg['recall_score_grand_mean']:.3f}")
    lines.append(f"  Comprehension active on {agg['comprehension_active_turns']}/{agg['total_turns']} turns")
    lines.append("")

    # Key findings
    lines.append("")
    lines.append("KEY FINDINGS")
    lines.append("-" * 40)

    # Per-scenario highlights
    for sid, data in sorted(scenarios.items()):
        a = data["analysis"]
        cog = a["cognitive"]
        base = a["baseline"]
        coh_delta = cog["coherence_mean"] - base["coherence_mean"]
        lat_ratio = cog["latency_mean_ms"] / base["latency_mean_ms"]
        comp = cog.get("comprehension_turns", 0)
        n = base["n_turns"]
        name = scenario_names.get(sid, (data.get("scenario_name", "?"), ""))[0]
        lines.append(f"  S{sid} ({name}): {lat_ratio:.1f}x latency, coherence delta {coh_delta:+.3f}, comprehension {comp}/{n} turns")

    lines.append("")
    lines.append("Configuration:")
    lines.append(f"  settle_k=100, online_eta_w=1e-05")
    lines.append(f"  max_tokens=64")
    lines.append("")

    with open(OUT_DIR / "FINDINGS.txt", "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote FINDINGS.txt")


if __name__ == "__main__":
    main()
