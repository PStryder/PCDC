#!/usr/bin/env python3
"""Post-hoc analysis of PCDC long-session test results.

Reads JSON output from long_session_test.py and produces 7 plots + summary markdown.

Usage:
    python scripts/analyse_session.py --input results/session_20260228.json
    python scripts/analyse_session.py --input results/session_20260228.json \
        --output-dir results/analysis/ --summary results/summary.md
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_session(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_series(turns: list[dict]) -> dict:
    """Extract all metric series from turn records."""
    n = len(turns)
    s = {
        "turn": list(range(1, n + 1)),
        "scenario": [t["scenario"] for t in turns],
        "domain": [t["domain"] for t in turns],
        "complexity": [t["complexity"] for t in turns],
        "e_recon": [],
        "e_predict": [],
        "e_blend": [],
        "temperature": [],
        "cosine_distance": [],
        "deviation_match": [],
        "elapsed": [t.get("elapsed_s", 0) for t in turns],
    }
    for t in turns:
        p = t.get("pcdc") or {}
        s["e_recon"].append(p.get("reconstruction_energy"))
        s["e_predict"].append(p.get("predictive_energy"))
        s["e_blend"].append(p.get("settling_energy"))
        s["temperature"].append(p.get("adjusted_temperature"))
        s["cosine_distance"].append(p.get("cosine_distance"))
        s["deviation_match"].append(p.get("deviation_match_score"))
    return s


def scenario_boundaries(scenarios: list[str]) -> list[tuple[int, str]]:
    """Find turn indices where scenario changes."""
    boundaries = [(0, scenarios[0])]
    for i in range(1, len(scenarios)):
        if scenarios[i] != scenarios[i - 1]:
            boundaries.append((i, scenarios[i]))
    return boundaries


def safe_array(values: list, default=0.0) -> np.ndarray:
    """Convert list with Nones to numpy array, replacing None with default."""
    return np.array([v if v is not None else default for v in values])


def valid_mask(values: list) -> np.ndarray:
    """Boolean mask for non-None values."""
    return np.array([v is not None for v in values])


# ── Plot 1: Energy Trajectory ──────────────────────────────────────────


def plot_energy_trajectory(s: dict, out: Path):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    turns = s["turn"]
    er = safe_array(s["e_recon"])
    ep = safe_array(s["e_predict"])
    eb = safe_array(s["e_blend"])

    # Panel 1: E_recon + E_predict with crossing markers
    ax1.plot(turns, er, "-", color="#e74c3c", linewidth=1.2, label="E_recon", alpha=0.9)
    ax1.plot(turns, ep, "-", color="#3498db", linewidth=1.2, label="E_predict", alpha=0.9)
    # Mark crossings
    for i in range(1, len(turns)):
        prev = er[i - 1] - ep[i - 1]
        curr = er[i] - ep[i]
        if prev * curr < 0:
            ax1.axvline(x=turns[i], color="#e67e22", linestyle="--", alpha=0.4, linewidth=0.8)
    ax1.set_ylabel("Energy")
    ax1.set_title("E_recon vs E_predict")
    ax1.legend(loc="upper right")

    # Panel 2: Blended energy with rolling median/IQR
    ax2.plot(turns, eb, "-", color="#9b59b6", linewidth=1.2, label="E_blend", alpha=0.9)
    if len(turns) >= 10:
        window = min(20, len(turns) // 2)
        medians = []
        iqr_lo = []
        iqr_hi = []
        for i in range(len(turns)):
            start = max(0, i - window + 1)
            w = eb[start : i + 1]
            q1, med, q3 = np.percentile(w, [25, 50, 75])
            medians.append(med)
            iqr_lo.append(q1)
            iqr_hi.append(q3)
        ax2.plot(turns, medians, "--", color="#2c3e50", linewidth=1, label=f"Rolling median (w={window})")
        ax2.fill_between(turns, iqr_lo, iqr_hi, color="#2c3e50", alpha=0.1, label="IQR band")
    ax2.set_ylabel("Blended Energy")
    ax2.set_title("Blended Energy with Adaptive Scale")
    ax2.legend(loc="upper right")

    # Panel 3: Cosine distance bars
    cos = s["cosine_distance"]
    cos_mask = valid_mask(cos)
    cos_turns = [turns[i] for i in range(len(turns)) if cos_mask[i]]
    cos_vals = [cos[i] for i in range(len(turns)) if cos_mask[i]]
    ax3.bar(cos_turns, cos_vals, color="#2ecc71", alpha=0.7, width=0.8)
    ax3.set_ylabel("Cosine Distance")
    ax3.set_xlabel("Turn")
    ax3.set_title("Cosine Distance")

    # Annotate scenario boundaries on all panels
    boundaries = scenario_boundaries(s["scenario"])
    for ax in (ax1, ax2, ax3):
        for idx, name in boundaries:
            ax.axvline(x=turns[idx], color="gray", linestyle=":", alpha=0.5, linewidth=0.5)
            ax.text(
                turns[idx] + 0.5, ax.get_ylim()[1] * 0.95, name,
                fontsize=6, rotation=45, va="top", alpha=0.6,
            )

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Plot 2: Temperature Distribution ──────────────────────────────────


def plot_temperature_dist(s: dict, out: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    temps = safe_array(s["temperature"])
    turns = s["turn"]
    domains = s["domain"]

    # Trajectory colored by domain
    unique_domains = sorted(set(domains))
    cmap = plt.cm.get_cmap("tab20", len(unique_domains))
    domain_colors = {d: cmap(i) for i, d in enumerate(unique_domains)}

    for i in range(len(turns)):
        ax1.scatter(turns[i], temps[i], color=domain_colors[domains[i]], s=15, alpha=0.7)
    ax1.plot(turns, temps, "-", color="gray", linewidth=0.5, alpha=0.4)
    ax1.set_ylabel("Temperature")
    ax1.set_xlabel("Turn")
    ax1.set_title("Temperature Trajectory (colored by domain)")

    # Legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=domain_colors[d], markersize=6, label=d)
        for d in unique_domains
    ]
    ax1.legend(handles=handles, loc="upper right", fontsize=6, ncol=3)

    # Histogram
    valid_temps = temps[temps > 0]
    if len(valid_temps) > 0:
        ax2.hist(valid_temps, bins=30, color="#3498db", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Temperature")
    ax2.set_ylabel("Count")
    ax2.set_title("Temperature Distribution")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Plot 3: IQR/Median Evolution ──────────────────────────────────────


def plot_iqr_evolution(s: dict, out: Path):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    er = safe_array(s["e_recon"])
    turns = s["turn"]

    medians = []
    iqrs = []
    q1s = []
    q3s = []
    for i in range(len(turns)):
        history = er[: i + 1]
        q1, med, q3 = np.percentile(history, [25, 50, 75])
        medians.append(med)
        iqrs.append(q3 - q1)
        q1s.append(q1)
        q3s.append(q3)

    ax1.plot(turns, medians, "-", color="#2c3e50", linewidth=1.5, label="Cumulative Median")
    ax1.set_ylabel("Median E_recon")
    ax1.set_title("Cumulative Median (adaptive scale center)")
    ax1.legend()

    ax2.plot(turns, iqrs, "-", color="#e74c3c", linewidth=1.5, label="Cumulative IQR")
    ax2.set_ylabel("IQR")
    ax2.set_title("Cumulative IQR (adaptive scale width)")
    ax2.legend()

    # IQR rate of change
    if len(iqrs) > 1:
        iqr_arr = np.array(iqrs)
        iqr_delta = np.diff(iqr_arr)
        ax3.plot(turns[1:], iqr_delta, "-", color="#9b59b6", linewidth=1, alpha=0.8)
        ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax3.set_ylabel("ΔIQR")
        ax3.set_title("IQR Change per Turn (convergence → 0)")

    ax3.set_xlabel("Turn")

    # Mark key checkpoints
    for ax in (ax1, ax2, ax3):
        for cp in [25, 50, 100, 200]:
            if cp <= len(turns):
                ax.axvline(x=cp, color="green", linestyle=":", alpha=0.4)
                ax.text(cp, ax.get_ylim()[1] * 0.9, f"t={cp}", fontsize=7, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Plot 4: Deviation Match Score ─────────────────────────────────────


def plot_deviation_match(s: dict, out: Path):
    fig, ax = plt.subplots(figsize=(14, 5))
    turns = s["turn"]
    dev = s["deviation_match"]
    dev_mask = valid_mask(dev)
    dev_turns = [turns[i] for i in range(len(turns)) if dev_mask[i]]
    dev_vals = [dev[i] for i in range(len(turns)) if dev_mask[i]]
    domains = s["domain"]

    if not dev_vals:
        ax.text(0.5, 0.5, "No deviation match data", transform=ax.transAxes, ha="center")
    else:
        unique_domains = sorted(set(domains))
        cmap = plt.cm.get_cmap("tab20", len(unique_domains))
        domain_colors = {d: cmap(i) for i, d in enumerate(unique_domains)}

        for i, t_idx in enumerate([j for j in range(len(turns)) if dev_mask[j]]):
            ax.scatter(turns[t_idx], dev[t_idx], color=domain_colors[domains[t_idx]], s=20, alpha=0.7)

        ax.plot(dev_turns, dev_vals, "-", color="gray", linewidth=0.5, alpha=0.4)
        ax.axhline(y=0.6, color="red", linestyle="--", alpha=0.6, label="Threshold (0.6)")
        ax.set_ylabel("Deviation Match Score")
        ax.set_title("Deviation Match Score (domain annotations)")
        ax.legend()

        # Annotate scenario boundaries
        boundaries = scenario_boundaries(s["scenario"])
        for idx, name in boundaries:
            ax.axvline(x=turns[idx], color="gray", linestyle=":", alpha=0.4)

    ax.set_xlabel("Turn")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Plot 5: Signal Crossing Analysis ─────────────────────────────────


def plot_signal_crossings(s: dict, out: Path):
    fig, ax = plt.subplots(figsize=(10, 10))
    er = s["e_recon"]
    ep = s["e_predict"]
    domains = s["domain"]
    mask = valid_mask(er) & valid_mask(ep)

    unique_domains = sorted(set(domains))
    cmap = plt.cm.get_cmap("tab20", len(unique_domains))
    domain_colors = {d: cmap(i) for i, d in enumerate(unique_domains)}

    for i in range(len(er)):
        if mask[i]:
            ax.scatter(er[i], ep[i], color=domain_colors[domains[i]], s=20, alpha=0.7)

    # Diagonal
    valid_er = [er[i] for i in range(len(er)) if mask[i]]
    valid_ep = [ep[i] for i in range(len(ep)) if mask[i]]
    if valid_er and valid_ep:
        lo = min(min(valid_er), min(valid_ep))
        hi = max(max(valid_er), max(valid_ep))
        ax.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.5, label="E_recon = E_predict")

    ax.set_xlabel("E_recon")
    ax.set_ylabel("E_predict")
    ax.set_title("Signal Crossing: E_recon vs E_predict")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=domain_colors[d], markersize=6, label=d)
        for d in unique_domains
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=6, ncol=2)
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Plot 6: Per-Domain Statistics ─────────────────────────────────────


def plot_domain_stats(s: dict, out: Path):
    domains = s["domain"]
    unique_domains = sorted(set(domains))
    metrics = {
        "E_recon": s["e_recon"],
        "E_predict": s["e_predict"],
        "Cos Dist": s["cosine_distance"],
        "Dev Match": s["deviation_match"],
        "Temperature": s["temperature"],
    }

    domain_means = {m: [] for m in metrics}
    domain_stds = {m: [] for m in metrics}

    for d in unique_domains:
        idxs = [i for i, dom in enumerate(domains) if dom == d]
        for m, vals in metrics.items():
            dv = [vals[i] for i in idxs if vals[i] is not None]
            if dv:
                domain_means[m].append(np.mean(dv))
                domain_stds[m].append(np.std(dv))
            else:
                domain_means[m].append(0)
                domain_stds[m].append(0)

    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 3 * len(metrics)), sharex=True)
    x = np.arange(len(unique_domains))

    for ax, (m, _) in zip(axes, metrics.items()):
        ax.bar(x, domain_means[m], yerr=domain_stds[m], capsize=3, alpha=0.7, color="#3498db")
        ax.set_ylabel(m)
        ax.set_title(f"Mean {m} per Domain")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(unique_domains, rotation=45, ha="right", fontsize=8)
    axes[-1].set_xlabel("Domain")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Plot 7: Deviation Norm Stability ─────────────────────────────────


def plot_deviation_norm(s: dict, out: Path):
    """Plot deviation norm trajectory from server stats if available, else from deviation match."""
    fig, ax = plt.subplots(figsize=(14, 5))
    turns = s["turn"]

    # Use deviation_match as proxy for deviation norm trajectory
    dev = s["deviation_match"]
    dev_mask = valid_mask(dev)
    dev_turns = [turns[i] for i in range(len(turns)) if dev_mask[i]]
    dev_vals = [dev[i] for i in range(len(turns)) if dev_mask[i]]

    if dev_vals:
        ax.plot(dev_turns, dev_vals, "-o", color="#e74c3c", markersize=3, linewidth=1, alpha=0.8)
        mean_val = np.mean(dev_vals)
        std_val = np.std(dev_vals)
        cv = (std_val / mean_val * 100) if mean_val > 0 else 0
        ax.axhline(y=mean_val, color="#2c3e50", linestyle="--", alpha=0.6,
                    label=f"Mean={mean_val:.3f}, CV={cv:.1f}%")
        ax.fill_between(
            dev_turns,
            [mean_val - std_val] * len(dev_turns),
            [mean_val + std_val] * len(dev_turns),
            color="#2c3e50", alpha=0.1, label="±1σ"
        )
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No deviation data available", transform=ax.transAxes, ha="center")

    ax.set_xlabel("Turn")
    ax.set_ylabel("Deviation Match Score")
    ax.set_title("Deviation Norm Stability")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Summary Markdown ──────────────────────────────────────────────────


def compute_stats(values: list) -> dict:
    """Compute summary stats for a list, ignoring Nones."""
    valid = [v for v in values if v is not None]
    if not valid:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None, "n": 0}
    arr = np.array(valid)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": len(valid),
    }


def fmt(v, decimals=2):
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def generate_summary(data: dict, s: dict) -> str:
    meta = data["meta"]
    turns = data["turns"]
    n = len(turns)

    lines = []
    lines.append(f"# Long-Session Analysis: {meta.get('label', 'unknown')}")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append(f"- **Started**: {meta.get('started', 'N/A')}")
    lines.append(f"- **Finished**: {meta.get('finished', 'N/A')}")
    lines.append(f"- **Total turns**: {n}")
    lines.append(f"- **Errors**: {meta.get('errors', 0)}")
    lines.append(f"- **Server**: {meta.get('url', 'N/A')}")
    lines.append(f"- **Max tokens**: {meta.get('max_tokens', 'N/A')}")
    lines.append(f"- **Delay**: {meta.get('delay', 'N/A')}s")

    # Scenarios used
    scenarios_used = sorted(set(s["scenario"]))
    lines.append(f"- **Scenarios**: {', '.join(scenarios_used)}")
    lines.append("")

    # Global statistics
    lines.append("## Global Statistics")
    lines.append("")
    metrics = {
        "E_recon": s["e_recon"],
        "E_predict": s["e_predict"],
        "E_blend": s["e_blend"],
        "Temperature": s["temperature"],
        "Cosine Distance": s["cosine_distance"],
        "Deviation Match": s["deviation_match"],
    }
    lines.append("| Metric | Mean | Median | Std | Min | Max | N |")
    lines.append("|--------|------|--------|-----|-----|-----|---|")
    for name, vals in metrics.items():
        st = compute_stats(vals)
        lines.append(
            f"| {name} | {fmt(st['mean'])} | {fmt(st['median'])} | "
            f"{fmt(st['std'])} | {fmt(st['min'])} | {fmt(st['max'])} | {st['n']} |"
        )
    lines.append("")

    # IQR stability at checkpoints
    lines.append("## IQR Stability")
    lines.append("")
    er = safe_array(s["e_recon"])
    lines.append("| Checkpoint | Median | IQR | IQR/Median |")
    lines.append("|------------|--------|-----|------------|")
    for cp in [25, 50, 100, 200]:
        if cp <= n:
            history = er[:cp]
            q1, med, q3 = np.percentile(history, [25, 50, 75])
            iqr = q3 - q1
            ratio = iqr / med if med > 0 else 0
            lines.append(f"| Turn {cp} | {med:.1f} | {iqr:.1f} | {ratio:.3f} |")
    lines.append("")

    # Signal crossing rate
    lines.append("## Signal Crossings")
    lines.append("")
    crossings = 0
    crossing_by_scenario = {}
    for i in range(1, n):
        r0 = s["e_recon"][i - 1]
        r1 = s["e_recon"][i]
        p0 = s["e_predict"][i - 1]
        p1 = s["e_predict"][i]
        if all(v is not None for v in [r0, r1, p0, p1]):
            if (r0 - p0) * (r1 - p1) < 0:
                crossings += 1
                sc = s["scenario"][i]
                crossing_by_scenario[sc] = crossing_by_scenario.get(sc, 0) + 1
    valid_turns = sum(1 for i in range(1, n) if s["e_recon"][i] is not None and s["e_predict"][i] is not None)
    rate = crossings / valid_turns * 100 if valid_turns > 0 else 0
    lines.append(f"- **Total crossings**: {crossings} / {valid_turns} transitions ({rate:.1f}%)")
    lines.append("")
    if crossing_by_scenario:
        lines.append("| Scenario | Crossings |")
        lines.append("|----------|-----------|")
        for sc, cnt in sorted(crossing_by_scenario.items()):
            lines.append(f"| {sc} | {cnt} |")
        lines.append("")

    # Per-domain summary
    lines.append("## Per-Domain Summary")
    lines.append("")
    unique_domains = sorted(set(s["domain"]))
    lines.append("| Domain | N | E_recon | E_predict | Cos Dist | Dev Match | Temp |")
    lines.append("|--------|---|---------|-----------|----------|-----------|------|")
    for d in unique_domains:
        idxs = [i for i, dom in enumerate(s["domain"]) if dom == d]
        n_d = len(idxs)
        er_d = compute_stats([s["e_recon"][i] for i in idxs])
        ep_d = compute_stats([s["e_predict"][i] for i in idxs])
        cos_d = compute_stats([s["cosine_distance"][i] for i in idxs])
        dev_d = compute_stats([s["deviation_match"][i] for i in idxs])
        temp_d = compute_stats([s["temperature"][i] for i in idxs])
        lines.append(
            f"| {d} | {n_d} | {fmt(er_d['mean'])} | {fmt(ep_d['mean'])} | "
            f"{fmt(cos_d['mean'], 3)} | {fmt(dev_d['mean'], 3)} | {fmt(temp_d['mean'], 3)} |"
        )
    lines.append("")

    # Auto-detected observations
    lines.append("## Auto-Detected Observations")
    lines.append("")
    observations = []

    # Energy monotonicity check
    er_valid = [v for v in s["e_recon"] if v is not None]
    if len(er_valid) > 10:
        rises = sum(1 for i in range(1, len(er_valid)) if er_valid[i] > er_valid[i - 1])
        falls = sum(1 for i in range(1, len(er_valid)) if er_valid[i] < er_valid[i - 1])
        total = rises + falls
        if total > 0:
            rise_pct = rises / total * 100
            observations.append(
                f"E_recon rises on {rises}/{total} transitions ({rise_pct:.0f}%) — "
                f"{'NOT monotonically declining' if rise_pct > 20 else 'mostly declining'}"
            )

    # Temperature floor/ceiling hits
    temps_valid = [v for v in s["temperature"] if v is not None]
    if temps_valid:
        floor_hits = sum(1 for t in temps_valid if t <= 0.1)
        ceiling_hits = sum(1 for t in temps_valid if t >= 1.95)
        if floor_hits > 0:
            observations.append(f"Temperature hit floor (≤0.1) on {floor_hits} turns")
        if ceiling_hits > 0:
            observations.append(f"Temperature hit ceiling (≥1.95) on {ceiling_hits} turns")

    # Deviation norm drift
    dev_valid = [v for v in s["deviation_match"] if v is not None]
    if len(dev_valid) > 20:
        first_half = dev_valid[: len(dev_valid) // 2]
        second_half = dev_valid[len(dev_valid) // 2 :]
        drift = abs(np.mean(second_half) - np.mean(first_half))
        if drift > 0.05:
            observations.append(
                f"Deviation match drifted {drift:.3f} between first and second half "
                f"({np.mean(first_half):.3f} → {np.mean(second_half):.3f})"
            )
        else:
            observations.append(f"Deviation match stable (drift={drift:.3f})")

    # Elapsed time trend
    if s["elapsed"]:
        first_10 = np.mean(s["elapsed"][:10]) if len(s["elapsed"]) >= 10 else None
        last_10 = np.mean(s["elapsed"][-10:]) if len(s["elapsed"]) >= 10 else None
        if first_10 and last_10:
            slowdown = (last_10 - first_10) / first_10 * 100
            if abs(slowdown) > 20:
                observations.append(
                    f"Response time {'increased' if slowdown > 0 else 'decreased'} "
                    f"{abs(slowdown):.0f}% (first 10: {first_10:.1f}s, last 10: {last_10:.1f}s)"
                )

    for obs in observations:
        lines.append(f"- {obs}")
    if not observations:
        lines.append("- No notable anomalies detected")
    lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Analyse PCDC long-session results")
    parser.add_argument("--input", required=True, help="Session JSON file")
    parser.add_argument("--output-dir", default=None, help="Directory for plots")
    parser.add_argument("--summary", default=None, help="Summary markdown output path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Default output paths
    stem = input_path.stem
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = input_path.parent / f"{stem}_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = Path(args.summary) if args.summary else out_dir / f"{stem}_summary.md"

    print(f"Loading {input_path}...")
    data = load_session(input_path)
    turns = data.get("turns", [])
    if not turns:
        print("No turns found in session data.", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(turns)} turns loaded")
    s = extract_series(turns)

    print("Generating plots...")
    plot_energy_trajectory(s, out_dir / "energy_trajectory.png")
    plot_temperature_dist(s, out_dir / "temperature_dist.png")
    plot_iqr_evolution(s, out_dir / "iqr_evolution.png")
    plot_deviation_match(s, out_dir / "deviation_match.png")
    plot_signal_crossings(s, out_dir / "signal_crossings.png")
    plot_domain_stats(s, out_dir / "domain_stats.png")
    plot_deviation_norm(s, out_dir / "deviation_norm.png")

    print("Generating summary...")
    summary = generate_summary(data, s)
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"  Saved {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
