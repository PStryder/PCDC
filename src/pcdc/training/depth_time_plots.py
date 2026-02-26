"""Visualization for depth-vs-time experiment results."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _aggregate(results: list[dict]) -> dict:
    """Group by (hidden_sizes_str, K), compute mean/std across seeds."""
    grouped: dict[tuple, list] = defaultdict(list)
    for r in results:
        key = (str(r["hidden_sizes"]), r["K"])
        grouped[key].append(r)

    agg = {}
    for (hs_str, K), cells in grouped.items():
        agg[(hs_str, K)] = {
            "acc_mean": np.mean([c["avg_accuracy"] for c in cells]),
            "acc_std": np.std([c["avg_accuracy"] for c in cells]),
            "forget_mean": np.mean([c["avg_forgetting"] for c in cells]),
            "forget_std": np.std([c["avg_forgetting"] for c in cells]),
            "time_mean": np.mean([c["avg_settle_wall_s"] for c in cells]) * 1000,
            "time_std": np.std([c["avg_settle_wall_s"] for c in cells]) * 1000,
            "steps_mean": np.mean([c["avg_steps_used"] for c in cells]),
            "n_params": cells[0]["n_params"],
        }
    return agg


def plot_heatmaps(results_path: str | Path, save_dir: str | Path | None = None):
    """Generate 2x2 heatmap grid: accuracy, forgetting, settle time, steps."""
    results = load_results(results_path)
    agg = _aggregate(results)

    depth_labels = sorted(set(hs for (hs, _) in agg.keys()))
    k_values = sorted(set(K for (_, K) in agg.keys()))

    metrics = [
        ("acc_mean", "Average Accuracy", "RdYlGn"),
        ("forget_mean", "Average Forgetting", "RdYlGn_r"),
        ("time_mean", "Settle Time (ms)", "YlOrRd"),
        ("steps_mean", "Steps Used", "YlOrRd"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Depth vs Settling Steps (K)", fontsize=14)

    for ax, (metric_key, title, cmap) in zip(axes.flat, metrics):
        matrix = np.zeros((len(depth_labels), len(k_values)))
        for i, hs in enumerate(depth_labels):
            for j, K in enumerate(k_values):
                if (hs, K) in agg:
                    matrix[i, j] = agg[(hs, K)][metric_key]

        im = ax.imshow(matrix, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(k_values)))
        ax.set_xticklabels(k_values)
        ax.set_yticks(range(len(depth_labels)))
        ax.set_yticklabels(depth_labels, fontsize=8)
        ax.set_xlabel("K (settling steps)")
        ax.set_ylabel("Hidden Sizes")
        ax.set_title(title)

        for i in range(len(depth_labels)):
            for j in range(len(k_values)):
                val = matrix[i, j]
                fmt = f"{val:.3f}" if "acc" in metric_key or "forget" in metric_key else f"{val:.1f}"
                ax.text(j, i, fmt, ha="center", va="center", fontsize=7)

        plt.colorbar(im, ax=ax)

    plt.tight_layout()

    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "depth_vs_time_heatmaps.png", dpi=150)
        print(f"Saved heatmaps to {out / 'depth_vs_time_heatmaps.png'}")

    plt.close(fig)


def plot_iso_accuracy(results_path: str | Path, save_dir: str | Path | None = None):
    """Accuracy vs K, one line per depth config.

    The key plot: if shallow+high-K lines converge toward deep+low-K,
    that's the depth-time tradeoff result.
    """
    results = load_results(results_path)
    agg = _aggregate(results)

    depth_labels = sorted(set(hs for (hs, _) in agg.keys()))
    k_values = sorted(set(K for (_, K) in agg.keys()))

    fig, ax = plt.subplots(figsize=(10, 6))
    for hs in depth_labels:
        accs = [agg.get((hs, K), {}).get("acc_mean", 0) for K in k_values]
        stds = [agg.get((hs, K), {}).get("acc_std", 0) for K in k_values]
        ax.errorbar(k_values, accs, yerr=stds, marker="o", label=hs, capsize=3)

    ax.set_xlabel("K (settling steps)")
    ax.set_ylabel("Average Accuracy")
    ax.set_title("Accuracy vs Settling Steps by Depth")
    ax.legend(title="Hidden Sizes", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "iso_accuracy_curves.png", dpi=150)
        print(f"Saved iso-accuracy to {out / 'iso_accuracy_curves.png'}")

    plt.close(fig)
