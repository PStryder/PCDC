#!/usr/bin/env python3
"""Plot E_recon and E_predict over time from PCDC server stats.

Usage:
    python scripts/plot_session.py [--url http://localhost:8001] [--output session_plot.png]

Fetches /v1/pcdc/stats and plots energy histories + cosine distances.
"""

import argparse
import json
import sys
import urllib.request


def fetch_stats(base_url: str) -> dict:
    url = f"{base_url}/v1/pcdc/stats"
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read())


def plot(stats: dict, output: str | None = None):
    import matplotlib.pyplot as plt

    recon = stats.get("recon_energy_history", [])
    predict = stats.get("predict_energy_history", [])
    cosine = stats.get("cosine_distance_history", [])

    if not recon and not predict:
        print("No energy history data to plot.")
        sys.exit(1)

    turns = list(range(1, len(recon) + 1))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- Energy plot ---
    ax1 = axes[0]
    if recon:
        ax1.plot(turns, recon, "o-", color="#e74c3c", label="E_recon (content novelty)", linewidth=2)
    if predict:
        ax1.plot(turns[:len(predict)], predict, "s-", color="#3498db", label="E_predict (transition surprise)", linewidth=2)

    # Mark crossover points
    for i in range(len(recon)):
        if i < len(predict) and i > 0:
            prev_diff = recon[i - 1] - predict[i - 1]
            curr_diff = recon[i] - predict[i]
            if prev_diff * curr_diff < 0:  # sign change = crossover
                ax1.axvline(x=turns[i], color="#e67e22", linestyle="--", alpha=0.7, label="Signal crossing")

    ax1.set_ylabel("Energy", fontsize=12)
    ax1.set_title("Two-Phase Settling Energy Over Time", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- Cosine distance plot ---
    ax2 = axes[1]
    if cosine:
        cos_turns = list(range(2, 2 + len(cosine)))
        ax2.bar(cos_turns, cosine, color="#2ecc71", alpha=0.8, label="Cosine distance")
        ax2.set_ylabel("Cosine Distance", fontsize=12)
        ax2.set_xlabel("Turn", fontsize=12)
        ax2.set_title("Embedding Cosine Distance Between Consecutive Turns", fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        ax2.text(0.5, 0.5, "No cosine distance data yet", transform=ax2.transAxes,
                 ha="center", va="center", fontsize=12, color="gray")
        ax2.set_xlabel("Turn", fontsize=12)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot PCDC session energy dynamics")
    parser.add_argument("--url", default="http://localhost:8001", help="PCDC server URL")
    parser.add_argument("--output", "-o", default=None, help="Output file (PNG). If omitted, shows interactively.")
    parser.add_argument("--json", default=None, help="Load from JSON file instead of server")
    args = parser.parse_args()

    if args.json:
        import json as json_mod
        with open(args.json) as f:
            stats = json_mod.load(f)
    else:
        stats = fetch_stats(args.url)

    plot(stats, args.output)


if __name__ == "__main__":
    main()
