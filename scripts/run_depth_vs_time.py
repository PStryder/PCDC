"""CLI entry point for depth-vs-time experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision import datasets, transforms

from pcdc.gguf.datasets import SplitDatasetSequence
from pcdc.training.depth_time_experiment import (
    print_summary_table,
    run_depth_time_grid,
    save_results,
)
from pcdc.utils.config import DepthTimeConfig


def main():
    parser = argparse.ArgumentParser(description="Depth vs Time experiment")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output-dir", type=str, default="./experiments/depth_vs_time")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plots")
    parser.add_argument(
        "--plot-only",
        type=str,
        default="",
        help="Path to existing results JSON; skip training, just plot",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20, 40, 80])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n-tasks", type=int, default=5)
    args = parser.parse_args()

    if args.plot_only:
        from pcdc.training.depth_time_plots import plot_heatmaps, plot_iso_accuracy

        plot_heatmaps(args.plot_only, args.output_dir)
        plot_iso_accuracy(args.plot_only, args.output_dir)
        return

    config = DepthTimeConfig(
        k_values=args.k_values,
        seeds=args.seeds,
        epochs_per_task=args.epochs,
        n_tasks=args.n_tasks,
        output_dir=args.output_dir,
    )

    # Load MNIST as flat features
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_features = train_ds.data.float().view(-1, 784) / 255.0
    train_labels = train_ds.targets
    test_features = test_ds.data.float().view(-1, 784) / 255.0
    test_labels = test_ds.targets

    task_seq = SplitDatasetSequence(
        train_features,
        train_labels,
        test_features,
        test_labels,
        n_tasks=config.n_tasks,
        classes_per_task=config.classes_per_task,
    )

    print(f"Depth vs Time Experiment")
    print(f"  Depths: {config.depth_configs}")
    print(f"  K values: {config.k_values}")
    print(f"  Seeds: {config.seeds}")
    print(f"  Tasks: {config.n_tasks}, Epochs/task: {config.epochs_per_task}")
    print(f"  Device: {args.device}")
    total = len(config.depth_configs) * len(config.k_values) * len(config.seeds)
    print(f"  Total cells: {total}")
    print()

    results = run_depth_time_grid(
        task_seq,
        feature_dim=784,
        num_classes=10,
        config=config,
        device=args.device,
    )

    path = save_results(results, config.output_dir)
    print(f"\nResults saved to {path}")
    print_summary_table(results)

    if not args.no_plot:
        try:
            from pcdc.training.depth_time_plots import plot_heatmaps, plot_iso_accuracy

            plot_heatmaps(str(path), config.output_dir)
            plot_iso_accuracy(str(path), config.output_dir)
        except ImportError:
            print("\nmatplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
