"""Depth vs Time experiment: sweep (hidden_sizes, K) grid for PCHead.

Tests whether increasing settling steps K can compensate for reducing
network depth — evidence that iterative relaxation substitutes for
structural depth.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from tqdm import tqdm

from pcdc.gguf.baselines import ReplayBuffer
from pcdc.gguf.datasets import SplitDatasetSequence
from pcdc.gguf.pc_head import PCHead
from pcdc.training.continual_runner import (
    compute_forgetting,
    eval_pchead_on_features,
    train_pchead_on_features,
)
from pcdc.utils.config import DepthTimeConfig


@dataclass
class CellResult:
    """Results for one (depth, K, seed) grid cell."""

    hidden_sizes: list[int]
    K: int
    seed: int
    n_params: int
    avg_accuracy: float
    avg_forgetting: float
    avg_settle_wall_s: float
    avg_steps_used: float
    acc_matrix: list[list[float]]
    forgetting: list[float]
    per_task_accs: list[float]


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def run_single_cell(
    hidden_sizes: list[int],
    K: int,
    task_sequence: SplitDatasetSequence,
    feature_dim: int,
    num_classes: int,
    config: DepthTimeConfig,
    device: str,
) -> CellResult:
    """Train and evaluate one (depth, K) configuration."""
    pc_head = PCHead(
        feature_dim=feature_dim,
        num_classes=num_classes,
        hidden_sizes=hidden_sizes if hidden_sizes else None,
        eta_x=config.eta_x,
        eta_w=config.eta_w,
        K=K,
        activation=config.activation,
    )
    # PCHead defaults None -> [256]. For truly zero hidden layers,
    # we need to construct directly.
    if not hidden_sizes:
        from pcdc.core.pc_network import PCNetwork
        from pcdc.utils.config import PCConfig

        pc_config = PCConfig(
            layer_sizes=[feature_dim, num_classes],
            eta_x=config.eta_x,
            eta_w=config.eta_w,
            K=K,
            activation=config.activation,
        )
        pc_head = PCNetwork(pc_config)

    n_params = _count_parameters(pc_head)
    replay = ReplayBuffer(config.replay_size)
    acc_matrix: list[list[float]] = []

    for task_id in range(task_sequence.n_tasks):
        task = task_sequence.get_task(task_id)

        train_pchead_on_features(
            pc_head,
            task.train_features,
            task.train_labels,
            num_classes=num_classes,
            device=device,
            epochs=config.epochs_per_task,
            batch_size=config.batch_size,
            replay_buffer=replay,
            replay_mix=config.replay_mix,
            replay_mode=config.replay_mode,
        )
        replay.add(task.train_features, task.train_labels)

        task_accs = []
        for eval_id in range(task_id + 1):
            eval_task = task_sequence.get_task(eval_id)
            acc = eval_pchead_on_features(
                pc_head,
                eval_task.test_features,
                eval_task.test_labels,
                device,
                batch_size=config.batch_size,
            )
            task_accs.append(acc)
        acc_matrix.append(task_accs)

    # Dedicated timing pass: measure pure settle dynamics on inference
    pc_head = pc_head.to(device)
    timing_task = task_sequence.get_task(0)
    sample_feats = timing_task.test_features[: config.batch_size].to(device)

    settle_times: list[float] = []
    settle_steps: list[int] = []
    with torch.no_grad():
        for _ in range(10):
            _, metrics = pc_head.infer(sample_feats, K=K)
            settle_times.append(metrics.wall_clock_s)
            settle_steps.append(metrics.steps_used)

    forgetting = compute_forgetting(acc_matrix)
    final_accs = acc_matrix[-1]
    avg_acc = sum(final_accs) / len(final_accs)
    avg_forget = sum(forgetting) / len(forgetting) if forgetting else 0.0

    return CellResult(
        hidden_sizes=hidden_sizes,
        K=K,
        seed=0,
        n_params=n_params,
        avg_accuracy=avg_acc,
        avg_forgetting=avg_forget,
        avg_settle_wall_s=sum(settle_times) / len(settle_times),
        avg_steps_used=sum(settle_steps) / len(settle_steps),
        acc_matrix=acc_matrix,
        forgetting=forgetting,
        per_task_accs=final_accs,
    )


def run_depth_time_grid(
    task_sequence: SplitDatasetSequence,
    feature_dim: int,
    num_classes: int,
    config: DepthTimeConfig,
    device: str = "cuda",
) -> list[CellResult]:
    """Run the full (depth, K) grid experiment."""
    results: list[CellResult] = []
    total = len(config.depth_configs) * len(config.k_values) * len(config.seeds)

    with tqdm(total=total, desc="Grid sweep") as pbar:
        for seed in config.seeds:
            torch.manual_seed(seed)
            for hidden_sizes in config.depth_configs:
                for K in config.k_values:
                    depth_str = str(hidden_sizes) if hidden_sizes else "[]"
                    pbar.set_postfix_str(f"h={depth_str} K={K} s={seed}")

                    cell = run_single_cell(
                        hidden_sizes,
                        K,
                        task_sequence,
                        feature_dim,
                        num_classes,
                        config,
                        device,
                    )
                    cell.seed = seed
                    results.append(cell)
                    pbar.update(1)

    return results


def save_results(results: list[CellResult], output_dir: str | Path) -> Path:
    """Save results to JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "depth_time_results.json"
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    return path


def print_summary_table(results: list[CellResult]) -> None:
    """Print console summary table, averaged across seeds."""
    grouped: dict[tuple, list[CellResult]] = defaultdict(list)
    for r in results:
        key = (tuple(r.hidden_sizes), r.K)
        grouped[key].append(r)

    print(
        f"\n{'Hidden Sizes':<20} {'K':>5} {'Params':>8} {'Acc':>7} "
        f"{'Forget':>7} {'Settle(ms)':>10} {'Steps':>6}"
    )
    print("-" * 75)

    for (hs, K), cells in sorted(grouped.items()):
        avg_acc = sum(c.avg_accuracy for c in cells) / len(cells)
        avg_fgt = sum(c.avg_forgetting for c in cells) / len(cells)
        avg_t = sum(c.avg_settle_wall_s for c in cells) / len(cells)
        avg_s = sum(c.avg_steps_used for c in cells) / len(cells)
        n_params = cells[0].n_params
        hs_str = str(list(hs)) if hs else "[]"

        print(
            f"{hs_str:<20} {K:>5} {n_params:>8} {avg_acc:>7.4f} "
            f"{avg_fgt:>7.4f} {avg_t * 1000:>10.2f} {avg_s:>6.1f}"
        )


def main_cli():
    """CLI entry point for depth-vs-time experiment."""
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

    from torchvision import datasets, transforms

    from pcdc.gguf.datasets import SplitDatasetSequence

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

    print("Depth vs Time Experiment")
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
        except Exception as e:
            print(f"\nSkipping plots: {e}")
