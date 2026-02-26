"""Continual learning experiment runner.

Orchestrates sequential task training with PCHead and baselines,
measuring forgetting and accuracy across tasks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from pcdc.core.pc_network import PCNetwork
from pcdc.gguf.baselines import (
    LinearProbe,
    ReplayBuffer,
    SGDBaseline,
    eval_baseline_on_features,
    train_baseline_on_features,
)
from pcdc.gguf.datasets import SplitDatasetSequence
from pcdc.gguf.pc_head import PCHead
from pcdc.utils.config import ContinualConfig


def train_pchead_on_features(
    head: PCHead,
    features: Tensor,
    labels: Tensor,
    num_classes: int,
    epochs: int = 10,
    batch_size: int = 64,
    device: str = "cuda",
    replay_buffer: ReplayBuffer | None = None,
    replay_mix: float = 0.25,
    replay_mode: str = "joint",
) -> dict:
    """Train PCHead on pre-extracted features using local learning.

    Args:
        replay_mode: "joint" — concatenate replay with current batch before
            settling (recommended, acts as implicit regularization).
            "separate" — settle and update on current and replay batches
            independently.

    Returns:
        Dict with energies, and per-source accuracy when replay is used.
    """
    head = head.to(device)
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    energies = []
    current_correct_total = [0, 0]
    replay_correct_total = [0, 0]

    for epoch in range(epochs):
        epoch_energy = 0.0
        n = 0
        for batch_feats, batch_labels in loader:
            batch_feats = batch_feats.to(device)
            batch_labels = batch_labels.to(device)
            current_size = batch_feats.shape[0]

            replay_feats = None
            replay_labels_tensor = None

            # Get replay data
            if replay_buffer is not None and len(replay_buffer) > 0:
                replay_n = int(batch_size * replay_mix)
                replay_data = replay_buffer.sample(replay_n)
                if replay_data is not None:
                    replay_feats, replay_labels_tensor = replay_data
                    replay_feats = replay_feats.to(device)
                    replay_labels_tensor = replay_labels_tensor.to(device)

            if replay_mode == "joint" and replay_feats is not None:
                # §4: Joint settling — concatenate into single batch
                joint_feats = torch.cat([batch_feats, replay_feats])
                joint_labels = torch.cat([batch_labels, replay_labels_tensor])
                y_target = F.one_hot(joint_labels, num_classes).float()
                result = head.train_step(joint_feats, y_target)
                epoch_energy += result["energy"]

                # Track accuracy separately for current vs replay portions
                with torch.no_grad():
                    pred, _ = head.infer(joint_feats)
                    pred_labels = pred.argmax(dim=-1)
                    # Current portion
                    current_correct_total[0] += (pred_labels[:current_size] == batch_labels).sum().item()
                    current_correct_total[1] += current_size
                    # Replay portion
                    replay_size = replay_feats.shape[0]
                    replay_correct_total[0] += (pred_labels[current_size:] == replay_labels_tensor).sum().item()
                    replay_correct_total[1] += replay_size

            elif replay_mode == "separate" and replay_feats is not None:
                # Settle current batch
                y_current = F.one_hot(batch_labels, num_classes).float()
                result = head.train_step(batch_feats, y_current)
                epoch_energy += result["energy"]

                # Settle replay batch separately
                y_replay = F.one_hot(replay_labels_tensor, num_classes).float()
                head.train_step(replay_feats, y_replay)

            else:
                # No replay available
                y_target = F.one_hot(batch_labels, num_classes).float()
                result = head.train_step(batch_feats, y_target)
                epoch_energy += result["energy"]

            n += 1

        energies.append(epoch_energy / n if n > 0 else 0.0)

    result_dict = {"energies": energies}
    if current_correct_total[1] > 0:
        result_dict["current_acc"] = current_correct_total[0] / current_correct_total[1]
    if replay_correct_total[1] > 0:
        result_dict["replay_acc"] = replay_correct_total[0] / replay_correct_total[1]
    return result_dict


@torch.no_grad()
def eval_pchead_on_features(
    head: PCHead,
    features: Tensor,
    labels: Tensor,
    device: str = "cuda",
    batch_size: int = 256,
) -> float:
    """Evaluate PCHead accuracy on features."""
    head = head.to(device)
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    for feats, labs in loader:
        feats = feats.to(device)
        labs = labs.to(device)
        pred, _ = head.infer(feats)
        correct += (pred.argmax(dim=-1) == labs).sum().item()
        total += labs.size(0)

    return correct / total if total > 0 else 0.0


def run_continual_experiment(
    task_sequence: SplitDatasetSequence,
    feature_dim: int,
    num_classes: int,
    config: ContinualConfig,
    device: str = "cuda",
) -> dict:
    """Run full continual learning experiment comparing PCHead vs baselines.

    Returns:
        Dict with accuracy matrices and forgetting metrics for each model.
    """
    n_tasks = task_sequence.n_tasks

    # Initialize models
    pc_head = PCHead(
        feature_dim=feature_dim,
        num_classes=num_classes,
        hidden_sizes=config.hidden_sizes,
        eta_x=0.05,
        eta_w=0.001,
        K=15,
        activation="tanh",
    )
    linear = LinearProbe(feature_dim, num_classes)
    sgd_mlp = SGDBaseline(feature_dim, num_classes, hidden_sizes=config.hidden_sizes)

    # Replay buffers (one per model)
    pc_replay = ReplayBuffer(config.replay_size)
    linear_replay = ReplayBuffer(config.replay_size)
    sgd_replay = ReplayBuffer(config.replay_size)

    # Accuracy matrices: acc[model][after_task][eval_task]
    results = {
        "pc_head": {"acc_matrix": [], "energies": [], "current_accs": [], "replay_accs": []},
        "linear": {"acc_matrix": [], "losses": []},
        "sgd_mlp": {"acc_matrix": [], "losses": []},
    }

    prev_linear_weights = None
    prev_sgd_weights = None

    for task_id in range(n_tasks):
        task = task_sequence.get_task(task_id)
        print(f"\n{'='*50}")
        print(f"Task {task_id}: classes {task.classes}")
        print(f"  Train: {len(task.train_labels)}, Test: {len(task.test_labels)}")

        # --- Train PCHead ---
        print("  Training PCHead...")
        pc_result = train_pchead_on_features(
            pc_head, task.train_features, task.train_labels,
            num_classes=num_classes, device=device,
            replay_buffer=pc_replay, replay_mix=config.replay_mix,
            replay_mode=config.replay_mode,
        )
        results["pc_head"]["energies"].append(pc_result["energies"])
        if "current_acc" in pc_result:
            results["pc_head"]["current_accs"].append(pc_result["current_acc"])
        if "replay_acc" in pc_result:
            results["pc_head"]["replay_accs"].append(pc_result["replay_acc"])
        pc_replay.add(task.train_features, task.train_labels)

        # --- Train Linear Probe ---
        print("  Training Linear Probe...")
        lin_losses = train_baseline_on_features(
            linear, task.train_features, task.train_labels,
            device=device, replay_buffer=linear_replay, replay_mix=config.replay_mix,
            prev_weights=prev_linear_weights, l2_prev=config.l2_prev_weight,
        )
        results["linear"]["losses"].append(lin_losses)
        linear_replay.add(task.train_features, task.train_labels)
        prev_linear_weights = {n: p.clone().cpu() for n, p in linear.named_parameters()}

        # --- Train SGD MLP ---
        print("  Training SGD MLP...")
        sgd_losses = train_baseline_on_features(
            sgd_mlp, task.train_features, task.train_labels,
            device=device, replay_buffer=sgd_replay, replay_mix=config.replay_mix,
            prev_weights=prev_sgd_weights, l2_prev=config.l2_prev_weight,
        )
        results["sgd_mlp"]["losses"].append(sgd_losses)
        sgd_replay.add(task.train_features, task.train_labels)
        prev_sgd_weights = {n: p.clone().cpu() for n, p in sgd_mlp.named_parameters()}

        # --- Evaluate all models on all tasks seen so far ---
        pc_accs = []
        lin_accs = []
        sgd_accs = []

        for eval_task_id in range(task_id + 1):
            eval_task = task_sequence.get_task(eval_task_id)

            pc_acc = eval_pchead_on_features(
                pc_head, eval_task.test_features, eval_task.test_labels, device
            )
            lin_acc = eval_baseline_on_features(
                linear, eval_task.test_features, eval_task.test_labels, device
            )
            sgd_acc = eval_baseline_on_features(
                sgd_mlp, eval_task.test_features, eval_task.test_labels, device
            )

            pc_accs.append(pc_acc)
            lin_accs.append(lin_acc)
            sgd_accs.append(sgd_acc)

        results["pc_head"]["acc_matrix"].append(pc_accs)
        results["linear"]["acc_matrix"].append(lin_accs)
        results["sgd_mlp"]["acc_matrix"].append(sgd_accs)

        # Print current state
        print(f"  After task {task_id}:")
        for name in ["pc_head", "linear", "sgd_mlp"]:
            accs = results[name]["acc_matrix"][-1]
            avg = sum(accs) / len(accs)
            print(f"    {name:12s}: avg={avg:.4f} | {['%.3f' % a for a in accs]}")

    # Compute forgetting
    for name in results:
        matrix = results[name]["acc_matrix"]
        forgetting = compute_forgetting(matrix)
        results[name]["forgetting"] = forgetting

    return results


def compute_forgetting(acc_matrix: list[list[float]]) -> list[float]:
    """Compute backward transfer (forgetting) per task.

    Forgetting for task t = max accuracy on t across all evaluations - final accuracy on t.
    """
    if not acc_matrix:
        return []

    n_tasks = len(acc_matrix)
    forgetting = []

    for t in range(n_tasks - 1):  # last task has no forgetting
        best = max(acc_matrix[after_t][t] for after_t in range(t, n_tasks))
        final = acc_matrix[-1][t]
        forgetting.append(best - final)

    return forgetting


def print_summary(results: dict):
    """Print formatted summary of continual learning results."""
    print("\n" + "=" * 60)
    print("CONTINUAL LEARNING SUMMARY")
    print("=" * 60)

    for name in ["pc_head", "linear", "sgd_mlp"]:
        data = results[name]
        matrix = data["acc_matrix"]
        forgetting = data.get("forgetting", [])

        final_accs = matrix[-1] if matrix else []
        avg_final = sum(final_accs) / len(final_accs) if final_accs else 0
        avg_forget = sum(forgetting) / len(forgetting) if forgetting else 0

        print(f"\n{name}:")
        print(f"  Final avg accuracy:  {avg_final:.4f}")
        print(f"  Avg forgetting:      {avg_forget:.4f}")
        if forgetting:
            print(f"  Per-task forgetting:  {['%.3f' % f for f in forgetting]}")

    # Print replay-specific metrics for PCHead
    pc_data = results.get("pc_head", {})
    if pc_data.get("current_accs"):
        print(f"\n  PCHead current-task train acc: {pc_data['current_accs']}")
    if pc_data.get("replay_accs"):
        print(f"  PCHead replay train acc:       {pc_data['replay_accs']}")


def main():
    """Run continual learning on MNIST features (for testing without GGUF)."""
    parser = argparse.ArgumentParser(description="Continual learning experiment")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-tasks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./experiments")
    parser.add_argument("--use-gguf", action="store_true", help="Use GGUF features (requires model)")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--replay-mode", type=str, default="joint", choices=["joint", "separate"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = ContinualConfig(n_tasks=args.n_tasks, replay_mode=args.replay_mode)

    if args.use_gguf:
        raise NotImplementedError("GGUF continual learning requires feature precomputation. See README.")

    # Default: use raw MNIST pixels as "features" for testing the pipeline
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_features = train_ds.data.float().view(-1, 784) / 255.0
    train_labels = train_ds.targets
    test_features = test_ds.data.float().view(-1, 784) / 255.0
    test_labels = test_ds.targets

    task_seq = SplitDatasetSequence(
        train_features, train_labels, test_features, test_labels,
        n_tasks=args.n_tasks, classes_per_task=2, seed=args.seed,
    )

    results = run_continual_experiment(
        task_seq, feature_dim=784, num_classes=10,
        config=config, device=args.device,
    )

    print_summary(results)

    with open(out / "continual_results.json", "w") as f:
        json.dump({
            k: {kk: vv for kk, vv in v.items() if kk not in ("energies", "losses")}
            for k, v in results.items()
        }, f, indent=2)
    print(f"\nResults saved to {out / 'continual_results.json'}")


if __name__ == "__main__":
    main()
