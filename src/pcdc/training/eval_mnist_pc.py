"""Evaluation and visualization for PC MNIST training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pcdc.core.clamp import ClampMode, make_clamp_mask
from pcdc.core.energy import compute_energy
from pcdc.core.pc_network import PCNetwork
from pcdc.utils.config import PCConfig


def plot_energy_curve(energy_traces: list[list[float]], save_path: str | None = None):
    """Plot energy vs settling iteration for multiple batches."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, trace in enumerate(energy_traces[:10]):  # plot first 10 batches
        ax.plot(trace, alpha=0.5, label=f"batch {i}" if i < 5 else None)
    ax.set_xlabel("Settling Iteration")
    ax.set_ylabel("Energy")
    ax.set_title("Energy Convergence During Settling")
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved energy plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_layer_errors(layer_mse_history: list[list[float]], save_path: str | None = None):
    """Plot per-layer MSE over settling steps."""
    if not layer_mse_history:
        return
    n_layers = len(layer_mse_history[0])
    fig, ax = plt.subplots(figsize=(8, 5))
    for l in range(n_layers):
        values = [step[l] for step in layer_mse_history]
        ax.plot(values, label=f"Layer {l}")
    ax.set_xlabel("Settling Iteration")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("Per-Layer Error During Settling")
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def detailed_settle_analysis(net: PCNetwork, loader: DataLoader, device: str, n_batches: int = 5):
    """Run settling with full metric recording on a few batches."""
    import torch.nn.functional as F

    from pcdc.training.train_mnist_pc import to_onehot

    energy_traces = []
    all_layer_mse = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if i >= n_batches:
                break
            x_input = images.view(images.size(0), -1).to(device)
            y_target = to_onehot(labels).to(device)

            clamp_mask = make_clamp_mask(net.n_nodes, ClampMode.SUPERVISED)
            x = net.init_states(x_input.shape[0], x0=x_input, xL=y_target)
            metrics = net.settle(x, clamp_mask, record_metrics=True)

            energy_traces.append(metrics.energy_trace)
            if metrics.layer_mse:
                all_layer_mse.extend(metrics.layer_mse)

    return energy_traces, all_layer_mse


def main():
    parser = argparse.ArgumentParser(description="Evaluate and visualize PC MNIST")
    parser.add_argument("--checkpoint", type=str, help="Path to saved model state dict")
    parser.add_argument("--output-dir", type=str, default="./experiments")
    parser.add_argument("--layers", type=int, nargs="+", default=[784, 256, 256, 10])
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = PCConfig(layer_sizes=args.layers, activation=args.activation, device=args.device)
    net = PCNetwork(config).to(args.device)

    if args.checkpoint:
        net.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True))
        print(f"Loaded checkpoint from {args.checkpoint}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # Detailed settling analysis
    print("Running detailed settling analysis...")
    energy_traces, layer_mse = detailed_settle_analysis(net, test_loader, args.device)

    plot_energy_curve(energy_traces, save_path=str(out / "energy_convergence.png"))
    if layer_mse:
        plot_layer_errors(layer_mse, save_path=str(out / "layer_errors.png"))

    # Full evaluation
    from pcdc.training.train_mnist_pc import evaluate

    acc = evaluate(net, test_loader, args.device)
    print(f"Test accuracy: {acc:.4f}")

    results = {"accuracy": acc, "energy_traces": energy_traces}
    with open(out / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
