"""CLI entry point: train PCNetwork on MNIST and compare to backprop baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from pcdc.core.pc_network import PCNetwork
from pcdc.training.baseline_mlp_backprop import BaselineMLP, eval_baseline, train_baseline_epoch
from pcdc.training.train_mnist_pc import evaluate, get_mnist_loaders, train_epoch
from pcdc.utils.config import PCConfig


def main():
    parser = argparse.ArgumentParser(description="PCDC MNIST Benchmark")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eta-x", type=float, default=0.1)
    parser.add_argument("--eta-w", type=float, default=0.001)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--layers", type=int, nargs="+", default=[784, 256, 256, 10])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--save-dir", type=str, default="./experiments")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_mnist_loaders(args.batch_size)

    # --- PC Network ---
    print("=" * 60)
    print("PREDICTIVE CODING NETWORK")
    print("=" * 60)

    config = PCConfig(
        layer_sizes=args.layers,
        activation=args.activation,
        eta_x=args.eta_x,
        eta_w=args.eta_w,
        K=args.K,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
    )
    pc_net = PCNetwork(config).to(args.device)

    print(f"Architecture: {args.layers}")
    print(f"Activation: {args.activation}, K={args.K}")
    print(f"eta_x={args.eta_x}, eta_w={args.eta_w}")
    print()

    pc_val_accs = []
    for epoch in range(1, args.epochs + 1):
        epoch_metrics = train_epoch(pc_net, train_loader, args.device, epoch)
        val_acc = evaluate(pc_net, test_loader, args.device)
        epoch_metrics.val_accuracy = val_acc
        pc_val_accs.append(val_acc)
        print(epoch_metrics.summary())

    torch.save(pc_net.state_dict(), save_dir / "pc_network.pt")
    print(f"\nPC best val accuracy: {max(pc_val_accs):.4f}")

    # --- Backprop Baseline ---
    if not args.skip_baseline:
        print()
        print("=" * 60)
        print("BACKPROP MLP BASELINE")
        print("=" * 60)

        torch.manual_seed(args.seed)
        if args.device == "cuda":
            torch.cuda.manual_seed(args.seed)

        mlp = BaselineMLP(args.layers, activation=args.activation).to(args.device)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=args.eta_w)

        bp_val_accs = []
        for epoch in range(1, args.epochs + 1):
            train_res = train_baseline_epoch(mlp, train_loader, optimizer, args.device)
            val_res = eval_baseline(mlp, test_loader, args.device)
            bp_val_accs.append(val_res["accuracy"])
            print(
                f"Epoch {epoch:3d} | "
                f"loss={train_res['loss']:.4f} | "
                f"train_acc={train_res['accuracy']:.4f} | "
                f"val_acc={val_res['accuracy']:.4f}"
            )

        print(f"\nBaseline best val accuracy: {max(bp_val_accs):.4f}")

    # --- Summary ---
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"PC Network:      {max(pc_val_accs):.4f}")
    if not args.skip_baseline:
        print(f"Backprop MLP:    {max(bp_val_accs):.4f}")


if __name__ == "__main__":
    main()
