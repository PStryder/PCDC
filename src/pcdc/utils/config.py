"""Configuration dataclasses for PCDC."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PCConfig:
    """All hyperparameters for a predictive coding network."""

    # Architecture
    layer_sizes: list[int] = field(default_factory=lambda: [784, 256, 256, 10])
    activation: str = "tanh"  # tanh | relu | gelu | linear
    tied_weights: bool = True  # R[l] = W[l]^T

    # Settling dynamics
    eta_x: float = 0.1  # base state update step size
    eta_x_scale: str = "inv_dim"  # "uniform" | "inv_dim" | "learned"
    K: int = 20  # max settling iterations
    converge_tol: float = 1e-4  # relative energy change threshold
    converge_patience: int = 3  # consecutive steps below tol to stop

    # State normalization
    state_norm: str = "layernorm"  # "none" | "layernorm" | "rmsnorm"

    # Stability
    leaky_rate: float = 0.0  # leaky integration coefficient (0 = off)
    dx_clip: float = 50.0  # gradient clipping on state updates
    error_norm: bool = False  # normalize errors by sqrt(dim)
    stability_mode: str = "adaptive"  # "strict" | "adaptive" | "none"

    # Learning
    eta_w: float = 0.001  # weight learning rate
    weight_decay: float = 0.0  # L2 on weights

    # Training
    batch_size: int = 64
    epochs: int = 50
    device: str = "cuda"
    seed: int = 42
    deterministic: bool = False

    # Diagnostics
    verbose: bool = False  # print per-iteration energy during settling

    @property
    def n_layers(self) -> int:
        """Number of weight layers (L)."""
        return len(self.layer_sizes) - 1


@dataclass
class ContinualConfig:
    """Configuration for continual learning experiments."""

    # PCHead architecture
    hidden_sizes: list[int] = field(default_factory=lambda: [1024, 256])
    num_classes: int = 10

    # Replay buffer
    replay_size: int = 1000
    replay_mix: float = 0.25  # fraction of batch from replay
    replay_mode: str = "joint"  # "joint" | "separate"

    # Regularization
    l2_prev_weight: float = 0.01  # L2 toward previous-task weights

    # GGUF backend
    model_path: str = ""
    n_ctx: int = 2048
    n_threads: int = 8
    n_gpu_layers: int = 0
    feature_layer: str | int = "last"  # "last" | "middle" | int

    # Task sequence
    n_tasks: int = 5
    classes_per_task: int = 2


@dataclass
class DepthTimeConfig:
    """Configuration for depth-vs-time experiment grid.

    Sweeps (hidden_sizes, K) to test whether settling steps
    can substitute for structural depth.
    """

    # Grid axes
    depth_configs: list[list[int]] = field(default_factory=lambda: [
        [],            # no hidden layers (direct input -> output)
        [128],         # 1 hidden, narrow
        [256],         # 1 hidden, wider
        [256, 128],    # 2 hidden
        [1024, 256],   # 2 hidden, current default
    ])
    k_values: list[int] = field(default_factory=lambda: [5, 10, 20, 40, 80])

    # PCHead hyperparameters (held constant across grid)
    eta_x: float = 0.05
    eta_w: float = 0.001
    activation: str = "tanh"

    # Continual learning settings
    n_tasks: int = 5
    classes_per_task: int = 2
    replay_size: int = 1000
    replay_mix: float = 0.25
    replay_mode: str = "joint"

    # Training
    epochs_per_task: int = 10
    batch_size: int = 64
    seeds: list[int] = field(default_factory=lambda: [42, 123, 7])

    # Output
    output_dir: str = "./experiments/depth_vs_time"
