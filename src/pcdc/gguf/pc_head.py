"""PCHead — small predictive coding network on frozen LLM features."""

from __future__ import annotations

from pcdc.core.pc_network import PCNetwork
from pcdc.utils.config import PCConfig


class PCHead(PCNetwork):
    """Predictive coding head operating on pre-extracted features.

    This is a thin wrapper around PCNetwork that constructs the right
    layer_sizes from feature_dim, hidden_sizes, and num_classes.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_sizes: list[int] | None = None,
        **config_kwargs,
    ):
        if hidden_sizes is None:
            hidden_sizes = [256]

        layer_sizes = [feature_dim] + hidden_sizes + [num_classes]
        config = PCConfig(layer_sizes=layer_sizes, **config_kwargs)
        super().__init__(config)
        self.feature_dim = feature_dim
        self.num_classes = num_classes
