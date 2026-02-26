"""Clamp mode definitions for predictive coding networks."""

from __future__ import annotations

from enum import Enum


class ClampMode(Enum):
    """Determines which layers are clamped during settling."""

    SUPERVISED = "supervised"  # Clamp x[0] (input) and x[L] (target)
    INFERENCE = "inference"  # Clamp x[0] only; x[L] free
    GENERATIVE = "generative"  # Clamp x[L] only; x[0] free


def make_clamp_mask(n_layers: int, mode: ClampMode) -> list[bool]:
    """Create a boolean mask indicating which layers are clamped.

    Args:
        n_layers: Total number of layers (L+1, including input and output).
        mode: The clamping mode.

    Returns:
        List of booleans, length n_layers. True = clamped (skip update).
    """
    mask = [False] * n_layers

    if mode == ClampMode.SUPERVISED:
        mask[0] = True
        mask[-1] = True
    elif mode == ClampMode.INFERENCE:
        mask[0] = True
    elif mode == ClampMode.GENERATIVE:
        mask[-1] = True

    return mask
