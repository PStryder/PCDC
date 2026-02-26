"""Tests for clamping modes."""

from pcdc.core.clamp import ClampMode, make_clamp_mask


def test_supervised_clamp():
    mask = make_clamp_mask(4, ClampMode.SUPERVISED)
    assert mask == [True, False, False, True]


def test_inference_clamp():
    mask = make_clamp_mask(4, ClampMode.INFERENCE)
    assert mask == [True, False, False, False]


def test_generative_clamp():
    mask = make_clamp_mask(4, ClampMode.GENERATIVE)
    assert mask == [False, False, False, True]


def test_two_layer_supervised():
    mask = make_clamp_mask(2, ClampMode.SUPERVISED)
    assert mask == [True, True]
