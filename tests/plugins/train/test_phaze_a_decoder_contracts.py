#!/usr/bin/env python3
"""Tests for shared Phaze-A decoder helper contracts."""

from __future__ import annotations

import numpy as np
import pytest
from keras import Input, Model
from keras import layers as kl

from lib.model.layers import Swish
from lib.model.normalization import (
    GroupNormalization,
    InstanceNormalization,
    RMSNormalization,
)
from plugins.train import train_config as train_cfg
from plugins.train.model.phaze_a import (
    _decoder_residual_block,
    _get_activation,
    _get_norm_layer,
)


@pytest.fixture(autouse=True)
def _patch_global_train_config(patch_config) -> None:
    """Keep global train config deterministic for decoder contract tests."""
    patch_config(train_cfg, {"conv_aware_init": False, "reflect_padding": False})


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("none", None),
        ("instance", InstanceNormalization),
        ("group", GroupNormalization),
        ("rms", RMSNormalization),
        ("layer", kl.LayerNormalization),
    ],
    ids=["none", "instance", "group", "rms", "layer"],
)
def test_get_norm_layer_accepts_supported_options(
    name: str, expected: type[kl.Layer] | None
) -> None:
    """Supported normalization names should resolve to layers or None."""
    layer = _get_norm_layer(name, layer_name=f"norm_{name}")

    if expected is None:
        assert layer is None
    else:
        assert isinstance(layer, expected)


@pytest.mark.parametrize(
    ("name", "expected"),
    [("leakyrelu", kl.LeakyReLU), ("swish", Swish), ("silu", Swish)],
    ids=["leakyrelu", "swish", "silu"],
)
def test_get_activation_accepts_supported_options(
    name: str, expected: type[kl.Layer]
) -> None:
    """Supported activations should resolve to layers."""
    layer = _get_activation(name, layer_name=f"activation_{name}")

    assert isinstance(layer, expected)


def test_invalid_normalization_fails_clearly() -> None:
    """Unknown normalization names should raise a clear error."""
    with pytest.raises(ValueError, match="Unsupported normalization"):
        _get_norm_layer("batch")


def test_invalid_activation_fails_clearly() -> None:
    """Unknown activation names should raise a clear error."""
    with pytest.raises(ValueError, match="Unsupported activation"):
        _get_activation("relu")


def test_default_decoder_residual_contract_preserves_legacy_semantics() -> None:
    """The shared helper should keep the legacy residual block path at default settings."""
    inputs = Input(shape=(8, 8, 16))
    outputs = _decoder_residual_block(
        inputs,
        16,
        norm="none",
        activation="leakyrelu",
        residual_scale=1.0,
        name="decoder_contract",
    )
    model = Model(inputs, outputs)

    data = np.random.random((2, 8, 8, 16)).astype("float32")
    result = model.predict(data, verbose=0)

    assert result.shape == data.shape
    assert sum(isinstance(layer, kl.Conv2D) for layer in model.layers) == 2
    assert not any(
        isinstance(
            layer,
            (
                GroupNormalization,
                InstanceNormalization,
                RMSNormalization,
                kl.LayerNormalization,
            ),
        )
        for layer in model.layers
    )


def test_decoder_residual_contract_builds_non_default_norm_and_activation() -> None:
    """Non-default helper paths should construct without requiring a full model build."""
    inputs = Input(shape=(8, 8, 16))
    outputs = _decoder_residual_block(
        inputs,
        16,
        norm="layer",
        activation="swish",
        residual_scale=1.0,
        name="decoder_contract_swish",
    )
    model = Model(inputs, outputs)

    data = np.random.random((2, 8, 8, 16)).astype("float32")
    result = model.predict(data, verbose=0)

    assert result.shape == data.shape
    assert any(isinstance(layer, kl.LayerNormalization) for layer in model.layers)
    assert any(isinstance(layer, Swish) for layer in model.layers)
