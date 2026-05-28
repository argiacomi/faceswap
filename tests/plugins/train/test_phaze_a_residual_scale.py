#!/usr/bin/env python3
"""Tests for decoder residual scaling in Phaze-A."""

from __future__ import annotations

import json

import numpy as np
import pytest
from keras import Input, Model
from keras import layers as kl

from plugins.train import train_config as train_cfg
from plugins.train.model import phaze_a_defaults as model_cfg
from plugins.train.model._base.state import State
from plugins.train.model.phaze_a import UpscaleBlocks, _decoder_residual_block


@pytest.fixture(autouse=True)
def _patch_global_train_config(patch_config) -> None:
    """Keep global train config deterministic for decoder residual scale tests."""
    patch_config(train_cfg, {"conv_aware_init": False, "reflect_padding": False})


def _set_constant_residual_weights(model: Model) -> None:
    """Force the residual branch to output a constant tensor of ones."""
    conv_layers = [layer for layer in model.layers if isinstance(layer, kl.Conv2D)]
    assert len(conv_layers) == 2

    for layer in conv_layers:
        kernel, bias = layer.get_weights()
        layer.set_weights([np.zeros_like(kernel), np.ones_like(bias)])


@pytest.mark.parametrize(
    "residual_scale",
    [0.1, 0.2, 0.5, 1.0],
    ids=["scale_0_1", "scale_0_2", "scale_0_5", "scale_1_0"],
)
def test_decoder_residual_scale_only_scales_residual_branch(
    residual_scale: float,
) -> None:
    """Residual scaling should attenuate only F(inputs), not the full residual output."""
    inputs = Input(shape=(4, 4, 1))
    outputs = _decoder_residual_block(
        inputs,
        1,
        norm="none",
        activation="none",
        residual_scale=residual_scale,
        name=f"decoder_scale_{str(residual_scale).replace('.', '_')}",
    )
    model = Model(inputs, outputs)
    _set_constant_residual_weights(model)

    data = np.full((1, 4, 4, 1), 2.0, dtype="float32")
    result = model.predict(data, verbose=0)

    expected = np.full_like(data, 2.0 + residual_scale)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("residual_scale", "expected_scale_layers"),
    [(0.5, 1), (1.0, 0)],
    ids=["scaled", "legacy_default"],
)
def test_upscale_block_uses_configured_decoder_residual_scale(
    patch_config, residual_scale: float, expected_scale_layers: int
) -> None:
    """The decoder upscale path should pass the configured residual scale into the helper."""
    patch_config(
        model_cfg,
        {
            "dec_gaussian": False,
            "dec_norm": "none",
            "dec_res_block_norm": "none",
            "dec_res_blocks": 1,
            "dec_residual_scale": residual_scale,
            "dec_upscale_method": "upsample2d",
        },
    )

    inputs = Input(shape=(4, 4, 8))
    outputs = UpscaleBlocks("a")._upscale_block(inputs, 8, 0)  # pylint:disable=protected-access
    model = Model(inputs, outputs)

    scale_layers = [layer for layer in model.layers if layer.name.endswith("_scale")]

    assert outputs.shape == (None, 8, 8, 8)
    assert len(scale_layers) == expected_scale_layers


@pytest.mark.parametrize(
    "residual_scale", [0.0, -0.1, 1.1], ids=["zero", "negative", "greater_than_one"]
)
def test_invalid_decoder_residual_scale_fails_clearly(residual_scale: float) -> None:
    """Decoder residual scales outside the supported range should fail clearly."""
    inputs = Input(shape=(4, 4, 1))

    with pytest.raises(ValueError, match="Decoder residual scale must be greater than 0"):
        _decoder_residual_block(
            inputs,
            1,
            norm="none",
            activation="none",
            residual_scale=residual_scale,
            name="decoder_invalid_scale",
        )


def test_state_defaults_missing_decoder_residual_scale_to_one(tmp_path) -> None:
    """Older Phaze-A state files should pick up the new residual scale option as `1.0`."""
    assert model_cfg.dec_residual_scale() == 1.0

    state_path = tmp_path / "phaze_a_state.json"
    state_path.write_text(
        json.dumps(
            {
                "name": "phaze_a",
                "config": {},
                "sessions": {},
                "iterations": 0,
                "mixed_precision_layers": [],
                "lr_finder": -1.0,
                "lowest_avg_loss": 0.0,
            }
        ),
        encoding="utf-8",
    )

    state = State(str(tmp_path), "phaze_a", no_logs=True)
    state.save()
    saved = json.loads(state_path.read_text(encoding="utf-8"))

    assert state._config["dec_residual_scale"] == 1.0
    assert saved["config"]["dec_residual_scale"] == 1.0
