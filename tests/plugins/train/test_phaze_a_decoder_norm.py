#!/usr/bin/env python3
"""Tests for decoder residual block normalization in Phaze-A."""

from __future__ import annotations

import json

import pytest
from keras import Input, Model
from keras import layers as kl

from lib.model.normalization import (
    GroupNormalization,
    InstanceNormalization,
    RMSNormalization,
)
from plugins.train import train_config as train_cfg
from plugins.train.model import phaze_a_defaults as model_cfg
from plugins.train.model._base.state import State
from plugins.train.model.phaze_a import UpscaleBlocks


@pytest.fixture(autouse=True)
def _patch_global_train_config(patch_config) -> None:
    """Keep train config deterministic for decoder normalization tests."""
    patch_config(train_cfg, {"conv_aware_init": False, "reflect_padding": False})


@pytest.mark.parametrize(
    ("norm_name", "norm_type"),
    [
        ("group", GroupNormalization),
        ("instance", InstanceNormalization),
        ("rms", RMSNormalization),
        ("layer", kl.LayerNormalization),
    ],
    ids=["group", "instance", "rms", "layer"],
)
def test_decoder_residual_norm_applies_only_inside_residual_blocks(
    patch_config, norm_name: str, norm_type: type[kl.Layer]
) -> None:
    """Residual normalization should only be introduced by decoder residual blocks."""
    patch_config(
        model_cfg,
        {
            "dec_gaussian": False,
            "dec_norm": "none",
            "dec_res_block_norm": norm_name,
            "dec_res_blocks": 1,
            "dec_upscale_method": "upsample2d",
        },
    )

    inputs = Input(shape=(4, 4, 8))
    outputs = UpscaleBlocks("a")._upscale_block(inputs, 8, 0)  # pylint:disable=protected-access
    model = Model(inputs, outputs)

    norms = [layer for layer in model.layers if isinstance(layer, norm_type)]

    assert outputs.shape == (None, 8, 8, 8)
    assert len(norms) == 2


def test_decoder_residual_norm_is_ignored_when_residual_blocks_disabled(
    patch_config,
) -> None:
    """The residual-only normalization setting should not affect non-residual decoder paths."""
    patch_config(
        model_cfg,
        {
            "dec_gaussian": False,
            "dec_norm": "none",
            "dec_res_block_norm": "layer",
            "dec_res_blocks": 0,
            "dec_upscale_method": "upsample2d",
        },
    )

    inputs = Input(shape=(4, 4, 8))
    outputs = UpscaleBlocks("a")._upscale_block(inputs, 8, 0)  # pylint:disable=protected-access
    model = Model(inputs, outputs)

    assert outputs.shape == (None, 8, 8, 8)
    assert not any(isinstance(layer, kl.LayerNormalization) for layer in model.layers)


def test_state_defaults_missing_decoder_residual_norm_to_none(tmp_path) -> None:
    """Older Phaze-A state files should pick up the new residual norm option as `none`."""
    assert model_cfg.dec_res_block_norm() == "none"

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

    assert state._config["dec_res_block_norm"] == "none"
    assert saved["config"]["dec_res_block_norm"] == "none"
