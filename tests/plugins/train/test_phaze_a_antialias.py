#!/usr/bin/env python3
"""Tests for decoder anti-alias blur in Phaze-A."""

from __future__ import annotations

import json

import numpy as np
import pytest
from keras import Input, Model
from keras import layers as kl

from plugins.train import train_config as train_cfg
from plugins.train.model import phaze_a_defaults as model_cfg
from plugins.train.model._base.state import State
from plugins.train.model.phaze_a import Decoder, UpscaleBlocks
from plugins.train.train_config import Loss as cfg_loss


@pytest.fixture(autouse=True)
def _patch_train_config(monkeypatch: pytest.MonkeyPatch, patch_config) -> None:
    """Keep global train config deterministic for decoder anti-alias tests."""
    patch_config(train_cfg, {"conv_aware_init": False, "reflect_padding": False})
    monkeypatch.setattr(cfg_loss, "learn_mask", lambda: False)
    UpscaleBlocks._filters = []  # pylint:disable=protected-access


def _patch_decoder_config(patch_config, **overrides) -> None:
    """Patch the minimal decoder config needed for anti-alias tests."""
    patch_config(
        model_cfg,
        {
            "dec_activation": "leakyrelu",
            "dec_antialias": "none",
            "dec_filter_slope": 0.0,
            "dec_gaussian": False,
            "dec_max_filters": 8,
            "dec_min_filters": 8,
            "dec_norm": "none",
            "dec_res_block_norm": "none",
            "dec_res_blocks": 0,
            "dec_residual_scale": 1.0,
            "dec_skip_last_residual": True,
            "dec_slope_mode": "full",
            "dec_upscale_method": "subpixel",
            "output_size": 8,
        }
        | overrides,
    )


@pytest.mark.parametrize(
    "upscale_method",
    ["subpixel", "resize_images", "upscale_fast", "upscale_hybrid", "upscale_dny"],
    ids=["subpixel", "resize_images", "upscale_fast", "upscale_hybrid", "upscale_dny"],
)
def test_decoder_antialias_light_adds_fixed_non_trainable_blur(
    patch_config, upscale_method: str
) -> None:
    """Light anti-alias mode should add a normalized non-trainable blur after learned upscales."""
    _patch_decoder_config(
        patch_config, dec_antialias="light", dec_upscale_method=upscale_method
    )

    inputs = Input(shape=(4, 4, 8))
    outputs = UpscaleBlocks("a")._upscale_block(inputs, 8, 0)  # pylint:disable=protected-access
    model = Model(inputs, outputs)

    blur_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, kl.DepthwiseConv2D) and layer.name.endswith("_antialias")
    ]

    assert outputs.shape == (None, 8, 8, 8)
    assert len(blur_layers) == 1
    assert blur_layers[0].trainable is False

    kernel = blur_layers[0].get_weights()[0]
    np.testing.assert_allclose(kernel.sum(axis=(0, 1, 3)), np.ones(kernel.shape[2]))


def test_decoder_antialias_is_absent_when_disabled(patch_config) -> None:
    """The default anti-alias mode should not add blur layers."""
    _patch_decoder_config(patch_config, dec_antialias="none")

    inputs = Input(shape=(4, 4, 8))
    outputs = UpscaleBlocks("a")._upscale_block(inputs, 8, 0)  # pylint:disable=protected-access
    model = Model(inputs, outputs)

    assert outputs.shape == (None, 8, 8, 8)
    assert not any(
        isinstance(layer, kl.DepthwiseConv2D) and layer.name.endswith("_antialias")
        for layer in model.layers
    )


def test_decoder_antialias_precedes_face_out(patch_config) -> None:
    """Anti-alias blur should remain upstream of the terminal face output layer."""
    _patch_decoder_config(patch_config, dec_antialias="light")

    decoder = Decoder("a", (4, 4, 8))()
    names = [layer.name for layer in decoder.layers]
    antialias_idx = [
        idx for idx, name in enumerate(names) if name.endswith("_antialias")
    ]
    face_out_idx = [
        idx for idx, name in enumerate(names) if name.startswith("face_out")
    ]

    assert antialias_idx
    assert face_out_idx
    assert max(antialias_idx) < min(face_out_idx)


def test_state_defaults_missing_decoder_antialias_to_none(tmp_path) -> None:
    """Older Phaze-A state files should pick up the new anti-alias config as `none`."""
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

    assert state._config["dec_antialias"] == "none"
    assert saved["config"]["dec_antialias"] == "none"
