#!/usr/bin/env python3
"""Tests for stage-specific activation configuration in Phaze-A."""

from __future__ import annotations

import json

from keras import Input, Model

from lib.model.layers import Swish
from plugins.train.model import phaze_a_defaults as model_cfg
from plugins.train.model._base.state import State
from plugins.train.model.phaze_a import Encoder, FullyConnected, UpscaleBlocks


def test_custom_encoder_supports_swish_activation(patch_config) -> None:
    """The custom Faceswap encoder path should use the configured Swish activation."""
    patch_config(
        model_cfg,
        {
            "enc_architecture": "fs_original",
            "enc_activation": "swish",
            "fs_original_depth": 2,
            "fs_original_min_filters": 16,
            "fs_original_max_filters": 32,
            "fs_original_use_alt": False,
            "bottleneck_in_encoder": False,
        },
    )

    model = Encoder((32, 32, 3))()

    assert any(isinstance(layer, Swish) for layer in model.layers)


def test_pretrained_encoder_activation_is_untouched_by_enc_activation(
    patch_config,
) -> None:
    """Pretrained encoder internals should not be rewritten by the custom activation config."""
    patch_config(
        model_cfg,
        {
            "enc_architecture": "mobilenet_v2",
            "enc_activation": "swish",
            "enc_load_weights": False,
            "bottleneck_in_encoder": False,
            "mobilenet_width": 1.0,
        },
    )

    model = Encoder((32, 32, 3))()

    assert not any(isinstance(layer, Swish) for layer in model.layers)


def test_fc_upsampling_supports_swish_activation(patch_config) -> None:
    """FC-owned upsampling should use the configured activation."""
    patch_config(
        model_cfg,
        {
            "fc_activation": "swish",
            "fc_dimensions": 4,
            "fc_upsampler": "upsample2d",
            "fc_upsamples": 1,
            "fc_upsample_filters": 64,
            "output_size": 64,
        },
    )

    fc_layer = FullyConnected("a", (4, 4, 8))
    inputs = Input(shape=(4, 4, 8))
    outputs = fc_layer._do_upsampling(inputs)  # pylint:disable=protected-access
    model = Model(inputs, outputs)

    assert any(isinstance(layer, Swish) for layer in model.layers)


def test_decoder_supports_swish_activation(patch_config) -> None:
    """Decoder-owned upsampling paths should use the configured activation."""
    patch_config(
        model_cfg,
        {
            "dec_activation": "swish",
            "dec_gaussian": False,
            "dec_norm": "none",
            "dec_res_block_norm": "none",
            "dec_res_blocks": 0,
            "dec_residual_scale": 1.0,
            "dec_upscale_method": "upsample2d",
        },
    )

    inputs = Input(shape=(4, 4, 8))
    outputs = UpscaleBlocks("a")._upscale_block(inputs, 8, 0)  # pylint:disable=protected-access
    model = Model(inputs, outputs)

    assert any(isinstance(layer, Swish) for layer in model.layers)


def test_state_defaults_missing_stage_activations_to_leakyrelu(tmp_path) -> None:
    """Older Phaze-A state files should pick up the new activation config defaults."""
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

    assert state._config["enc_activation"] == "leakyrelu"
    assert state._config["fc_activation"] == "leakyrelu"
    assert state._config["dec_activation"] == "leakyrelu"
    assert saved["config"]["enc_activation"] == "leakyrelu"
    assert saved["config"]["fc_activation"] == "leakyrelu"
    assert saved["config"]["dec_activation"] == "leakyrelu"
