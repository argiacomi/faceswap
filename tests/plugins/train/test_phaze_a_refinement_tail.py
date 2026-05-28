#!/usr/bin/env python3
"""Tests for decoder refinement tail in Phaze-A."""

from __future__ import annotations

import json

import pytest
from keras import layers as kl

from plugins.train.model import phaze_a_defaults as model_cfg
from plugins.train.model._base.state import State
from plugins.train.model.phaze_a import Decoder, UpscaleBlocks
from plugins.train.train_config import Loss as cfg_loss


@pytest.fixture(autouse=True)
def _patch_decoder_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep decoder refinement tail tests deterministic."""
    monkeypatch.setattr(cfg_loss, "learn_mask", lambda: False)
    UpscaleBlocks._filters = []  # pylint:disable=protected-access


def _patch_decoder_config(patch_config, **overrides) -> None:
    """Patch the minimal decoder config needed for refinement tail tests."""
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
            "dec_output_kernel": 3,
            "dec_res_block_norm": "none",
            "dec_res_blocks": 0,
            "dec_residual_scale": 1.0,
            "dec_skip_last_residual": True,
            "dec_slope_mode": "full",
            "dec_upscale_method": "subpixel",
            "output_size": 8,
            "refinement_blocks": 1,
            "refinement_filters": 64,
            "refinement_tail": "none",
        }
        | overrides,
    )


def test_refinement_tail_none_adds_no_tail_layers(patch_config) -> None:
    """The default-off refinement tail should leave the decoder path unchanged."""
    _patch_decoder_config(patch_config, refinement_tail="none")

    decoder = Decoder("a", (4, 4, 8))()

    assert not any("refinement" in layer.name for layer in decoder.layers)


@pytest.mark.parametrize("blocks", [1, 2], ids=["one_block", "two_blocks"])
def test_refinement_tail_light_inserts_before_face_out(
    patch_config, blocks: int
) -> None:
    """The refinement tail should be inserted immediately before the face output path."""
    _patch_decoder_config(
        patch_config,
        dec_res_block_norm="layer",
        refinement_tail="light",
        refinement_blocks=blocks,
    )

    decoder = Decoder("a", (4, 4, 8))()
    names = [layer.name for layer in decoder.layers]
    refinement_idx = [idx for idx, name in enumerate(names) if "refinement" in name]
    face_out_idx = [
        idx for idx, name in enumerate(names) if name.startswith("face_out")
    ]

    assert refinement_idx
    assert face_out_idx
    assert max(refinement_idx) < min(face_out_idx)
    assert (
        sum("_refinement_" in name and name.endswith("_add") for name in names)
        == blocks
    )


def test_refinement_tail_preserves_output_shape(patch_config) -> None:
    """Enabling the refinement tail should not change the decoder output tensor shape."""
    _patch_decoder_config(patch_config, refinement_tail="none")
    decoder_default = Decoder("a", (4, 4, 8))()

    UpscaleBlocks._filters = []  # pylint:disable=protected-access
    _patch_decoder_config(patch_config, refinement_tail="light", refinement_blocks=2)
    decoder_refined = Decoder("a", (4, 4, 8))()

    assert decoder_default.output_shape == decoder_refined.output_shape


def test_refinement_filters_are_capped_to_dec_min_and_sixty_four(patch_config) -> None:
    """Runtime refinement channels should be capped to `min(refinement_filters, dec_min, 64)`."""
    _patch_decoder_config(
        patch_config,
        dec_max_filters=96,
        dec_min_filters=96,
        refinement_tail="light",
        refinement_filters=128,
    )

    decoder = Decoder("a", (4, 4, 96))()
    projection = next(
        layer for layer in decoder.layers if "refinement_proj" in layer.name
    )

    assert isinstance(projection, kl.Conv2D)
    assert projection.filters == 64


def test_state_defaults_missing_refinement_tail_config(tmp_path) -> None:
    """Older Phaze-A state files should pick up the refinement tail defaults."""
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

    assert state._config["refinement_tail"] == "none"
    assert state._config["refinement_blocks"] == 1
    assert state._config["refinement_filters"] == 64
    assert saved["config"]["refinement_tail"] == "none"
    assert saved["config"]["refinement_blocks"] == 1
    assert saved["config"]["refinement_filters"] == 64
