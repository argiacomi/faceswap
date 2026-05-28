#!/usr/bin/env python3
"""Tests for the Phaze-A bottleneck latent normalization flow."""

from __future__ import annotations

import json

import numpy as np
import pytest
from keras import Input, Model, models

from lib.utils import get_backend
from plugins.train.model import phaze_a_defaults as model_cfg
from plugins.train.model._base.state import State
from plugins.train.model.phaze_a import _bottleneck

_BACKEND_ID = [get_backend().upper()]


_BOTTLENECK_SHAPES = (
    ("flatten", (2, 128)),
    ("dense", (2, 16)),
    ("average_pooling", (2, 8)),
    ("max_pooling", (2, 8)),
)
_LATENT_NORM_MODES = ("none", "layer", "rms", "group", "instance")

# Representative per-commit smoke: each bottleneck builds + predicts with the
# default ("layer") latent normalization.  The exhaustive bottleneck x latent
# norm matrix below is gated behind the ``slow`` marker so per-commit CI does
# not pay for it.
_REPRESENTATIVE = [(bn, shape, "layer") for bn, shape in _BOTTLENECK_SHAPES]
_FULL_MATRIX = [
    (bn, shape, norm)
    for bn, shape in _BOTTLENECK_SHAPES
    for norm in _LATENT_NORM_MODES
    if norm != "layer"
]


def _run_bottleneck_predict(bottleneck: str, expected_shape, normalization: str) -> None:
    inputs = Input(shape=(4, 4, 8))
    outputs = _bottleneck(inputs, bottleneck, 16, normalization)
    model = Model(inputs, outputs)
    data = np.random.random((2, 4, 4, 8)).astype("float32")
    assert model.predict(data, verbose=0).shape == expected_shape


@pytest.mark.parametrize("dummy", [None], ids=_BACKEND_ID)
@pytest.mark.parametrize(
    ("bottleneck", "expected_shape", "normalization"),
    _REPRESENTATIVE,
    ids=[f"{bn}-layer" for bn, _, _ in _REPRESENTATIVE],
)
def test_bottleneck_builds_all_modes_with_default_latent_normalization(
    dummy, bottleneck, expected_shape, normalization
) -> None:
    """Every bottleneck mode should construct and predict with the default latent norm."""
    del dummy
    _run_bottleneck_predict(bottleneck, expected_shape, normalization)


@pytest.mark.slow
@pytest.mark.parametrize("dummy", [None], ids=_BACKEND_ID)
@pytest.mark.parametrize(
    ("bottleneck", "expected_shape", "normalization"),
    _FULL_MATRIX,
    ids=[f"{bn}-{norm}" for bn, _, norm in _FULL_MATRIX],
)
def test_bottleneck_full_normalization_matrix(
    dummy, bottleneck, expected_shape, normalization
) -> None:
    """Full bottleneck x latent norm matrix.  Slow tier only."""
    del dummy
    _run_bottleneck_predict(bottleneck, expected_shape, normalization)


@pytest.mark.parametrize(
    ("bottleneck", "expected_layers"),
    [
        ("flatten", ["Flatten", "LayerNormalization"]),
        ("dense", ["Flatten", "Dense", "LayerNormalization"]),
        ("average_pooling", ["GlobalAveragePooling2D", "LayerNormalization"]),
        ("max_pooling", ["GlobalMaxPooling2D", "LayerNormalization"]),
    ],
    ids=["flatten", "dense", "average_pooling", "max_pooling"],
)
def test_bottleneck_applies_latent_normalization_after_projection(
    bottleneck, expected_layers
) -> None:
    """Layer order should place latent normalization after flatten/pooling/dense work."""
    inputs = Input(shape=(4, 4, 8))
    outputs = _bottleneck(inputs, bottleneck, 16, "layer")
    model = Model(inputs, outputs)

    layer_names = [layer.__class__.__name__ for layer in model.layers[1:]]

    assert layer_names == expected_layers


def test_state_migrates_bottleneck_norm_to_latent_norm(tmp_path) -> None:
    """Existing state files should migrate legacy bottleneck_norm to latent_norm."""
    if not model_cfg.latent_norm._name:  # pylint:disable=protected-access
        model_cfg.latent_norm.set_name("train.model.phaze_a.latent_norm")

    state_path = tmp_path / "phaze_a_state.json"
    state_path.write_text(
        json.dumps(
            {
                "name": "phaze_a",
                "config": {"bottleneck_norm": "group"},
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
    saved = json.loads(state_path.read_text(encoding="utf-8"))

    assert "bottleneck_norm" not in state._config
    assert state._config["latent_norm"] == "group"
    assert "bottleneck_norm" not in saved["config"]
    assert saved["config"]["latent_norm"] == "group"


def test_bottleneck_model_roundtrips_with_latent_norm(tmp_path) -> None:
    """A model using post-bottleneck latent normalization should save and reload cleanly."""
    inputs = Input(shape=(4, 4, 8))
    outputs = _bottleneck(inputs, "dense", 16, "group")
    model = Model(inputs, outputs)

    model_path = tmp_path / "latent_norm.keras"
    model.save(model_path)
    loaded = models.load_model(model_path)

    data = np.random.random((2, 4, 4, 8)).astype("float32")
    result = loaded.predict(data, verbose=0)

    assert result.shape == (2, 16)
