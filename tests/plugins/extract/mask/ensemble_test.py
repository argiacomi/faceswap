#!/usr/bin/env python3
"""Unit tests for :mod:`plugins.extract.mask.ensemble`."""

from __future__ import annotations

import typing as T
from dataclasses import dataclass

import numpy as np
import pytest

from lib.utils import FaceswapError
from plugins.extract.mask._output import MaskPluginOutput
from plugins.extract.mask.ensemble import (
    Ensemble,
    _confidence_weighted_intersection,
    _confidence_weighted_union,
    _normalised_parser_weights,
)
from plugins.plugin_loader import PluginLoader

if T.TYPE_CHECKING:
    import numpy.typing as npt


@dataclass
class _FakeSource:
    """Minimal source masker for ensemble process tests."""

    storage_name: str
    output: MaskPluginOutput
    input_size: int = 512
    _segment_indices: tuple[int, ...] = (1,)

    def pre_process(self, batch: np.ndarray) -> np.ndarray:
        return batch

    def process(self, batch: np.ndarray) -> np.ndarray:
        return batch

    def post_process(self, batch: np.ndarray) -> MaskPluginOutput:
        del batch
        return self.output


class _BinaryOnlySource:
    """Source plugin without the ensemble probability capability flag."""

    storage_name = "custom_face"

    def load_model(self) -> object:
        raise AssertionError("Binary-only sources should be rejected before model load")


def _patch_config(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_models: list[str] | None = None,
    strategy: str = "confidence-weighted-union",
    centering: str = "face",
) -> None:
    """Patch ensemble config accessors for constructor-only tests."""
    monkeypatch.setattr("plugins.extract.mask.ensemble.cfg.batch_size", lambda: 2)
    monkeypatch.setattr(
        "plugins.extract.mask.ensemble.cfg.source_models",
        lambda: source_models or ["bisenet-fp", "segnext-fp"],
    )
    monkeypatch.setattr("plugins.extract.mask.ensemble.cfg.strategy", lambda: strategy)
    monkeypatch.setattr("plugins.extract.mask.ensemble.cfg.centering", lambda: centering)


def _output(
    foreground: npt.NDArray[np.float32],
    selected_confidence: npt.NDArray[np.float32],
    *,
    source_id: str,
) -> MaskPluginOutput:
    """Build a two-class parser output with a selected foreground class."""
    probs = np.stack([1.0 - foreground, selected_confidence], axis=-1).astype("float32")
    return MaskPluginOutput(
        (foreground >= 0.5).astype("float32"),
        source_id=source_id,
        per_class_probs=probs,
    )


def test_plugin_loader_discovers_ensemble() -> None:
    """The ensemble masker must be selectable through normal mask discovery."""
    assert "ensemble" in PluginLoader.get_available_extractors("mask")


def test_plugin_loader_extends_ensemble_storage_names() -> None:
    """Convert/train mask choices should expose ensemble face/head storage names."""
    maskers = PluginLoader.get_available_extractors("mask", extend_plugin=True)

    assert "ensemble_face" in maskers
    assert "ensemble_head" in maskers


def test_constructor_uses_configured_storage_centering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensemble masks should store under a stable centering-specific name."""
    _patch_config(monkeypatch, centering="head")

    plugin = Ensemble()

    assert plugin.centering == "head"
    assert plugin.storage_name == "ensemble_head"


def test_rejects_less_than_two_source_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """A useful ensemble requires two or more sources."""
    _patch_config(monkeypatch, source_models=["bisenet-fp"])

    with pytest.raises(FaceswapError, match="at least two source models"):
        Ensemble()


def test_confidence_weighted_union_is_deterministic() -> None:
    """Union should preserve agreement and weighted unique regions."""
    primary = np.array([[[0.8, 0.2], [0.4, 0.0]]], dtype=np.float32)
    secondary = np.array([[[0.5, 0.6], [0.1, 0.3]]], dtype=np.float32)
    weights = np.array([[0.75], [0.25]], dtype=np.float32)

    result = _confidence_weighted_union([primary, secondary], weights)

    expected = np.array([[[0.7625, 0.45], [0.3625, 0.1875]]], dtype=np.float32)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_confidence_weighted_intersection_is_deterministic() -> None:
    """Intersection should keep only mutually supported regions."""
    primary = np.array([[[0.8, 0.2], [0.4, 0.0]]], dtype=np.float32)
    secondary = np.array([[[0.5, 0.6], [0.1, 0.3]]], dtype=np.float32)
    weights = np.array([[0.75], [0.25]], dtype=np.float32)

    result = _confidence_weighted_intersection([primary, secondary], weights)

    expected = np.array([[[0.25, 0.1], [0.05, 0.0]]], dtype=np.float32)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_process_combines_sources_and_exposes_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plugin should emit a normal ndarray-compatible mask with debug metadata."""
    _patch_config(monkeypatch)
    plugin = Ensemble()
    plugin.input_size = 2
    primary = np.array([[[0.8, 0.2], [0.4, 0.0]]], dtype=np.float32)
    secondary = np.array([[[0.5, 0.6], [0.1, 0.3]]], dtype=np.float32)
    plugin._sources = [  # pylint:disable=protected-access
        _FakeSource("bisenet-fp_face", _output(primary, primary, source_id="bisenet-fp_face")),  # type: ignore[list-item]
        _FakeSource(  # type: ignore[list-item]
            "segnext-fp_face",
            _output(secondary, secondary, source_id="segnext-fp_face"),
        ),
    ]

    result = plugin.process(np.zeros((1, 2, 2, 3), dtype=np.float32))

    assert isinstance(result, MaskPluginOutput)
    assert result.shape == (1, 2, 2)
    assert result.source_id == "ensemble_face"
    assert result.per_class_probs is not None
    assert result.per_class_probs.shape == (1, 2, 2, 2)
    assert result.metadata["strategy"] == "confidence-weighted-union"
    assert result.metadata["sources"] == ["bisenet-fp_face", "segnext-fp_face"]


def test_process_rejects_source_without_probability_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Binary-only maskers should fail clearly rather than silently lowering quality."""
    _patch_config(monkeypatch)
    plugin = Ensemble()
    binary_only = MaskPluginOutput(
        np.ones((1, 2, 2), dtype=np.float32),
        source_id="custom_face",
        per_class_probs=None,
    )
    parser = _output(
        np.ones((1, 2, 2), dtype=np.float32),
        np.ones((1, 2, 2), dtype=np.float32),
        source_id="bisenet-fp_face",
    )
    plugin._sources = [  # pylint:disable=protected-access
        _FakeSource("custom_face", binary_only),  # type: ignore[list-item]
        _FakeSource("bisenet-fp_face", parser),  # type: ignore[list-item]
    ]

    with pytest.raises(FaceswapError, match="does not expose per-class probabilities"):
        plugin.process(np.zeros((1, 2, 2, 3), dtype=np.float32))


def test_load_model_rejects_sources_without_probability_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Incompatible sources should fail before loading model weights."""
    _patch_config(monkeypatch, source_models=["custom", "bisenet-fp"])
    plugin = Ensemble()
    monkeypatch.setattr(
        "plugins.extract.mask.ensemble.PluginLoader.get_extractor",
        lambda plugin_type, name: _BinaryOnlySource(),
    )

    with pytest.raises(FaceswapError, match="does not expose per-class probabilities"):
        plugin.load_model()


def test_parser_weights_are_normalized_per_batch() -> None:
    """Parser weights should be computed independently per batch item."""
    primary_conf = np.array(
        [
            [[0.8, 0.8], [0.8, 0.8]],
            [[0.2, 0.2], [0.2, 0.2]],
        ],
        dtype=np.float32,
    )
    secondary_conf = np.array(
        [
            [[0.2, 0.2], [0.2, 0.2]],
            [[0.8, 0.8], [0.8, 0.8]],
        ],
        dtype=np.float32,
    )

    weights = _normalised_parser_weights([primary_conf, secondary_conf])

    np.testing.assert_allclose(weights[:, 0], np.array([0.8, 0.2], dtype=np.float32))
    np.testing.assert_allclose(weights[:, 1], np.array([0.2, 0.8], dtype=np.float32))
