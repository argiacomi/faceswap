#!/usr/bin/env python3
"""Tests for BiSeNet-FP mouth class selection."""

from __future__ import annotations

import numpy as np
import pytest

from plugins.extract.mask._output import MaskPluginOutput
from plugins.extract.mask.bisenet_fp import BiSeNetFP


def _patch_plugin_config(
    monkeypatch: pytest.MonkeyPatch,
    *,
    weights: str,
    include_mouth: bool,
) -> None:
    """Patch BiSeNet-FP config accessors for constructor-only tests."""
    monkeypatch.setattr("plugins.extract.mask.bisenet_fp.cfg.batch_size", lambda: 1)
    monkeypatch.setattr("plugins.extract.mask.bisenet_fp.cfg.cpu", lambda: True)
    monkeypatch.setattr("plugins.extract.mask.bisenet_fp.cfg.weights", lambda: weights)
    monkeypatch.setattr("plugins.extract.mask.bisenet_fp.cfg.include_ears", lambda: False)
    monkeypatch.setattr("plugins.extract.mask.bisenet_fp.cfg.include_hair", lambda: False)
    monkeypatch.setattr("plugins.extract.mask.bisenet_fp.cfg.include_glasses", lambda: False)
    monkeypatch.setattr("plugins.extract.mask.bisenet_fp.cfg.include_mouth", lambda: include_mouth)


def test_original_weights_include_mouth_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Original 19-class weights should preserve the historical mouth-included mask."""
    _patch_plugin_config(monkeypatch, weights="original", include_mouth=True)

    plugin = BiSeNetFP()

    assert 11 in plugin._segment_indices  # pylint:disable=protected-access
    assert {12, 13}.issubset(plugin._segment_indices)  # pylint:disable=protected-access


def test_original_weights_exclude_only_inner_mouth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabling mouth on original weights should preserve upper/lower lips."""
    _patch_plugin_config(monkeypatch, weights="original", include_mouth=False)

    plugin = BiSeNetFP()

    assert 11 not in plugin._segment_indices  # pylint:disable=protected-access
    assert {12, 13}.issubset(plugin._segment_indices)  # pylint:disable=protected-access


def test_faceswap_weights_do_not_select_mouth_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Faceswap 5-class weights do not expose a separate mouth class."""
    _patch_plugin_config(monkeypatch, weights="faceswap", include_mouth=True)

    plugin = BiSeNetFP()

    assert plugin._segment_indices == [1]  # pylint:disable=protected-access


def test_post_process_exposes_per_class_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BiSeNet-FP should keep ndarray behavior while exposing class probabilities."""
    _patch_plugin_config(monkeypatch, weights="faceswap", include_mouth=True)
    plugin = BiSeNetFP()
    logits: np.ndarray = np.zeros((1, 2, 2, 5), dtype=np.float32)
    logits[..., 1] = 10.0

    output = plugin.post_process(logits)

    assert isinstance(output, MaskPluginOutput)
    assert output.shape == (1, 2, 2)
    assert np.all(output == 1.0)
    assert output.source_id == "bisenet-fp_face"
    assert output.per_class_probs is not None
    assert output.per_class_probs.shape == (1, 2, 2, 5)
