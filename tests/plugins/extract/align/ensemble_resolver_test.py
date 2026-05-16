#!/usr/bin/env python3
"""Plugin-level tests for the geometry-risk alignment resolver wiring (#78)."""

from __future__ import annotations

import numpy as np

from lib.landmarks.adapters import LandmarkAdapterConfig, StaticLandmarkAdapter
from plugins.extract.align.ensemble import Ensemble


def _face() -> np.ndarray:
    """Plausible iBUG-style 68-point face used so AlignedFace returns sane geometry."""
    points = np.zeros((68, 2), dtype="float32")
    points[0:17, 0] = np.linspace(40, 160, 17)
    points[0:17, 1] = 120 + 30 * np.sin(np.linspace(0, np.pi, 17))
    points[17:22, 0] = np.linspace(50, 90, 5)
    points[17:22, 1] = 70
    points[22:27, 0] = np.linspace(110, 150, 5)
    points[22:27, 1] = 70
    points[27:36, 0] = 100
    points[27:36, 1] = np.linspace(75, 110, 9)
    points[36:42, 0] = np.linspace(60, 80, 6)
    points[36:42, 1] = 85
    points[42:48, 0] = np.linspace(120, 140, 6)
    points[42:48, 1] = 85
    points[48:60, 0] = np.linspace(70, 130, 12)
    points[48:60, 1] = 130
    points[60:68, 0] = np.linspace(80, 120, 8)
    points[60:68, 1] = 130
    return points


def _three_adapters(offsets: tuple[float, float, float]) -> list[StaticLandmarkAdapter]:
    """Three static adapters returning the canonical face shifted by ``offsets``."""
    base = _face()
    return [
        StaticLandmarkAdapter(
            LandmarkAdapterConfig(name, coordinate_space="frame", weight=1.0),
            base + offset,
        )
        for name, offset in zip(("hrnet", "spiga", "orformer"), offsets, strict=True)
    ]


def test_resolver_disabled_uses_legacy_fusion(tmp_path) -> None:
    """``use_alignment_resolver=False`` (default) preserves existing fusion."""
    adapters = _three_adapters((0.0, 0.5, 1.0))
    plugin = Ensemble(
        adapters=adapters,
        crop_scale=1.0,
        strategy="plain_average",
        use_alignment_resolver=False,
    )
    plugin.model = plugin.load_model()
    plugin.predict_landmarks_68(np.zeros((256, 256, 3), dtype="float32"))
    debug = plugin.last_debug_metadata[0]
    assert debug["weight_source"] in {"adapter_config", "promoted_setup"}
    assert "resolver" not in debug


def test_resolver_enabled_routes_low_risk_path(tmp_path) -> None:
    """Closely-agreeing adapters take the resolver's low-risk path."""
    adapters = _three_adapters((0.0, 0.3, 0.6))
    plugin = Ensemble(
        adapters=adapters,
        crop_scale=1.0,
        strategy="plain_average",
        use_alignment_resolver=True,
        resolver_hard_case_strategy="static_weighted_downweight",
        resolver_high_disagreement_px=50.0,  # generous so we land in low_risk
    )
    plugin.model = plugin.load_model()
    plugin.predict_landmarks_68(np.zeros((256, 256, 3), dtype="float32"))
    debug = plugin.last_debug_metadata[0]
    assert debug["weight_source"] == "geometry_resolver"
    assert debug["resolver"]["risk_route"] == "low_risk"
    assert debug["strategy"] == "plain_average"
    assert "active_models" in debug


def test_resolver_high_risk_swaps_in_hard_case_strategy(tmp_path) -> None:
    """Large disagreement steers the resolver to the hard-case strategy."""
    # 25-pixel shifts across adapters drive mean pairwise distance well above
    # the configured 10-px threshold.
    adapters = _three_adapters((0.0, 25.0, -25.0))
    plugin = Ensemble(
        adapters=adapters,
        crop_scale=1.0,
        strategy="plain_average",
        use_alignment_resolver=True,
        resolver_hard_case_strategy="static_weighted_downweight",
        resolver_high_disagreement_px=10.0,
    )
    plugin.model = plugin.load_model()
    plugin.predict_landmarks_68(np.zeros((256, 256, 3), dtype="float32"))
    debug = plugin.last_debug_metadata[0]
    assert debug["resolver"]["risk_route"] == "high_risk"
    assert debug["strategy"] == "static_weighted_downweight"
    assert "high_disagreement" in debug["resolver"]["geometry_flags"]


def test_resolver_metadata_carries_per_model_disagreement(tmp_path) -> None:
    """Per-model disagreement values appear in ``last_debug_metadata`` for debugging."""
    adapters = _three_adapters((0.0, 0.5, 1.0))
    plugin = Ensemble(
        adapters=adapters,
        crop_scale=1.0,
        strategy="plain_average",
        use_alignment_resolver=True,
    )
    plugin.model = plugin.load_model()
    plugin.predict_landmarks_68(np.zeros((256, 256, 3), dtype="float32"))
    debug = plugin.last_debug_metadata[0]
    assert "max_disagreement_px" in debug["resolver"]
    assert debug["resolver"]["max_disagreement_px"] >= 0.0
    assert debug["resolver"]["geometry_confidence"] >= 0.0
