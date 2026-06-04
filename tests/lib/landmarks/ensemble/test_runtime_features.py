#!/usr/bin/env python3
"""Tests for runtime-visible feature builders, including stacked regression inputs."""

from __future__ import annotations

import numpy as np

from lib.landmarks.ensemble.runtime_features import (
    STACKED_REGRESSION_FEATURE_MODELS,
    forbidden_runtime_features,
    stacked_regression_feature_map,
)


def _face(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    face: np.ndarray = (rng.random((68, 2)) * 100.0 + 50.0).astype("float64")
    return face


def test_forbidden_runtime_features_flags_offline_only_fields() -> None:
    names = [
        "candidate_is_fusion",
        "candidate_nme",
        "oracle_gap",
        "transform_regret_v3",
        "is_selected_by_policy",
        "ground_truth_distance",
        "region_disagreement_jaw",
    ]
    leaked = set(forbidden_runtime_features(names))
    assert "candidate_nme" in leaked
    assert "oracle_gap" in leaked
    assert "transform_regret_v3" in leaked
    assert "is_selected_by_policy" in leaked
    assert "ground_truth_distance" in leaked
    # Legitimate runtime features are not flagged.
    assert "candidate_is_fusion" not in leaked
    assert "region_disagreement_jaw" not in leaked


def test_stacked_feature_map_is_runtime_safe() -> None:
    base = _face(1)
    models = {"fan": _face(2), "hrnet": _face(3)}
    features = stacked_regression_feature_map(
        base_landmarks=base,
        model_landmarks=models,
        runtime_bucket="frontal",
        roll_estimate=1.0,
        yaw_estimate=2.0,
        candidate_yaw_disagreement=3.0,
        max_disagreement_px=4.0,
        hard_case_tags=("frontal",),
        model_predictions_available={"fan": True, "hrnet": True},
    )
    assert forbidden_runtime_features(features) == ()
    assert all(np.isfinite(value) for value in features.values())


def test_stacked_feature_map_fixed_width_for_missing_models() -> None:
    """Missing models contribute zeros plus an availability flag of 0."""
    base = _face(1)
    only_fan = stacked_regression_feature_map(
        base_landmarks=base,
        model_landmarks={"fan": _face(2)},
    )
    assert only_fan["model_available_fan"] == 1.0
    for model in STACKED_REGRESSION_FEATURE_MODELS:
        if model != "fan":
            assert only_fan[f"model_available_{model}"] == 0.0
            assert only_fan[f"model_{model}_region_jaw_dx"] == 0.0


def test_stacked_feature_map_region_disagreement_present() -> None:
    base = _face(1)
    models = {"fan": _face(2), "hrnet": _face(3), "orformer": _face(4)}
    features = stacked_regression_feature_map(base_landmarks=base, model_landmarks=models)
    for region in ("jaw", "brows", "nose", "eyes", "mouth"):
        assert f"region_disagreement_{region}" in features
    assert features["available_model_count"] == 3.0
