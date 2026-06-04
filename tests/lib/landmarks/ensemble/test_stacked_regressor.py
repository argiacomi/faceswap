#!/usr/bin/env python3
"""Tests for the stacked residual regressor artifact contract and residual application."""

from __future__ import annotations

import numpy as np
import pytest

from lib.landmarks.ensemble import runtime_features
from lib.landmarks.ensemble import stacked_regressor as sr
from lib.landmarks.ensemble.stacked_regressor import (
    OUTPUT_MODE_GLOBAL_TRANSFORM,
    OUTPUT_MODE_LANDMARK_RESIDUAL_68,
    OUTPUT_MODE_REGION_RESIDUAL,
    RuntimeStackedLandmarkRegressor,
    StackedRegressorInvalid,
    apply_residual,
    load_stacked_regressor,
    output_dim_for_mode,
    write_stacked_regressor,
)


def _make_regressor(
    *,
    output_mode: str = OUTPUT_MODE_GLOBAL_TRANSFORM,
    feature_names: tuple[str, ...] = ("f0", "f1"),
    coef: np.ndarray | None = None,
    intercept: np.ndarray | None = None,
    clip: float = 0.05,
) -> RuntimeStackedLandmarkRegressor:
    dim = output_dim_for_mode(output_mode)
    n = len(feature_names)
    coef = np.zeros((n, dim)) if coef is None else np.asarray(coef, dtype="float64")
    intercept = np.zeros(dim) if intercept is None else np.asarray(intercept, dtype="float64")
    return RuntimeStackedLandmarkRegressor(
        candidate_name="stacked_residual",
        base_candidate_policy="static_weighted",
        feature_names=feature_names,
        output_mode=output_mode,
        output_dim=dim,
        residual_clip_fraction=clip,
        coef=coef,
        intercept=intercept,
        feature_mean=np.zeros(n),
        feature_std=np.ones(n),
    )


def _square_face() -> np.ndarray:
    rng = np.random.default_rng(0)
    face: np.ndarray = (rng.random((68, 2)) * 100.0 + 50.0).astype("float64")
    return face


def test_region_index_drift_guard() -> None:
    """Region partitions must match between the regressor and feature builder."""
    assert sr.STACKED_REGION_INDICES == runtime_features.STACKED_REGRESSION_REGION_INDICES


def test_artifact_roundtrip(tmp_path) -> None:
    regressor = _make_regressor()
    path = write_stacked_regressor(regressor, tmp_path / "stacked_regressor.json")
    loaded = load_stacked_regressor(path)
    assert loaded.candidate_name == "stacked_residual"
    assert loaded.feature_names == ("f0", "f1")
    assert loaded.output_mode == OUTPUT_MODE_GLOBAL_TRANSFORM
    np.testing.assert_allclose(loaded.coef, regressor.coef)


def test_load_missing_file_is_invalid(tmp_path) -> None:
    with pytest.raises(StackedRegressorInvalid):
        load_stacked_regressor(tmp_path / "does_not_exist.json")


def test_rejects_unsupported_model_type() -> None:
    regressor = _make_regressor()
    payload = regressor.to_payload()
    payload["model_type"] = "totally_made_up"
    with pytest.raises(StackedRegressorInvalid):
        RuntimeStackedLandmarkRegressor.from_payload(payload)


def test_rejects_output_dim_mismatch() -> None:
    payload = _make_regressor(output_mode=OUTPUT_MODE_GLOBAL_TRANSFORM).to_payload()
    payload["output_dim"] = 99
    with pytest.raises(StackedRegressorInvalid):
        RuntimeStackedLandmarkRegressor.from_payload(payload)


def test_rejects_coef_shape_mismatch() -> None:
    payload = _make_regressor(feature_names=("a", "b", "c")).to_payload()
    # coef rows no longer match feature_names length.
    payload["coef"] = [[0.0, 0.0, 0.0, 0.0]]
    with pytest.raises(StackedRegressorInvalid):
        RuntimeStackedLandmarkRegressor.from_payload(payload)


def test_rejects_runtime_feature_contract_mismatch() -> None:
    payload = _make_regressor().to_payload()
    payload["runtime_feature_contract_version"] = "runtime_features_v0"
    with pytest.raises(StackedRegressorInvalid):
        RuntimeStackedLandmarkRegressor.from_payload(payload)


def test_rejects_empty_feature_names() -> None:
    payload = _make_regressor().to_payload()
    payload["feature_names"] = []
    with pytest.raises(StackedRegressorInvalid):
        RuntimeStackedLandmarkRegressor.from_payload(payload)


def test_zero_output_is_identity_correction() -> None:
    base = _square_face()
    result = apply_residual(
        base,
        np.zeros(4),
        output_mode=OUTPUT_MODE_GLOBAL_TRANSFORM,
        clip_fraction=0.05,
    )
    np.testing.assert_allclose(result.landmarks.astype("float64"), base, atol=1e-5)
    assert result.residual_norm_max == pytest.approx(0.0, abs=1e-6)
    assert not result.clip_applied


def test_global_transform_translation() -> None:
    base = _square_face()
    diag = float(np.hypot(*(base.max(0) - base.min(0))))
    # tx=ty=0.01 of the bbox diagonal.
    result = apply_residual(
        base,
        np.array([0.01, 0.01, 0.0, 0.0]),
        output_mode=OUTPUT_MODE_GLOBAL_TRANSFORM,
        clip_fraction=1.0,
        bbox_diagonal=diag,
    )
    delta = result.landmarks.astype("float64") - base
    np.testing.assert_allclose(delta[:, 0], 0.01 * diag, atol=1e-4)
    np.testing.assert_allclose(delta[:, 1], 0.01 * diag, atol=1e-4)


def test_residual_clipping_caps_magnitude() -> None:
    base = _square_face()
    diag = float(np.hypot(*(base.max(0) - base.min(0))))
    # Request a translation of 0.5 * diag but clip to 0.02 * diag.
    result = apply_residual(
        base,
        np.array([0.5, 0.0, 0.0, 0.0]),
        output_mode=OUTPUT_MODE_GLOBAL_TRANSFORM,
        clip_fraction=0.02,
        bbox_diagonal=diag,
    )
    magnitudes = np.linalg.norm(result.landmarks.astype("float64") - base, axis=1)
    assert result.clip_applied
    # The internal (float64) residual norm is the precise clip check; the
    # recomputed magnitude carries float32 landmark-storage rounding.
    assert result.residual_norm_max <= 0.02 + 1e-9
    assert magnitudes.max() <= 0.02 * diag + 1e-3


def test_region_residual_broadcast() -> None:
    base = _square_face()
    diag = float(np.hypot(*(base.max(0) - base.min(0))))
    # Only the jaw region (first region) gets a dx offset.
    raw = np.zeros(sr.REGION_RESIDUAL_OUTPUT_DIM)
    raw[0] = 0.01  # jaw dx
    result = apply_residual(
        base,
        raw,
        output_mode=OUTPUT_MODE_REGION_RESIDUAL,
        clip_fraction=1.0,
        bbox_diagonal=diag,
    )
    delta = result.landmarks.astype("float64") - base
    jaw = list(sr.STACKED_REGION_INDICES["jaw"])
    np.testing.assert_allclose(delta[jaw, 0], 0.01 * diag, atol=1e-4)
    # Non-jaw points untouched.
    assert np.allclose(delta[48:68], 0.0, atol=1e-5)


def test_landmark_residual_68_applies_per_point() -> None:
    base = _square_face()
    diag = float(np.hypot(*(base.max(0) - base.min(0))))
    raw = np.zeros(sr.LANDMARK_RESIDUAL_68_OUTPUT_DIM)
    raw[0] = 0.01  # point 0, dx
    result = apply_residual(
        base,
        raw,
        output_mode=OUTPUT_MODE_LANDMARK_RESIDUAL_68,
        clip_fraction=1.0,
        bbox_diagonal=diag,
    )
    delta = result.landmarks.astype("float64") - base
    assert delta[0, 0] == pytest.approx(0.01 * diag, abs=1e-4)
    assert np.allclose(delta[1:], 0.0, atol=1e-5)


def test_predict_applies_standardization() -> None:
    regressor = _make_regressor(
        feature_names=("f0",),
        coef=np.array([[1.0, 0.0, 0.0, 0.0]]),
        intercept=np.array([0.0, 0.0, 0.0, 0.0]),
    )
    object.__setattr__(regressor, "feature_mean", np.array([2.0]))
    object.__setattr__(regressor, "feature_std", np.array([4.0]))
    # standardized = (10 - 2) / 4 = 2 -> output[0] = 2
    out = regressor.predict({"f0": 10.0})
    assert out[0] == pytest.approx(2.0)
