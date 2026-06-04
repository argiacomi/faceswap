#!/usr/bin/env python3
"""Integration tests for the stacked residual candidate inside the runtime resolver."""

from __future__ import annotations

import numpy as np

from lib.landmarks.ensemble.runtime_resolver import (
    ModelPrediction,
    RuntimeResolverConfig,
    resolve_runtime,
)
from lib.landmarks.ensemble.stacked_regressor import (
    OUTPUT_MODE_GLOBAL_TRANSFORM,
    RuntimeStackedLandmarkRegressor,
    output_dim_for_mode,
)
from tests.lib.landmarks.ensemble.scorer_test_utils import LinearTestScorer


def _face(jitter: float = 0.0) -> np.ndarray:
    points: np.ndarray = np.zeros((68, 2), dtype="float32")
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
    jittered: np.ndarray = points + jitter
    return jittered


def _regressor(
    *,
    intercept: list[float] | None = None,
    clip: float = 0.05,
) -> RuntimeStackedLandmarkRegressor:
    dim = output_dim_for_mode(OUTPUT_MODE_GLOBAL_TRANSFORM)
    feature_names = ("base_centroid_x_norm", "base_centroid_y_norm")
    return RuntimeStackedLandmarkRegressor(
        candidate_name="stacked_residual",
        base_candidate_policy="static_weighted",
        feature_names=feature_names,
        output_mode=OUTPUT_MODE_GLOBAL_TRANSFORM,
        output_dim=dim,
        residual_clip_fraction=clip,
        coef=np.zeros((len(feature_names), dim)),
        intercept=np.zeros(dim) if intercept is None else np.asarray(intercept, dtype="float64"),
        feature_mean=np.zeros(len(feature_names)),
        feature_std=np.ones(len(feature_names)),
    )


def _predictions() -> list[ModelPrediction]:
    return [
        ModelPrediction(model="fan", landmarks=_face(0.0)),
        ModelPrediction(model="hrnet", landmarks=_face(1.5)),
    ]


def test_stacked_candidate_added_when_enabled() -> None:
    config = RuntimeResolverConfig(policy="roll_aware_veto", use_stacked_regressor=True)
    result = resolve_runtime(_predictions(), config, stacked_regressor=_regressor())
    assert result.metadata["stacked_regressor_candidate_appended"] is True
    meta = result.metadata["stacked_regressor"]
    assert meta["stacked_regressor_name"] == "stacked_residual"
    assert meta["stacked_output_mode"] == OUTPUT_MODE_GLOBAL_TRANSFORM
    assert meta["stacked_base_candidate"] in result.metadata["candidate_priority"]
    assert "stacked_residual" in result.metadata["candidate_priority"]


def test_stacked_candidate_absent_when_disabled() -> None:
    config = RuntimeResolverConfig(policy="roll_aware_veto", use_stacked_regressor=False)
    result = resolve_runtime(_predictions(), config, stacked_regressor=_regressor())
    assert result.metadata["stacked_regressor_candidate_appended"] is False
    assert "stacked_residual" not in result.metadata["candidate_priority"]


def test_stacked_candidate_absent_without_regressor() -> None:
    config = RuntimeResolverConfig(policy="roll_aware_veto", use_stacked_regressor=True)
    result = resolve_runtime(_predictions(), config, stacked_regressor=None)
    assert result.metadata["stacked_regressor_candidate_appended"] is False
    assert "stacked_residual" not in result.metadata["candidate_priority"]


def test_runaway_stacked_correction_is_vetoed_not_selected() -> None:
    """A huge unclipped correction must be geometry-vetoed and never selected."""
    config = RuntimeResolverConfig(
        policy="roll_aware_veto",
        use_stacked_regressor=True,
        stacked_regressor_max_residual=0.0,  # defer to artifact clip
    )
    # Translate every point by ~10 bbox-diagonals; clip is effectively disabled.
    regressor = _regressor(intercept=[10.0, 10.0, 0.0, 0.0], clip=100.0)
    result = resolve_runtime(_predictions(), config, stacked_regressor=regressor)
    assert result.metadata["stacked_regressor_candidate_appended"] is True
    assert "stacked_residual" in result.metadata["vetoed"]
    assert result.selected_candidate != "stacked_residual"


def test_stacked_candidate_participates_in_learned_scoring() -> None:
    scorer = LinearTestScorer(
        features=("candidate_is_stacked_residual",),
        coefficients=(-1.0,),
    )
    config = RuntimeResolverConfig(policy="learned_quality_v3", use_stacked_regressor=True)
    result = resolve_runtime(
        _predictions(),
        config,
        preloaded_scorer=scorer,
        stacked_regressor=_regressor(),
    )
    scores = result.metadata["candidate_scores"]
    assert "stacked_residual" in scores
    # Negative coefficient on the stacked flag makes it the lowest-cost candidate.
    assert result.selected_candidate == "stacked_residual"


def test_stacked_clip_applied_metadata_reports_clipping() -> None:
    config = RuntimeResolverConfig(
        policy="roll_aware_veto",
        use_stacked_regressor=True,
        stacked_regressor_max_residual=0.01,
    )
    regressor = _regressor(intercept=[0.5, 0.0, 0.0, 0.0], clip=0.05)
    result = resolve_runtime(_predictions(), config, stacked_regressor=regressor)
    assert result.metadata["stacked_regressor"]["stacked_clip_applied"] is True
