#!/usr/bin/env python3
"""Tests for stacked regressor training, evaluation, and promotion gates (#223)."""

from __future__ import annotations

import typing as T
from dataclasses import dataclass, field

import numpy as np
import pytest

from lib.landmarks.ensemble.runtime_resolver import CandidateRecord
from lib.landmarks.ensemble.stacked_regressor import (
    OUTPUT_MODE_GLOBAL_TRANSFORM,
    OUTPUT_MODE_LANDMARK_RESIDUAL_68,
)
from lib.landmarks.ensemble.stacked_regressor_evaluation import (
    evaluate_promotion_gates,
    evaluate_stacked_candidate,
)
from lib.landmarks.ensemble.stacked_regressor_training import (
    StackedRegressorTrainingError,
    StackedTrainingExample,
    stacked_feature_order,
    train_stacked_regressor,
)


@dataclass
class _Ctx:
    sample_id: str
    candidates: tuple[CandidateRecord, ...]
    truth_landmarks: np.ndarray
    normalizer: float
    runtime_bucket: str = "frontal"
    visibility: T.Any = None
    roll_estimate: float = 0.0
    yaw_estimate: float = 0.0
    candidate_yaw_disagreement: float = 0.0
    max_disagreement_px: float = 2.0
    hard_case_tags: tuple[str, ...] = ("frontal",)
    model_predictions_available: dict[str, bool] = field(
        default_factory=lambda: {"fan": True, "hrnet": True}
    )


def _make_contexts(
    n: int,
    *,
    bias: np.ndarray | None = None,
    seed: int = 0,
    bucket: str = "frontal",
) -> list[_Ctx]:
    bias = np.array([5.0, 3.0]) if bias is None else bias
    rng = np.random.default_rng(seed)
    contexts: list[_Ctx] = []
    for i in range(n):
        gt = (rng.random((68, 2)) * 100.0 + 50.0).astype("float32")
        fan = (gt + rng.normal(0, 0.5, (68, 2))).astype("float32")
        hrnet = (gt + rng.normal(0, 0.5, (68, 2))).astype("float32")
        base = (gt + bias).astype("float32")
        candidates = (
            CandidateRecord("fan", fan, False, ("fan",)),
            CandidateRecord("hrnet", hrnet, False, ("hrnet",)),
            CandidateRecord("static_weighted", base, True, ("fan", "hrnet")),
        )
        contexts.append(
            _Ctx(
                sample_id=f"s{i}",
                candidates=candidates,
                truth_landmarks=gt,
                normalizer=100.0,
                runtime_bucket=bucket,
                hard_case_tags=(bucket,),
            )
        )
    return contexts


def test_train_reduces_eval_error_below_baseline() -> None:
    contexts = _make_contexts(48)
    regressor, metrics = train_stacked_regressor(
        contexts,
        output_mode=OUTPUT_MODE_GLOBAL_TRANSFORM,
        eval_fraction=0.25,
    )
    assert metrics["n_train"] > 0
    assert metrics["n_eval"] > 0
    assert metrics["eval_mse"] <= metrics["baseline_eval_mse"]
    assert regressor.output_mode == OUTPUT_MODE_GLOBAL_TRANSFORM
    assert regressor.training_metadata["n_examples"] == 48


def test_trained_candidate_improves_nme_and_passes_gates() -> None:
    contexts = _make_contexts(48)
    regressor, _ = train_stacked_regressor(contexts, output_mode=OUTPUT_MODE_GLOBAL_TRANSFORM)
    report = evaluate_stacked_candidate(contexts, regressor)
    assert report.overall["count"] == 48
    assert report.overall["corrected_nme_median"] < report.overall["base_nme_median"]
    assert report.overall["win_rate"] > 0.8
    gate = evaluate_promotion_gates(report)
    assert gate.passed, gate.reasons


def test_split_is_by_identity_no_leakage() -> None:
    contexts = _make_contexts(40)
    _, metrics = train_stacked_regressor(contexts, eval_fraction=0.25)
    eval_ids = set(metrics["eval_sample_ids"])
    all_ids = {ctx.sample_id for ctx in contexts}
    train_ids = all_ids - eval_ids
    # No sample identity appears in both splits.
    assert eval_ids.isdisjoint(train_ids)
    assert eval_ids


def test_landmark_residual_68_is_gated() -> None:
    contexts = _make_contexts(10)
    with pytest.raises(StackedRegressorTrainingError):
        train_stacked_regressor(contexts, output_mode=OUTPUT_MODE_LANDMARK_RESIDUAL_68)
    # Explicit opt-in is accepted.
    regressor, _ = train_stacked_regressor(
        contexts,
        output_mode=OUTPUT_MODE_LANDMARK_RESIDUAL_68,
        allow_landmark_residual_68=True,
    )
    assert regressor.output_mode == OUTPUT_MODE_LANDMARK_RESIDUAL_68


def test_no_usable_examples_raises() -> None:
    with pytest.raises(StackedRegressorTrainingError):
        train_stacked_regressor([], output_mode=OUTPUT_MODE_GLOBAL_TRANSFORM)


def test_feature_order_rejects_offline_only_fields() -> None:
    example = StackedTrainingExample(
        sample_id="s0",
        base_candidate="static_weighted",
        features={"base_centroid_x_norm": 0.5, "candidate_nme": 0.1},
        target=np.zeros(4),
    )
    with pytest.raises(StackedRegressorTrainingError):
        stacked_feature_order([example])


def test_promotion_gate_blocks_frontal_regression() -> None:
    """A report with a frontal NME regression fails the gate."""
    contexts = _make_contexts(40)
    regressor, _ = train_stacked_regressor(contexts)
    report = evaluate_stacked_candidate(contexts, regressor)
    # Inject a frontal regression and a catastrophic increase.
    report.by_slice["frontal"]["nme_median_change"] = 0.01
    report.by_slice["frontal"]["catastrophic_rate_change"] = 0.2
    gate = evaluate_promotion_gates(report)
    assert not gate.passed
    assert any("frontal" in reason for reason in gate.reasons)
