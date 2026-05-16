#!/usr/bin/env python3
"""Unit tests for :mod:`lib.landmarks.eval.candidate_search` (issue #69)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.ensemble.weight_generators import GENERATOR_NAMES
from lib.landmarks.ensemble.weights import LANDMARK_COUNT
from lib.landmarks.eval.candidate_search import (
    DEFAULT_OBJECTIVE,
    Candidate,
    candidate_id,
    enumerate_candidates,
    evaluate_candidate,
    expand_model_subsets,
    load_split_samples,
    run_candidate_search,
    score_v1,
    select_best_candidate,
    weights_hash,
)
from lib.landmarks.eval.prediction_cache import DiskPredictionCache
from lib.landmarks.eval.splits import SplitAssignment
from lib.landmarks.schema import LandmarkPrediction

MODELS = ("hrnet", "spiga", "orformer")


def _truth_points() -> np.ndarray:
    return np.stack(
        (
            np.linspace(20.0, 70.0, LANDMARK_COUNT, dtype="float32"),
            np.linspace(25.0, 75.0, LANDMARK_COUNT, dtype="float32"),
        ),
        axis=1,
    )


def _prediction(noise: float) -> np.ndarray:
    return _truth_points() + noise


def _write_manifest_and_cache(
    tmp_path: Path,
    *,
    samples_per_bucket: dict[str, int],
    model_noises: dict[str, float],
) -> tuple[Path, Path, list[str]]:
    """Build a manifest, ground-truth file, and prediction cache for the tests."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    truth_path = dataset_dir / "truth.npy"
    np.save(str(truth_path), _truth_points())
    cache_dir = tmp_path / "cache"
    cache = DiskPredictionCache(cache_dir)
    samples: list[dict[str, str]] = []
    ordered_ids: list[str] = []
    for bucket, count in samples_per_bucket.items():
        dataset, condition = bucket.split(":", 1)
        for idx in range(count):
            sid = f"{bucket}-{idx:03d}"
            ordered_ids.append(sid)
            samples.append(
                {
                    "sample_id": sid,
                    "dataset": dataset,
                    "condition": condition,
                    "image": "image.png",
                    "landmarks": "truth.npy",
                }
            )
            for model, noise in model_noises.items():
                cache.write(sid, LandmarkPrediction(_prediction(noise), model_name=model))
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"samples": samples}) + "\n", encoding="utf-8")
    return manifest_path, cache_dir, ordered_ids


def _equal_split(ids: list[str]) -> SplitAssignment:
    cut1 = len(ids) // 3
    cut2 = 2 * len(ids) // 3
    return SplitAssignment(
        fit=tuple(ids[:cut1]),
        select=tuple(ids[cut1:cut2]),
        report=tuple(ids[cut2:]),
    )


def test_expand_model_subsets_supports_all_pairs_triples() -> None:
    """Subset presets expand into the correct combinations."""
    subsets = expand_model_subsets(MODELS, ("all", "pairs"))
    assert (MODELS[0], MODELS[1], MODELS[2]) in subsets
    assert any(set(subset) == {"hrnet", "spiga"} for subset in subsets)
    assert any(set(subset) == {"spiga", "orformer"} for subset in subsets)
    triples = expand_model_subsets(MODELS, ("triples",))
    assert len(triples) == 1
    assert set(triples[0]) == set(MODELS)


def test_expand_model_subsets_rejects_unknown_preset() -> None:
    """Unknown subset preset names must raise with the supported list."""
    with pytest.raises(ValueError, match="unknown"):
        expand_model_subsets(MODELS, ("quadruples",))


def test_enumerate_candidates_skips_threshold_for_non_threshold_strategies() -> None:
    """Threshold-free strategies appear once even with multiple --outlier-thresholds."""
    candidates = enumerate_candidates(
        models=MODELS,
        model_subset_presets=("all",),
        weight_generators=("inverse_mean_error",),
        strategies=("static_weighted", "static_weighted_hard_drop"),
        outlier_thresholds=(2.5, 3.5),
    )
    static_weighted = [c for c in candidates if c.strategy == "static_weighted"]
    hard_drop = [c for c in candidates if c.strategy == "static_weighted_hard_drop"]
    assert len(static_weighted) == 1
    assert static_weighted[0].outlier_threshold is None
    assert len(hard_drop) == 2
    assert {c.outlier_threshold for c in hard_drop} == {2.5, 3.5}


def test_enumerate_candidates_requires_threshold_when_strategy_uses_it() -> None:
    """A threshold-using strategy with no thresholds supplied raises clearly."""
    with pytest.raises(ValueError, match="outlier-thresholds"):
        enumerate_candidates(
            models=MODELS,
            model_subset_presets=("all",),
            weight_generators=("inverse_mean_error",),
            strategies=("static_weighted_downweight",),
            outlier_thresholds=(),
        )


def test_candidate_id_is_stable_and_includes_split_and_objective_hash() -> None:
    """Same inputs yield the same hash; changing splits or objective invalidates it."""
    candidate = Candidate(
        models=MODELS,
        weight_generator="inverse_mean_error",
        strategy="static_weighted",
    )
    a = candidate_id(
        candidate,
        weights_hash="sha256:abc",
        split_assignment_hash="sha256:111",
        objective="extract_alignment_v1",
    )
    b = candidate_id(
        candidate,
        weights_hash="sha256:abc",
        split_assignment_hash="sha256:111",
        objective="extract_alignment_v1",
    )
    diff_split = candidate_id(
        candidate,
        weights_hash="sha256:abc",
        split_assignment_hash="sha256:222",
        objective="extract_alignment_v1",
    )
    diff_objective = candidate_id(
        candidate,
        weights_hash="sha256:abc",
        split_assignment_hash="sha256:111",
        objective="extract_alignment_v2",
    )
    assert a == b
    assert a.startswith("sha256:")
    assert a != diff_split
    assert a != diff_objective


def test_score_v1_combines_components_with_documented_coefficients() -> None:
    """Objective v1 formula: nme + 0.5*failure + 0.25*sample_reg + 0.25*bucket_reg."""
    from lib.landmarks.eval.candidate_search import CandidateMetrics

    metrics = CandidateMetrics(
        sample_count=10,
        overall_nme=0.03,
        failure_rate=0.10,
        auc=0.5,
        regression_rate_vs_best_single=0.20,
        bucket_regression_rate_vs_best_single=0.04,
        per_bucket={},
        best_single_model="hrnet",
    )
    expected = 0.03 + 0.5 * 0.10 + 0.25 * 0.20 + 0.25 * 0.04
    assert score_v1(metrics) == pytest.approx(expected)


def test_evaluate_candidate_returns_metrics_and_fit_diagnostics(tmp_path: Path) -> None:
    """A single candidate evaluation surfaces NME, failure rate, and regression rate."""
    manifest, cache_dir, ids = _write_manifest_and_cache(
        tmp_path,
        samples_per_bucket={"fixture:clean": 6, "fixture:occluded": 6},
        model_noises={"hrnet": 0.5, "spiga": 1.0, "orformer": 2.0},
    )
    assignment = _equal_split(ids)
    cache = DiskPredictionCache(cache_dir)
    fit = load_split_samples(manifest, assignment, "fit")
    select = load_split_samples(manifest, assignment, "select")

    candidate = Candidate(
        models=MODELS,
        weight_generator="inverse_mean_error",
        strategy="static_weighted",
    )
    fit_result, metrics = evaluate_candidate(
        candidate, fit_samples=fit, select_samples=select, cache=cache
    )

    assert metrics.sample_count == len(select)
    assert metrics.best_single_model == "hrnet"
    assert metrics.overall_nme > 0
    assert "mean_errors" in fit_result.diagnostics


def test_run_candidate_search_orders_results_by_score(tmp_path: Path) -> None:
    """run_candidate_search ranks candidates by ascending objective score."""
    manifest, cache_dir, ids = _write_manifest_and_cache(
        tmp_path,
        samples_per_bucket={"fixture:clean": 9, "fixture:occluded": 9},
        model_noises={"hrnet": 0.5, "spiga": 1.0, "orformer": 2.5},
    )
    assignment = _equal_split(ids)
    cache = DiskPredictionCache(cache_dir)
    fit = load_split_samples(manifest, assignment, "fit")
    select = load_split_samples(manifest, assignment, "select")

    candidates = enumerate_candidates(
        models=MODELS,
        model_subset_presets=("all",),
        weight_generators=("equal", "inverse_mean_error"),
        strategies=("static_weighted",),
        outlier_thresholds=(),
    )
    results = run_candidate_search(
        candidates,
        fit_samples=fit,
        select_samples=select,
        cache=cache,
        split_assignment_hash="sha256:test",
    )

    assert len(results) == len(candidates)
    scores = [result.score for result in results]
    assert scores == sorted(scores)
    winner = select_best_candidate(results)
    assert winner.score == scores[0]


def test_run_candidate_search_handles_threshold_aware_strategies(tmp_path: Path) -> None:
    """Threshold-aware strategies emit one candidate per supplied threshold."""
    manifest, cache_dir, ids = _write_manifest_and_cache(
        tmp_path,
        samples_per_bucket={"fixture:clean": 9, "fixture:occluded": 9},
        model_noises={"hrnet": 0.5, "spiga": 1.0, "orformer": 2.5},
    )
    assignment = _equal_split(ids)
    cache = DiskPredictionCache(cache_dir)
    fit = load_split_samples(manifest, assignment, "fit")
    select = load_split_samples(manifest, assignment, "select")

    candidates = enumerate_candidates(
        models=MODELS,
        model_subset_presets=("all",),
        weight_generators=("inverse_mean_error",),
        strategies=("static_weighted_downweight",),
        outlier_thresholds=(2.5, 3.5),
    )
    results = run_candidate_search(
        candidates,
        fit_samples=fit,
        select_samples=select,
        cache=cache,
        split_assignment_hash="sha256:test",
    )

    assert {result.candidate.outlier_threshold for result in results} == {2.5, 3.5}


def test_weights_hash_is_stable_for_identical_weights() -> None:
    """weights_hash treats identical numeric content as identical."""
    a = {
        "hrnet": [1.0 / 3] * LANDMARK_COUNT,
        "spiga": [1.0 / 3] * LANDMARK_COUNT,
        "orformer": [1.0 / 3] * LANDMARK_COUNT,
    }
    b = {
        "hrnet": [1.0 / 3] * LANDMARK_COUNT,
        "spiga": [1.0 / 3] * LANDMARK_COUNT,
        "orformer": [1.0 / 3] * LANDMARK_COUNT,
    }
    assert weights_hash(a) == weights_hash(b)
    c = dict(a)
    c["hrnet"] = [0.5] * LANDMARK_COUNT
    assert weights_hash(a) != weights_hash(c)


def test_default_objective_constant_pins_extract_alignment_v1() -> None:
    """The default objective name is stable; changing it breaks candidate IDs."""
    assert DEFAULT_OBJECTIVE == "extract_alignment_v1"


def test_candidate_rejects_invalid_threshold_for_strategy() -> None:
    """Constructing a Candidate with a threshold for a non-threshold strategy fails."""
    with pytest.raises(ValueError):
        Candidate(
            models=MODELS,
            weight_generator="equal",
            strategy="static_weighted",
            outlier_threshold=3.5,
        )


def test_candidate_rejects_unknown_generator() -> None:
    """A Candidate referencing an unknown generator fails at construction."""
    with pytest.raises(ValueError):
        Candidate(models=MODELS, weight_generator="made_up_gen", strategy="static_weighted")


def test_candidate_rejects_duplicate_models() -> None:
    """Models list must be unique."""
    with pytest.raises(ValueError):
        Candidate(models=("hrnet", "hrnet"), weight_generator="equal", strategy="static_weighted")


def test_enumerate_candidates_validates_generator_names() -> None:
    """Unknown generator names are rejected during enumeration."""
    with pytest.raises(ValueError):
        enumerate_candidates(
            models=MODELS,
            model_subset_presets=("all",),
            weight_generators=("unknown_generator",),
            strategies=("static_weighted",),
            outlier_thresholds=(),
        )


def test_generator_registry_lists_at_least_three_options() -> None:
    """The default search space contract relies on a non-trivial generator registry."""
    assert {"equal", "inverse_mean_error", "regularized_inverse_error"}.issubset(GENERATOR_NAMES)


def test_single_model_baselines_can_be_appended_via_enumerate(tmp_path: Path) -> None:
    """``include_single_model_baselines=True`` appends one plain_average per model."""
    candidates = enumerate_candidates(
        models=MODELS,
        model_subset_presets=("all",),
        weight_generators=("inverse_mean_error",),
        strategies=("static_weighted",),
        outlier_thresholds=(),
        include_single_model_baselines=True,
    )

    baselines = [
        c
        for c in candidates
        if len(c.models) == 1 and c.strategy == "plain_average"
    ]
    assert {b.models[0] for b in baselines} == set(MODELS)
    # No duplicates when a baseline already exists in the regular enumeration.
    again = enumerate_candidates(
        models=MODELS,
        model_subset_presets=("all",),
        weight_generators=("inverse_mean_error",),
        strategies=("plain_average",),
        outlier_thresholds=(),
        include_single_model_baselines=True,
    )
    seen = {(tuple(sorted(c.models)), c.strategy) for c in again}
    assert len(again) == len(seen)


def test_run_candidate_search_attaches_effective_ensemble_diagnostics(tmp_path: Path) -> None:
    """Search results carry effective-ensemble diagnostics for every candidate."""
    manifest, cache_dir, ids = _write_manifest_and_cache(
        tmp_path,
        samples_per_bucket={"fixture:clean": 9, "fixture:occluded": 9},
        model_noises={"hrnet": 0.5, "spiga": 1.0, "orformer": 2.5},
    )
    assignment = _equal_split(ids)
    cache = DiskPredictionCache(cache_dir)
    fit = load_split_samples(manifest, assignment, "fit")
    select = load_split_samples(manifest, assignment, "select")

    candidates = enumerate_candidates(
        models=MODELS,
        model_subset_presets=("all",),
        weight_generators=("equal", "inverse_mean_error"),
        strategies=("static_weighted",),
        outlier_thresholds=(),
        include_single_model_baselines=True,
    )
    results = run_candidate_search(
        candidates,
        fit_samples=fit,
        select_samples=select,
        cache=cache,
        split_assignment_hash="sha256:test",
    )

    for result in results:
        assert result.effective_ensemble is not None
    # Single-model baselines must self-identify and self-flag as collapsed.
    baselines = [result for result in results if result.is_single_model_baseline]
    assert {result.candidate.models[0] for result in baselines} == set(MODELS)
    for baseline in baselines:
        assert baseline.effective_ensemble.collapsed is True
