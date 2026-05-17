#!/usr/bin/env python3
"""Tests for :mod:`lib.landmarks.search.geometry_candidate_eval` (Ticket 2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.eval.candidate_search import Candidate, CandidateResult
from lib.landmarks.eval.geometry_metrics import GeometryAggregate
from lib.landmarks.eval.prediction_cache import DiskPredictionCache
from lib.landmarks.manifest import load_manifest
from lib.landmarks.schema import LandmarkPrediction
from lib.landmarks.search.geometry_candidate_eval import (
    GeometryContextRow,
    build_geometry_context,
    evaluate_candidate_geometry,
    fuse_candidate,
    geometry_score_from_aggregate,
)


def _truth_face() -> np.ndarray:
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


def _build_fixture(tmp_path: Path, models: tuple[str, ...] = ("hrnet", "spiga")) -> dict:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    truth = _truth_face()
    np.save(str(dataset / "truth.npy"), truth)
    cache_dir = tmp_path / "cache"
    cache = DiskPredictionCache(cache_dir)
    samples = []
    for idx in range(3):
        sid = f"face-{idx:03d}"
        samples.append(
            {
                "sample_id": sid,
                "dataset": "fixture",
                "condition": "clean" if idx < 2 else "profile",
                "image": "image.png",
                "landmarks": "truth.npy",
                "metadata": {"face_bbox": [40.0, 60.0, 160.0, 150.0]},
            }
        )
        for noise_idx, model in enumerate(models):
            cache.write(sid, LandmarkPrediction(truth + float(noise_idx) * 0.5, model_name=model))
    manifest_path = dataset / "manifest.json"
    manifest_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    return {"manifest": manifest_path, "cache": cache_dir}


def _candidate_result(
    *, models: tuple[str, ...], strategy: str = "static_weighted"
) -> CandidateResult:
    candidate = Candidate(
        models=models,
        weight_generator="equal",
        weight_generator_params=(),
        strategy=strategy,
        outlier_threshold=None,
        bbox_source="manifest",
        crop_scale=1.6,
    )
    weights = {model: [1.0 / len(models)] * 68 for model in models}
    return CandidateResult(
        candidate=candidate,
        candidate_id=f"sha256:{strategy}-{'-'.join(models)}",
        weights=weights,
        weights_hash="sha256:test",
        score=0.0,
        objective="extract_alignment_v1",
        regression_epsilon_nme=0.001,
        metrics=None,  # type: ignore[arg-type]
        fit_diagnostics={},
        effective_ensemble=None,
        is_single_model_baseline=len(models) == 1,
    )


def test_build_geometry_context_includes_truth_summary_per_sample(tmp_path: Path) -> None:
    """Each context row carries truth, bbox, summary, and per-model predictions."""
    fixture = _build_fixture(tmp_path)
    samples = load_manifest(fixture["manifest"])
    cache = DiskPredictionCache(fixture["cache"])

    rows = build_geometry_context(
        samples, cache=cache, models=("hrnet", "spiga"), aligned_size=128
    )

    assert len(rows) == len(samples)
    for row in rows:
        assert isinstance(row, GeometryContextRow)
        assert row.truth.shape == (68, 2)
        assert row.bbox == (40.0, 60.0, 160.0, 150.0)
        assert set(row.predictions) == {"hrnet", "spiga"}
        # AlignedFace summary is precomputed (vs being None / lazy).
        # Default build sweeps just the canonical 1.0 crop scale; per-crop-scale
        # summaries are precomputed so the geometry evaluator can pick the right one.
        summaries = row.truth_summary_by_crop_scale
        assert summaries
        assert next(iter(summaries.values())).aligned_landmarks.shape == (68, 2)


def test_build_geometry_context_skips_missing_truth(tmp_path: Path) -> None:
    """Samples whose truth file is missing are dropped instead of crashing."""
    fixture = _build_fixture(tmp_path)
    samples = load_manifest(fixture["manifest"])
    # Synthesize a fourth sample that points at a non-existent landmarks file
    # so the context builder skips it without breaking the existing fixture.
    from lib.landmarks.manifest import LandmarkSample

    missing = LandmarkSample(
        sample_id="missing-gt",
        image="",
        landmarks=str(tmp_path / "does_not_exist.npy"),
        dataset="fixture",
        condition="clean",
        face_bbox=(40.0, 60.0, 160.0, 150.0),
    )
    cache = DiskPredictionCache(fixture["cache"])

    rows = build_geometry_context(
        [*samples, missing],
        cache=cache,
        models=("hrnet", "spiga"),
        aligned_size=128,
    )

    # Only the real samples produce context rows; the missing-truth one is dropped.
    assert len(rows) == len(samples)


def test_fuse_candidate_returns_canonical_68_landmarks(tmp_path: Path) -> None:
    """``fuse_candidate`` returns a ``(68, 2)`` array for valid strategies."""
    truth = _truth_face()
    result = _candidate_result(models=("hrnet", "spiga"))
    fused = fuse_candidate(result.candidate, [truth, truth + 0.5], weights=result.weights)
    assert fused.shape == (68, 2)


def test_build_geometry_context_caches_truth_summary_per_crop_scale(tmp_path: Path) -> None:
    """One truth ``AlignmentSummary`` is precached per requested crop scale.

    The crop_scale sweep would otherwise rebuild GT AlignedFace on every
    (sample, candidate) pair; caching per crop scale is what keeps the
    fanout cheap.
    """
    fixture = _build_fixture(tmp_path)
    samples = load_manifest(fixture["manifest"])
    cache = DiskPredictionCache(fixture["cache"])

    rows = build_geometry_context(
        samples,
        cache=cache,
        models=("hrnet", "spiga"),
        aligned_size=128,
        crop_scales=(1.0, 1.4, 1.8),
    )

    assert rows
    for row in rows:
        assert set(row.truth_summary_by_crop_scale) == {1.0, 1.4, 1.8}


def test_evaluate_candidate_geometry_shifts_with_crop_scale(tmp_path: Path) -> None:
    """Geometry aggregates differ when the candidate's crop scale changes.

    With landmark predictions held fixed (cached), varying ``crop_scale``
    shifts the AlignedFace ROI which feeds ``roi_delta.iou``, the per-region
    aligned-space error normalization, and the ROI diagnostics. This is the
    selection signal the crop_scale fanout relies on.
    """
    fixture = _build_fixture(tmp_path)
    samples = load_manifest(fixture["manifest"])
    cache = DiskPredictionCache(fixture["cache"])
    context = build_geometry_context(
        samples,
        cache=cache,
        models=("hrnet", "spiga"),
        aligned_size=128,
        crop_scales=(1.0, 1.6),
    )

    def _with_scale(scale: float) -> GeometryAggregate:
        result = _candidate_result(models=("hrnet", "spiga"))
        # ``Candidate`` is frozen; build a sibling with the chosen crop scale.
        candidate = Candidate(
            models=result.candidate.models,
            weight_generator=result.candidate.weight_generator,
            strategy=result.candidate.strategy,
            outlier_threshold=result.candidate.outlier_threshold,
            bbox_source=result.candidate.bbox_source,
            crop_scale=scale,
        )
        scaled_result = CandidateResult(
            candidate=candidate,
            candidate_id=f"sha256:scaled-{scale}",
            weights=result.weights,
            weights_hash=result.weights_hash,
            score=result.score,
            objective=result.objective,
            regression_epsilon_nme=result.regression_epsilon_nme,
            metrics=result.metrics,
            fit_diagnostics=result.fit_diagnostics,
            effective_ensemble=result.effective_ensemble,
            is_single_model_baseline=result.is_single_model_baseline,
        )
        return evaluate_candidate_geometry(
            scaled_result, context=context, aligned_size=128, region_failure_threshold=0.05
        )

    tight = _with_scale(1.0)
    wide = _with_scale(1.6)
    # Either the ROI IoU or the aligned-space per-region error shifts when the
    # crop scale changes (the two are coupled through AlignedFace's padding).
    # We don't pin the direction — the point is that crop_scale is no longer a
    # no-op in the scoring path.
    assert tight.mean_roi_iou != wide.mean_roi_iou or tight.overall_score != wide.overall_score


def test_evaluate_candidate_geometry_aggregates_across_context(tmp_path: Path) -> None:
    """Geometry aggregate spans every context row and surfaces per-bucket scores."""
    fixture = _build_fixture(tmp_path)
    samples = load_manifest(fixture["manifest"])
    cache = DiskPredictionCache(fixture["cache"])
    context = build_geometry_context(
        samples, cache=cache, models=("hrnet", "spiga"), aligned_size=128
    )

    result = _candidate_result(models=("hrnet", "spiga"))
    aggregate = evaluate_candidate_geometry(
        result, context=context, aligned_size=128, region_failure_threshold=0.05
    )

    assert isinstance(aggregate, GeometryAggregate)
    assert aggregate.sample_count == len(context)
    assert set(aggregate.per_bucket) == {"fixture:clean", "fixture:profile"}


def test_geometry_score_from_aggregate_no_baseline_returns_zero_regression() -> None:
    """Absent a single-model baseline, ``max_bucket_regression_score`` is zero."""
    aggregate = GeometryAggregate(
        label="ensemble",
        sample_count=1,
        overall_score=0.5,
        catastrophic_failure_rate=0.0,
        mean_scale_delta=0.0,
        mean_relative_scale_delta=0.0,
        mean_rotation_degrees_delta=0.0,
        mean_translation_normalized=0.0,
        p95_translation_normalized=0.02,
        p95_rotation_degrees_delta=0.0,
        p95_relative_scale_delta=0.0,
        p95_roll_degrees_delta=1.0,
        mean_roi_iou=0.9,
        p05_roi_iou=0.8,
        mean_roi_center_normalized=0.0,
        p95_roi_center_normalized=0.02,
        mean_hull_iou=0.85,
        p05_hull_iou=0.7,
        mean_pitch_delta_degrees=0.0,
        mean_yaw_delta_degrees=0.0,
        mean_roll_delta_degrees=0.0,
        mean_average_distance_delta=0.0,
        per_region_error={},
        per_region_failure_rate={},
        per_bucket={"fixture:profile": {"overall_score": 0.9}},
    )
    score = geometry_score_from_aggregate(aggregate, baseline_score=None)
    assert score.max_bucket_regression_score == pytest.approx(0.0)
    assert score.overall_score == pytest.approx(0.5)


def test_geometry_score_from_aggregate_reports_worst_bucket_regression() -> None:
    """When the baseline is supplied, worst-bucket regression is clamped at zero."""
    aggregate = GeometryAggregate(
        label="ensemble",
        sample_count=1,
        overall_score=0.20,
        catastrophic_failure_rate=0.0,
        mean_scale_delta=0.0,
        mean_relative_scale_delta=0.0,
        mean_rotation_degrees_delta=0.0,
        mean_translation_normalized=0.0,
        p95_translation_normalized=0.0,
        p95_rotation_degrees_delta=0.0,
        p95_relative_scale_delta=0.0,
        p95_roll_degrees_delta=0.0,
        mean_roi_iou=1.0,
        p05_roi_iou=1.0,
        mean_roi_center_normalized=0.0,
        p95_roi_center_normalized=0.0,
        mean_hull_iou=1.0,
        p05_hull_iou=1.0,
        mean_pitch_delta_degrees=0.0,
        mean_yaw_delta_degrees=0.0,
        mean_roll_delta_degrees=0.0,
        mean_average_distance_delta=0.0,
        per_region_error={},
        per_region_failure_rate={},
        per_bucket={
            "fixture:clean": {"overall_score": 0.05},
            "fixture:profile": {"overall_score": 0.25},
        },
    )
    score = geometry_score_from_aggregate(aggregate, baseline_score=0.10)
    # Worst bucket (0.25) exceeds baseline (0.10) by 0.15.
    assert score.max_bucket_regression_score == pytest.approx(0.15)


def test_geometry_score_from_aggregate_zero_clamp_when_no_regression() -> None:
    aggregate = GeometryAggregate(
        label="ensemble",
        sample_count=1,
        overall_score=0.05,
        catastrophic_failure_rate=0.0,
        mean_scale_delta=0.0,
        mean_relative_scale_delta=0.0,
        mean_rotation_degrees_delta=0.0,
        mean_translation_normalized=0.0,
        p95_translation_normalized=0.0,
        p95_rotation_degrees_delta=0.0,
        p95_relative_scale_delta=0.0,
        p95_roll_degrees_delta=0.0,
        mean_roi_iou=1.0,
        p05_roi_iou=1.0,
        mean_roi_center_normalized=0.0,
        p95_roi_center_normalized=0.0,
        mean_hull_iou=1.0,
        p05_hull_iou=1.0,
        mean_pitch_delta_degrees=0.0,
        mean_yaw_delta_degrees=0.0,
        mean_roll_delta_degrees=0.0,
        mean_average_distance_delta=0.0,
        per_region_error={},
        per_region_failure_rate={},
        per_bucket={"fixture:clean": {"overall_score": 0.05}},
    )
    # Every bucket is at or below baseline → no regression.
    score = geometry_score_from_aggregate(aggregate, baseline_score=0.10)
    assert score.max_bucket_regression_score == pytest.approx(0.0)


def _aggregate_with_buckets(per_bucket: dict[str, dict[str, float]]) -> GeometryAggregate:
    return GeometryAggregate(
        label="ensemble",
        sample_count=1,
        overall_score=0.2,
        catastrophic_failure_rate=0.0,
        mean_scale_delta=0.0,
        mean_relative_scale_delta=0.0,
        mean_rotation_degrees_delta=0.0,
        mean_translation_normalized=0.0,
        p95_translation_normalized=0.0,
        p95_rotation_degrees_delta=0.0,
        p95_relative_scale_delta=0.0,
        p95_roll_degrees_delta=0.0,
        mean_roi_iou=1.0,
        p05_roi_iou=1.0,
        mean_roi_center_normalized=0.0,
        p95_roi_center_normalized=0.0,
        mean_hull_iou=1.0,
        p05_hull_iou=1.0,
        mean_pitch_delta_degrees=0.0,
        mean_yaw_delta_degrees=0.0,
        mean_roll_delta_degrees=0.0,
        mean_average_distance_delta=0.0,
        per_region_error={},
        per_region_failure_rate={},
        per_bucket=per_bucket,
    )


def test_geometry_score_from_aggregate_records_worst_bucket_identity() -> None:
    """``worst_bucket`` is the bucket driving ``max_bucket_regression_score``."""
    aggregate = _aggregate_with_buckets(
        {
            "ds:clean": {"overall_score": 0.05},
            "ds:profile": {"overall_score": 0.25},
        }
    )
    score = geometry_score_from_aggregate(aggregate, baseline_score=0.10)
    assert score.worst_bucket == "ds:profile"
    assert score.worst_bucket_score == pytest.approx(0.25)
    assert score.worst_bucket_baseline_score == pytest.approx(0.10)
    assert score.max_bucket_regression_score == pytest.approx(0.15)


def test_geometry_score_from_aggregate_per_bucket_baseline_overrides_scalar() -> None:
    """When per-bucket baselines are supplied, each bucket uses its own reference."""
    aggregate = _aggregate_with_buckets(
        {
            "ds:clean": {"overall_score": 0.20},
            "ds:profile": {"overall_score": 0.40},
        }
    )
    # Scalar baseline alone would flag ``ds:profile`` as worst by 0.30 over 0.10;
    # the per-bucket baseline lifts the profile reference to 0.38 (worst single
    # model on profile faces was already 0.38), so the clean bucket — which
    # regressed by 0.20 vs its 0.0 baseline — becomes the worst slice.
    score = geometry_score_from_aggregate(
        aggregate,
        baseline_score=0.10,
        baseline_per_bucket={"ds:clean": 0.0, "ds:profile": 0.38},
    )
    assert score.worst_bucket == "ds:clean"
    assert score.worst_bucket_score == pytest.approx(0.20)
    assert score.worst_bucket_baseline_score == pytest.approx(0.0)
    assert score.max_bucket_regression_score == pytest.approx(0.20)


def test_geometry_score_from_aggregate_exposes_per_bucket_map() -> None:
    """The score carries a defensive copy of the per-bucket payload for persistence."""
    per_bucket = {"ds:clean": {"overall_score": 0.05, "p95_translation_normalized": 0.02}}
    aggregate = _aggregate_with_buckets(per_bucket)
    score = geometry_score_from_aggregate(aggregate, baseline_score=0.10)
    assert score.per_bucket is not None
    assert score.per_bucket["ds:clean"]["p95_translation_normalized"] == pytest.approx(0.02)
    # Defensive copy: mutating the source should not affect the score payload.
    per_bucket["ds:clean"]["overall_score"] = 99.0
    assert score.per_bucket["ds:clean"]["overall_score"] == pytest.approx(0.05)
