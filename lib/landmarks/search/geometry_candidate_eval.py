#!/usr/bin/env python3
"""Geometry-candidate evaluation primitives (Ticket 2).

The search-ensemble-setup CLI used to own the geometry-context cache, the
per-candidate aggregate builder, and the score-packaging helper inline.
Moving them here keeps the select-vs-report split testable in the library
instead of buried in a CLI, and gives Ticket 3 (shared fusion helpers) and
Ticket 4 (move candidate_search out of eval) a stable seam to land on.

Public surface:

* :class:`GeometryContextRow` — typed per-sample bundle (truth + summary +
  bbox + per-model cached predictions).
* :func:`build_geometry_context` — preloads the per-sample bundles once so
  candidate iteration doesn't re-read npy / rebuild AlignedFace per pair.
* :func:`evaluate_candidate_geometry` — fuses a candidate against the
  context and aggregates per-sample geometry into a
  :class:`GeometryAggregate`.
* :func:`geometry_score_from_aggregate` — packs an aggregate into the
  :class:`GeometryScore` shape the promotion-gate framework consumes.
* :func:`fuse_candidate` — the candidate-aware fusion shim used by the
  geometry evaluator. Ticket 3 will consolidate this with the other
  ``_fuse_variant`` clones into a single canonical helper.
"""

from __future__ import annotations

import typing as T
from dataclasses import dataclass

import numpy as np

from lib.landmarks.eval.geometry_metrics import (
    GeometryAggregate,
    aggregate_geometry_samples,
    evaluate_geometry_sample,
)
from lib.landmarks.eval.geometry_signals import AlignmentSummary, alignment_summary
from lib.landmarks.eval.prediction_cache import DiskPredictionCache
from lib.landmarks.manifest import LandmarkSample, bbox_for_sample
from lib.landmarks.search.candidate_search import CandidateResult
from lib.landmarks.search.fusion_variants import fuse_candidate
from lib.landmarks.search.promotion_gates import GeometryScore


def _crop_scale_key(value: float) -> float:
    """Normalize a crop_scale value into a stable dict key.

    Floats round-tripped through argparse / JSON can pick up tiny mantissa
    drift; the search enumerates a small discrete set of crop scales so a
    coarse rounding to 6 decimal places is plenty to deduplicate while
    surviving accidental ``1.5999999999`` style inputs.
    """
    return round(float(value), 6)


@dataclass(frozen=True)
class GeometryContextRow:
    """Per-sample bundle reused across every candidate the geometry stage scores.

    ``truth_summary_by_crop_scale`` holds one precomputed ``AlignmentSummary``
    per searched crop scale so the geometry evaluator can pick the summary
    matching the candidate's ``crop_scale`` without rebuilding ``AlignedFace``
    on every (sample, candidate) pair. Keys are rounded via
    :func:`_crop_scale_key`.
    """

    sample: LandmarkSample
    truth: np.ndarray
    truth_summary_by_crop_scale: dict[float, AlignmentSummary]
    bbox: tuple[float, float, float, float]
    predictions: dict[str, np.ndarray]


def build_geometry_context(
    samples: T.Sequence[LandmarkSample],
    *,
    cache: DiskPredictionCache,
    models: T.Sequence[str],
    aligned_size: int,
    crop_scales: T.Sequence[float] = (1.0,),
) -> list[GeometryContextRow]:
    """Preload truth landmarks, AlignedFace summary, bbox, and cached predictions.

    Building each :class:`GeometryContextRow` once amortizes the ``AlignedFace``
    (Umeyama + solvePnP) cost across every candidate scored against the row.
    Samples whose truth file is unreadable or whose bbox cannot be resolved
    are skipped silently — callers ought to surface that earlier in the
    pipeline (typically the manifest loader).

    ``crop_scales`` is the set of crop scales the search will sweep over.
    Each row precomputes one truth ``AlignmentSummary`` per crop scale so
    :func:`evaluate_candidate_geometry` can score every candidate in its
    own coverage frame without rebuilding the GT summary per pair.
    """
    unique_crop_scales = list(dict.fromkeys(_crop_scale_key(scale) for scale in crop_scales))
    if not unique_crop_scales:
        unique_crop_scales = [_crop_scale_key(1.0)]
    rows: list[GeometryContextRow] = []
    for sample in samples:
        try:
            truth = np.load(sample.landmarks).astype("float32")
        except OSError:
            continue
        bbox = bbox_for_sample(sample, allow_truth_fallback=True)
        if bbox is None:
            continue
        rows.append(
            GeometryContextRow(
                sample=sample,
                truth=truth,
                truth_summary_by_crop_scale={
                    scale: alignment_summary(truth, size=aligned_size, coverage_ratio=scale)
                    for scale in unique_crop_scales
                },
                bbox=bbox,
                predictions={
                    model: cache.read(sample.sample_id, model).landmarks for model in models
                },
            )
        )
    return rows


# ``fuse_candidate`` lives in lib.landmarks.search.fusion_variants now; the
# import above gives this module a backwards-compatible alias.


def evaluate_candidate_geometry(
    result: CandidateResult,
    *,
    context: T.Sequence[GeometryContextRow],
    aligned_size: int,
    region_failure_threshold: float,
) -> GeometryAggregate:
    """Fuse one candidate against every context row and aggregate the geometry metrics.

    ``context`` is built once per search-stage invocation (see
    :func:`build_geometry_context`); every candidate scored shares the same
    rows so we never rebuild GT AlignedFace or re-read GT npy per pair. The
    candidate's ``crop_scale`` selects which precomputed truth summary to
    consume — the row falls back to building a fresh summary if the search
    asks for a crop scale that wasn't precached (unit tests, ad-hoc calls).
    """
    crop_scale = float(result.candidate.crop_scale)
    key = _crop_scale_key(crop_scale)
    per_sample: list[T.Any] = []
    for row in context:
        cached_points = [row.predictions[model] for model in result.candidate.models]
        fused = fuse_candidate(result.candidate, cached_points, weights=result.weights)
        truth_summary = row.truth_summary_by_crop_scale.get(key)
        if truth_summary is None:
            truth_summary = alignment_summary(
                row.truth, size=aligned_size, coverage_ratio=crop_scale
            )
        per_sample.append(
            evaluate_geometry_sample(
                fused,
                row.truth,
                sample_id=row.sample.sample_id,
                dataset=row.sample.dataset,
                condition=row.sample.condition,
                bbox=row.bbox,
                visibility=row.sample.visibility,
                aligned_size=aligned_size,
                region_failure_threshold=region_failure_threshold,
                truth_summary=truth_summary,
                crop_scale=crop_scale,
            )
        )
    return aggregate_geometry_samples(result.candidate_id, per_sample)


def geometry_score_from_aggregate(
    aggregate: GeometryAggregate,
    *,
    baseline_score: float | None,
    baseline_per_bucket: T.Mapping[str, float] | None = None,
) -> GeometryScore:
    """Pack a :class:`GeometryAggregate` into the gate framework's score shape.

    ``baseline_score`` is the lowest single-model overall_score across the
    same context; bucket scores above it become the
    ``max_bucket_regression_score`` the hard-slice gate consumes.

    ``baseline_per_bucket`` lets the caller supply per-bucket baseline scores
    (e.g., the best single-model score per scenario bucket) so the worst-
    bucket regression is reported against the bucket-specific reference
    rather than the global minimum. When omitted, ``baseline_score`` is used
    as the reference for every bucket.
    """
    worst_bucket = ""
    worst_bucket_score = 0.0
    worst_bucket_baseline = 0.0
    max_bucket_regression = 0.0
    if aggregate.per_bucket and baseline_score is not None:
        for bucket, values in aggregate.per_bucket.items():
            bucket_score = float(values.get("overall_score", 0.0))
            bucket_baseline = (
                float(baseline_per_bucket.get(bucket, baseline_score))
                if baseline_per_bucket is not None
                else baseline_score
            )
            regression = max(0.0, bucket_score - bucket_baseline)
            if regression > max_bucket_regression or worst_bucket == "":
                worst_bucket = bucket
                worst_bucket_score = bucket_score
                worst_bucket_baseline = bucket_baseline
                if regression > max_bucket_regression:
                    max_bucket_regression = regression
    return GeometryScore(
        overall_score=aggregate.overall_score,
        catastrophic_failure_rate=aggregate.catastrophic_failure_rate,
        p95_translation_normalized=aggregate.p95_translation_normalized,
        p95_roi_center_normalized=aggregate.p95_roi_center_normalized,
        p95_roll_degrees=aggregate.p95_roll_degrees_delta,
        mean_hull_iou=aggregate.mean_hull_iou,
        p05_hull_iou=aggregate.p05_hull_iou,
        max_bucket_regression_score=max_bucket_regression,
        worst_bucket=worst_bucket,
        worst_bucket_score=worst_bucket_score,
        worst_bucket_baseline_score=worst_bucket_baseline,
        per_bucket={bucket: dict(values) for bucket, values in aggregate.per_bucket.items()}
        if aggregate.per_bucket
        else None,
    )


__all__ = [
    "GeometryContextRow",
    "build_geometry_context",
    "evaluate_candidate_geometry",
    "fuse_candidate",
    "geometry_score_from_aggregate",
]
