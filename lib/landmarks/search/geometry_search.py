#!/usr/bin/env python3
"""Geometry-side candidate scoring used by the search promotion stage.

The per-sample geometry context primitives live alongside the alignment
resolver (:mod:`lib.landmarks.alignment.geometry_context`) since they are
alignment-decisioning data. This module hosts the search-engine layer
that consumes those primitives: per-candidate aggregation and the
:class:`GeometryScore` packaging the promotion-gate framework eats.

Public surface:

* :func:`evaluate_candidate_geometry` — fuses a candidate against the
  context and aggregates per-sample geometry into a
  :class:`GeometryAggregate`.
* :func:`geometry_score_from_aggregate` — packs an aggregate into the
  :class:`GeometryScore` shape the promotion-gate framework consumes.
"""

from __future__ import annotations

import typing as T

from lib.landmarks.alignment.geometry_context import (
    GeometryContextRow,
    build_geometry_context,
    crop_scale_key,
)
from lib.landmarks.core.fusion_variants import fuse_candidate
from lib.landmarks.evaluation.geometry_metrics import (
    GeometryAggregate,
    aggregate_geometry_samples,
    evaluate_geometry_sample,
)
from lib.landmarks.evaluation.geometry_signals import alignment_summary
from lib.landmarks.search.candidate_search import CandidateResult
from lib.landmarks.search.promotion_gates import GeometryScore


def evaluate_candidate_geometry(
    result: CandidateResult,
    *,
    context: T.Sequence[GeometryContextRow],
    aligned_size: int,
    region_failure_threshold: float,
) -> GeometryAggregate:
    """Fuse one candidate against every context row and aggregate the geometry metrics.

    ``context`` is built once per search-stage invocation (see
    :func:`lib.landmarks.alignment.geometry_context.build_geometry_context`);
    every candidate scored shares the same rows so we never rebuild GT
    AlignedFace or re-read GT npy per pair. The candidate's ``crop_scale``
    selects which precomputed truth summary to consume — the row falls
    back to building a fresh summary if the search asks for a crop scale
    that wasn't precached (unit tests, ad-hoc calls).
    """
    if not context:
        raise ValueError(
            "geometry candidate evaluation requires at least one context row; "
            "check the manifest, truth landmarks, bboxes, and prediction cache"
        )
    crop_scale = float(result.candidate.crop_scale)
    key = crop_scale_key(crop_scale)
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
    if aggregate.sample_count <= 0:
        raise ValueError(
            f"geometry aggregate {aggregate.label!r} has zero samples; refusing to score"
        )
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
    "geometry_score_from_aggregate",
]
