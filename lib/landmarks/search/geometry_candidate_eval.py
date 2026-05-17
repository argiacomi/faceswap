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


@dataclass(frozen=True)
class GeometryContextRow:
    """Per-sample bundle reused across every candidate the geometry stage scores."""

    sample: LandmarkSample
    truth: np.ndarray
    truth_summary: AlignmentSummary
    bbox: tuple[float, float, float, float]
    predictions: dict[str, np.ndarray]


def build_geometry_context(
    samples: T.Sequence[LandmarkSample],
    *,
    cache: DiskPredictionCache,
    models: T.Sequence[str],
    aligned_size: int,
) -> list[GeometryContextRow]:
    """Preload truth landmarks, AlignedFace summary, bbox, and cached predictions.

    Building each :class:`GeometryContextRow` once amortizes the ``AlignedFace``
    (Umeyama + solvePnP) cost across every candidate scored against the row.
    Samples whose truth file is unreadable or whose bbox cannot be resolved
    are skipped silently — callers ought to surface that earlier in the
    pipeline (typically the manifest loader).
    """
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
                truth_summary=alignment_summary(truth, size=aligned_size),
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
    rows so we never rebuild GT AlignedFace or re-read GT npy per pair.
    """
    per_sample: list[T.Any] = []
    for row in context:
        cached_points = [row.predictions[model] for model in result.candidate.models]
        fused = fuse_candidate(result.candidate, cached_points, weights=result.weights)
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
                truth_summary=row.truth_summary,
            )
        )
    return aggregate_geometry_samples(result.candidate_id, per_sample)


def geometry_score_from_aggregate(
    aggregate: GeometryAggregate,
    *,
    baseline_score: float | None,
) -> GeometryScore:
    """Pack a :class:`GeometryAggregate` into the gate framework's score shape.

    ``baseline_score`` is the lowest single-model overall_score across the
    same context; bucket scores above it become the
    ``max_bucket_regression_score`` the hard-slice gate consumes.
    """
    if aggregate.per_bucket:
        max_bucket = max(
            float(values.get("overall_score", 0.0)) for values in aggregate.per_bucket.values()
        )
    else:
        max_bucket = 0.0
    max_bucket_regression = (
        max(0.0, max_bucket - baseline_score) if baseline_score is not None else 0.0
    )
    return GeometryScore(
        overall_score=aggregate.overall_score,
        catastrophic_failure_rate=aggregate.catastrophic_failure_rate,
        p95_translation_normalized=aggregate.p95_translation_normalized,
        p95_roi_center_normalized=aggregate.p95_roi_center_normalized,
        p95_roll_degrees=aggregate.p95_roll_degrees_delta,
        mean_hull_iou=aggregate.mean_hull_iou,
        p05_hull_iou=aggregate.p05_hull_iou,
        max_bucket_regression_score=max_bucket_regression,
    )


__all__ = [
    "GeometryContextRow",
    "build_geometry_context",
    "evaluate_candidate_geometry",
    "fuse_candidate",
    "geometry_score_from_aggregate",
]
