#!/usr/bin/env python3
"""Per-sample geometry context shared by every candidate the search scores.

The geometry context — truth landmarks, the precomputed truth
``AlignmentSummary`` for each searched crop scale, the resolved bbox, and
the per-model cached predictions — is alignment-decisioning data, not
search-engine data. It belongs alongside the alignment resolver because
both consume the same primitives (``AlignedFace``, ``bbox_for_sample``,
``DiskPredictionCache``); the search-side scorer just iterates the
context.
"""

from __future__ import annotations

import typing as T
from dataclasses import dataclass

import numpy as np

from lib.landmarks.cache.prediction_cache import DiskPredictionCache
from lib.landmarks.datasets.manifest_io import LandmarkSample, bbox_for_sample
from lib.landmarks.evaluation.geometry_signals import (
    AlignmentSummary,
    alignment_summary,
    visible_hull,
)


def crop_scale_key(value: float) -> float:
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
    :func:`crop_scale_key`.
    """

    sample: LandmarkSample
    truth: np.ndarray
    truth_summary_by_crop_scale: dict[float, AlignmentSummary]
    bbox: tuple[float, float, float, float]
    predictions: dict[str, np.ndarray]
    # Precomputed truth-side convex hulls reused by every candidate scored
    # against this row. ``truth_landmarks_hull`` is the full-landmark hull
    # consumed by ROI diagnostics (``aligned_crop_visible_hull_iou``);
    # ``truth_visible_hull`` honors the sample's visibility mask and is
    # consumed by :func:`visible_hull_iou`. Both are ``None`` when the
    # GT cloud has fewer than three usable points.
    truth_landmarks_hull: np.ndarray | None
    truth_visible_hull: np.ndarray | None


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
    the geometry evaluator can score every candidate in its own coverage
    frame without rebuilding the GT summary per pair.
    """
    unique_crop_scales = list(dict.fromkeys(crop_scale_key(scale) for scale in crop_scales))
    if not unique_crop_scales:
        unique_crop_scales = [crop_scale_key(1.0)]
    rows: list[GeometryContextRow] = []
    for sample in samples:
        try:
            truth = np.load(sample.landmarks).astype("float32")
        except OSError:
            continue
        bbox = bbox_for_sample(sample, allow_truth_fallback=True)
        if bbox is None:
            continue
        truth_landmarks_hull = visible_hull(truth)
        truth_visible_hull = (
            truth_landmarks_hull
            if sample.visibility is None
            else visible_hull(truth, visibility=sample.visibility)
        )
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
                truth_landmarks_hull=truth_landmarks_hull,
                truth_visible_hull=truth_visible_hull,
            )
        )
    return rows


__all__ = [
    "GeometryContextRow",
    "build_geometry_context",
    "crop_scale_key",
]
