#!/usr/bin/env python3
"""GT-derived alignment-geometry metrics for extract quality (#76).

For each (predicted, truth) landmark pair this module computes the deviation
between the **GT-derived** Faceswap alignment geometry and the **prediction-
derived** alignment geometry. It is the foundation of Phase 1 of the
alignment-geometry roadmap: every signal below corresponds to something
``AlignedFace`` would actually consume at extract time.

Design notes:

* All comparisons are GT-vs-prediction. We never flag a prediction as
  "anatomically wrong" just because the visible projection looks unusual —
  projected/profile landmarks are valid if they match the GT convention.
* The primary score is :func:`score_alignment_geometry_v1`. NME / PCK
  remain available via :mod:`lib.landmarks.eval.profile_metrics` for
  diagnostic reporting only.
* Per-region geometry errors live in *aligned-face* space (the same space
  Faceswap's crop and mask code see), so they translate directly to swap
  placement / mask quality.
* Catastrophic failure rate is reported separately from continuous deltas so
  it can drive its own promotion gate (#77).
"""

from __future__ import annotations

import math
import typing as T
from dataclasses import dataclass, field

import numpy as np

from lib.landmarks.eval.geometry_signals import (
    DEFAULT_ALIGNED_SIZE,
    DEFAULT_BBOX_INFLATION,
    DEFAULT_COLLAPSE_RATIO,
    AlignmentSummary,
    CatastrophicFlags,
    MatrixDelta,
    PoseDelta,
    RoiDelta,
    alignment_matrix_delta,
    alignment_summary,
    average_distance_delta,
    evaluate_catastrophic_flags,
    pose_delta,
    roi_delta,
    visible_hull_iou,
)
from lib.landmarks.eval.roi_diagnostics import (
    DEFAULT_LANDMARK_COVERAGE_FLOOR,
    RoiDiagnostics,
    evaluate_roi_diagnostics,
)

GEOMETRY_OBJECTIVE: str = "alignment_geometry_v1"
DEFAULT_FAILURE_THRESHOLD: float = 0.5  # 1 - roi_iou ≥ this counts as a soft failure

REGION_DEFINITIONS: dict[str, tuple[tuple[int, int], ...]] = {
    "eyes": ((36, 42), (42, 48)),
    "nose": ((27, 36),),
    "mouth": ((48, 60), (60, 68)),
    "jaw": ((0, 17),),
    "brows": ((17, 22), (22, 27)),
}


def _region_indices(region: str) -> tuple[int, ...]:
    """Return the canonical landmark indices for a named geometry region."""
    spans = REGION_DEFINITIONS.get(region)
    if spans is None:
        raise KeyError(f"unknown geometry region {region!r}")
    indices: list[int] = []
    for start, end in spans:
        indices.extend(range(start, end))
    return tuple(indices)


def _normalize_bbox(
    bbox: T.Sequence[float] | None,
) -> tuple[float, float, float, float] | None:
    """Return bbox as ``(left, top, right, bottom)`` when possible.

    Thin wrapper over :func:`lib.landmarks.manifest.coerce_bbox` — the
    canonical xywh / ltrb coercion lives in :mod:`lib.landmarks.manifest`
    so every layer of the pipeline normalizes the same way (the COFW-68
    xywh shape used to silently drift between layers).
    """
    from lib.landmarks.manifest import coerce_bbox

    return coerce_bbox(bbox)


def aligned_face_size_from_summary(summary: AlignmentSummary) -> float:
    """Return the side length of the aligned-face coordinate system."""
    points = summary.aligned_landmarks
    if points.size == 0:
        return float(DEFAULT_ALIGNED_SIZE)
    width = float(points[:, 0].max() - points[:, 0].min())
    height = float(points[:, 1].max() - points[:, 1].min())
    return max(width, height, 1.0)


@dataclass(frozen=True)
class GeometrySampleMetrics:
    """Geometry deltas for one (predicted, truth) landmark pair."""

    sample_id: str
    dataset: str
    condition: str
    matrix_delta: MatrixDelta
    relative_scale_delta: float
    roi_delta: RoiDelta
    pose_delta: PoseDelta
    average_distance_delta: float
    relative_eye_mouth_position_predicted: float
    relative_eye_mouth_position_truth: float
    hull_iou: float
    per_region_error: dict[str, float]
    per_region_failure: dict[str, bool]
    catastrophic_flags: CatastrophicFlags
    points_outside_bbox: int
    roi_diagnostics: RoiDiagnostics | None
    overall_score: float

    def to_payload(self) -> dict[str, T.Any]:
        return {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "condition": self.condition,
            "scenario_bucket": f"{self.dataset or 'unspecified'}:{self.condition or 'unspecified'}",
            "scale_delta": self.matrix_delta.scale_delta,
            "relative_scale_delta": self.relative_scale_delta,
            "rotation_degrees_delta": self.matrix_delta.rotation_degrees_delta,
            "translation_pixel_distance": self.matrix_delta.translation_pixel_distance,
            "translation_normalized_distance": self.matrix_delta.translation_normalized_distance,
            "roi_iou": self.roi_delta.iou,
            "roi_center_pixel_distance": self.roi_delta.center_pixel_distance,
            "roi_center_normalized_distance": self.roi_delta.center_normalized_distance,
            "pitch_delta_degrees": self.pose_delta.pitch_delta_degrees,
            "yaw_delta_degrees": self.pose_delta.yaw_delta_degrees,
            "roll_delta_degrees": self.pose_delta.roll_delta_degrees,
            "average_distance_delta": self.average_distance_delta,
            "relative_eye_mouth_position_predicted": self.relative_eye_mouth_position_predicted,
            "relative_eye_mouth_position_truth": self.relative_eye_mouth_position_truth,
            "hull_iou": self.hull_iou,
            "per_region_error": dict(self.per_region_error),
            "per_region_failure": dict(self.per_region_failure),
            "points_outside_bbox": int(self.points_outside_bbox),
            "cloud_collapse": bool(self.catastrophic_flags.cloud_collapse),
            "eye_mouth_flip": bool(self.catastrophic_flags.eye_mouth_flip),
            "catastrophic": bool(self.catastrophic_flags.any),
            "roi_diagnostics": (
                self.roi_diagnostics.to_payload() if self.roi_diagnostics is not None else None
            ),
            "overall_score": float(self.overall_score),
        }


@dataclass(frozen=True)
class GeometryAggregate:
    """Aggregate geometry metrics across one label's sample evaluations."""

    label: str
    sample_count: int
    overall_score: float
    catastrophic_failure_rate: float
    mean_scale_delta: float
    mean_relative_scale_delta: float
    mean_rotation_degrees_delta: float
    mean_translation_normalized: float
    p95_translation_normalized: float
    p95_rotation_degrees_delta: float
    p95_relative_scale_delta: float
    p95_roll_degrees_delta: float
    mean_roi_iou: float
    p05_roi_iou: float
    mean_roi_center_normalized: float
    p95_roi_center_normalized: float
    mean_hull_iou: float
    p05_hull_iou: float
    mean_pitch_delta_degrees: float
    mean_yaw_delta_degrees: float
    mean_roll_delta_degrees: float
    mean_average_distance_delta: float
    per_region_error: dict[str, float] = field(default_factory=dict)
    per_region_failure_rate: dict[str, float] = field(default_factory=dict)
    per_bucket: dict[str, dict[str, float]] = field(default_factory=dict)
    mean_aligned_crop_visible_hull_iou: float = 0.0
    mean_landmarks_inside_aligned_crop_fraction: float = 0.0
    aligned_crop_miss_rate: float = 0.0
    mean_bbox_aspect_ratio: float = 0.0

    def to_payload(self) -> dict[str, T.Any]:
        return {
            "label": self.label,
            "sample_count": int(self.sample_count),
            "overall_score": float(self.overall_score),
            "catastrophic_failure_rate": float(self.catastrophic_failure_rate),
            "mean_scale_delta": float(self.mean_scale_delta),
            "mean_relative_scale_delta": float(self.mean_relative_scale_delta),
            "mean_rotation_degrees_delta": float(self.mean_rotation_degrees_delta),
            "mean_translation_normalized": float(self.mean_translation_normalized),
            "p95_translation_normalized": float(self.p95_translation_normalized),
            "p95_rotation_degrees_delta": float(self.p95_rotation_degrees_delta),
            "p95_relative_scale_delta": float(self.p95_relative_scale_delta),
            "p95_roll_degrees_delta": float(self.p95_roll_degrees_delta),
            "mean_roi_iou": float(self.mean_roi_iou),
            "p05_roi_iou": float(self.p05_roi_iou),
            "mean_roi_center_normalized": float(self.mean_roi_center_normalized),
            "p95_roi_center_normalized": float(self.p95_roi_center_normalized),
            "mean_hull_iou": float(self.mean_hull_iou),
            "p05_hull_iou": float(self.p05_hull_iou),
            "mean_pitch_delta_degrees": float(self.mean_pitch_delta_degrees),
            "mean_yaw_delta_degrees": float(self.mean_yaw_delta_degrees),
            "mean_roll_delta_degrees": float(self.mean_roll_delta_degrees),
            "mean_average_distance_delta": float(self.mean_average_distance_delta),
            "per_region_error": dict(self.per_region_error),
            "per_region_failure_rate": dict(self.per_region_failure_rate),
            "per_bucket": {bucket: dict(values) for bucket, values in self.per_bucket.items()},
            "mean_aligned_crop_visible_hull_iou": float(self.mean_aligned_crop_visible_hull_iou),
            "mean_landmarks_inside_aligned_crop_fraction": float(
                self.mean_landmarks_inside_aligned_crop_fraction
            ),
            "aligned_crop_miss_rate": float(self.aligned_crop_miss_rate),
            "mean_bbox_aspect_ratio": float(self.mean_bbox_aspect_ratio),
        }


def evaluate_geometry_sample(
    predicted: np.ndarray,
    truth: np.ndarray,
    *,
    sample_id: str,
    dataset: str = "",
    condition: str = "",
    bbox: T.Sequence[float] | None = None,
    visibility: T.Sequence[bool] | None = None,
    aligned_size: int = DEFAULT_ALIGNED_SIZE,
    bbox_inflation: float = DEFAULT_BBOX_INFLATION,
    collapse_ratio: float = DEFAULT_COLLAPSE_RATIO,
    region_failure_threshold: float = 0.05,
    bbox_source: str = "manifest",
    landmark_coverage_floor: float = DEFAULT_LANDMARK_COVERAGE_FLOOR,
    truth_summary: AlignmentSummary | None = None,
) -> GeometrySampleMetrics:
    """Return the GT-vs-prediction geometry deltas for one sample.

    ``region_failure_threshold`` is the per-region aligned-space normalized
    error above which a region is flagged as failed. The default ``0.05`` is
    5% of the aligned-face side length — large enough that small wobble does
    not trip the gate, small enough that a misaligned eye/mouth region does.

    ``truth_summary`` lets callers that evaluate many candidates against the
    same GT (candidate search, signal validation, geometry CLI) precompute
    the GT-side :func:`alignment_summary` once and reuse it across calls,
    avoiding redundant ``AlignedFace`` (Umeyama + solvePnP) builds.
    """
    bbox = _normalize_bbox(bbox)
    pred_summary = alignment_summary(predicted, size=aligned_size)
    if truth_summary is None:
        truth_summary = alignment_summary(truth, size=aligned_size)
    normalizer = _bbox_diagonal(truth)
    if normalizer <= 0:
        normalizer = max(aligned_size, 1.0)

    matrix_delta = alignment_matrix_delta(pred_summary, truth_summary, normalizer=normalizer)
    relative_scale_delta = matrix_delta.scale_delta / max(abs(truth_summary.scale), 1e-9)
    roi = roi_delta(pred_summary, truth_summary, normalizer=normalizer)
    poses = pose_delta(pred_summary, truth_summary)
    avg_dist_delta = average_distance_delta(pred_summary, truth_summary)
    hull_iou = visible_hull_iou(predicted, truth, visibility=visibility)
    cat_flags = evaluate_catastrophic_flags(
        predicted,
        truth,
        pred_summary,
        bbox=bbox,
        collapse_ratio=collapse_ratio,
        bbox_inflation=bbox_inflation,
    )
    outside = (
        0 if bbox is None else _points_outside(predicted, bbox=bbox, inflation=bbox_inflation)
    )
    roi_diag = (
        None
        if bbox is None
        else evaluate_roi_diagnostics(
            predicted_summary=pred_summary,
            truth_landmarks=np.asarray(truth, dtype="float64"),
            bbox=bbox,
            bbox_source=bbox_source,
            coverage_floor=landmark_coverage_floor,
        )
    )

    aligned_face_size = aligned_face_size_from_summary(truth_summary)
    per_region_error: dict[str, float] = {}
    per_region_failure: dict[str, bool] = {}
    for region in REGION_DEFINITIONS:
        indices = _region_indices(region)
        pred_pts = pred_summary.aligned_landmarks[list(indices)]
        truth_pts = truth_summary.aligned_landmarks[list(indices)]
        distances = np.linalg.norm(pred_pts - truth_pts, axis=1)
        normalized_error = float(distances.mean() / aligned_face_size)
        per_region_error[region] = normalized_error
        per_region_failure[region] = normalized_error > region_failure_threshold

    overall = score_alignment_geometry_v1(
        translation_normalized=matrix_delta.translation_normalized_distance,
        relative_scale_delta=relative_scale_delta,
        rotation_degrees_delta=matrix_delta.rotation_degrees_delta,
        roi_iou=roi.iou,
        hull_iou=hull_iou,
        catastrophic=cat_flags.any,
    )

    return GeometrySampleMetrics(
        sample_id=sample_id,
        dataset=dataset,
        condition=condition,
        matrix_delta=matrix_delta,
        relative_scale_delta=relative_scale_delta,
        roi_delta=roi,
        pose_delta=poses,
        average_distance_delta=avg_dist_delta,
        relative_eye_mouth_position_predicted=pred_summary.relative_eye_mouth_position,
        relative_eye_mouth_position_truth=truth_summary.relative_eye_mouth_position,
        hull_iou=hull_iou,
        per_region_error=per_region_error,
        per_region_failure=per_region_failure,
        catastrophic_flags=cat_flags,
        points_outside_bbox=outside,
        roi_diagnostics=roi_diag,
        overall_score=overall,
    )


def score_alignment_geometry_v1(
    *,
    translation_normalized: float,
    relative_scale_delta: float,
    rotation_degrees_delta: float,
    roi_iou: float,
    hull_iou: float,
    catastrophic: bool,
) -> float:
    """Composite ``alignment_geometry_v1`` per-sample score (lower is better).

    Coefficients chosen so each non-catastrophic component lands in
    ``O(0.01-0.10)`` range for "decent" predictions, while a catastrophic
    failure adds a full 1.0 penalty so even a single hard failure dominates
    the score and surfaces immediately in the rankings.
    """
    return (
        5.0 * float(translation_normalized)
        + 1.0 * float(relative_scale_delta)
        + 0.02 * float(rotation_degrees_delta)
        + 0.5 * (1.0 - float(roi_iou))
        + 0.5 * (1.0 - float(hull_iou))
        + (1.0 if catastrophic else 0.0)
    )


def aggregate_geometry_samples(
    label: str,
    samples: T.Sequence[GeometrySampleMetrics],
) -> GeometryAggregate:
    """Aggregate a label's per-sample geometry metrics into a single row."""
    if not samples:
        return GeometryAggregate(
            label=label,
            sample_count=0,
            overall_score=0.0,
            catastrophic_failure_rate=0.0,
            mean_scale_delta=0.0,
            mean_relative_scale_delta=0.0,
            mean_rotation_degrees_delta=0.0,
            mean_translation_normalized=0.0,
            p95_translation_normalized=0.0,
            p95_rotation_degrees_delta=0.0,
            p95_relative_scale_delta=0.0,
            p95_roll_degrees_delta=0.0,
            mean_roi_iou=0.0,
            p05_roi_iou=0.0,
            mean_roi_center_normalized=0.0,
            p95_roi_center_normalized=0.0,
            mean_hull_iou=0.0,
            p05_hull_iou=0.0,
            mean_pitch_delta_degrees=0.0,
            mean_yaw_delta_degrees=0.0,
            mean_roll_delta_degrees=0.0,
            mean_average_distance_delta=0.0,
        )

    scale_deltas = np.array([s.matrix_delta.scale_delta for s in samples])
    relative_scale = np.array([s.relative_scale_delta for s in samples])
    rotation = np.array([s.matrix_delta.rotation_degrees_delta for s in samples])
    translation = np.array([s.matrix_delta.translation_normalized_distance for s in samples])
    roi_ious = np.array([s.roi_delta.iou for s in samples])
    roi_centers = np.array([s.roi_delta.center_normalized_distance for s in samples])
    hull_ious = np.array([s.hull_iou for s in samples])
    pitch = np.array([s.pose_delta.pitch_delta_degrees for s in samples])
    yaw = np.array([s.pose_delta.yaw_delta_degrees for s in samples])
    roll = np.array([s.pose_delta.roll_delta_degrees for s in samples])
    avg_dist = np.array([s.average_distance_delta for s in samples])
    overall = np.array([s.overall_score for s in samples])
    catastrophic = np.array([s.catastrophic_flags.any for s in samples], dtype=bool)

    per_region_error: dict[str, list[float]] = {region: [] for region in REGION_DEFINITIONS}
    per_region_failures: dict[str, list[bool]] = {region: [] for region in REGION_DEFINITIONS}
    per_bucket: dict[str, list[GeometrySampleMetrics]] = {}
    for sample in samples:
        for region in REGION_DEFINITIONS:
            per_region_error[region].append(sample.per_region_error.get(region, 0.0))
            per_region_failures[region].append(sample.per_region_failure.get(region, False))
        bucket = f"{sample.dataset or 'unspecified'}:{sample.condition or 'unspecified'}"
        per_bucket.setdefault(bucket, []).append(sample)

    per_bucket_payload: dict[str, dict[str, float]] = {}
    for bucket, bucket_samples in per_bucket.items():
        bucket_overall = np.array([s.overall_score for s in bucket_samples])
        bucket_catastrophic = np.array(
            [s.catastrophic_flags.any for s in bucket_samples], dtype=bool
        )
        bucket_translation = np.array(
            [s.matrix_delta.translation_normalized_distance for s in bucket_samples]
        )
        bucket_rotation = np.array([s.matrix_delta.rotation_degrees_delta for s in bucket_samples])
        bucket_roi = np.array([s.roi_delta.iou for s in bucket_samples])
        per_bucket_payload[bucket] = {
            "sample_count": float(len(bucket_samples)),
            "overall_score": float(bucket_overall.mean()),
            "catastrophic_failure_rate": float(bucket_catastrophic.mean()),
            "mean_translation_normalized": float(bucket_translation.mean()),
            "p95_translation_normalized": float(np.percentile(bucket_translation, 95)),
            "mean_rotation_degrees_delta": float(bucket_rotation.mean()),
            "mean_roi_iou": float(bucket_roi.mean()),
        }

    roi_with_diag = [s for s in samples if s.roi_diagnostics is not None]
    if roi_with_diag:
        hull_coverage = np.array(
            [s.roi_diagnostics.aligned_crop_visible_hull_iou for s in roi_with_diag]
        )
        coverage = np.array(
            [s.roi_diagnostics.landmarks_inside_aligned_crop_fraction for s in roi_with_diag]
        )
        miss_flags = np.array(
            [s.roi_diagnostics.aligned_crop_misses_visible_face for s in roi_with_diag],
            dtype=bool,
        )
        aspect_ratios = np.array([s.roi_diagnostics.bbox_aspect_ratio for s in roi_with_diag])
        mean_aligned_crop_visible_hull_iou = float(hull_coverage.mean())
        mean_landmarks_inside_aligned_crop_fraction = float(coverage.mean())
        aligned_crop_miss_rate = float(miss_flags.mean())
        mean_bbox_aspect_ratio = float(aspect_ratios.mean())
    else:
        mean_aligned_crop_visible_hull_iou = 0.0
        mean_landmarks_inside_aligned_crop_fraction = 0.0
        aligned_crop_miss_rate = 0.0
        mean_bbox_aspect_ratio = 0.0

    return GeometryAggregate(
        label=label,
        sample_count=len(samples),
        overall_score=float(overall.mean()),
        catastrophic_failure_rate=float(catastrophic.mean()),
        mean_scale_delta=float(scale_deltas.mean()),
        mean_relative_scale_delta=float(relative_scale.mean()),
        mean_rotation_degrees_delta=float(rotation.mean()),
        mean_translation_normalized=float(translation.mean()),
        p95_translation_normalized=float(np.percentile(translation, 95)),
        p95_rotation_degrees_delta=float(np.percentile(rotation, 95)),
        p95_relative_scale_delta=float(np.percentile(relative_scale, 95)),
        p95_roll_degrees_delta=float(np.percentile(roll, 95)),
        mean_roi_iou=float(roi_ious.mean()),
        p05_roi_iou=float(np.percentile(roi_ious, 5)),
        mean_roi_center_normalized=float(roi_centers.mean()),
        p95_roi_center_normalized=float(np.percentile(roi_centers, 95)),
        mean_hull_iou=float(hull_ious.mean()),
        p05_hull_iou=float(np.percentile(hull_ious, 5)),
        mean_pitch_delta_degrees=float(pitch.mean()),
        mean_yaw_delta_degrees=float(yaw.mean()),
        mean_roll_delta_degrees=float(roll.mean()),
        mean_average_distance_delta=float(avg_dist.mean()),
        per_region_error={
            region: float(np.mean(values)) for region, values in per_region_error.items()
        },
        per_region_failure_rate={
            region: float(np.mean(values)) for region, values in per_region_failures.items()
        },
        per_bucket=per_bucket_payload,
        mean_aligned_crop_visible_hull_iou=mean_aligned_crop_visible_hull_iou,
        mean_landmarks_inside_aligned_crop_fraction=mean_landmarks_inside_aligned_crop_fraction,
        aligned_crop_miss_rate=aligned_crop_miss_rate,
        mean_bbox_aspect_ratio=mean_bbox_aspect_ratio,
    )


def _bbox_diagonal(points: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return math.hypot(float(maxs[0] - mins[0]), float(maxs[1] - mins[1]))


def _points_outside(
    predicted: np.ndarray,
    *,
    bbox: T.Sequence[float],
    inflation: float,
) -> int:
    from lib.landmarks.eval.geometry_signals import points_outside_bbox

    return points_outside_bbox(predicted, bbox=bbox, inflation=inflation)


__all__ = [
    "DEFAULT_FAILURE_THRESHOLD",
    "GEOMETRY_OBJECTIVE",
    "GeometryAggregate",
    "GeometrySampleMetrics",
    "REGION_DEFINITIONS",
    "aggregate_geometry_samples",
    "aligned_face_size_from_summary",
    "evaluate_geometry_sample",
    "score_alignment_geometry_v1",
]
