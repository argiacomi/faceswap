#!/usr/bin/env python3
"""Profile-aware alignment metrics (#76).

Profile / AFLW samples are scored against a face-bbox normalizer (default
diagonal) rather than the iBUG interocular distance, which is undefined on
extreme yaw faces where one eye is non-visible. The objective is tuned for
extract alignment quality: stable visible-face geometry — jaw silhouette,
nose, visible eye, mouth — rather than chasing average point error across
ambiguous projected-contour points.

Key design choices (deliberate departures from the ticket where ambiguous):

* ``visible_jaw`` and ``visible_eye`` regions count only points marked visible
  in the sample manifest. When no visibility flags are present the regions
  fall back to the full anatomical span (the metric is then no better than
  classical region NME, but it still works for datasets without visibility
  annotations).
* The bbox normalizer defaults to the bbox diagonal because that survives
  extreme aspect ratios on profile faces; callers can switch to
  ``sqrt(area)`` if their downstream report needs a face-width-like scale.
* ``profile_alignment_v1 = weighted_region_score + 0.5 * region_failure_rate
  + 0.25 * p90_visible_error``. All terms are bbox-normalized and land in
  the same O(0.01–0.10) range so no single dimension dominates by scale.
"""

from __future__ import annotations

import math
import typing as T
from dataclasses import dataclass, field

import numpy as np

from lib.landmarks.core.metrics import per_landmark_error
from lib.landmarks.core.schema import to_canonical_68

DEFAULT_NORMALIZER: str = "bbox_diagonal"
NORMALIZERS: tuple[str, ...] = ("bbox_diagonal", "bbox_sqrt_area")
DEFAULT_PCK_THRESHOLDS: tuple[float, ...] = (0.02, 0.03, 0.05)
DEFAULT_REGION_FAILURE_THRESHOLD: float = 0.05
DEFAULT_PRIORITY_FAILURE_REGIONS: tuple[str, ...] = (
    "visible_jaw",
    "nose",
    "visible_eye",
    "mouth_outer",
)
PROFILE_OBJECTIVE: str = "profile_alignment_v1"

# Canonical 68-point region spans (left inclusive, right exclusive).
_JAW_INDICES: tuple[int, ...] = tuple(range(0, 17))
_RIGHT_BROW: tuple[int, ...] = tuple(range(17, 22))
_LEFT_BROW: tuple[int, ...] = tuple(range(22, 27))
_NOSE_INDICES: tuple[int, ...] = tuple(range(27, 36))
_RIGHT_EYE: tuple[int, ...] = tuple(range(36, 42))
_LEFT_EYE: tuple[int, ...] = tuple(range(42, 48))
_MOUTH_OUTER: tuple[int, ...] = tuple(range(48, 60))
_MOUTH_INNER: tuple[int, ...] = tuple(range(60, 68))

#: Index set per region. ``visible_*`` regions span all anatomical landmarks
#: but downstream code only counts the ones flagged visible. The non-visible
#: variants below are kept for tests/diagnostics.
REGION_INDICES: dict[str, tuple[int, ...]] = {
    "visible_jaw": _JAW_INDICES,
    "nose": _NOSE_INDICES,
    "visible_eye": _RIGHT_EYE + _LEFT_EYE,
    "mouth_outer": _MOUTH_OUTER,
    "brows": _RIGHT_BROW + _LEFT_BROW,
    "mouth_inner": _MOUTH_INNER,
}

#: Default region weights for ``weighted_region_score``. Sums to 1.0. Jaw and
#: nose dominate because they anchor the alignment matrix and silhouette;
#: brows / inner mouth contribute as low-weight support.
DEFAULT_REGION_WEIGHTS: dict[str, float] = {
    "visible_jaw": 0.25,
    "nose": 0.25,
    "visible_eye": 0.20,
    "mouth_outer": 0.20,
    "brows": 0.05,
    "mouth_inner": 0.05,
}


@dataclass(frozen=True)
class ProfileSampleMetrics:
    """Profile metrics for one (prediction, truth) pair."""

    sample_id: str
    normalizer: float
    visible_landmark_count: int
    visible_landmark_errors: tuple[float, ...]  # normalized
    per_region_error: dict[str, float] = field(default_factory=dict)
    region_failures: dict[str, bool] = field(default_factory=dict)
    pck_at: dict[str, float] = field(default_factory=dict)
    p90_visible_error: float = 0.0
    weighted_region_score: float = 0.0
    overall_score: float = 0.0  # profile_alignment_v1


@dataclass(frozen=True)
class ProfileAggregate:
    """Aggregated profile metrics across a set of samples for one label."""

    label: str
    sample_count: int
    overall_score: float
    weighted_region_score: float
    region_failure_rate: float  # priority-region failure rate
    p90_visible_error: float
    pck_at: dict[str, float]
    per_region_error: dict[str, float]
    per_region_failure_rate: dict[str, float]


def normalizer_from_bbox(
    face_bbox: T.Sequence[float] | None,
    *,
    method: str = DEFAULT_NORMALIZER,
) -> float:
    """Return a scalar normalizer derived from a manifest face bounding box.

    ``bbox_diagonal`` uses ``sqrt(w^2 + h^2)``, which survives the extreme
    aspect ratios common in profile faces. ``bbox_sqrt_area`` returns
    ``sqrt(w * h)`` for callers who want a face-width-like scale.
    """
    if face_bbox is None:
        raise ValueError("face_bbox is required to compute a profile normalizer")
    if method not in NORMALIZERS:
        raise ValueError(f"unsupported normalizer {method!r}; supported: {', '.join(NORMALIZERS)}")
    left, top, right, bottom = (float(value) for value in face_bbox)
    width = max(0.0, right - left)
    height = max(0.0, bottom - top)
    if width <= 0 or height <= 0:
        raise ValueError(f"face_bbox must have positive width and height, got {face_bbox!r}")
    if method == "bbox_diagonal":
        return math.sqrt(width * width + height * height)
    return math.sqrt(width * height)


def _resolve_region_indices(
    region: str,
    visibility: T.Sequence[bool] | None,
) -> tuple[int, ...]:
    """Return the indices that contribute to ``region``.

    The two ``visible_*`` regions consult ``visibility`` and drop non-visible
    points so projected/occluded landmarks do not bias the region error.
    """
    indices = REGION_INDICES[region]
    if visibility is None or region not in {"visible_jaw", "visible_eye"}:
        return indices
    return tuple(index for index in indices if visibility[index])


def evaluate_profile_sample(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    sample_id: str,
    face_bbox: T.Sequence[float] | None,
    visibility: T.Sequence[bool] | None = None,
    normalizer_method: str = DEFAULT_NORMALIZER,
    region_weights: T.Mapping[str, float] = DEFAULT_REGION_WEIGHTS,
    region_failure_threshold: float = DEFAULT_REGION_FAILURE_THRESHOLD,
    priority_failure_regions: T.Sequence[str] = DEFAULT_PRIORITY_FAILURE_REGIONS,
    pck_thresholds: T.Sequence[float] = DEFAULT_PCK_THRESHOLDS,
) -> ProfileSampleMetrics:
    """Score one prediction against the profile objective.

    All distance values in the returned record are bbox-normalized so they are
    comparable across samples with different face scales.
    """
    pred = to_canonical_68(predicted)
    truth = to_canonical_68(target)
    normalizer = normalizer_from_bbox(face_bbox, method=normalizer_method)
    point_errors = per_landmark_error(pred, truth).astype("float32") / normalizer

    if visibility is None:
        visible_mask = np.ones(point_errors.shape, dtype=bool)
    else:
        if len(visibility) != point_errors.size:
            raise ValueError(
                "visibility length must match landmark count "
                f"{point_errors.size}, got {len(visibility)}"
            )
        visible_mask = np.array(list(visibility), dtype=bool)

    visible_errors = point_errors[visible_mask].astype("float32")

    per_region_error: dict[str, float] = {}
    region_failures: dict[str, bool] = {}
    for region in REGION_INDICES:
        indices = _resolve_region_indices(region, visibility)
        if not indices:
            per_region_error[region] = 0.0
            region_failures[region] = False
            continue
        region_errors = point_errors[list(indices)]
        per_region_error[region] = float(region_errors.mean())
        region_failures[region] = bool(per_region_error[region] > region_failure_threshold)

    weighted_region_score = float(
        sum(region_weights.get(name, 0.0) * per_region_error[name] for name in per_region_error)
    )
    pck_at = {
        f"{threshold:.2f}": float(np.mean(visible_errors <= threshold))
        if visible_errors.size
        else 0.0
        for threshold in pck_thresholds
    }
    p90 = float(np.percentile(visible_errors, 90)) if visible_errors.size else 0.0

    sample_failure_rate = (
        float(np.mean([region_failures[name] for name in priority_failure_regions]))
        if priority_failure_regions
        else 0.0
    )
    overall_score = float(weighted_region_score + 0.5 * sample_failure_rate + 0.25 * p90)

    return ProfileSampleMetrics(
        sample_id=sample_id,
        normalizer=normalizer,
        visible_landmark_count=int(visible_mask.sum()),
        visible_landmark_errors=tuple(float(value) for value in visible_errors.tolist()),
        per_region_error=per_region_error,
        region_failures=region_failures,
        pck_at=pck_at,
        p90_visible_error=p90,
        weighted_region_score=weighted_region_score,
        overall_score=overall_score,
    )


def aggregate_profile_samples(
    label: str,
    samples: T.Sequence[ProfileSampleMetrics],
    *,
    region_weights: T.Mapping[str, float] = DEFAULT_REGION_WEIGHTS,
    priority_failure_regions: T.Sequence[str] = DEFAULT_PRIORITY_FAILURE_REGIONS,
    pck_thresholds: T.Sequence[float] = DEFAULT_PCK_THRESHOLDS,
) -> ProfileAggregate:
    """Aggregate per-sample ProfileSampleMetrics into one ProfileAggregate row."""
    if not samples:
        return ProfileAggregate(
            label=label,
            sample_count=0,
            overall_score=0.0,
            weighted_region_score=0.0,
            region_failure_rate=0.0,
            p90_visible_error=0.0,
            pck_at={f"{t:.2f}": 0.0 for t in pck_thresholds},
            per_region_error={},
            per_region_failure_rate={},
        )

    per_region_errors: dict[str, list[float]] = {region: [] for region in REGION_INDICES}
    per_region_failures: dict[str, list[bool]] = {region: [] for region in REGION_INDICES}
    pck_values: dict[str, list[float]] = {f"{t:.2f}": [] for t in pck_thresholds}
    sample_p90s: list[float] = []
    visible_errors_all: list[float] = []
    weighted_scores: list[float] = []
    priority_failures: list[float] = []

    for sample in samples:
        for region, value in sample.per_region_error.items():
            per_region_errors.setdefault(region, []).append(value)
        for region, flag in sample.region_failures.items():
            per_region_failures.setdefault(region, []).append(flag)
        for key, value in sample.pck_at.items():
            pck_values.setdefault(key, []).append(value)
        sample_p90s.append(sample.p90_visible_error)
        visible_errors_all.extend(sample.visible_landmark_errors)
        weighted_scores.append(sample.weighted_region_score)
        priority_failures.append(
            float(
                np.mean(
                    [sample.region_failures.get(name, False) for name in priority_failure_regions]
                )
            )
            if priority_failure_regions
            else 0.0
        )

    per_region_error = {
        region: float(np.mean(values)) for region, values in per_region_errors.items() if values
    }
    per_region_failure_rate = {
        region: float(np.mean(values)) for region, values in per_region_failures.items() if values
    }
    pck_at = {
        threshold: float(np.mean(values)) if values else 0.0
        for threshold, values in pck_values.items()
    }
    p90 = (
        float(np.percentile(np.array(visible_errors_all, dtype="float32"), 90))
        if visible_errors_all
        else 0.0
    )
    weighted_region_score = float(np.mean(weighted_scores)) if weighted_scores else 0.0
    region_failure_rate = float(np.mean(priority_failures)) if priority_failures else 0.0
    overall_score = float(weighted_region_score + 0.5 * region_failure_rate + 0.25 * p90)

    return ProfileAggregate(
        label=label,
        sample_count=len(samples),
        overall_score=overall_score,
        weighted_region_score=weighted_region_score,
        region_failure_rate=region_failure_rate,
        p90_visible_error=p90,
        pck_at=pck_at,
        per_region_error=per_region_error,
        per_region_failure_rate=per_region_failure_rate,
    )


def score_profile_alignment_v1(aggregate: ProfileAggregate) -> float:
    """Re-derive the ``profile_alignment_v1`` score from a ProfileAggregate.

    Identical formula to the per-sample compute path; exposed as a separate
    function so callers can re-score an aggregate they loaded from disk.
    """
    return float(
        aggregate.weighted_region_score
        + 0.5 * aggregate.region_failure_rate
        + 0.25 * aggregate.p90_visible_error
    )


def is_priority_region(region: str) -> bool:
    """Return True when ``region`` is one of the extract-critical regions."""
    return region in DEFAULT_PRIORITY_FAILURE_REGIONS


__all__ = [
    "DEFAULT_NORMALIZER",
    "DEFAULT_PCK_THRESHOLDS",
    "DEFAULT_PRIORITY_FAILURE_REGIONS",
    "DEFAULT_REGION_FAILURE_THRESHOLD",
    "DEFAULT_REGION_WEIGHTS",
    "NORMALIZERS",
    "PROFILE_OBJECTIVE",
    "REGION_INDICES",
    "ProfileAggregate",
    "ProfileSampleMetrics",
    "aggregate_profile_samples",
    "evaluate_profile_sample",
    "is_priority_region",
    "normalizer_from_bbox",
    "score_profile_alignment_v1",
]
