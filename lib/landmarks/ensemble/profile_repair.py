#!/usr/bin/env python3
"""Generate a profile-safe repaired candidate for all-invalid profile groups (#219).

Most GT-hard invalid selections are not avoidable scorer mistakes - the
candidate set simply has no v3-valid/rankable candidate. The validity-first
selector (#218) can stop choosing a hard-invalid candidate when a valid one
exists, but it cannot reduce the all-invalid group count when none exists.

This module builds a real ``profile_visible_side_repaired`` candidate (actual
landmarks, not a faked scorer row) for profile/occlusion contexts where every
existing candidate is hard-invalid/vetoed. The repair keeps the visible-side
and stable-centre landmarks from a chosen source candidate and replaces the
occluded-side landmarks with a compressed mirror of the visible side. The
repaired candidate is only appended when it passes basic sanity gates; the
existing metric/shape/v3 pipeline then scores it like any other candidate, so
no CSV fields are spoofed.
"""

from __future__ import annotations

import typing as T

import numpy as np

from lib.landmarks.ensemble.profile_features import profile_side_from_context
from lib.landmarks.ensemble.profile_routing import (
    condition_tags,
    is_profile_or_occlusion_context,
)

PROFILE_REPAIR_CANDIDATE_NAME = "profile_visible_side_repaired"
PROFILE_REPAIR_METHOD = "compressed_mirror_v1"
DEFAULT_REPAIR_COMPRESSION = 0.35

# Source-candidate preference order for the repair base. The final choice is
# metric-based (visible-side stability); names only break ties.
PROFILE_REPAIR_SOURCE_PREFERENCE: tuple[str, ...] = (
    "spiga",
    "hrnet",
    "orformer",
    "static_weighted_downweight",
    "weighted_median",
    "plain_average",
)

NOSE_TIP_68 = 30
# Symmetric jaw index pairs (subject-left jaw point, subject-right jaw point).
_JAW_MIRROR_PAIRS: tuple[tuple[int, int], ...] = (
    (16, 0),
    (15, 1),
    (14, 2),
    (13, 3),
    (12, 4),
    (11, 5),
    (10, 6),
    (9, 7),
)


def is_profile_repair_context(context_or_bucket: T.Any) -> bool:
    """Return ``True`` when a context is eligible for profile repair generation.

    Eligibility requires a profile/large-yaw/occlusion route and explicitly excludes
    ``anchor`` contexts.  Anchor is the mined hard-negative bucket for clean frontal
    reference faces (and a report bucket); a repair candidate should never be generated
    for one even if a runtime bucket was mislabelled with a profile hint.
    """
    if any(tag == "anchor" for tag in condition_tags(context_or_bucket)):
        return False
    return is_profile_or_occlusion_context(context_or_bucket)


def _as_68(landmarks: T.Any) -> np.ndarray | None:
    points = np.asarray(landmarks, dtype="float32")
    if points.ndim != 2 or points.shape[0] < 68 or points.shape[1] < 2:
        return None
    return T.cast(np.ndarray | None, np.array(points[:68, :2], dtype="float32"))


def repair_candidate_passes_gates(landmarks: T.Any) -> bool:
    """Return ``True`` when repaired landmarks are a finite 68x2 array."""
    points = _as_68(landmarks)
    if points is None:
        return False
    return bool(np.all(np.isfinite(points)))


def make_profile_visible_side_repair(
    source_landmarks: T.Any,
    *,
    visible_side: str,
    compression: float = DEFAULT_REPAIR_COMPRESSION,
) -> np.ndarray:
    """Return repaired 68x2 landmarks for ``visible_side`` (``'left'``/``'right'``).

    The occluded-side jaw is synthesized by reflecting the visible-side jaw across the
    chin's vertical axis (landmark 8, the jaw vertex) and compressing it toward that
    axis, then forcing the full jaw contour to be strictly x-monotonic. A strictly
    x-monotonic open jaw polyline is a function graph and therefore cannot
    self-intersect, which is the topology check the repair previously failed. Non-jaw
    landmarks (eyes, brows, nose, mouth) are kept from the source candidate. Returns the
    source landmarks unchanged for an unknown side.
    """
    repaired = _as_68(source_landmarks)
    if repaired is None:
        raise ValueError("source_landmarks must be a 68x2 landmark array")
    if visible_side not in {"left", "right"}:
        return repaired

    # Trust the source geometry over a possibly mis-inferred yaw sign: mirror the half
    # that is actually wider (the real visible jaw), not the half the label points at.
    visible_side = _resolve_visible_side(repaired, visible_side)

    # Reflect about the chin (jaw vertex) so the two jaw halves straddle the chin and
    # stay monotonic; reflecting about the nose tip (offset on a profile) is what pushed
    # occluded points to the wrong side and produced crossing contours.
    chin_x = float(repaired[8, 0])
    if visible_side == "left":
        # subject-left visible (jaw 9-16), subject-right occluded (jaw 0-7).
        pairs = _JAW_MIRROR_PAIRS
    else:
        # subject-right visible (jaw 0-7), subject-left occluded (jaw 9-16).
        pairs = tuple((dst, src) for src, dst in _JAW_MIRROR_PAIRS)

    for src_idx, dst_idx in pairs:
        src = repaired[src_idx].copy()
        repaired[dst_idx, 0] = chin_x - (float(src[0]) - chin_x) * compression
        repaired[dst_idx, 1] = float(src[1])

    _clamp_occluded_jaw_band(repaired, pairs)
    return _enforce_monotonic_jawline(repaired)


def _resolve_visible_side(repaired: np.ndarray, visible_side: str) -> str:
    """Return the visible side implied by jaw geometry, overriding a wrong label.

    The visible jaw half spans farther from the chin than the collapsed occluded half.
    When the source geometry clearly contradicts the passed ``visible_side`` (a wrong
    yaw-sign inference), trust the geometry; otherwise keep the label.
    """
    chin_x = float(repaired[8, 0])
    left_spread = float(np.mean(np.abs(repaired[9:17, 0] - chin_x)))
    right_spread = float(np.mean(np.abs(repaired[0:8, 0] - chin_x)))
    larger = max(left_spread, right_spread)
    smaller = min(left_spread, right_spread)
    if larger <= 1e-6 or smaller >= 0.85 * larger:
        # Ambiguous (near-symmetric) source: keep the caller's side.
        return visible_side
    return "left" if left_spread >= right_spread else "right"


def _clamp_occluded_jaw_band(repaired: np.ndarray, pairs: tuple[tuple[int, int], ...]) -> None:
    """Keep synthesized occluded jaw points within the lower-face y band.

    The occluded y is copied from the mirrored visible point so it already sits in the
    jaw band; this clamps any stragglers to ``[min visible jaw y, chin y]`` so the
    synthesized contour cannot rise into the eye/nose region.
    """
    visible_indices = [src for src, _dst in pairs]
    visible_ys = repaired[visible_indices, 1]
    top = float(np.min(visible_ys))
    bottom = float(repaired[8, 1])
    lo, hi = (top, bottom) if top <= bottom else (bottom, top)
    for _src_idx, dst_idx in pairs:
        repaired[dst_idx, 1] = float(np.clip(repaired[dst_idx, 1], lo, hi))


def _enforce_monotonic_jawline(repaired: np.ndarray) -> np.ndarray:
    """Force the jaw (0..16) to be strictly x-monotonic so it cannot self-intersect."""
    xs = repaired[0:17, 0].astype(np.float64).copy()
    span = abs(float(xs[-1] - xs[0]))
    eps = max(span, 1.0) * 1e-3
    increasing = xs[-1] >= xs[0]
    work = xs if increasing else xs[::-1].copy()
    for idx in range(1, work.shape[0]):
        if work[idx] <= work[idx - 1]:
            work[idx] = work[idx - 1] + eps
    repaired[0:17, 0] = work if increasing else work[::-1]
    return repaired


def _source_has_fatal_visible_core_issue(metric: T.Any) -> bool:
    """Return ``True`` when a source candidate has a fatal visible-core defect.

    A repair keeps the source's visible-side eyes/brows/nose/mouth, so a source whose
    visible core is eye/mouth-flipped or otherwise fatally broken must not seed a repair.
    """
    reasons = " ".join(
        [
            *getattr(metric, "geometry_veto_reasons", ()),
            *getattr(metric, "shape_veto_reasons", ()),
        ]
    ).lower()
    if getattr(metric, "eye_mouth_order_valid_after_deroll", None) is False:
        return True
    return bool("eye_mouth" in reasons or "flip" in reasons)


def choose_profile_repair_source(
    candidates: T.Sequence[T.Any],
    metrics: T.Mapping[str, T.Any],
    candidate_extra_features: T.Mapping[str, T.Mapping[str, float]] | None = None,
) -> T.Any | None:
    """Return the best base candidate for repair, or ``None`` when none is usable.

    Prefers candidates with low visible-side consensus distance / high shape
    plausibility; ``PROFILE_REPAIR_SOURCE_PREFERENCE`` only breaks ties.
    """
    extra = candidate_extra_features or {}
    scored: list[tuple[tuple[float, float, int], T.Any]] = []
    for candidate in candidates:
        name = str(candidate.name)
        if _as_68(getattr(candidate, "landmarks", None)) is None:
            continue
        features = extra.get(name, {})
        visible_distance = float(features.get("visible_side_consensus_distance", 0.0) or 0.0)
        metric = metrics.get(name)
        if metric is not None and _source_has_fatal_visible_core_issue(metric):
            continue
        plausibility = float(getattr(metric, "shape_plausibility_score", 0.0) or 0.0)
        try:
            preference = PROFILE_REPAIR_SOURCE_PREFERENCE.index(name)
        except ValueError:
            preference = len(PROFILE_REPAIR_SOURCE_PREFERENCE)
        scored.append(((visible_distance, plausibility, preference), candidate))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0][0], item[0][1], item[0][2], str(item[1].name)))
    return scored[0][1]


def profile_repair_provenance(
    *,
    source_candidate: str,
    visible_side: str,
    reason: str,
    compression: float = DEFAULT_REPAIR_COMPRESSION,
) -> dict[str, float | str]:
    """Return provenance metadata for an appended repair candidate."""
    return {
        "profile_repair_used": 1.0,
        "profile_repair_source_candidate": source_candidate,
        "profile_repair_visible_side": visible_side,
        "profile_repair_method": PROFILE_REPAIR_METHOD,
        "profile_repair_reason": reason,
        "profile_repair_compression": float(compression),
    }


def profile_repair_features(
    *,
    visible_side: str,
    source_rank: int,
    shape_score: float,
) -> dict[str, float]:
    """Return scorer feature values describing the repair candidate."""
    return {
        "candidate_is_profile_repaired": 1.0,
        "profile_repair_source_rank": float(source_rank),
        "profile_repair_visible_side_left": 1.0 if visible_side == "left" else 0.0,
        "profile_repair_visible_side_right": 1.0 if visible_side == "right" else 0.0,
        "profile_repair_candidate_shape_score": float(shape_score),
    }


def build_profile_repair_landmarks(
    candidates: T.Sequence[T.Any],
    metrics: T.Mapping[str, T.Any],
    *,
    runtime_bucket: str,
    condition: str,
    yaw_estimate: float | None,
    candidate_extra_features: T.Mapping[str, T.Mapping[str, float]] | None = None,
    compression: float = DEFAULT_REPAIR_COMPRESSION,
) -> tuple[np.ndarray, str, str] | None:
    """Return ``(repaired_landmarks, source_name, visible_side)`` or ``None``.

    Returns ``None`` when the context is not repair-eligible, no visible side can
    be resolved, no usable source candidate exists, or the repaired landmarks
    fail the sanity gates. Eligibility/all-invalid gating is the caller's
    responsibility; this builds the candidate once the caller decides to repair.
    """
    if not is_profile_repair_context((runtime_bucket, condition)):
        return None
    visible_side = profile_side_from_context(
        runtime_bucket=runtime_bucket, condition=condition, yaw_estimate=yaw_estimate
    )
    if visible_side not in {"left", "right"}:
        return None
    source = choose_profile_repair_source(candidates, metrics, candidate_extra_features)
    if source is None:
        return None
    repaired = make_profile_visible_side_repair(
        source.landmarks, visible_side=visible_side, compression=compression
    )
    if not repair_candidate_passes_gates(repaired):
        return None
    return repaired, str(source.name), visible_side


__all__ = [
    "DEFAULT_REPAIR_COMPRESSION",
    "PROFILE_REPAIR_CANDIDATE_NAME",
    "PROFILE_REPAIR_METHOD",
    "build_profile_repair_landmarks",
    "choose_profile_repair_source",
    "is_profile_repair_context",
    "make_profile_visible_side_repair",
    "profile_repair_features",
    "profile_repair_provenance",
    "repair_candidate_passes_gates",
]
