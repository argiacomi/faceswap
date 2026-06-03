#!/usr/bin/env python3
"""Partial-schema (39-point profile) scoring for the profile specialist (#218/#219).

39-point MultiPIE/Menpo profile ground truth cannot be scored against full
68-point model output, so it is deliberately excluded from the canonical-68
NME/v3 scorer paths (:func:`lib.landmarks.datasets.manifest_io
.filter_canonical_68_samples`). This module provides the *partial-schema*
alternative: project a 68-point candidate prediction onto the side-specific
39-point profile layout using the MultiPIE 68/39 correspondence, then compare it
pointwise to the 39-point GT.

The resulting ``profile39_*`` metrics are intentionally kept separate from the
canonical ``candidate_nme`` / ``transform_cost_v3`` fields: production still
emits full 68-point landmarks, and only ``learned_quality_v3_profile`` consumes
the 39-point objective.

Final contract::

    68-point candidate prediction
      -> side-specific 39-point projection (PROFILE39_TO_CANONICAL68)
      -> compare projected prediction to 39-point GT
      -> profile-only cost / regret / oracle
      -> train the profile specialist only
"""

from __future__ import annotations

import typing as T

import numpy as np

# 39-point profile index order -> canonical 68 index, zero-based. The layout is
# from the MultiPIE 68/39 landmark configuration.
#
# "left"  = left cheek/eye visible (face facing left in the image).
# "right" = the opposite profile side.
PROFILE39_TO_CANONICAL68: dict[str, tuple[int, ...]] = {
    "left": (
        16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5,
        26, 25, 24, 23,
        27, 28, 29, 30,
        33, 35,
        43, 44, 45, 46, 47,
        51, 52, 53, 54, 55, 56, 57,
        62, 63, 64, 65, 66,
    ),
    "right": (
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        17, 18, 19, 20,
        27, 28, 29, 30,
        33, 31,
        38, 37, 36, 35, 34,
        51, 50, 49, 48,
        59, 58, 57,
        62, 61, 60,
        67, 66,
    ),
}  # fmt: skip

PROFILE39_POINT_COUNT = 39


def project_68_to_profile39(pred68: T.Any, *, side: str) -> np.ndarray:
    """Project a ``(68, 2)`` prediction onto the side-specific ``(39, 2)`` layout."""
    side = side.lower().strip()
    if side not in PROFILE39_TO_CANONICAL68:
        raise ValueError(f"unknown profile side: {side!r}")
    points = np.asarray(pred68, dtype="float32")
    if points.shape != (68, 2):
        raise ValueError(f"expected pred68 shape (68, 2), got {points.shape}")
    idx = np.asarray(PROFILE39_TO_CANONICAL68[side], dtype=np.int64)
    return T.cast("np.ndarray", points[idx])


def _as_truth39(truth39: T.Any) -> np.ndarray:
    points = np.asarray(truth39, dtype="float32")
    if points.shape != (PROFILE39_POINT_COUNT, 2):
        raise ValueError(f"expected truth39 shape (39, 2), got {points.shape}")
    return T.cast("np.ndarray", points)


def _bbox_diag(points: np.ndarray) -> float:
    span = np.ptp(points, axis=0)
    return max(float(np.linalg.norm(span)), 1e-6)


def _center(points: np.ndarray) -> np.ndarray:
    return T.cast("np.ndarray", np.mean(points, axis=0))


def profile39_point_error(
    pred68: T.Any,
    truth39: T.Any,
    *,
    side: str,
    normalizer: float,
) -> float:
    """Return the normalized visible-side point error (partial-schema NME analog)."""
    truth = _as_truth39(truth39)
    pred39 = project_68_to_profile39(pred68, side=side)
    dists = np.linalg.norm(pred39 - truth, axis=1)
    return float(np.mean(dists) / max(float(normalizer), 1e-6))


def profile39_transform_cost(
    pred68: T.Any,
    truth39: T.Any,
    *,
    side: str,
    normalizer: float | None = None,
    fit_weight: float = 1.0,
    center_weight: float = 0.25,
    scale_weight: float = 0.25,
) -> dict[str, float]:
    """Return a profile-specific transform cost (separate from ``transform_cost_v3``).

    Deliberately simple - fit + centre + scale deltas on the projected 39-point
    geometry - so it is hard to overfit and easy to debug. Roll / profile-outline
    terms can be layered on later.
    """
    truth = _as_truth39(truth39)
    pred39 = project_68_to_profile39(pred68, side=side)

    if normalizer is None:
        normalizer = _bbox_diag(truth)
    normalizer = max(float(normalizer), 1e-6)

    point_dists = np.linalg.norm(pred39 - truth, axis=1)
    fit_delta = float(np.mean(point_dists) / normalizer)
    center_delta = float(np.linalg.norm(_center(pred39) - _center(truth)) / normalizer)
    truth_scale = _bbox_diag(truth)
    scale_delta = float(abs(_bbox_diag(pred39) - truth_scale) / max(truth_scale, 1e-6))

    total = fit_weight * fit_delta + center_weight * center_delta + scale_weight * scale_delta
    return {
        "profile39_transform_cost": float(total),
        "profile39_fit_delta": fit_delta,
        "profile39_center_delta": center_delta,
        "profile39_scale_delta": scale_delta,
    }


def profile39_side_from_sample(sample: T.Mapping[str, T.Any]) -> str | None:
    """Return ``'left'`` / ``'right'`` for a 39-point sample, or ``None`` to skip.

    Resolves the visible side from ``profile_left`` / ``profile_right`` condition
    labels and ``metadata.yaw_side``. Unknown side returns ``None`` so the caller
    skips the partial row (the conservative first implementation - both-map
    evaluation with an uncertainty penalty can come later).
    """
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    explicit = (
        str((metadata or {}).get("yaw_side") or sample.get("yaw_side") or "").strip().lower()
    )
    if explicit in {"left", "right"}:
        return explicit

    labels: list[str] = [str(sample.get("condition") or "")]
    raw_conditions = sample.get("conditions") or ()
    if isinstance(raw_conditions, (list, tuple, set)):
        labels.extend(str(item) for item in raw_conditions)
    blob = " ".join(labels).lower()
    has_left = "left" in blob
    has_right = "right" in blob
    if has_left and not has_right:
        return "left"
    if has_right and not has_left:
        return "right"
    return None


def profile39_candidate_costs(
    candidates: T.Sequence[T.Any],
    truth39: T.Any,
    *,
    side: str,
    normalizer: float | None = None,
) -> dict[str, float]:
    """Return ``{candidate_name: profile39_transform_cost}`` for usable candidates."""
    truth = _as_truth39(truth39)
    costs: dict[str, float] = {}
    for candidate in candidates:
        landmarks = np.asarray(getattr(candidate, "landmarks", None), dtype="float32")
        if landmarks.shape != (68, 2) or not np.all(np.isfinite(landmarks)):
            continue
        costs[str(candidate.name)] = profile39_transform_cost(
            landmarks, truth, side=side, normalizer=normalizer
        )["profile39_transform_cost"]
    return costs


def profile39_rows(
    candidates: T.Sequence[T.Any],
    truth39: T.Any,
    *,
    side: str,
    normalizer: float | None = None,
) -> list[dict[str, T.Any]]:
    """Return per-candidate profile-39 ranking rows (cost / regret / oracle).

    The oracle is the lowest-cost candidate; ``profile39_transform_regret`` is
    ``candidate_cost - oracle_cost``. These feed the ``learned_quality_v3_profile``
    ranking target and are kept entirely separate from canonical NME/v3 fields.
    """
    costs = profile39_candidate_costs(candidates, truth39, side=side, normalizer=normalizer)
    if not costs:
        return []
    oracle = min(costs, key=lambda name: costs[name])
    oracle_cost = costs[oracle]
    truth = _as_truth39(truth39)
    rows: list[dict[str, T.Any]] = []
    for candidate in candidates:
        name = str(candidate.name)
        if name not in costs:
            continue
        landmarks = np.asarray(candidate.landmarks, dtype="float32")
        rows.append(
            {
                "candidate_name": name,
                "profile39_side": side,
                "profile39_transform_cost": costs[name],
                "profile39_transform_regret": max(costs[name] - oracle_cost, 0.0),
                "profile39_visible_side_error": profile39_point_error(
                    landmarks,
                    truth,
                    side=side,
                    normalizer=normalizer if normalizer is not None else _bbox_diag(truth),
                ),
                "profile39_oracle_candidate": oracle,
                "profile39_is_oracle": name == oracle,
                "profile39_rankable": True,
            }
        )
    return rows


__all__ = [
    "PROFILE39_POINT_COUNT",
    "PROFILE39_TO_CANONICAL68",
    "profile39_candidate_costs",
    "profile39_point_error",
    "profile39_rows",
    "profile39_side_from_sample",
    "profile39_transform_cost",
    "project_68_to_profile39",
]
