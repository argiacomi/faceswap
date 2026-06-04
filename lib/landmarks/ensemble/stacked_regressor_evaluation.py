#!/usr/bin/env python3
"""Evaluation and promotion gates for the stacked residual regressor (#223).

Two views are produced:

- *Standalone candidate*: apply the regressor's clipped correction to each
  sample's base candidate and compare NME against the base candidate, broken out
  by runtime bucket and coarse hard-case slice. This isolates whether the
  correction itself helps without depending on the scorer.
- *Pipeline selection*: run the full runtime resolver with and without the
  stacked candidate to measure how often it is selected, its veto rate, and the
  net NME of the selected candidate. This confirms safe behavior when the
  candidate is present but not chosen.

Promotion gates translate the standalone report into a pass/fail decision that
blocks regressions on easy/frontal cases and any rise in catastrophic outliers.
"""

from __future__ import annotations

import logging
import math
import typing as T
from dataclasses import dataclass, field

import numpy as np

from lib.landmarks.ensemble.runtime_features import stacked_regression_feature_map
from lib.landmarks.ensemble.stacked_regressor import (
    RuntimeStackedLandmarkRegressor,
    apply_residual,
)
from lib.landmarks.ensemble.stacked_regressor_training import (
    _base_extent_diagonal,
    select_base_candidate,
)

logger = logging.getLogger(__name__)

ContextLike = T.Any

#: NME above which a candidate is considered a catastrophic alignment failure.
DEFAULT_CATASTROPHIC_NME = 0.08

#: Coarse hard-case slices used for promotion gating.
SLICE_FRONTAL = "frontal"
SLICE_LARGE_YAW = "large_yaw"
SLICE_PROFILE = "profile"
SLICE_ROLLED = "rolled"
SLICE_OTHER = "other"


def _slice_for_bucket(bucket: str) -> str:
    name = (bucket or "").lower()
    if "profile" in name:
        return SLICE_PROFILE
    if "large_yaw" in name or "yaw" in name:
        return SLICE_LARGE_YAW
    if "roll" in name:
        return SLICE_ROLLED
    if name in {"frontal", "intermediate", "no_pose", ""}:
        return SLICE_FRONTAL
    return SLICE_OTHER


def _nme(pred: np.ndarray, gt: np.ndarray, normalizer: T.Any, visibility: T.Any) -> float | None:
    if normalizer is None or not math.isfinite(float(normalizer)) or float(normalizer) <= 0:
        return None
    distances = np.linalg.norm(pred - gt, axis=1)
    if visibility is not None and len(visibility) == distances.shape[0]:
        mask = np.asarray(visibility, dtype=bool)
        if mask.any():
            distances = distances[mask]
    return float(np.mean(distances) / float(normalizer))


@dataclass
class _BucketAccumulator:
    base_nme: list[float] = field(default_factory=list)
    corrected_nme: list[float] = field(default_factory=list)
    residual_norm: list[float] = field(default_factory=list)
    clip_count: int = 0
    base_catastrophic: int = 0
    corrected_catastrophic: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0


def _percentile(values: T.Sequence[float], pct: float) -> float:
    return (
        float(np.percentile(np.asarray(values, dtype="float64"), pct)) if values else float("nan")
    )


def _median(values: T.Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype="float64"))) if values else float("nan")


def _gate_value(value: T.Any) -> float | None:
    """Return ``value`` as a finite float for gating, or ``None`` if unusable."""
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return float(value)
    return None


def _summarize(acc: _BucketAccumulator) -> dict[str, T.Any]:
    n = len(acc.base_nme)
    return {
        "count": n,
        "base_nme_median": _median(acc.base_nme),
        "corrected_nme_median": _median(acc.corrected_nme),
        "base_nme_p95": _percentile(acc.base_nme, 95),
        "corrected_nme_p95": _percentile(acc.corrected_nme, 95),
        "nme_median_change": _median(acc.corrected_nme) - _median(acc.base_nme),
        "win_rate": (acc.wins / n) if n else float("nan"),
        "loss_rate": (acc.losses / n) if n else float("nan"),
        "tie_rate": (acc.ties / n) if n else float("nan"),
        "base_catastrophic_rate": (acc.base_catastrophic / n) if n else float("nan"),
        "corrected_catastrophic_rate": (acc.corrected_catastrophic / n) if n else float("nan"),
        "catastrophic_rate_change": (
            (acc.corrected_catastrophic - acc.base_catastrophic) / n if n else float("nan")
        ),
        "residual_norm_mean": (
            float(np.mean(acc.residual_norm)) if acc.residual_norm else float("nan")
        ),
        "clip_rate": (acc.clip_count / n) if n else float("nan"),
    }


@dataclass
class StackedEvaluationReport:
    """Standalone candidate evaluation aggregates."""

    overall: dict[str, T.Any]
    by_bucket: dict[str, dict[str, T.Any]]
    by_slice: dict[str, dict[str, T.Any]]

    def to_dict(self) -> dict[str, T.Any]:
        return {
            "overall": self.overall,
            "by_bucket": self.by_bucket,
            "by_slice": self.by_slice,
        }


def evaluate_stacked_candidate(
    contexts: T.Iterable[ContextLike],
    regressor: RuntimeStackedLandmarkRegressor,
    *,
    catastrophic_nme: float = DEFAULT_CATASTROPHIC_NME,
    residual_clip_fraction: float | None = None,
) -> StackedEvaluationReport:
    """Compare the corrected candidate against its base candidate, by bucket/slice."""
    clip = (
        float(residual_clip_fraction)
        if residual_clip_fraction is not None
        else float(regressor.residual_clip_fraction)
    )
    overall = _BucketAccumulator()
    by_bucket: dict[str, _BucketAccumulator] = {}
    by_slice: dict[str, _BucketAccumulator] = {}

    for context in contexts:
        truth = getattr(context, "truth_landmarks", None)
        if truth is None:
            continue
        gt = np.asarray(truth, dtype="float64")
        if gt.shape != (68, 2) or not np.all(np.isfinite(gt)):
            continue
        base = select_base_candidate(context, regressor.base_candidate_policy)
        if base is None:
            continue
        base_landmarks = np.asarray(base.landmarks, dtype="float64")
        if base_landmarks.shape != (68, 2) or not np.all(np.isfinite(base_landmarks)):
            continue
        normalizer = getattr(context, "normalizer", None)
        visibility = getattr(context, "visibility", None)
        base_nme = _nme(base_landmarks, gt, normalizer, visibility)
        if base_nme is None:
            continue

        model_landmarks = {
            candidate.name: np.asarray(candidate.landmarks, dtype="float64")
            for candidate in getattr(context, "candidates", ())
            if not getattr(candidate, "is_fusion", False)
        }
        features = stacked_regression_feature_map(
            base_landmarks=base_landmarks,
            model_landmarks=model_landmarks,
            reference_bbox=None,
            runtime_bucket=str(getattr(context, "runtime_bucket", "") or ""),
            roll_estimate=getattr(context, "roll_estimate", None),
            yaw_estimate=getattr(context, "yaw_estimate", None),
            candidate_yaw_disagreement=getattr(context, "candidate_yaw_disagreement", None),
            max_disagreement_px=getattr(context, "max_disagreement_px", None),
            hard_case_tags=tuple(getattr(context, "hard_case_tags", ()) or ()),
            model_predictions_available=getattr(context, "model_predictions_available", None),
        )
        raw_output = regressor.predict(features)
        result = apply_residual(
            base_landmarks,
            raw_output,
            output_mode=regressor.output_mode,
            clip_fraction=clip,
            bbox_diagonal=_base_extent_diagonal(base_landmarks),
        )
        corrected_nme = _nme(
            np.asarray(result.landmarks, dtype="float64"), gt, normalizer, visibility
        )
        if corrected_nme is None:
            continue

        bucket = str(getattr(context, "runtime_bucket", "") or "unknown")
        slice_name = _slice_for_bucket(bucket)
        for acc in (
            overall,
            by_bucket.setdefault(bucket, _BucketAccumulator()),
            by_slice.setdefault(slice_name, _BucketAccumulator()),
        ):
            acc.base_nme.append(base_nme)
            acc.corrected_nme.append(corrected_nme)
            acc.residual_norm.append(result.residual_norm_mean)
            acc.clip_count += 1 if result.clip_applied else 0
            acc.base_catastrophic += 1 if base_nme > catastrophic_nme else 0
            acc.corrected_catastrophic += 1 if corrected_nme > catastrophic_nme else 0
            if corrected_nme < base_nme - 1e-9:
                acc.wins += 1
            elif corrected_nme > base_nme + 1e-9:
                acc.losses += 1
            else:
                acc.ties += 1

    return StackedEvaluationReport(
        overall=_summarize(overall),
        by_bucket={bucket: _summarize(acc) for bucket, acc in sorted(by_bucket.items())},
        by_slice={name: _summarize(acc) for name, acc in sorted(by_slice.items())},
    )


@dataclass(frozen=True)
class PromotionGateResult:
    """Outcome of applying promotion gates to an evaluation report."""

    passed: bool
    reasons: tuple[str, ...]
    details: dict[str, T.Any]


def evaluate_promotion_gates(
    report: StackedEvaluationReport,
    *,
    max_frontal_catastrophic_increase: float = 0.0,
    max_overall_catastrophic_increase: float = 0.0,
    max_frontal_nme_regression: float = 0.0005,
    min_hard_slice_win_rate: float = 0.5,
    hard_slices: T.Sequence[str] = (SLICE_PROFILE, SLICE_LARGE_YAW, SLICE_ROLLED),
    min_hard_slice_count: int = 1,
) -> PromotionGateResult:
    """Return a pass/fail promotion decision from the standalone evaluation report.

    Gates:

    - No increase in catastrophic outliers overall or on the frontal slice.
    - No material NME regression on the frontal/easy slice.
    - Targeted hard slices must show a win rate at or above the floor (only when
      enough samples exist to judge).
    """
    reasons: list[str] = []
    overall = report.overall
    frontal = report.by_slice.get(SLICE_FRONTAL, {})

    overall_cat = _gate_value(overall.get("catastrophic_rate_change"))
    if overall_cat is not None and overall_cat > max_overall_catastrophic_increase:
        reasons.append(
            f"overall catastrophic rate increased by {overall_cat:.4f} "
            f"> {max_overall_catastrophic_increase:.4f}"
        )

    frontal_cat = _gate_value(frontal.get("catastrophic_rate_change"))
    if frontal_cat is not None and frontal_cat > max_frontal_catastrophic_increase:
        reasons.append(
            f"frontal catastrophic rate increased by {frontal_cat:.4f} "
            f"> {max_frontal_catastrophic_increase:.4f}"
        )

    frontal_change = _gate_value(frontal.get("nme_median_change"))
    if frontal_change is not None and frontal_change > max_frontal_nme_regression:
        reasons.append(
            f"frontal median NME regressed by {frontal_change:.5f} "
            f"> {max_frontal_nme_regression:.5f}"
        )

    hard_details: dict[str, T.Any] = {}
    for slice_name in hard_slices:
        stats = report.by_slice.get(slice_name)
        if not stats or int(stats.get("count", 0)) < min_hard_slice_count:
            continue
        win_rate = _gate_value(stats.get("win_rate"))
        hard_details[slice_name] = win_rate
        if win_rate is not None and win_rate < min_hard_slice_win_rate:
            reasons.append(
                f"hard slice {slice_name!r} win rate {win_rate:.3f} "
                f"< {min_hard_slice_win_rate:.3f}"
            )

    details = {
        "overall_catastrophic_rate_change": overall_cat,
        "frontal_catastrophic_rate_change": frontal_cat,
        "frontal_nme_median_change": frontal_change,
        "hard_slice_win_rates": hard_details,
    }
    return PromotionGateResult(passed=not reasons, reasons=tuple(reasons), details=details)


__all__ = [
    "DEFAULT_CATASTROPHIC_NME",
    "SLICE_FRONTAL",
    "SLICE_LARGE_YAW",
    "SLICE_PROFILE",
    "SLICE_ROLLED",
    "PromotionGateResult",
    "StackedEvaluationReport",
    "evaluate_promotion_gates",
    "evaluate_stacked_candidate",
]
