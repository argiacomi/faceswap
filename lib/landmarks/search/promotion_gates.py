#!/usr/bin/env python3
"""Promotion gates for the ensemble candidate search."""

from __future__ import annotations

import typing as T
from dataclasses import dataclass

from lib.landmarks.evaluation.effective_ensemble import DEFAULT_EFFECTIVE_MODELS_FLOOR
from lib.landmarks.search.candidate_search import CandidateResult

DEFAULT_REPORT_IMPROVEMENT_TOLERANCE: float = 0.001


@dataclass(frozen=True)
class GateConfig:
    """Configuration for the suite of promotion gates."""

    require_report_improvement: bool = False
    report_improvement_tolerance: float = DEFAULT_REPORT_IMPROVEMENT_TOLERANCE
    max_overall_regression_nme: float | None = None
    max_bucket_regression_rate: float | None = None
    require_profile_improvement: bool = False
    max_profile_region_failure_rate: float | None = None
    require_effective_ensemble: bool = False
    effective_models_floor: float = DEFAULT_EFFECTIVE_MODELS_FLOOR
    allow_single_model_baselines: bool = False
    allow_single_model_promotion: bool = False
    require_geometry_improvement: bool = False
    max_catastrophic_geometry_failure_rate: float | None = None
    max_per_bucket_catastrophic_geometry_failure_rate: float | None = None
    max_p95_transform_error: float | None = None
    max_p95_crop_center_error: float | None = None
    max_p95_roll_error: float | None = None
    min_hull_iou: float | None = None
    min_per_bucket_p05_hull_iou: float | None = None
    max_hard_slice_regression_rate: float | None = None
    allow_nme_only_promotion: bool = True

    def is_active(self) -> bool:
        """Return True when at least one gate is configured."""
        return any(
            (
                self.require_report_improvement,
                self.max_overall_regression_nme is not None,
                self.max_bucket_regression_rate is not None,
                self.require_profile_improvement,
                self.max_profile_region_failure_rate is not None,
                self.require_effective_ensemble,
                self.require_geometry_improvement,
                self.max_catastrophic_geometry_failure_rate is not None,
                self.max_per_bucket_catastrophic_geometry_failure_rate is not None,
                self.max_p95_transform_error is not None,
                self.max_p95_crop_center_error is not None,
                self.max_p95_roll_error is not None,
                self.min_hull_iou is not None,
                self.min_per_bucket_p05_hull_iou is not None,
                self.max_hard_slice_regression_rate is not None,
            )
        )

    def requires_geometry(self) -> bool:
        """Return True when any active gate depends on geometry metrics."""
        return any(
            (
                self.require_geometry_improvement,
                self.max_catastrophic_geometry_failure_rate is not None,
                self.max_per_bucket_catastrophic_geometry_failure_rate is not None,
                self.max_p95_transform_error is not None,
                self.max_p95_crop_center_error is not None,
                self.max_p95_roll_error is not None,
                self.min_hull_iou is not None,
                self.min_per_bucket_p05_hull_iou is not None,
                self.max_hard_slice_regression_rate is not None,
            )
        )


@dataclass(frozen=True)
class ProfileScore:
    """Profile-side per-candidate inputs to the gate framework."""

    overall_score: float
    region_failure_rate: float


@dataclass(frozen=True)
class GeometryScore:
    """GT-derived geometry per-candidate inputs to the gate framework."""

    overall_score: float
    catastrophic_failure_rate: float
    p95_translation_normalized: float
    p95_roi_center_normalized: float
    p95_roll_degrees: float
    mean_hull_iou: float
    p05_hull_iou: float
    max_bucket_regression_score: float = 0.0
    worst_bucket: str = ""
    worst_bucket_score: float = 0.0
    worst_bucket_baseline_score: float = 0.0
    per_bucket: T.Mapping[str, T.Mapping[str, float]] | None = None


@dataclass(frozen=True)
class GateOutcome:
    """Per-candidate gate decision."""

    candidate_id: str
    rank: int
    passed: bool
    failed_gates: tuple[str, ...] = ()
    failure_reasons: tuple[str, ...] = ()
    report_nme: float = 0.0
    profile_score: float | None = None
    profile_region_failure_rate: float | None = None
    geometry_score: float | None = None
    catastrophic_geometry_failure_rate: float | None = None
    max_per_bucket_catastrophic_geometry_failure_rate: float | None = None
    worst_catastrophic_geometry_bucket: str = ""
    p95_transform_error: float | None = None
    p95_crop_center_error: float | None = None
    p95_roll_error: float | None = None
    mean_hull_iou: float | None = None
    min_per_bucket_p05_hull_iou: float | None = None
    worst_hull_iou_bucket: str = ""
    max_bucket_regression_score: float | None = None

    def to_payload(self) -> dict[str, T.Any]:
        return {
            "candidate_id": self.candidate_id,
            "rank": int(self.rank),
            "passed": bool(self.passed),
            "failed_gates": list(self.failed_gates),
            "failure_reasons": list(self.failure_reasons),
            "report_nme": float(self.report_nme),
            "profile_score": self.profile_score,
            "profile_region_failure_rate": self.profile_region_failure_rate,
            "geometry_score": self.geometry_score,
            "catastrophic_geometry_failure_rate": self.catastrophic_geometry_failure_rate,
            "max_per_bucket_catastrophic_geometry_failure_rate": (
                self.max_per_bucket_catastrophic_geometry_failure_rate
            ),
            "worst_catastrophic_geometry_bucket": self.worst_catastrophic_geometry_bucket,
            "p95_transform_error": self.p95_transform_error,
            "p95_crop_center_error": self.p95_crop_center_error,
            "p95_roll_error": self.p95_roll_error,
            "mean_hull_iou": self.mean_hull_iou,
            "min_per_bucket_p05_hull_iou": self.min_per_bucket_p05_hull_iou,
            "worst_hull_iou_bucket": self.worst_hull_iou_bucket,
            "max_bucket_regression_score": self.max_bucket_regression_score,
        }


@dataclass(frozen=True)
class GateApplication:
    """Result of applying gates to a candidate result list."""

    outcomes: tuple[GateOutcome, ...]
    promoted: CandidateResult | None
    promoted_outcome: GateOutcome | None
    baseline_report_nme: float | None
    baseline_profile_score: float | None
    baseline_profile_region_failure_rate: float | None
    baseline_geometry_score: float | None = None
    baseline_geometry_catastrophic_rate: float | None = None
    baseline_geometry_hull_iou: float | None = None

    @property
    def passed_count(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for outcome in self.outcomes if not outcome.passed)

    def to_payload(self) -> dict[str, T.Any]:
        return {
            "outcomes": [outcome.to_payload() for outcome in self.outcomes],
            "promoted_candidate_id": self.promoted.candidate_id if self.promoted else "",
            "baseline_report_nme": self.baseline_report_nme,
            "baseline_profile_score": self.baseline_profile_score,
            "baseline_profile_region_failure_rate": self.baseline_profile_region_failure_rate,
            "baseline_geometry_score": self.baseline_geometry_score,
            "baseline_geometry_catastrophic_rate": self.baseline_geometry_catastrophic_rate,
            "baseline_geometry_hull_iou": self.baseline_geometry_hull_iou,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
        }


def _best_baseline_nme(results: T.Sequence[CandidateResult]) -> tuple[float | None, str]:
    baselines = [result for result in results if result.is_single_model_baseline]
    if not baselines:
        return None, ""
    winner = min(baselines, key=lambda result: result.metrics.overall_nme)
    return float(winner.metrics.overall_nme), winner.candidate.models[0]


def _bucket_extremes(
    geometry: GeometryScore | None,
) -> tuple[float | None, str, float | None, str]:
    if geometry is None or not geometry.per_bucket:
        return None, "", None, ""
    max_catastrophic: float | None = None
    max_catastrophic_bucket = ""
    min_hull: float | None = None
    min_hull_bucket = ""
    for bucket, values in geometry.per_bucket.items():
        catastrophic = float(values.get("catastrophic_failure_rate", 0.0))
        hull = float(values.get("p05_hull_iou", 1.0))
        if max_catastrophic is None or catastrophic > max_catastrophic:
            max_catastrophic = catastrophic
            max_catastrophic_bucket = bucket
        if min_hull is None or hull < min_hull:
            min_hull = hull
            min_hull_bucket = bucket
    return max_catastrophic, max_catastrophic_bucket, min_hull, min_hull_bucket


def apply_gates(
    results: T.Sequence[CandidateResult],
    config: GateConfig,
    *,
    profile_scores: T.Mapping[str, ProfileScore] | None = None,
    geometry_scores: T.Mapping[str, GeometryScore] | None = None,
) -> GateApplication:
    """Evaluate ``config`` against every candidate, in original score order."""
    baseline_nme, baseline_model_name = _best_baseline_nme(results)
    baseline_candidate_id = (
        baseline_model_candidate_id(results, baseline_model_name) if baseline_model_name else ""
    )
    baseline_profile = (
        profile_scores.get(baseline_candidate_id)
        if (profile_scores and baseline_candidate_id)
        else None
    )
    baseline_profile_score = baseline_profile.overall_score if baseline_profile else None
    baseline_profile_failure = baseline_profile.region_failure_rate if baseline_profile else None
    baseline_geometry = (
        geometry_scores.get(baseline_candidate_id)
        if (geometry_scores and baseline_candidate_id)
        else None
    )
    baseline_geometry_score = baseline_geometry.overall_score if baseline_geometry else None
    baseline_geometry_catastrophic = (
        baseline_geometry.catastrophic_failure_rate if baseline_geometry else None
    )
    baseline_geometry_hull = baseline_geometry.mean_hull_iou if baseline_geometry else None

    outcomes: list[GateOutcome] = []
    promoted: CandidateResult | None = None
    promoted_outcome: GateOutcome | None = None
    for rank, result in enumerate(results, start=1):
        report_nme = float(result.metrics.overall_nme)
        profile = profile_scores.get(result.candidate_id) if profile_scores else None
        profile_score = profile.overall_score if profile else None
        profile_failure = profile.region_failure_rate if profile else None
        geometry = geometry_scores.get(result.candidate_id) if geometry_scores else None
        (
            per_bucket_catastrophic,
            per_bucket_catastrophic_bucket,
            per_bucket_p05_hull,
            per_bucket_hull_bucket,
        ) = _bucket_extremes(geometry)
        failed_gates: list[str] = []
        failure_reasons: list[str] = []

        if config.require_report_improvement and baseline_nme is not None:
            margin = baseline_nme + config.report_improvement_tolerance
            if report_nme > margin:
                failed_gates.append("require_report_improvement")
                failure_reasons.append(
                    f"report NME {report_nme:.6f} exceeds best-single baseline "
                    f"{baseline_nme:.6f} + tolerance {config.report_improvement_tolerance:.6f}"
                )
        if config.max_overall_regression_nme is not None and baseline_nme is not None:
            delta = report_nme - baseline_nme
            if delta > config.max_overall_regression_nme:
                failed_gates.append("max_overall_regression_nme")
                failure_reasons.append(
                    f"report NME regression {delta:.6f} exceeds "
                    f"max_overall_regression_nme {config.max_overall_regression_nme:.6f}"
                )
        if config.max_bucket_regression_rate is not None:
            bucket_rate = float(result.metrics.bucket_regression_rate_vs_best_single)
            if bucket_rate > config.max_bucket_regression_rate:
                failed_gates.append("max_bucket_regression_rate")
                failure_reasons.append(
                    f"bucket regression rate {bucket_rate:.6f} exceeds "
                    f"max_bucket_regression_rate {config.max_bucket_regression_rate:.6f}"
                )
        if config.require_profile_improvement:
            if profile_score is None or baseline_profile_score is None:
                failed_gates.append("require_profile_improvement")
                failure_reasons.append(
                    "profile improvement gate is configured but profile metrics are "
                    "missing for this candidate or the baseline"
                )
            elif profile_score > baseline_profile_score:
                failed_gates.append("require_profile_improvement")
                failure_reasons.append(
                    f"profile score {profile_score:.6f} exceeds best-single profile "
                    f"score {baseline_profile_score:.6f}"
                )
        if config.max_profile_region_failure_rate is not None:
            if profile_failure is None:
                failed_gates.append("max_profile_region_failure_rate")
                failure_reasons.append(
                    "profile region failure rate gate is configured but profile metrics "
                    "are missing for this candidate"
                )
            elif profile_failure > config.max_profile_region_failure_rate:
                failed_gates.append("max_profile_region_failure_rate")
                failure_reasons.append(
                    f"profile region failure rate {profile_failure:.6f} exceeds "
                    f"max_profile_region_failure_rate {config.max_profile_region_failure_rate:.6f}"
                )
        if config.require_effective_ensemble:
            diagnostics = result.effective_ensemble
            if diagnostics is None or diagnostics.collapsed:
                failed_gates.append("require_effective_ensemble")
                if diagnostics is None:
                    failure_reasons.append(
                        "effective-ensemble diagnostics unavailable for this candidate"
                    )
                else:
                    failure_reasons.append(
                        "effective ensemble collapsed: "
                        f"mean_effective_models={diagnostics.mean_effective_models:.3f} "
                        f"(floor={diagnostics.effective_models_floor:.3f}), "
                        f"dominant={diagnostics.collapsed_dominant_model or 'n/a'}"
                    )

        if config.requires_geometry() and geometry is None:
            failed_gates.append("geometry_metrics_unavailable")
            failure_reasons.append(
                "geometry-side gate(s) configured but geometry metrics are missing "
                "for this candidate; pass --include-geometry-metrics or supply scores"
            )
        if config.require_geometry_improvement and baseline_geometry_score is None:
            failed_gates.append("geometry_baseline_unavailable")
            failure_reasons.append(
                "geometry improvement gate requires at least one single-model baseline "
                "with geometry metrics"
            )
        if geometry is not None:
            if (
                config.require_geometry_improvement
                and baseline_geometry_score is not None
                and geometry.overall_score > baseline_geometry_score
            ):
                failed_gates.append("require_geometry_improvement")
                failure_reasons.append(
                    f"geometry score {geometry.overall_score:.6f} exceeds "
                    f"best-single geometry score {baseline_geometry_score:.6f}"
                )
            if (
                config.max_catastrophic_geometry_failure_rate is not None
                and geometry.catastrophic_failure_rate
                > config.max_catastrophic_geometry_failure_rate
            ):
                failed_gates.append("max_catastrophic_geometry_failure_rate")
                failure_reasons.append(
                    f"catastrophic geometry failure rate "
                    f"{geometry.catastrophic_failure_rate:.6f} exceeds "
                    f"{config.max_catastrophic_geometry_failure_rate:.6f}"
                )
            if (
                config.max_per_bucket_catastrophic_geometry_failure_rate is not None
                and per_bucket_catastrophic is not None
                and per_bucket_catastrophic
                > config.max_per_bucket_catastrophic_geometry_failure_rate
            ):
                failed_gates.append("max_per_bucket_catastrophic_geometry_failure_rate")
                failure_reasons.append(
                    f"bucket {per_bucket_catastrophic_bucket!r} catastrophic geometry "
                    f"failure rate {per_bucket_catastrophic:.6f} exceeds "
                    f"{config.max_per_bucket_catastrophic_geometry_failure_rate:.6f}"
                )
            if (
                config.max_p95_transform_error is not None
                and geometry.p95_translation_normalized > config.max_p95_transform_error
            ):
                failed_gates.append("max_p95_transform_error")
                failure_reasons.append(
                    f"P95 transform error {geometry.p95_translation_normalized:.6f} exceeds "
                    f"{config.max_p95_transform_error:.6f}"
                )
            if (
                config.max_p95_crop_center_error is not None
                and geometry.p95_roi_center_normalized > config.max_p95_crop_center_error
            ):
                failed_gates.append("max_p95_crop_center_error")
                failure_reasons.append(
                    f"P95 crop-center error {geometry.p95_roi_center_normalized:.6f} exceeds "
                    f"{config.max_p95_crop_center_error:.6f}"
                )
            if (
                config.max_p95_roll_error is not None
                and geometry.p95_roll_degrees > config.max_p95_roll_error
            ):
                failed_gates.append("max_p95_roll_error")
                failure_reasons.append(
                    f"P95 roll error {geometry.p95_roll_degrees:.3f}° exceeds "
                    f"{config.max_p95_roll_error:.3f}°"
                )
            if config.min_hull_iou is not None and geometry.mean_hull_iou < config.min_hull_iou:
                failed_gates.append("min_hull_iou")
                failure_reasons.append(
                    f"mean hull IoU {geometry.mean_hull_iou:.6f} below floor "
                    f"{config.min_hull_iou:.6f}"
                )
            if (
                config.min_per_bucket_p05_hull_iou is not None
                and per_bucket_p05_hull is not None
                and per_bucket_p05_hull < config.min_per_bucket_p05_hull_iou
            ):
                failed_gates.append("min_per_bucket_p05_hull_iou")
                failure_reasons.append(
                    f"bucket {per_bucket_hull_bucket!r} P05 hull IoU "
                    f"{per_bucket_p05_hull:.6f} below floor "
                    f"{config.min_per_bucket_p05_hull_iou:.6f}"
                )
            if (
                config.max_hard_slice_regression_rate is not None
                and geometry.max_bucket_regression_score > config.max_hard_slice_regression_rate
            ):
                failed_gates.append("max_hard_slice_regression_rate")
                failure_reasons.append(
                    f"worst-bucket geometry regression "
                    f"{geometry.max_bucket_regression_score:.6f} exceeds "
                    f"{config.max_hard_slice_regression_rate:.6f}"
                )

        if result.is_single_model_baseline:
            if not config.allow_single_model_promotion:
                failed_gates.append("single_model_promotion_not_allowed")
                failure_reasons.append(
                    "single-model baselines are enabled for comparison but are not "
                    "allowed as promoted outputs; keep the best single-model baseline"
                )
            else:
                if baseline_nme is not None and report_nme >= baseline_nme:
                    failed_gates.append("single_model_does_not_beat_best_baseline_report")
                    failure_reasons.append(
                        f"single-model report NME {report_nme:.6f} does not beat "
                        f"best single-model baseline {baseline_nme:.6f}"
                    )
                if geometry is not None and baseline_geometry_score is not None:
                    if geometry.overall_score >= baseline_geometry_score:
                        failed_gates.append("single_model_does_not_beat_best_baseline_geometry")
                        failure_reasons.append(
                            f"single-model geometry score {geometry.overall_score:.6f} "
                            f"does not beat best single-model baseline {baseline_geometry_score:.6f}"
                        )

        passed = not failed_gates
        outcome = GateOutcome(
            candidate_id=result.candidate_id,
            rank=rank,
            passed=passed,
            failed_gates=tuple(failed_gates),
            failure_reasons=tuple(failure_reasons),
            report_nme=report_nme,
            profile_score=profile_score,
            profile_region_failure_rate=profile_failure,
            geometry_score=(geometry.overall_score if geometry is not None else None),
            catastrophic_geometry_failure_rate=(
                geometry.catastrophic_failure_rate if geometry is not None else None
            ),
            max_per_bucket_catastrophic_geometry_failure_rate=per_bucket_catastrophic,
            worst_catastrophic_geometry_bucket=per_bucket_catastrophic_bucket,
            p95_transform_error=(
                geometry.p95_translation_normalized if geometry is not None else None
            ),
            p95_crop_center_error=(
                geometry.p95_roi_center_normalized if geometry is not None else None
            ),
            p95_roll_error=(geometry.p95_roll_degrees if geometry is not None else None),
            mean_hull_iou=(geometry.mean_hull_iou if geometry is not None else None),
            min_per_bucket_p05_hull_iou=per_bucket_p05_hull,
            worst_hull_iou_bucket=per_bucket_hull_bucket,
            max_bucket_regression_score=(
                geometry.max_bucket_regression_score if geometry is not None else None
            ),
        )
        outcomes.append(outcome)

        if (
            promoted is None
            and passed
            and (not result.is_single_model_baseline or config.allow_single_model_promotion)
        ):
            promoted = result
            promoted_outcome = outcome

    return GateApplication(
        outcomes=tuple(outcomes),
        promoted=promoted,
        promoted_outcome=promoted_outcome,
        baseline_report_nme=baseline_nme,
        baseline_profile_score=baseline_profile_score,
        baseline_profile_region_failure_rate=baseline_profile_failure,
        baseline_geometry_score=baseline_geometry_score,
        baseline_geometry_catastrophic_rate=baseline_geometry_catastrophic,
        baseline_geometry_hull_iou=baseline_geometry_hull,
    )


def baseline_model_candidate_id(results: T.Sequence[CandidateResult], model_name: str) -> str:
    """Return the candidate_id of the single-model baseline for ``model_name``."""
    if not model_name:
        return ""
    for result in results:
        if result.is_single_model_baseline and result.candidate.models == (model_name,):
            return result.candidate_id
    return ""


def no_promotion_payload(application: GateApplication, *, top_n: int = 5) -> dict[str, T.Any]:
    """Build the payload written when no candidate passes the gates."""
    failing = [outcome for outcome in application.outcomes if not outcome.passed][:top_n]
    baseline_blocked = any(
        "single_model_promotion_not_allowed" in outcome.failed_gates
        for outcome in application.outcomes
    )
    return {
        "status": "no_promotion",
        "reason": (
            "no ensemble candidate passed all configured promotion gates; keep "
            "the current best single-model baseline"
            if baseline_blocked
            else "no candidate passed all configured promotion gates"
        ),
        "baseline_report_nme": application.baseline_report_nme,
        "baseline_profile_score": application.baseline_profile_score,
        "baseline_profile_region_failure_rate": application.baseline_profile_region_failure_rate,
        "top_failing_candidates": [outcome.to_payload() for outcome in failing],
        "passed_count": application.passed_count,
        "failed_count": application.failed_count,
    }


__all__ = [
    "DEFAULT_REPORT_IMPROVEMENT_TOLERANCE",
    "GateApplication",
    "GateConfig",
    "GateOutcome",
    "GeometryScore",
    "ProfileScore",
    "apply_gates",
    "baseline_model_candidate_id",
    "no_promotion_payload",
]
