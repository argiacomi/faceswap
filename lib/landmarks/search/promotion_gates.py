#!/usr/bin/env python3
"""Promotion gates for the ensemble candidate search (#77).

The gate framework protects extract-alignment quality by only promoting
candidates that match or beat a reference single-model baseline on the
metrics that actually matter for Faceswap output:

* general report NME and bucket-level regression rate (classical)
* profile alignment score and profile region failure rate (from #76)
* effective-ensemble status (from #79) — refuse promotion of collapsed
  ensembles unless explicitly allowed

Gate logic is intentionally hardware-cheap: it only consumes data the
candidate search already produced (report metrics, profile metrics computed
from the same cached predictions, effective-ensemble diagnostics). No
adapter re-runs.
"""

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
    # Geometry-first gates (alignment-geometry roadmap, Phase 5 / #77).
    require_geometry_improvement: bool = False
    max_catastrophic_geometry_failure_rate: float | None = None
    max_p95_transform_error: float | None = None
    max_p95_crop_center_error: float | None = None
    max_p95_roll_error: float | None = None
    min_hull_iou: float | None = None
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
                self.max_p95_transform_error is not None,
                self.max_p95_crop_center_error is not None,
                self.max_p95_roll_error is not None,
                self.min_hull_iou is not None,
                self.max_hard_slice_regression_rate is not None,
            )
        )

    def requires_geometry(self) -> bool:
        """Return True when any active gate depends on geometry metrics."""
        return any(
            (
                self.require_geometry_improvement,
                self.max_catastrophic_geometry_failure_rate is not None,
                self.max_p95_transform_error is not None,
                self.max_p95_crop_center_error is not None,
                self.max_p95_roll_error is not None,
                self.min_hull_iou is not None,
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
    """GT-derived geometry per-candidate inputs to the gate framework.

    The bucket-level fields (``worst_bucket``, ``worst_bucket_score``,
    ``worst_bucket_baseline_score``, ``per_bucket``) are populated by
    :func:`lib.landmarks.search.geometry_search.geometry_score_from_aggregate`
    so downstream artifacts (candidate_results.json, no_promotion.json)
    can persist the slice that drives ``max_bucket_regression_score``
    without recomputing anything.
    """

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
    rank: int  # 1-indexed position in the score-sorted result list
    passed: bool
    failed_gates: tuple[str, ...] = ()
    failure_reasons: tuple[str, ...] = ()
    report_nme: float = 0.0
    profile_score: float | None = None
    profile_region_failure_rate: float | None = None
    geometry_score: float | None = None
    catastrophic_geometry_failure_rate: float | None = None
    p95_transform_error: float | None = None
    p95_crop_center_error: float | None = None
    p95_roll_error: float | None = None
    mean_hull_iou: float | None = None
    max_bucket_regression_score: float | None = None

    def to_payload(self) -> dict[str, T.Any]:
        return {
            "candidate_id": self.candidate_id,
            "rank": int(self.rank),
            "passed": bool(self.passed),
            "failed_gates": list(self.failed_gates),
            "failure_reasons": list(self.failure_reasons),
            "report_nme": float(self.report_nme),
            "profile_score": (
                float(self.profile_score) if self.profile_score is not None else None
            ),
            "profile_region_failure_rate": (
                float(self.profile_region_failure_rate)
                if self.profile_region_failure_rate is not None
                else None
            ),
            "geometry_score": (
                float(self.geometry_score) if self.geometry_score is not None else None
            ),
            "catastrophic_geometry_failure_rate": (
                float(self.catastrophic_geometry_failure_rate)
                if self.catastrophic_geometry_failure_rate is not None
                else None
            ),
            "p95_transform_error": (
                float(self.p95_transform_error) if self.p95_transform_error is not None else None
            ),
            "p95_crop_center_error": (
                float(self.p95_crop_center_error)
                if self.p95_crop_center_error is not None
                else None
            ),
            "p95_roll_error": (
                float(self.p95_roll_error) if self.p95_roll_error is not None else None
            ),
            "mean_hull_iou": (
                float(self.mean_hull_iou) if self.mean_hull_iou is not None else None
            ),
            "max_bucket_regression_score": (
                float(self.max_bucket_regression_score)
                if self.max_bucket_regression_score is not None
                else None
            ),
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


def _best_baseline_nme(
    results: T.Sequence[CandidateResult],
) -> tuple[float | None, str]:
    """Return the lowest single-model-baseline report NME among ``results``."""
    baselines = [result for result in results if result.is_single_model_baseline]
    if not baselines:
        return None, ""
    winner = min(baselines, key=lambda result: result.metrics.overall_nme)
    return float(winner.metrics.overall_nme), winner.candidate.models[0]


def apply_gates(
    results: T.Sequence[CandidateResult],
    config: GateConfig,
    *,
    profile_scores: T.Mapping[str, ProfileScore] | None = None,
    geometry_scores: T.Mapping[str, GeometryScore] | None = None,
) -> GateApplication:
    """Evaluate ``config`` against every candidate, in original score order.

    ``profile_scores`` / ``geometry_scores`` are candidate-id → score lookups
    for the profile and GT-derived-geometry gate families respectively. Absent
    keys disable the matching gates for that candidate even if those gates are
    configured. The first candidate that passes every active gate is selected
    as the promoted setup. When ``config.allow_single_model_baselines`` is
    false (the default), single-model baseline candidates are skipped during
    promotion selection — they are still scored in the outcomes for reporting.
    """
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
                    f"max_profile_region_failure_rate "
                    f"{config.max_profile_region_failure_rate:.6f}"
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

        # Geometry-first gates (alignment-geometry roadmap, Phase 5).
        if config.requires_geometry() and geometry is None:
            failed_gates.append("geometry_metrics_unavailable")
            failure_reasons.append(
                "geometry-side gate(s) configured but geometry metrics are missing "
                "for this candidate; pass --include-geometry-metrics or supply scores"
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
                config.max_hard_slice_regression_rate is not None
                and geometry.max_bucket_regression_score > config.max_hard_slice_regression_rate
            ):
                failed_gates.append("max_hard_slice_regression_rate")
                failure_reasons.append(
                    f"worst-bucket geometry regression "
                    f"{geometry.max_bucket_regression_score:.6f} exceeds "
                    f"{config.max_hard_slice_regression_rate:.6f}"
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
            p95_transform_error=(
                geometry.p95_translation_normalized if geometry is not None else None
            ),
            p95_crop_center_error=(
                geometry.p95_roi_center_normalized if geometry is not None else None
            ),
            p95_roll_error=(geometry.p95_roll_degrees if geometry is not None else None),
            mean_hull_iou=(geometry.mean_hull_iou if geometry is not None else None),
            max_bucket_regression_score=(
                geometry.max_bucket_regression_score if geometry is not None else None
            ),
        )
        outcomes.append(outcome)

        if (
            promoted is None
            and passed
            and (config.allow_single_model_baselines or not result.is_single_model_baseline)
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


def baseline_model_candidate_id(
    results: T.Sequence[CandidateResult],
    model_name: str,
) -> str:
    """Return the candidate_id of the single-model baseline for ``model_name``."""
    if not model_name:
        return ""
    for result in results:
        if result.is_single_model_baseline and result.candidate.models == (model_name,):
            return result.candidate_id
    return ""


def no_promotion_payload(
    application: GateApplication,
    *,
    top_n: int = 5,
) -> dict[str, T.Any]:
    """Build the payload written when no candidate passes the gates.

    The payload includes the top ``top_n`` failing candidates by score (the
    leading edge of the search space) plus their per-gate failure reasons so
    operators can act on the regression instead of blindly promoting.
    """
    failing = [outcome for outcome in application.outcomes if not outcome.passed][:top_n]
    return {
        "status": "no_promotion",
        "reason": "no candidate passed all configured promotion gates",
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
