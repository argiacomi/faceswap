#!/usr/bin/env python3
"""Evaluate the bucket-aware production resolver policy."""

from __future__ import annotations

import sys
import typing as T
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.landmarks import evaluate_runtime_resolver as base

DEFAULT_PRIORITY: tuple[str, ...] = (
    "weighted_median",
    "static_weighted_downweight",
    "static_weighted",
    "static_weighted_hard_drop",
    "spiga",
    "orformer",
    "hrnet",
    "fan",
)

BUCKET_PRIORITIES: dict[str, tuple[str, ...]] = {
    "large_roll": (
        "static_weighted_downweight",
        "static_weighted",
        "weighted_median",
        "spiga",
        "orformer",
        "hrnet",
    ),
    "extreme_roll": (
        "hrnet",
        "spiga",
        "orformer",
        "static_weighted_downweight",
        "static_weighted",
    ),
    "large_yaw_left": (
        "spiga",
        "static_weighted_downweight",
        "static_weighted",
        "hrnet",
        "orformer",
    ),
    "large_yaw_right": (
        "spiga",
        "static_weighted_downweight",
        "static_weighted",
        "hrnet",
        "orformer",
    ),
    "profile_left": (
        "static_weighted_downweight",
        "static_weighted",
        "spiga",
        "hrnet",
        "orformer",
    ),
    "profile_right": (
        "static_weighted_downweight",
        "static_weighted",
        "hrnet",
        "spiga",
        "orformer",
    ),
    "rolled_large_yaw_left": ("spiga", "hrnet", "static_weighted_downweight", "orformer"),
    "rolled_large_yaw_right": ("spiga", "hrnet", "orformer", "static_weighted_downweight"),
    "rolled_profile_left": ("hrnet", "spiga", "static_weighted_downweight", "orformer"),
    "rolled_profile_right": (
        "spiga",
        "hrnet",
        "static_weighted_downweight",
        "orformer",
        "static_weighted_hard_drop",
    ),
}

_ORIGINAL_EVALUATE_SAMPLE = base.evaluate_sample


def _priority_for_bucket(condition: str) -> tuple[str, ...]:
    priority = list(BUCKET_PRIORITIES.get(condition, DEFAULT_PRIORITY))
    priority.extend(name for name in DEFAULT_PRIORITY if name not in priority)
    return tuple(priority)


def _available_by_priority(priority: T.Sequence[str], available: T.AbstractSet[str]) -> str:
    for name in priority:
        if name in available:
            return name
    if not available:
        raise ValueError("bucket_aware_veto received no candidates")
    return sorted(available)[0]


def _roll_vetoes(
    candidates: T.Sequence[base.CandidateRecord],
    metrics: T.Mapping[str, base.CandidateMetrics],
    *,
    threshold_deg: float = base.DEFAULT_ROLL_VETO_THRESHOLD_DEG,
) -> tuple[set[str], float | None]:
    rolls = [metric.roll_degrees for metric in metrics.values() if metric.roll_degrees is not None]
    if not rolls:
        return set(), None
    consensus = base._circular_median(rolls)
    fusion_names = {candidate.name for candidate in candidates if candidate.is_fusion}
    names: set[str] = set()
    for name in fusion_names:
        metric = metrics[name]
        if metric.roll_degrees is None:
            names.add(name)
            continue
        if abs(base._signed_degree_delta(metric.roll_degrees, consensus)) > threshold_deg:
            names.add(name)
    return names, consensus


def _shape_reasons(condition: str, name: str, metric: base.CandidateMetrics) -> tuple[str, ...]:
    reasons: list[str] = []
    if metric.cloud_area_ratio is None:
        reasons.append("missing_cloud_area_ratio")
    elif metric.cloud_area_ratio < base.DEFAULT_MIN_CLOUD_AREA_RATIO:
        reasons.append("cloud_area_too_small")
    elif metric.cloud_area_ratio > base.DEFAULT_MAX_CLOUD_AREA_RATIO:
        reasons.append("cloud_area_too_large")
    if metric.hull_area_ratio is None:
        reasons.append("missing_hull_area_ratio")
    elif metric.hull_area_ratio < base.DEFAULT_MIN_HULL_AREA_RATIO:
        reasons.append("hull_area_too_small")
    elif metric.hull_area_ratio > base.DEFAULT_MAX_HULL_AREA_RATIO:
        reasons.append("hull_area_too_large")
    if (
        metric.points_outside_expanded_bbox_fraction is not None
        and metric.points_outside_expanded_bbox_fraction
        > base.DEFAULT_MAX_POINTS_OUTSIDE_EXPANDED_BBOX_FRACTION
    ):
        reasons.append("too_many_points_outside_expanded_bbox")
    if (
        condition == "rolled_large_yaw_left"
        and name == "spiga"
        and metric.cloud_area_ratio is not None
        and metric.cloud_area_ratio < 0.55
    ):
        reasons.append("rolled_left_spiga_cloud_area_low")
    if (
        condition == "rolled_large_yaw_right"
        and name == "spiga"
        and metric.cloud_area_ratio is not None
        and metric.cloud_area_ratio < 0.60
    ):
        reasons.append("rolled_right_spiga_cloud_area_low")
    return tuple(reasons)


def _apply_shape_reasons(
    condition: str,
    metrics: T.MutableMapping[str, base.CandidateMetrics],
) -> set[str]:
    names: set[str] = set()
    for name, metric in metrics.items():
        metric.geometry_veto_reasons = _shape_reasons(condition, name, metric)
        if metric.geometry_veto_reasons:
            names.add(name)
    return names


def _geometry_reasons(metrics: T.Mapping[str, base.CandidateMetrics]) -> dict[str, list[str]]:
    return {
        name: list(metric.geometry_veto_reasons)
        for name, metric in metrics.items()
        if metric.geometry_veto_reasons
    }


def resolve_bucket_aware_veto(
    condition: str,
    candidates: T.Sequence[base.CandidateRecord],
    metrics: T.Mapping[str, base.CandidateMetrics],
) -> base.PolicyDecision:
    available = {candidate.name for candidate in candidates}
    priority = _priority_for_bucket(condition)
    roll_names, consensus = _roll_vetoes(candidates, metrics)
    shape_names = {name for name, metric in metrics.items() if metric.geometry_veto_reasons}
    vetoed = roll_names | shape_names
    survivors = available - vetoed
    diagnostics: dict[str, T.Any] = {
        "bucket_priority": list(priority),
        "bucket_priority_applied": condition in BUCKET_PRIORITIES,
        "roll_veto_scope": "fusion_only",
        "roll_vetoed": sorted(roll_names & available),
        "geometry_vetoed": sorted(shape_names & available),
        "geometry_veto_reasons": _geometry_reasons(metrics),
        "rolls": {name: metric.roll_degrees for name, metric in metrics.items()},
    }
    if survivors:
        chosen = _available_by_priority(priority, survivors)
    else:
        diagnostics["fallback_reason"] = "all_candidates_vetoed"
        chosen = _available_by_priority(priority, available)
    return base.PolicyDecision(
        policy="bucket_aware_veto",
        chosen=chosen,
        vetoed=tuple(sorted(vetoed & available)),
        consensus_roll_deg=consensus,
        diagnostics=diagnostics,
    )


def evaluate_sample(
    sample: base.LandmarkSample,
    *,
    cache: base.DiskPredictionCache,
    requested: T.Sequence[str],
    weights: T.Mapping[str, T.Sequence[float]] | None,
    policy: str,
    outlier_threshold: float,
    failure_threshold: float,
) -> base.SampleReport:
    if policy != "bucket_aware_veto":
        return _ORIGINAL_EVALUATE_SAMPLE(
            sample,
            cache=cache,
            requested=requested,
            weights=weights,
            policy=policy,
            outlier_threshold=outlier_threshold,
            failure_threshold=failure_threshold,
        )
    truth = base._load_truth(sample)
    candidates = base._build_candidates(
        sample, cache, requested, weights, outlier_threshold=outlier_threshold
    )
    reference_bbox = sample.face_bbox or base.bbox_from_truth_fallback(truth)
    metrics = {
        candidate.name: base._evaluate_candidate(
            candidate.landmarks,
            truth,
            normalizer=sample.normalizer,
            visibility=sample.visibility,
            failure_threshold=failure_threshold,
            face_bbox=reference_bbox,
        )
        for candidate in candidates
    }
    base._populate_consensus_geometry(candidates, metrics, reference_bbox=reference_bbox)
    _apply_shape_reasons(sample.condition, metrics)
    decision = resolve_bucket_aware_veto(sample.condition, candidates, metrics)
    oracle = min(metrics.items(), key=lambda item: item[1].nme)[0]
    decision.diagnostics["oracle_vetoed"] = oracle in decision.vetoed
    return base.SampleReport(
        sample_id=sample.sample_id,
        dataset=sample.dataset,
        condition=sample.condition,
        candidates=tuple(candidate.name for candidate in candidates),
        metrics=metrics,
        decision=decision,
        oracle=oracle,
        image_path=sample.image,
        truth=truth,
        landmarks_by_candidate={candidate.name: candidate.landmarks for candidate in candidates},
    )


def main(argv: T.Sequence[str] | None = None) -> int:
    base.POLICY_REGISTRY.setdefault("bucket_aware_veto", lambda candidates, metrics: candidates[0])
    base.evaluate_sample = evaluate_sample
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
