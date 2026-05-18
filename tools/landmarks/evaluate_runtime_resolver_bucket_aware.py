#!/usr/bin/env python3
"""Evaluate the bucket-aware production resolver policy.

This is a thin wrapper around ``evaluate_runtime_resolver.py``. It adds the
``bucket_aware_veto`` policy without duplicating the report writers.
"""

from __future__ import annotations

import sys
import typing as T
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.landmarks import evaluate_runtime_resolver as base

DEFAULT_PRIORITY: tuple[str, ...] = (
    "static_weighted_downweight",
    "static_weighted",
    "static_weighted_hard_drop",
    "weighted_median",
    "spiga",
    "hrnet",
    "orformer",
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
    "rolled_large_yaw_left": (
        "spiga",
        "hrnet",
        "static_weighted_downweight",
        "orformer",
    ),
    "rolled_large_yaw_right": (
        "hrnet",
        "spiga",
        "orformer",
        "static_weighted_downweight",
    ),
    "rolled_profile_left": (
        "hrnet",
        "spiga",
        "static_weighted_downweight",
        "orformer",
    ),
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
    """Return fusion candidates whose roll disagrees with cohort consensus.

    Single-model predictions are intentionally not roll-vetoed. The current
    validation showed that roll consensus can reject the oracle single model on
    rolled/profile faces, while fusion candidates are the main source of
    avoidable geometry catastrophics.
    """
    rolls = [metric.roll_degrees for metric in metrics.values() if metric.roll_degrees is not None]
    if not rolls:
        return set(), None
    consensus = base._circular_median(rolls)
    fusion_names = {candidate.name for candidate in candidates if candidate.is_fusion}
    vetoed: set[str] = set()
    for name in fusion_names:
        metric = metrics[name]
        if metric.roll_degrees is None:
            vetoed.add(name)
            continue
        if abs(base._signed_degree_delta(metric.roll_degrees, consensus)) > threshold_deg:
            vetoed.add(name)
    return vetoed, consensus


def resolve_bucket_aware_veto(
    condition: str,
    candidates: T.Sequence[base.CandidateRecord],
    metrics: T.Mapping[str, base.CandidateMetrics],
) -> base.PolicyDecision:
    """Choose by bucket priority, after vetoing fusion roll outliers."""
    available = {candidate.name for candidate in candidates}
    priority = _priority_for_bucket(condition)
    vetoed, consensus = _roll_vetoes(candidates, metrics)
    survivors = available - vetoed
    diagnostics: dict[str, T.Any] = {
        "bucket_priority": list(priority),
        "bucket_priority_applied": condition in BUCKET_PRIORITIES,
        "roll_veto_scope": "fusion_only",
        "roll_veto_threshold_deg": base.DEFAULT_ROLL_VETO_THRESHOLD_DEG,
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
        sample,
        cache,
        requested,
        weights,
        outlier_threshold=outlier_threshold,
    )
    metrics = {
        candidate.name: base._evaluate_candidate(
            candidate.landmarks,
            truth,
            normalizer=sample.normalizer,
            visibility=sample.visibility,
            failure_threshold=failure_threshold,
        )
        for candidate in candidates
    }
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
