#!/usr/bin/env python3
"""Cache-only ensemble candidate search for the landmark static-weight pipeline (#69).

A "candidate" pins one ensemble configuration: a model subset, a weight
generator (with parameters), a fusion strategy, and a strategy-scoped outlier
threshold. The search runner fits weights on the ``fit`` split, scores each
candidate on the ``select`` split, picks the winner via an objective, and
defers final ``report`` metrics to the caller.

The search is cache-only: it consumes a populated prediction cache and never
re-runs landmark adapters. Re-prediction belongs in the cache-building stage,
not the search loop.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import typing as T
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lib.landmarks.ensemble.strategies import (
    canonical_strategy,
    strategy_outlier_method,
    strategy_requires_weights,
    strategy_uses_threshold,
    validate_threshold,
)
from lib.landmarks.ensemble.weight_generators import (
    ErrorTable,
    WeightFitResult,
    get_generator,
)
from lib.landmarks.ensemble.weights import weights_matrix_for_models
from lib.landmarks.eval.harness import LandmarkSample, load_manifest
from lib.landmarks.eval.metrics import evaluate_prediction
from lib.landmarks.eval.prediction_cache import DiskPredictionCache
from lib.landmarks.eval.splits import SplitAssignment, scenario_bucket
from lib.landmarks.fusion import (
    FusionResult,
    normalize_weight_matrix,
    plain_average,
    static_weighted,
)
from lib.landmarks.metrics import per_landmark_error
from lib.landmarks.rejection import weighted_median as weighted_median_fusion
from lib.landmarks.schema import CANONICAL_SCHEMA

DEFAULT_OBJECTIVE: str = "extract_alignment_v1"
DEFAULT_REGRESSION_EPSILON_NME: float = 0.001
SUBSET_PRESETS: tuple[str, ...] = ("all", "pairs", "triples")


@dataclass(frozen=True)
class Candidate:
    """One fully-specified ensemble configuration to evaluate."""

    models: tuple[str, ...]
    weight_generator: str
    weight_generator_params: tuple[tuple[str, float], ...] = ()
    strategy: str = "static_weighted"
    outlier_threshold: float | None = None
    bbox_source: str = "manifest"
    crop_scale: float = 1.6

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("Candidate must include at least one model")
        if len(set(self.models)) != len(self.models):
            raise ValueError(f"Candidate.models must be unique: {self.models!r}")
        canonical = canonical_strategy(self.strategy)
        object.__setattr__(self, "strategy", canonical)
        validate_threshold(canonical, self.outlier_threshold)
        # Generator name validated lazily inside fit; do a fast existence check now.
        get_generator(self.weight_generator)

    def generator_params_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in self.weight_generator_params}


@dataclass(frozen=True)
class CandidateMetrics:
    """Per-candidate metrics gathered on the select split."""

    sample_count: int
    overall_nme: float
    failure_rate: float
    auc: float
    regression_rate_vs_best_single: float
    bucket_regression_rate_vs_best_single: float
    per_bucket: dict[str, dict[str, float]] = field(default_factory=dict)
    best_single_model: str = ""

    def to_payload(self) -> dict[str, T.Any]:
        return {
            "sample_count": int(self.sample_count),
            "overall_nme": float(self.overall_nme),
            "failure_rate": float(self.failure_rate),
            "auc": float(self.auc),
            "regression_rate_vs_best_single": float(self.regression_rate_vs_best_single),
            "bucket_regression_rate_vs_best_single": float(
                self.bucket_regression_rate_vs_best_single
            ),
            "per_bucket": {bucket: dict(metrics) for bucket, metrics in self.per_bucket.items()},
            "best_single_model": self.best_single_model,
        }


@dataclass(frozen=True)
class CandidateResult:
    """Evaluation of one Candidate plus its identifying hash and score."""

    candidate: Candidate
    candidate_id: str
    weights: dict[str, list[float]]
    weights_hash: str
    score: float
    objective: str
    regression_epsilon_nme: float
    metrics: CandidateMetrics
    fit_diagnostics: dict[str, T.Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, T.Any]:
        return {
            "candidate_id": self.candidate_id,
            "models": list(self.candidate.models),
            "weight_generator": {
                "name": self.candidate.weight_generator,
                "params": self.candidate.generator_params_dict(),
            },
            "strategy": self.candidate.strategy,
            "outlier_threshold": self.candidate.outlier_threshold,
            "bbox_source": self.candidate.bbox_source,
            "crop_scale": self.candidate.crop_scale,
            "weights_hash": self.weights_hash,
            "objective": self.objective,
            "regression_epsilon_nme": self.regression_epsilon_nme,
            "score": float(self.score),
            "metrics": self.metrics.to_payload(),
            "fit_diagnostics": _json_ready(self.fit_diagnostics),
        }


def _json_ready(value: T.Any) -> T.Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def candidate_id(
    candidate: Candidate,
    *,
    weights_hash: str,
    split_assignment_hash: str,
    objective: str,
) -> str:
    """Return a stable ``sha256:...`` identifier for a candidate evaluation.

    The hash covers every dimension that changes how a candidate behaves: the
    model subset (order-insensitive), generator name + params, weights hash,
    strategy, outlier threshold (when used), bbox source and crop scale, and
    the split assignment + objective version (since changing the splits or the
    objective invalidates prior candidate IDs).
    """
    payload = {
        "models": sorted(candidate.models),
        "weight_generator": {
            "name": candidate.weight_generator,
            "params": candidate.generator_params_dict(),
        },
        "strategy": candidate.strategy,
        "outlier_threshold": candidate.outlier_threshold,
        "bbox_source": candidate.bbox_source,
        "crop_scale": candidate.crop_scale,
        "weights_hash": weights_hash,
        "split_assignment_hash": split_assignment_hash,
        "objective": objective,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return "sha256:" + digest


def weights_hash(weights: T.Mapping[str, T.Sequence[float]]) -> str:
    """Return a stable ``sha256:...`` hash of normalized per-landmark weights."""
    payload = {model: [float(value) for value in values] for model, values in weights.items()}
    return (
        "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    )


def expand_model_subsets(
    models: T.Sequence[str], presets: T.Sequence[str]
) -> tuple[tuple[str, ...], ...]:
    """Expand subset preset names into actual model-tuple subsets.

    Supported presets: ``all`` (the full set), ``pairs`` (every 2-combination),
    ``triples`` (every 3-combination). Unknown names raise ``ValueError`` with
    the supported list.
    """
    model_tuple = tuple(models)
    if not model_tuple:
        raise ValueError("models must be non-empty")
    unknown = [name for name in presets if name not in SUBSET_PRESETS]
    if unknown:
        raise ValueError(
            f"unknown model-subset preset(s) {unknown!r}; supported: {', '.join(SUBSET_PRESETS)}"
        )
    subsets: list[tuple[str, ...]] = []
    if "all" in presets:
        subsets.append(model_tuple)
    if "pairs" in presets:
        subsets.extend(tuple(combo) for combo in itertools.combinations(model_tuple, 2))
    if "triples" in presets and len(model_tuple) >= 3:
        subsets.extend(tuple(combo) for combo in itertools.combinations(model_tuple, 3))
    # Deduplicate while preserving order.
    seen: set[tuple[str, ...]] = set()
    ordered: list[tuple[str, ...]] = []
    for subset in subsets:
        key = tuple(sorted(subset))
        if key in seen:
            continue
        seen.add(key)
        ordered.append(subset)
    return tuple(ordered)


def enumerate_candidates(
    *,
    models: T.Sequence[str],
    model_subset_presets: T.Sequence[str],
    weight_generators: T.Sequence[str],
    strategies: T.Sequence[str],
    outlier_thresholds: T.Sequence[float],
    bbox_source: str = "manifest",
    crop_scale: float = 1.6,
) -> list[Candidate]:
    """Cartesian product of search dimensions, honoring strategy-scoped threshold rules.

    For strategies that do not consume a threshold the threshold dimension is
    skipped (one candidate per strategy/subset/generator). For strategies that
    do consume a threshold a candidate is emitted per supplied threshold.
    """
    subsets = expand_model_subsets(models, model_subset_presets)
    if not subsets:
        raise ValueError("no model subsets produced from the requested presets")
    if not weight_generators:
        raise ValueError("at least one weight generator is required")
    if not strategies:
        raise ValueError("at least one strategy is required")
    canonical_list = [canonical_strategy(name) for name in strategies]
    candidates: list[Candidate] = []
    for subset, generator, strategy in itertools.product(
        subsets, weight_generators, canonical_list
    ):
        if strategy_uses_threshold(strategy):
            if not outlier_thresholds:
                raise ValueError(
                    f"strategy {strategy!r} requires --outlier-thresholds but none were supplied"
                )
            for threshold in outlier_thresholds:
                candidates.append(
                    Candidate(
                        models=tuple(subset),
                        weight_generator=generator,
                        strategy=strategy,
                        outlier_threshold=float(threshold),
                        bbox_source=bbox_source,
                        crop_scale=crop_scale,
                    )
                )
        else:
            candidates.append(
                Candidate(
                    models=tuple(subset),
                    weight_generator=generator,
                    strategy=strategy,
                    outlier_threshold=None,
                    bbox_source=bbox_source,
                    crop_scale=crop_scale,
                )
            )
    return candidates


def _build_error_table(
    samples: T.Sequence[LandmarkSample],
    cache: DiskPredictionCache,
    models: T.Sequence[str],
) -> ErrorTable:
    """Assemble per-sample per-landmark error matrices from a populated cache."""
    if not samples:
        raise ValueError("fit-split samples must be non-empty")
    errors: dict[str, list[np.ndarray]] = {model: [] for model in models}
    for sample in samples:
        truth = np.load(sample.landmarks).astype("float32")
        for model in models:
            prediction = cache.read(sample.sample_id, model)
            errors[model].append(per_landmark_error(prediction.landmarks, truth))
    return ErrorTable.from_samples(errors)


def _fit_weights(
    candidate: Candidate,
    fit_samples: T.Sequence[LandmarkSample],
    cache: DiskPredictionCache,
) -> WeightFitResult:
    """Fit static weights for one candidate using its generator + the fit split."""
    table = _build_error_table(fit_samples, cache, candidate.models)
    generator_cls = type(get_generator(candidate.weight_generator))
    generator = generator_cls(**candidate.generator_params_dict())
    return generator.fit(table, models=candidate.models)


def _fuse_candidate(
    candidate: Candidate,
    cached_predictions: T.Sequence[np.ndarray],
    *,
    weight_matrix: np.ndarray,
) -> FusionResult:
    """Fuse one face's cached predictions using a candidate's strategy + weights."""
    from lib.landmarks.schema import LandmarkPrediction

    predictions = [
        LandmarkPrediction(points.astype("float32"), source=model)
        for points, model in zip(cached_predictions, candidate.models, strict=True)
    ]
    method = strategy_outlier_method(candidate.strategy)
    threshold = candidate.outlier_threshold if strategy_uses_threshold(candidate.strategy) else 3.5
    if not strategy_requires_weights(candidate.strategy):
        return plain_average(predictions, outlier_method=method, outlier_threshold=threshold)
    if candidate.strategy == "weighted_median":
        stack = np.stack([prediction.canonical_68().points for prediction in predictions], axis=0)
        normalized = normalize_weight_matrix(
            weight_matrix, model_count=stack.shape[0], landmark_count=stack.shape[1]
        )
        points = weighted_median_fusion(stack, normalized)
        return FusionResult(
            points=points,
            schema=CANONICAL_SCHEMA,
            strategy=candidate.strategy,
            weights=normalized,
            sources=tuple(candidate.models),
            kept_indices=tuple(range(len(predictions))),
        )
    return static_weighted(
        predictions,
        weight_matrix,
        outlier_method=method,
        outlier_threshold=threshold,
    )


def evaluate_candidate(
    candidate: Candidate,
    *,
    fit_samples: T.Sequence[LandmarkSample],
    select_samples: T.Sequence[LandmarkSample],
    cache: DiskPredictionCache,
    failure_threshold: float = 0.08,
    regression_epsilon_nme: float = DEFAULT_REGRESSION_EPSILON_NME,
) -> tuple[WeightFitResult, CandidateMetrics]:
    """Fit weights on ``fit_samples`` and evaluate the candidate on ``select_samples``.

    Returns the fit result alongside metrics that include sample-level and
    bucket-level regression rate against the best single model in the subset.
    """
    fit_result = _fit_weights(candidate, fit_samples, cache)
    weight_matrix = weights_matrix_for_models(fit_result.weights, candidate.models)

    nmes: list[float] = []
    failures: list[bool] = []
    regressions: list[bool] = []
    per_bucket_nmes: dict[str, list[float]] = {}
    per_bucket_regressions: dict[str, list[bool]] = {}
    single_total_nme: dict[str, float] = {model: 0.0 for model in candidate.models}
    single_counts: dict[str, int] = {model: 0 for model in candidate.models}

    for sample in select_samples:
        truth = np.load(sample.landmarks).astype("float32")
        cached_points = [
            cache.read(sample.sample_id, model).landmarks for model in candidate.models
        ]
        single_nmes: dict[str, float] = {}
        for model, points in zip(candidate.models, cached_points, strict=True):
            metrics = evaluate_prediction(
                points,
                truth,
                normalizer=sample.normalizer,
                failure_threshold=failure_threshold,
            )
            single_nmes[model] = float(metrics["nme"])
            single_total_nme[model] += single_nmes[model]
            single_counts[model] += 1
        best_single_nme = min(single_nmes.values())

        fused = _fuse_candidate(candidate, cached_points, weight_matrix=weight_matrix)
        fused_metrics = evaluate_prediction(
            fused.points,
            truth,
            normalizer=sample.normalizer,
            failure_threshold=failure_threshold,
        )
        ensemble_nme = float(fused_metrics["nme"])
        nmes.append(ensemble_nme)
        failures.append(bool(fused_metrics["failure"]))
        regressed = ensemble_nme > best_single_nme + regression_epsilon_nme
        regressions.append(regressed)
        bucket = scenario_bucket({"dataset": sample.dataset, "condition": sample.condition})
        per_bucket_nmes.setdefault(bucket, []).append(ensemble_nme)
        per_bucket_regressions.setdefault(bucket, []).append(regressed)

    overall_nme = float(np.mean(nmes)) if nmes else 0.0
    failure_rate = float(np.mean(failures)) if failures else 0.0
    auc = _auc_from_nmes(nmes, threshold=failure_threshold)
    sample_regression = float(np.mean(regressions)) if regressions else 0.0
    bucket_regression = (
        float(np.mean([float(np.mean(values)) for values in per_bucket_regressions.values()]))
        if per_bucket_regressions
        else 0.0
    )
    per_bucket: dict[str, dict[str, float]] = {}
    for bucket, values in per_bucket_nmes.items():
        per_bucket[bucket] = {
            "nme": float(np.mean(values)),
            "regression_rate": float(np.mean(per_bucket_regressions[bucket])),
            "count": float(len(values)),
        }

    if single_counts:
        mean_single = {
            model: single_total_nme[model] / single_counts[model]
            for model in candidate.models
            if single_counts[model] > 0
        }
        best_single_model = min(mean_single, key=lambda key: mean_single[key])
    else:
        best_single_model = ""

    metrics = CandidateMetrics(
        sample_count=len(select_samples),
        overall_nme=overall_nme,
        failure_rate=failure_rate,
        auc=auc,
        regression_rate_vs_best_single=sample_regression,
        bucket_regression_rate_vs_best_single=bucket_regression,
        per_bucket=per_bucket,
        best_single_model=best_single_model,
    )
    return fit_result, metrics


def _auc_from_nmes(nmes: T.Sequence[float], *, threshold: float = 0.08, steps: int = 100) -> float:
    """Trapezoidal AUC of the cumulative error distribution up to ``threshold``."""
    values = np.asarray(nmes, dtype="float32")
    if values.size == 0:
        return 0.0
    xs = np.linspace(0.0, threshold, steps, dtype="float32")
    ced = np.array([float(np.mean(values <= x)) for x in xs], dtype="float32")
    return float(np.trapezoid(ced, xs) / threshold)


def score_v1(metrics: CandidateMetrics) -> float:
    """Default ``extract_alignment_v1`` objective.

    ``score = overall_nme + 0.5 * failure_rate + 0.25 * sample_regression_rate
              + 0.25 * bucket_regression_rate``

    Coefficients are chosen so each rate term lands in roughly the same
    ``O(0.01 - 0.05)`` range as NME for typical landmark error distributions,
    rather than allowing any single dimension to dominate by scale alone.
    """
    return (
        float(metrics.overall_nme)
        + 0.5 * float(metrics.failure_rate)
        + 0.25 * float(metrics.regression_rate_vs_best_single)
        + 0.25 * float(metrics.bucket_regression_rate_vs_best_single)
    )


def select_best_candidate(
    results: T.Sequence[CandidateResult],
) -> CandidateResult:
    """Return the candidate with the lowest objective score (ties broken by lower NME)."""
    if not results:
        raise ValueError("results must be non-empty")
    return min(
        results,
        key=lambda result: (result.score, result.metrics.overall_nme),
    )


def run_candidate_search(
    candidates: T.Sequence[Candidate],
    *,
    fit_samples: T.Sequence[LandmarkSample],
    select_samples: T.Sequence[LandmarkSample],
    cache: DiskPredictionCache,
    split_assignment_hash: str,
    objective: str = DEFAULT_OBJECTIVE,
    regression_epsilon_nme: float = DEFAULT_REGRESSION_EPSILON_NME,
    failure_threshold: float = 0.08,
) -> list[CandidateResult]:
    """Evaluate every candidate and return CandidateResults sorted by score."""
    results: list[CandidateResult] = []
    for candidate in candidates:
        fit_result, metrics = evaluate_candidate(
            candidate,
            fit_samples=fit_samples,
            select_samples=select_samples,
            cache=cache,
            failure_threshold=failure_threshold,
            regression_epsilon_nme=regression_epsilon_nme,
        )
        whash = weights_hash(fit_result.weights)
        cid = candidate_id(
            candidate,
            weights_hash=whash,
            split_assignment_hash=split_assignment_hash,
            objective=objective,
        )
        score = score_v1(metrics) if objective == DEFAULT_OBJECTIVE else score_v1(metrics)
        results.append(
            CandidateResult(
                candidate=candidate,
                candidate_id=cid,
                weights=fit_result.weights,
                weights_hash=whash,
                score=score,
                objective=objective,
                regression_epsilon_nme=regression_epsilon_nme,
                metrics=metrics,
                fit_diagnostics=dict(fit_result.diagnostics),
            )
        )
    return sorted(results, key=lambda result: (result.score, result.metrics.overall_nme))


def load_split_samples(
    manifest_path: str | Path, assignment: SplitAssignment, split_name: str
) -> list[LandmarkSample]:
    """Load the LandmarkSample list for one named split from the full manifest."""
    full = load_manifest(manifest_path)
    by_id = {sample.sample_id: sample for sample in full}
    wanted = assignment.ids_for(split_name)
    missing = [sid for sid in wanted if sid not in by_id]
    if missing:
        raise ValueError(
            f"manifest is missing samples named in the {split_name!r} split: {missing}"
        )
    return [by_id[sid] for sid in wanted]


__all__ = [
    "Candidate",
    "CandidateMetrics",
    "CandidateResult",
    "DEFAULT_OBJECTIVE",
    "DEFAULT_REGRESSION_EPSILON_NME",
    "SUBSET_PRESETS",
    "candidate_id",
    "enumerate_candidates",
    "evaluate_candidate",
    "expand_model_subsets",
    "load_split_samples",
    "run_candidate_search",
    "score_v1",
    "select_best_candidate",
    "weights_hash",
]
