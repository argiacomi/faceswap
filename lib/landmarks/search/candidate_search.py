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

import typing as T
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lib.landmarks.cache.prediction_cache import DiskPredictionCache
from lib.landmarks.core.fusion import (
    FusionResult,
    normalize_weight_matrix,
    plain_average,
    static_weighted,
)
from lib.landmarks.core.metrics import per_landmark_error
from lib.landmarks.core.rejection import weighted_median as weighted_median_fusion
from lib.landmarks.core.schema import CANONICAL_SCHEMA, LandmarkPrediction
from lib.landmarks.ensemble.strategies import (
    strategy_outlier_method,
    strategy_requires_weights,
    strategy_uses_threshold,
)
from lib.landmarks.ensemble.weight_generators import (
    ErrorTable,
    WeightFitResult,
    get_generator,
)
from lib.landmarks.ensemble.weights import weights_matrix_for_models
from lib.landmarks.evaluation.effective_ensemble import (
    DEFAULT_EFFECTIVE_MODELS_FLOOR,
    EffectiveEnsembleDiagnostics,
)
from lib.landmarks.evaluation.effective_ensemble import (
    diagnose as diagnose_effective_ensemble,
)
from lib.landmarks.evaluation.harness import LandmarkSample, load_manifest
from lib.landmarks.evaluation.nme_metrics import evaluate_prediction
from lib.landmarks.evaluation.splits import SplitAssignment, scenario_bucket
from lib.landmarks.search.candidates import (
    SUBSET_PRESETS,
    Candidate,
    CandidateProgress,
    candidate_id,
    enumerate_candidates,
    expand_model_subsets,
    single_model_baseline_candidates,
    weights_hash,
)

DEFAULT_OBJECTIVE: str = "extract_alignment_v1"
DEFAULT_REGRESSION_EPSILON_NME: float = 0.001


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
    # P95 of the ensemble NME across the select split — fed to the overall
    # magnitude gates so tail behavior, not just the mean, can block a
    # promotion.
    overall_p95_nme: float = 0.0
    # Best-single overall baselines (lowest mean / p95 any member model
    # would have produced alone). The ``max_*_nme_regression`` fields below
    # are computed against these so a candidate isn't penalized for
    # absolute NME, only for *worsening* the slice vs the best alternative.
    best_single_overall_mean_nme: float = 0.0
    best_single_overall_p95_nme: float = 0.0
    # Magnitude-aware overall NME regressions (recommended gate inputs).
    # ``max_mean_nme_regression`` caps how much the ensemble's mean NME
    # may exceed the best-single baseline overall; ``max_p95_nme_regression``
    # does the same for the p95.
    max_mean_nme_regression: float = 0.0
    max_p95_nme_regression: float = 0.0
    # Magnitude-aware bucket regressions: max across buckets of (ensemble
    # mean/p95 NME minus the per-bucket best-single NME). Surfaces the
    # *size* of the worst worsened slice; the gate framework prefers these
    # over the count-only ``bucket_regression_rate_vs_best_single`` term
    # (kept as a diagnostic, no longer the recommended gate).
    max_bucket_mean_nme_regression: float = 0.0
    max_bucket_p95_nme_regression: float = 0.0

    def to_payload(self) -> dict[str, T.Any]:
        # Keep the explicit ``max_*`` field names used by the gate framework,
        # and expose concise aliases for report/JQ consumers that read the
        # metrics payload directly.
        return {
            "sample_count": int(self.sample_count),
            "overall_nme": float(self.overall_nme),
            "overall_p95_nme": float(self.overall_p95_nme),
            "failure_rate": float(self.failure_rate),
            "auc": float(self.auc),
            "regression_rate_vs_best_single": float(self.regression_rate_vs_best_single),
            "bucket_regression_rate_vs_best_single": float(
                self.bucket_regression_rate_vs_best_single
            ),
            "best_single_overall_mean_nme": float(self.best_single_overall_mean_nme),
            "best_single_overall_p95_nme": float(self.best_single_overall_p95_nme),
            "best_single_overall_nme": float(self.best_single_overall_mean_nme),
            "best_single_overall_p95_nme": float(self.best_single_overall_p95_nme),
            "max_mean_nme_regression": float(self.max_mean_nme_regression),
            "max_p95_nme_regression": float(self.max_p95_nme_regression),
            "mean_nme_regression": float(self.max_mean_nme_regression),
            "p95_nme_regression": float(self.max_p95_nme_regression),
            "max_bucket_mean_nme_regression": float(self.max_bucket_mean_nme_regression),
            "max_bucket_p95_nme_regression": float(self.max_bucket_p95_nme_regression),
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
    effective_ensemble: EffectiveEnsembleDiagnostics | None = None
    is_single_model_baseline: bool = False

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
            "effective_ensemble": (
                self.effective_ensemble.to_payload()
                if self.effective_ensemble is not None
                else None
            ),
            "is_single_model_baseline": bool(self.is_single_model_baseline),
        }


@dataclass(frozen=True)
class _SearchSampleData:
    """In-memory fit/select data for one sample.

    Candidate search evaluates many candidates against the same split data. Loading
    truth arrays, metadata JSON, and prediction npy files inside every candidate
    loop is slow and mostly idle. This object stores the reusable per-sample data
    once for the duration of a search.
    """

    sample: LandmarkSample
    truth: np.ndarray
    predictions: dict[str, LandmarkPrediction]
    single_nme: dict[str, float] = field(default_factory=dict)
    single_failure: dict[str, bool] = field(default_factory=dict)
    bucket: str = ""


class _CandidateSearchContext:
    """Cached fit/select data and reusable fit artifacts for one search run."""

    def __init__(
        self,
        *,
        fit_samples: T.Sequence[LandmarkSample],
        select_samples: T.Sequence[LandmarkSample],
        cache: DiskPredictionCache,
        models: T.Sequence[str],
        failure_threshold: float,
    ) -> None:
        self.cache = cache
        self.models = tuple(dict.fromkeys(models))
        self.failure_threshold = float(failure_threshold)
        self.fit_data = self._load_samples(fit_samples, include_single_metrics=False)
        self.select_data = self._load_samples(select_samples, include_single_metrics=True)
        self._error_tables: dict[tuple[str, ...], ErrorTable] = {}
        self._fit_results: dict[
            tuple[tuple[str, ...], str, tuple[tuple[str, float], ...]], WeightFitResult
        ] = {}

    def _load_samples(
        self,
        samples: T.Sequence[LandmarkSample],
        *,
        include_single_metrics: bool,
    ) -> list[_SearchSampleData]:
        rows: list[_SearchSampleData] = []
        for sample in samples:
            truth = np.load(sample.landmarks).astype("float32")
            predictions = {
                model: self.cache.read(sample.sample_id, model) for model in self.models
            }
            single_nme: dict[str, float] = {}
            single_failure: dict[str, bool] = {}
            if include_single_metrics:
                for model, prediction in predictions.items():
                    metrics = evaluate_prediction(
                        prediction.landmarks,
                        truth,
                        normalizer=sample.normalizer,
                        failure_threshold=self.failure_threshold,
                    )
                    single_nme[model] = float(metrics["nme"])
                    single_failure[model] = bool(metrics["failure"])
            rows.append(
                _SearchSampleData(
                    sample=sample,
                    truth=truth,
                    predictions=predictions,
                    single_nme=single_nme,
                    single_failure=single_failure,
                    bucket=scenario_bucket(
                        {"dataset": sample.dataset, "condition": sample.condition}
                    ),
                )
            )
        return rows

    def error_table_for(self, models: T.Sequence[str]) -> ErrorTable:
        """Return the fit-split error table for ``models``, building it once."""
        model_tuple = tuple(models)
        table = self._error_tables.get(model_tuple)
        if table is not None:
            return table
        if not self.fit_data:
            raise ValueError("fit-split samples must be non-empty")
        errors: dict[str, list[np.ndarray]] = {model: [] for model in model_tuple}
        for row in self.fit_data:
            for model in model_tuple:
                prediction = row.predictions[model]
                errors[model].append(per_landmark_error(prediction.landmarks, row.truth))
        table = ErrorTable.from_samples(errors)
        self._error_tables[model_tuple] = table
        return table

    def fit_weights(self, candidate: Candidate) -> WeightFitResult:
        """Fit/cache static weights for the candidate's model subset + generator."""
        key = (
            tuple(candidate.models),
            candidate.weight_generator,
            candidate.weight_generator_params,
        )
        fit_result = self._fit_results.get(key)
        if fit_result is not None:
            return fit_result
        table = self.error_table_for(candidate.models)
        generator_cls = type(get_generator(candidate.weight_generator))
        generator = generator_cls(**candidate.generator_params_dict())
        fit_result = generator.fit(table, models=candidate.models)
        self._fit_results[key] = fit_result
        return fit_result


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
    context: _CandidateSearchContext | None = None,
) -> tuple[WeightFitResult, CandidateMetrics]:
    """Fit weights on ``fit_samples`` and evaluate the candidate on ``select_samples``.

    Returns the fit result alongside metrics that include sample-level and
    bucket-level regression rate against the best single model in the subset.
    When ``context`` is supplied, truth arrays, predictions, single-model metrics,
    error tables, and fitted weights are reused across candidates.
    """
    if context is None:
        context = _CandidateSearchContext(
            fit_samples=fit_samples,
            select_samples=select_samples,
            cache=cache,
            models=candidate.models,
            failure_threshold=failure_threshold,
        )
    fit_result = context.fit_weights(candidate)
    weight_matrix = weights_matrix_for_models(fit_result.weights, candidate.models)

    nmes: list[float] = []
    failures: list[bool] = []
    regressions: list[bool] = []
    per_bucket_nmes: dict[str, list[float]] = {}
    per_bucket_regressions: dict[str, list[bool]] = {}
    # Per-bucket per-model NMEs let the post-loop aggregate compute a
    # best-single baseline (min mean / min p95 across the candidate's
    # models per bucket) so the gate framework can flag a worsened slice
    # in NME-magnitude terms instead of regression-rate terms alone.
    per_bucket_single_nmes: dict[str, dict[str, list[float]]] = {}
    # Overall (cross-bucket) per-model NME arrays so we can build the
    # global best-single baselines used by the magnitude-aware overall
    # NME gates (``max_mean_nme_regression`` / ``max_p95_nme_regression``).
    single_nme_arrays: dict[str, list[float]] = {model: [] for model in candidate.models}
    single_total_nme: dict[str, float] = {model: 0.0 for model in candidate.models}
    single_counts: dict[str, int] = {model: 0 for model in candidate.models}

    for row in context.select_data:
        cached_points = [row.predictions[model].landmarks for model in candidate.models]
        single_nmes = {model: row.single_nme[model] for model in candidate.models}
        for model in candidate.models:
            single_total_nme[model] += single_nmes[model]
            single_counts[model] += 1
            single_nme_arrays[model].append(float(single_nmes[model]))
            per_bucket_single_nmes.setdefault(row.bucket, {}).setdefault(model, []).append(
                float(single_nmes[model])
            )
        best_single_nme = min(single_nmes.values())

        fused = _fuse_candidate(candidate, cached_points, weight_matrix=weight_matrix)
        fused_metrics = evaluate_prediction(
            fused.points,
            row.truth,
            normalizer=row.sample.normalizer,
            failure_threshold=failure_threshold,
        )
        ensemble_nme = float(fused_metrics["nme"])
        nmes.append(ensemble_nme)
        failures.append(bool(fused_metrics["failure"]))
        regressed = ensemble_nme > best_single_nme + regression_epsilon_nme
        regressions.append(regressed)
        per_bucket_nmes.setdefault(row.bucket, []).append(ensemble_nme)
        per_bucket_regressions.setdefault(row.bucket, []).append(regressed)

    overall_nme = float(np.mean(nmes)) if nmes else 0.0
    overall_p95_nme = float(np.percentile(nmes, 95)) if nmes else 0.0
    failure_rate = float(np.mean(failures)) if failures else 0.0
    auc = _auc_from_nmes(nmes, threshold=failure_threshold)
    sample_regression = float(np.mean(regressions)) if regressions else 0.0
    # Overall best-single baselines (lowest mean / p95 NME any member model
    # would have produced alone, across the full select split). Feed the
    # magnitude-aware NME-regression gates so a tiny per-bucket count can't
    # mask an overall worsening, and vice versa.
    best_single_overall_mean = 0.0
    best_single_overall_p95 = 0.0
    if any(values for values in single_nme_arrays.values()):
        per_model_overall_means: dict[str, float] = {}
        per_model_overall_p95: dict[str, float] = {}
        for model, values in single_nme_arrays.items():
            if not values:
                continue
            arr = np.asarray(values, dtype="float32")
            per_model_overall_means[model] = float(arr.mean())
            per_model_overall_p95[model] = float(np.percentile(arr, 95))
        if per_model_overall_means:
            best_single_overall_mean = min(per_model_overall_means.values())
        if per_model_overall_p95:
            best_single_overall_p95 = min(per_model_overall_p95.values())
    max_mean_nme_regression = max(0.0, overall_nme - best_single_overall_mean)
    max_p95_nme_regression = max(0.0, overall_p95_nme - best_single_overall_p95)
    bucket_regression = (
        float(np.mean([float(np.mean(values)) for values in per_bucket_regressions.values()]))
        if per_bucket_regressions
        else 0.0
    )
    per_bucket: dict[str, dict[str, float]] = {}
    max_bucket_mean_nme_regression = 0.0
    max_bucket_p95_nme_regression = 0.0
    for bucket, values in per_bucket_nmes.items():
        values_array = np.asarray(values, dtype="float32")
        bucket_mean = float(values_array.mean())
        bucket_p95 = float(np.percentile(values_array, 95)) if values_array.size else 0.0
        # Best-single baseline per bucket: the candidate's model with the
        # lowest mean (or p95) NME on this bucket, so the regression term
        # is "ensemble worsened the slice's NME magnitude beyond what any
        # member model would have produced alone."
        best_single_mean = 0.0
        best_single_p95 = 0.0
        single_means: dict[str, float] = {}
        single_p95s: dict[str, float] = {}
        per_model = per_bucket_single_nmes.get(bucket, {})
        if per_model:
            for model, model_values in per_model.items():
                arr = np.asarray(model_values, dtype="float32")
                single_means[model] = float(arr.mean()) if arr.size else 0.0
                single_p95s[model] = float(np.percentile(arr, 95)) if arr.size else 0.0
            best_single_mean = min(single_means.values())
            best_single_p95 = min(single_p95s.values())
        per_bucket[bucket] = {
            "nme": bucket_mean,
            "p95_nme": bucket_p95,
            "regression_rate": float(np.mean(per_bucket_regressions[bucket])),
            "count": float(len(values)),
            "best_single_mean_nme": best_single_mean,
            "best_single_p95_nme": best_single_p95,
        }
        max_bucket_mean_nme_regression = max(
            max_bucket_mean_nme_regression, bucket_mean - best_single_mean
        )
        max_bucket_p95_nme_regression = max(
            max_bucket_p95_nme_regression, bucket_p95 - best_single_p95
        )

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
        overall_p95_nme=overall_p95_nme,
        failure_rate=failure_rate,
        auc=auc,
        regression_rate_vs_best_single=sample_regression,
        bucket_regression_rate_vs_best_single=bucket_regression,
        per_bucket=per_bucket,
        best_single_model=best_single_model,
        best_single_overall_mean_nme=best_single_overall_mean,
        best_single_overall_p95_nme=best_single_overall_p95,
        max_mean_nme_regression=max_mean_nme_regression,
        max_p95_nme_regression=max_p95_nme_regression,
        max_bucket_mean_nme_regression=max_bucket_mean_nme_regression,
        max_bucket_p95_nme_regression=max_bucket_p95_nme_regression,
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
    progress: CandidateProgress | None = None,
) -> list[CandidateResult]:
    """Evaluate every candidate and return CandidateResults sorted by score.

    ``progress`` optionally wraps the candidate sequence for user-facing tools
    such as CLIs that want a tqdm bar without making this library depend on
    tqdm directly. The runner preloads the fit/select truth arrays and cached
    predictions once, then reuses per-subset error tables and fitted weights
    across all strategies/thresholds that share the same subset + generator.
    """
    results: list[CandidateResult] = []
    all_models = tuple(
        dict.fromkeys(model for candidate in candidates for model in candidate.models)
    )
    context = _CandidateSearchContext(
        fit_samples=fit_samples,
        select_samples=select_samples,
        cache=cache,
        models=all_models,
        failure_threshold=failure_threshold,
    )
    iterator = progress(candidates) if progress is not None else candidates
    for candidate in iterator:
        fit_result, metrics = evaluate_candidate(
            candidate,
            fit_samples=fit_samples,
            select_samples=select_samples,
            cache=cache,
            failure_threshold=failure_threshold,
            regression_epsilon_nme=regression_epsilon_nme,
            context=context,
        )
        whash = weights_hash(fit_result.weights)
        cid = candidate_id(
            candidate,
            weights_hash=whash,
            split_assignment_hash=split_assignment_hash,
            objective=objective,
        )
        score = score_v1(metrics) if objective == DEFAULT_OBJECTIVE else score_v1(metrics)
        try:
            diagnostics = diagnose_effective_ensemble(
                fit_result.weights,
                strategy=candidate.strategy,
                models=candidate.models,
            )
        except ValueError:
            diagnostics = None
        is_single_baseline = len(candidate.models) == 1 and candidate.strategy == "plain_average"
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
                effective_ensemble=diagnostics,
                is_single_model_baseline=is_single_baseline,
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
    "CandidateProgress",
    "CandidateResult",
    "DEFAULT_EFFECTIVE_MODELS_FLOOR",
    "DEFAULT_OBJECTIVE",
    "DEFAULT_REGRESSION_EPSILON_NME",
    "EffectiveEnsembleDiagnostics",
    "SUBSET_PRESETS",
    "candidate_id",
    "enumerate_candidates",
    "evaluate_candidate",
    "expand_model_subsets",
    "single_model_baseline_candidates",
    "load_split_samples",
    "run_candidate_search",
    "score_v1",
    "select_best_candidate",
    "weights_hash",
]
