#!/usr/bin/env python3
"""Portable scorer artifact support for runtime resolver candidate quality."""

from __future__ import annotations

import json
import logging
import math
import typing as T
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from lib.landmarks.ensemble.scorer_target_config import (
    MODEL_TYPE_LIGHTGBM_LAMBDARANK,
    MODEL_TYPE_LINEAR_REGRESSION,
    MODEL_TYPE_LOGISTIC_REGRESSION,
    SCORE_SEMANTICS_PREDICTED_COST,
    SCORE_SEMANTICS_PREDICTED_RISK,
    TARGET_CANDIDATE_FAILURE_OR_HIGH_GAP,
)

logger = logging.getLogger(__name__)

CandidateLike = T.Any
MetricLike = T.Any


def _float(value: T.Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _bool(value: T.Any) -> float:
    return 1.0 if bool(value) else 0.0


def sigmoid(value: float) -> float:
    """Numerically stable logistic sigmoid."""
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def candidate_feature_map(
    candidate: CandidateLike,
    metric: MetricLike,
    *,
    runtime_bucket: str = "",
    risk_route: str = "",
    model_predictions_available: T.Mapping[str, bool] | T.Iterable[str] | None = None,
    roll_estimate: float | None = None,
    yaw_estimate: float | None = None,
    candidate_yaw_disagreement: float | None = None,
    max_disagreement_px: float | None = None,
    runtime_bucket_source: str = "",
    candidate_extra_features: T.Mapping[str, T.Mapping[str, float]] | None = None,
) -> dict[str, float]:
    """Return numeric and one-hot features for one runtime resolver candidate.

    Feature names are intentionally literal and stable because scorer artifacts
    store the ordered feature list. Unknown feature names evaluate to zero at
    scoring time, allowing old artifacts to load after new diagnostics are added.
    """
    name = str(getattr(candidate, "name", ""))
    is_fusion = bool(getattr(candidate, "is_fusion", False))
    veto_reasons = tuple(getattr(metric, "geometry_veto_reasons", ()) or ())
    shape_veto_reasons = tuple(getattr(metric, "shape_veto_reasons", ()) or ())
    model_available: dict[str, bool] = {}
    if isinstance(model_predictions_available, Mapping):
        model_available = {
            str(model): bool(available) for model, available in model_predictions_available.items()
        }
    elif model_predictions_available is not None:
        model_available = {str(model): True for model in model_predictions_available}

    roll = getattr(metric, "roll_degrees", None)
    yaw = getattr(metric, "yaw_degrees", None)
    features: dict[str, float] = {
        "candidate_is_single_model": 0.0 if is_fusion else 1.0,
        "candidate_is_fusion": 1.0 if is_fusion else 0.0,
        "cloud_area_ratio": _float(getattr(metric, "cloud_area_ratio", None)),
        "hull_area_ratio": _float(getattr(metric, "hull_area_ratio", None)),
        "points_outside_expanded_bbox_fraction": _float(
            getattr(metric, "points_outside_expanded_bbox_fraction", None)
        ),
        "eye_mouth_order_valid_after_deroll": _bool(
            getattr(metric, "eye_mouth_order_valid_after_deroll", False)
        ),
        "roi_center_consensus_distance": _float(
            getattr(metric, "roi_center_consensus_distance", None)
        ),
        "landmark_consensus_distance": _float(
            getattr(metric, "landmark_consensus_distance", None)
        ),
        "shape_plausibility_score": _float(getattr(metric, "shape_plausibility_score", None)),
        "max_edge_length_ratio": _float(getattr(metric, "max_edge_length_ratio", None)),
        "mean_shape_fit_error": _float(getattr(metric, "mean_shape_fit_error", None)),
        "topology_violation_count": _float(getattr(metric, "topology_violation_count", None)),
        "shape_veto_reason_count": float(len(shape_veto_reasons)),
        "roll_degrees": _float(roll),
        "yaw_degrees": _float(yaw),
        "roll_delta_to_consensus": abs(_float(roll) - _float(roll_estimate)),
        "yaw_delta_to_consensus": abs(_float(yaw) - _float(yaw_estimate)),
        "candidate_yaw_disagreement": _float(candidate_yaw_disagreement),
        "max_disagreement_px": _float(max_disagreement_px),
        "has_geometry_veto": 1.0 if veto_reasons else 0.0,
    }
    features[f"candidate_name={name}"] = 1.0
    if runtime_bucket:
        features[f"runtime_bucket={runtime_bucket}"] = 1.0
    if risk_route:
        features[f"risk_route={risk_route}"] = 1.0
    if runtime_bucket_source:
        features[f"runtime_bucket_source={runtime_bucket_source}"] = 1.0
    for model, available in model_available.items():
        if available:
            features[f"model_predictions_available={model}"] = 1.0
    for reason in veto_reasons:
        features[f"geometry_veto_reason={reason}"] = 1.0
    for reason in shape_veto_reasons:
        features[f"shape_veto_reason={reason}"] = 1.0
    if candidate_extra_features is not None:
        features.update(candidate_extra_features.get(name, {}))
    return features


@dataclass(frozen=True)
class RuntimeResolverScorer:
    """Loaded runtime resolver scorer artifact."""

    features: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    artifact_schema_version: int = 1
    model_type: str = MODEL_TYPE_LOGISTIC_REGRESSION
    target: str = TARGET_CANDIDATE_FAILURE_OR_HIGH_GAP
    score_semantics: str = SCORE_SEMANTICS_PREDICTED_RISK
    higher_is_better: bool = False
    failure_threshold: float = 0.08
    calibration: dict[str, T.Any] | None = None
    source_path: str = ""
    version: str = "learned_quality_v1"
    selection_target: str = "binary_failure_or_high_gap"
    promoted_from: str = ""
    objective: str = "minimize_candidate_failure_risk"
    training_mode: str = "binary_failure_or_high_gap"
    runtime_policy: str = "learned_quality_v1"

    def __post_init__(self) -> None:
        """Validate the score direction contract at construction time."""
        expected_semantics = (
            SCORE_SEMANTICS_PREDICTED_COST
            if self.model_type in {MODEL_TYPE_LINEAR_REGRESSION, MODEL_TYPE_LIGHTGBM_LAMBDARANK}
            else SCORE_SEMANTICS_PREDICTED_RISK
        )
        if self.score_semantics != expected_semantics:
            raise ValueError(
                "runtime resolver scorer score_semantics/model_type mismatch: "
                f"{self.score_semantics!r} for {self.model_type!r}"
            )
        if self.higher_is_better:
            raise ValueError("runtime resolver scorer scores must rank lower as better")

    @classmethod
    def from_payload(
        cls,
        payload: T.Mapping[str, T.Any],
        *,
        source_path: str = "",
    ) -> RuntimeResolverScorer:
        """Build a scorer from a JSON artifact payload."""
        features = tuple(str(item) for item in payload.get("features", ()))
        coefficients = tuple(float(item) for item in payload.get("coefficients", ()))
        if not features:
            raise ValueError("runtime resolver scorer artifact has no features")
        if len(features) != len(coefficients):
            raise ValueError(
                "runtime resolver scorer features/coefficients length mismatch: "
                f"{len(features)} != {len(coefficients)}"
            )
        model_type = str(payload.get("model_type", MODEL_TYPE_LOGISTIC_REGRESSION))
        supported_model_types = {MODEL_TYPE_LOGISTIC_REGRESSION, MODEL_TYPE_LINEAR_REGRESSION}
        if model_type not in supported_model_types:
            raise ValueError(f"unsupported runtime resolver scorer model_type {model_type!r}")
        default_score_semantics = (
            SCORE_SEMANTICS_PREDICTED_COST
            if model_type == MODEL_TYPE_LINEAR_REGRESSION
            else SCORE_SEMANTICS_PREDICTED_RISK
        )
        score_semantics = str(payload.get("score_semantics", default_score_semantics))
        if score_semantics != default_score_semantics:
            raise ValueError(
                "runtime resolver scorer score_semantics/model_type mismatch: "
                f"{score_semantics!r} for {model_type!r}"
            )
        higher_is_better = bool(payload.get("higher_is_better", False))
        if higher_is_better:
            raise ValueError("runtime resolver scorer scores must rank lower as better")
        calibration = payload.get("calibration", {"type": "none", "params": {}})
        return cls(
            features=features,
            coefficients=coefficients,
            intercept=float(payload.get("intercept", 0.0)),
            artifact_schema_version=int(payload.get("artifact_schema_version", 1)),
            model_type=model_type,
            target=str(payload.get("target", TARGET_CANDIDATE_FAILURE_OR_HIGH_GAP)),
            score_semantics=score_semantics,
            higher_is_better=higher_is_better,
            failure_threshold=float(payload.get("failure_threshold", 0.08)),
            calibration=calibration if isinstance(calibration, dict) else None,
            source_path=source_path,
            version=str(
                payload.get("version", payload.get("scorer_version", "learned_quality_v1"))
            ),
            selection_target=str(payload.get("selection_target", "binary_failure_or_high_gap")),
            promoted_from=str(payload.get("promoted_from", "")),
            objective=str(payload.get("objective", "minimize_candidate_failure_risk")),
            training_mode=str(payload.get("training_mode", "binary_failure_or_high_gap")),
            runtime_policy=str(payload.get("runtime_policy", "learned_quality_v1")),
        )

    def to_payload(self) -> dict[str, T.Any]:
        """Serialize this scorer to a portable JSON payload."""
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "model_type": self.model_type,
            "target": self.target,
            "score_semantics": self.score_semantics,
            "higher_is_better": self.higher_is_better,
            "failure_threshold": self.failure_threshold,
            "features": list(self.features),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "calibration": self.calibration or {"type": "none", "params": {}},
            "version": self.version,
            "scorer_version": self.version,
            "selection_target": self.selection_target,
            "promoted_from": self.promoted_from,
            "objective": self.objective,
            "training_mode": self.training_mode,
            "runtime_policy": self.runtime_policy,
        }

    def score_feature_map(self, features: T.Mapping[str, float]) -> float:
        """Return predicted candidate risk/cost for a precomputed feature map."""
        linear = self.intercept
        for name, coefficient in zip(self.features, self.coefficients, strict=True):
            linear += coefficient * _float(features.get(name))
        if self.model_type == MODEL_TYPE_LINEAR_REGRESSION:
            return float(linear)
        return sigmoid(linear)

    def score_candidate(
        self,
        candidate: CandidateLike,
        metric: MetricLike,
        **context: T.Any,
    ) -> float:
        """Return predicted risk for one candidate and diagnostic context."""
        return self.score_feature_map(candidate_feature_map(candidate, metric, **context))


@dataclass(frozen=True)
class RuntimeResolverLightGBMScorer:
    """Loaded LightGBM LambdaRank scorer artifact."""

    features: tuple[str, ...]
    model_data: str
    artifact_schema_version: int = 2
    model_type: str = MODEL_TYPE_LIGHTGBM_LAMBDARANK
    target: str = "selection_cost"
    score_semantics: str = SCORE_SEMANTICS_PREDICTED_COST
    higher_is_better: bool = False
    failure_threshold: float = 0.08
    calibration: dict[str, T.Any] | None = None
    source_path: str = ""
    version: str = "learned_quality_v2"
    selection_target: str = "inverse_selection_cost_rank"
    promoted_from: str = ""
    objective: str = "lambdarank_inverse_regret"
    training_mode: str = "grouped_lambdarank"
    runtime_policy: str = "learned_quality_v2"
    feature_importances: dict[str, float] | None = None
    _booster_cache: T.Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.model_type != MODEL_TYPE_LIGHTGBM_LAMBDARANK:
            raise ValueError(f"unsupported v2 scorer model_type {self.model_type!r}")
        if self.score_semantics != SCORE_SEMANTICS_PREDICTED_COST:
            raise ValueError("LightGBM runtime scorer must expose predicted_cost semantics")
        if self.higher_is_better:
            raise ValueError("runtime resolver scorer scores must rank lower as better")

    @classmethod
    def from_payload(
        cls,
        payload: T.Mapping[str, T.Any],
        *,
        source_path: str = "",
    ) -> RuntimeResolverLightGBMScorer:
        features = tuple(str(item) for item in payload.get("features", ()))
        if not features:
            raise ValueError("runtime resolver v2 scorer artifact has no features")
        model_data = str(payload.get("model_data") or "")
        if not model_data:
            raise ValueError("runtime resolver v2 scorer artifact has no model_data")
        calibration = payload.get("calibration", {"type": "none", "params": {}})
        importances = payload.get("feature_importances", {})
        return cls(
            features=features,
            model_data=model_data,
            artifact_schema_version=int(payload.get("artifact_schema_version", 2)),
            model_type=str(payload.get("model_type", MODEL_TYPE_LIGHTGBM_LAMBDARANK)),
            target=str(payload.get("target", "selection_cost")),
            score_semantics=str(payload.get("score_semantics", SCORE_SEMANTICS_PREDICTED_COST)),
            higher_is_better=bool(payload.get("higher_is_better", False)),
            failure_threshold=float(payload.get("failure_threshold", 0.08)),
            calibration=calibration if isinstance(calibration, dict) else None,
            source_path=source_path,
            version=str(
                payload.get("version", payload.get("scorer_version", "learned_quality_v2"))
            ),
            selection_target=str(payload.get("selection_target", "inverse_selection_cost_rank")),
            promoted_from=str(payload.get("promoted_from", "")),
            objective=str(payload.get("objective", "lambdarank_inverse_regret")),
            training_mode=str(payload.get("training_mode", "grouped_lambdarank")),
            runtime_policy=str(payload.get("runtime_policy", "learned_quality_v2")),
            feature_importances=(
                {str(key): float(value) for key, value in importances.items()}
                if isinstance(importances, dict)
                else None
            ),
        )

    def _booster(self) -> T.Any:
        if self._booster_cache is not None:
            logger.debug("[RuntimeResolver] LightGBM booster cache hit")
            return self._booster_cache
        # On macOS, LightGBM ships libomp.dylib and PyTorch ships its own
        # libomp.dylib; loading both into the same process triggers
        # "OMP: System error #22". KMP_DUPLICATE_LIB_OK lets the second
        # runtime no-op instead of aborting, and pinning OMP_NUM_THREADS
        # to 1 avoids LightGBM's thread pool fighting with Torch's.
        import os

        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("LIBOMP_NUM_THREADS", "1")
        logger.debug("[RuntimeResolver] importing lightgbm")
        try:
            import lightgbm as lgb
        except ModuleNotFoundError as err:  # pragma: no cover - depends on runtime env
            raise RuntimeError(
                "learned_quality_v2 requires lightgbm to load scorer artifacts"
            ) from err
        logger.debug("[RuntimeResolver] imported lightgbm")
        logger.debug("[RuntimeResolver] constructing LightGBM Booster")
        # Force single-threaded Booster construction *and* prediction. The
        # original crash on macOS was LightGBM bringing up its own OpenMP
        # runtime in a process where PyTorch's libomp was already loaded;
        # pinning num_threads=1 here keeps the second runtime from
        # spawning a thread pool. verbosity=-1 also silences LightGBM's
        # info logging on the extract path.
        booster = lgb.Booster(
            model_str=self.model_data,
            params={"num_threads": 1, "verbosity": -1},
        )
        logger.debug("[RuntimeResolver] constructed LightGBM Booster")
        object.__setattr__(self, "_booster_cache", booster)
        return booster

    def to_payload(self) -> dict[str, T.Any]:
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "model_type": self.model_type,
            "target": self.target,
            "score_semantics": self.score_semantics,
            "higher_is_better": self.higher_is_better,
            "failure_threshold": self.failure_threshold,
            "features": list(self.features),
            "model_data": self.model_data,
            "calibration": self.calibration or {"type": "none", "params": {}},
            "version": self.version,
            "scorer_version": self.version,
            "selection_target": self.selection_target,
            "promoted_from": self.promoted_from,
            "objective": self.objective,
            "training_mode": self.training_mode,
            "runtime_policy": self.runtime_policy,
            "feature_importances": self.feature_importances or {},
        }

    def score_feature_map(self, features: T.Mapping[str, float]) -> float:
        logger.debug("[RuntimeResolver] building LightGBM feature matrix")
        x = feature_matrix([features], self.features)
        logger.debug("[RuntimeResolver] predicting LightGBM x_shape=%s", x.shape)
        predicted_relevance = float(self._booster().predict(x, num_threads=1)[0])
        logger.debug("[RuntimeResolver] predicted LightGBM score=%s", predicted_relevance)
        return -predicted_relevance

    def score_feature_maps(self, feature_maps: T.Sequence[T.Mapping[str, float]]) -> list[float]:
        """Score N feature dicts in one LightGBM predict call.

        ``score_feature_map`` does a 1-row predict per candidate. Each call
        re-enters ``Booster.predict`` and rebuilds the row matrix, which
        adds latency that is linear in candidate count and pathological
        when LightGBM has thread-pool overhead per call. Batching keeps
        the cost roughly constant.
        """
        if not feature_maps:
            return []
        logger.debug(
            "[RuntimeResolver] building LightGBM feature matrix batch=%d", len(feature_maps)
        )
        x = feature_matrix(feature_maps, self.features)
        logger.debug("[RuntimeResolver] predicting LightGBM x_shape=%s", x.shape)
        predicted = self._booster().predict(x, num_threads=1)
        logger.debug("[RuntimeResolver] predicted LightGBM batch_size=%d", len(predicted))
        return [-float(value) for value in predicted]

    def score_candidate(
        self,
        candidate: CandidateLike,
        metric: MetricLike,
        **context: T.Any,
    ) -> float:
        return self.score_feature_map(candidate_feature_map(candidate, metric, **context))


@lru_cache(maxsize=8)
def _load_runtime_resolver_scorer_cached(
    path: str,
) -> RuntimeResolverScorer | RuntimeResolverLightGBMScorer:
    """Read and parse a scorer artifact, memoized by resolved path."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runtime resolver scorer artifact must be a JSON object")
    if str(payload.get("model_type", "")) == MODEL_TYPE_LIGHTGBM_LAMBDARANK:
        return RuntimeResolverLightGBMScorer.from_payload(payload, source_path=str(source))
    return RuntimeResolverScorer.from_payload(payload, source_path=str(source))


def load_runtime_resolver_scorer(
    path: str | Path,
) -> RuntimeResolverScorer | RuntimeResolverLightGBMScorer:
    """Load a runtime resolver scorer artifact from disk.

    Scorer artifacts are immutable once promoted, and ``runtime_resolver``
    can call this once per face during extract. Memoize by resolved path
    so we read + parse + (for LightGBM) reconstruct the booster once per
    process; the bounded cache size keeps the working set small.
    """
    return _load_runtime_resolver_scorer_cached(str(Path(path).resolve()))


def write_runtime_resolver_scorer(
    scorer: RuntimeResolverScorer | RuntimeResolverLightGBMScorer,
    path: str | Path,
) -> Path:
    """Write a runtime resolver scorer artifact to disk."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(scorer.to_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def candidate_scores(
    scorer: RuntimeResolverScorer | RuntimeResolverLightGBMScorer,
    candidates: T.Sequence[CandidateLike],
    metrics: T.Mapping[str, MetricLike],
    **context: T.Any,
) -> dict[str, float]:
    """Score all candidates that have matching metric payloads.

    For the LightGBM (learned_quality_v2) path we build one feature matrix
    and call ``predict`` once — the old code did N=len(candidates) 1-row
    predict calls and was the hot spot under load. Linear / logistic v1
    scorers still go per-candidate so the per-candidate diagnostic logs
    fire for them.
    """
    eligible = [c for c in candidates if str(c.name) in metrics]
    if isinstance(scorer, RuntimeResolverLightGBMScorer):
        if not eligible:
            return {}
        logger.debug(
            "[RuntimeResolver] batch scoring %d candidate(s) via %s",
            len(eligible),
            type(scorer).__name__,
        )
        feature_maps = [
            candidate_feature_map(candidate, metrics[str(candidate.name)], **context)
            for candidate in eligible
        ]
        scores = scorer.score_feature_maps(feature_maps)
        batched = {
            str(candidate.name): float(score)
            for candidate, score in zip(eligible, scores, strict=True)
        }
        logger.debug("[RuntimeResolver] batch score done scores=%s", batched)
        return batched

    per_candidate: dict[str, float] = {}
    for candidate in eligible:
        name = str(candidate.name)
        logger.debug("[RuntimeResolver] score start candidate=%s", name)
        score = float(scorer.score_candidate(candidate, metrics[name], **context))
        logger.debug("[RuntimeResolver] score done candidate=%s score=%s", name, score)
        per_candidate[name] = score
    return per_candidate


def feature_matrix(
    rows: T.Sequence[T.Mapping[str, float]],
    features: T.Sequence[str],
) -> np.ndarray:
    """Convert feature maps into a dense model matrix."""
    return np.asarray(
        [[_float(row.get(feature)) for feature in features] for row in rows],
        dtype="float64",
    )


__all__ = [
    "RuntimeResolverScorer",
    "RuntimeResolverLightGBMScorer",
    "candidate_feature_map",
    "candidate_scores",
    "feature_matrix",
    "load_runtime_resolver_scorer",
    "sigmoid",
    "write_runtime_resolver_scorer",
]
