#!/usr/bin/env python3
"""Portable scorer artifact support for runtime resolver candidate quality."""

from __future__ import annotations

import json
import logging
import math
import typing as T
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from lib.landmarks.ensemble.runtime_features import (
    RUNTIME_FEATURE_CONTRACT_VERSION,
    candidate_feature_map,
    runtime_candidate_feature_maps,
)
from lib.landmarks.ensemble.scorer_target_config import (
    MODEL_TYPE_LIGHTGBM_LAMBDARANK,
    SCORE_SEMANTICS_PREDICTED_COST,
    TARGET_TRANSFORM_REGRET_V3,
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


def sigmoid(value: float) -> float:
    """Numerically stable logistic sigmoid."""
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


#: Scorer policies the runtime loader accepts. The profile specialist (#218)
#: shares the v3 LambdaRank target/semantics and differs only by routed scope.
ACCEPTED_RUNTIME_POLICIES: frozenset[str] = frozenset(
    {"learned_quality_v3", "learned_quality_v3_profile"}
)


@dataclass(frozen=True)
class RuntimeResolverLearnedScorer:
    """Loaded learned-quality v3 LambdaRank scorer artifact."""

    features: tuple[str, ...]
    model_data: str
    artifact_schema_version: int = 2
    model_type: str = MODEL_TYPE_LIGHTGBM_LAMBDARANK
    target: str = TARGET_TRANSFORM_REGRET_V3
    score_semantics: str = SCORE_SEMANTICS_PREDICTED_COST
    higher_is_better: bool = False
    failure_threshold: float = 0.08
    calibration: dict[str, T.Any] | None = None
    source_path: str = ""
    version: str = "learned_quality_v3"
    selection_target: str = TARGET_TRANSFORM_REGRET_V3
    promoted_from: str = ""
    objective: str = "lambdarank_transform_regret_v3"
    training_mode: str = "grouped_lambdarank_v3"
    runtime_policy: str = "learned_quality_v3"
    runtime_feature_contract_version: str = RUNTIME_FEATURE_CONTRACT_VERSION
    feature_importances: dict[str, float] | None = None
    _booster_cache: T.Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.model_type != MODEL_TYPE_LIGHTGBM_LAMBDARANK:
            raise ValueError(f"unsupported runtime scorer model_type {self.model_type!r}")
        if (
            self.version not in ACCEPTED_RUNTIME_POLICIES
            or self.runtime_policy not in ACCEPTED_RUNTIME_POLICIES
        ):
            raise ValueError(
                "runtime scorer loader only accepts learned_quality_v3 / "
                "learned_quality_v3_profile artifacts"
            )
        if self.target != TARGET_TRANSFORM_REGRET_V3:
            raise ValueError(
                "runtime scorer loader only accepts transform_alignment_regret_v3 artifacts"
            )
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
    ) -> RuntimeResolverLearnedScorer:
        features = tuple(str(item) for item in payload.get("features", ()))
        if not features:
            raise ValueError("runtime resolver scorer artifact has no features")
        model_data = str(payload.get("model_data") or "")
        if not model_data:
            raise ValueError("runtime resolver scorer artifact has no model_data")
        calibration = payload.get("calibration", {"type": "none", "params": {}})
        importances = payload.get("feature_importances", {})
        version = str(payload.get("version", payload.get("scorer_version", "learned_quality_v3")))
        return cls(
            features=features,
            model_data=model_data,
            artifact_schema_version=int(payload.get("artifact_schema_version", 2)),
            model_type=str(payload.get("model_type", MODEL_TYPE_LIGHTGBM_LAMBDARANK)),
            target=str(payload.get("target", TARGET_TRANSFORM_REGRET_V3)),
            score_semantics=str(payload.get("score_semantics", SCORE_SEMANTICS_PREDICTED_COST)),
            higher_is_better=bool(payload.get("higher_is_better", False)),
            failure_threshold=float(payload.get("failure_threshold", 0.08)),
            calibration=calibration if isinstance(calibration, dict) else None,
            source_path=source_path,
            version=version,
            selection_target=str(payload.get("selection_target", TARGET_TRANSFORM_REGRET_V3)),
            promoted_from=str(payload.get("promoted_from", "")),
            objective=str(payload.get("objective", "lambdarank_transform_regret_v3")),
            training_mode=str(payload.get("training_mode", "grouped_lambdarank_v3")),
            runtime_policy=str(payload.get("runtime_policy", version)),
            runtime_feature_contract_version=str(
                payload.get("runtime_feature_contract_version", RUNTIME_FEATURE_CONTRACT_VERSION)
            ),
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
                "learned_quality_v3 requires lightgbm to load scorer artifacts"
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
            "runtime_feature_contract_version": self.runtime_feature_contract_version,
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
) -> RuntimeResolverLearnedScorer:
    """Read and parse a scorer artifact, memoized by resolved path."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runtime resolver scorer artifact must be a JSON object")
    if str(payload.get("model_type", "")) != MODEL_TYPE_LIGHTGBM_LAMBDARANK:
        raise ValueError("runtime resolver scorer must be a learned_quality_v3 LightGBM artifact")
    return RuntimeResolverLearnedScorer.from_payload(payload, source_path=str(source))


def load_runtime_resolver_scorer(
    path: str | Path,
) -> RuntimeResolverLearnedScorer:
    """Load a runtime resolver scorer artifact from disk.

    Scorer artifacts are immutable once promoted, and ``runtime_resolver``
    can call this once per face during extract. Memoize by resolved path
    so we read + parse + (for LightGBM) reconstruct the booster once per
    process; the bounded cache size keeps the working set small.
    """
    return _load_runtime_resolver_scorer_cached(str(Path(path).resolve()))


def write_runtime_resolver_scorer(
    scorer: RuntimeResolverLearnedScorer,
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
    scorer: RuntimeResolverLearnedScorer,
    candidates: T.Sequence[CandidateLike],
    metrics: T.Mapping[str, MetricLike],
    **context: T.Any,
) -> dict[str, float]:
    """Score all candidates that have matching metric payloads.

    Build one feature matrix and call ``predict`` once; per-candidate
    1-row prediction is the hot path this avoids.
    """
    eligible = [c for c in candidates if str(c.name) in metrics]
    if not eligible:
        return {}
    logger.debug(
        "[RuntimeResolver] batch scoring %d candidate(s) via %s",
        len(eligible),
        type(scorer).__name__,
    )
    feature_maps = runtime_candidate_feature_maps(eligible, metrics, **context)
    scores = scorer.score_feature_maps(feature_maps)
    batched = {
        str(candidate.name): float(score)
        for candidate, score in zip(eligible, scores, strict=True)
    }
    logger.debug("[RuntimeResolver] batch score done scores=%s", batched)
    return batched


def feature_matrix(
    rows: T.Sequence[T.Mapping[str, float]],
    features: T.Sequence[str],
) -> np.ndarray:
    """Convert feature maps into a dense model matrix."""
    return np.asarray(  # type: ignore[no-any-return]
        [[_float(row.get(feature)) for feature in features] for row in rows],
        dtype="float64",
    )


__all__ = [
    "RuntimeResolverLearnedScorer",
    "candidate_feature_map",
    "candidate_scores",
    "feature_matrix",
    "load_runtime_resolver_scorer",
    "sigmoid",
    "write_runtime_resolver_scorer",
]
