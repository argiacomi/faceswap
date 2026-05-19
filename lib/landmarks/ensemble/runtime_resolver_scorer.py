#!/usr/bin/env python3
"""Portable scorer artifact support for runtime resolver candidate quality."""

from __future__ import annotations

import json
import math
import typing as T
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
) -> dict[str, float]:
    """Return numeric and one-hot features for one runtime resolver candidate.

    Feature names are intentionally literal and stable because scorer artifacts
    store the ordered feature list. Unknown feature names evaluate to zero at
    scoring time, allowing old artifacts to load after new diagnostics are added.
    """
    name = str(getattr(candidate, "name", ""))
    is_fusion = bool(getattr(candidate, "is_fusion", False))
    veto_reasons = tuple(getattr(metric, "geometry_veto_reasons", ()) or ())
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
    for model, available in model_available.items():
        if available:
            features[f"model_predictions_available={model}"] = 1.0
    for reason in veto_reasons:
        features[f"geometry_veto_reason={reason}"] = 1.0
    return features


@dataclass(frozen=True)
class RuntimeResolverScorer:
    """Loaded logistic-regression scorer artifact."""

    features: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    artifact_schema_version: int = 1
    model_type: str = "logistic_regression"
    target: str = "candidate_failure_or_high_gap"
    failure_threshold: float = 0.08
    calibration: dict[str, T.Any] | None = None
    source_path: str = ""
    version: str = "learned_quality_v1"

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
        model_type = str(payload.get("model_type", "logistic_regression"))
        if model_type != "logistic_regression":
            raise ValueError(f"unsupported runtime resolver scorer model_type {model_type!r}")
        calibration = payload.get("calibration", {"type": "none", "params": {}})
        return cls(
            features=features,
            coefficients=coefficients,
            intercept=float(payload.get("intercept", 0.0)),
            artifact_schema_version=int(payload.get("artifact_schema_version", 1)),
            model_type=model_type,
            target=str(payload.get("target", "candidate_failure_or_high_gap")),
            failure_threshold=float(payload.get("failure_threshold", 0.08)),
            calibration=calibration if isinstance(calibration, dict) else None,
            source_path=source_path,
            version=str(
                payload.get("version", payload.get("scorer_version", "learned_quality_v1"))
            ),
        )

    def to_payload(self) -> dict[str, T.Any]:
        """Serialize this scorer to a portable JSON payload."""
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "model_type": self.model_type,
            "target": self.target,
            "failure_threshold": self.failure_threshold,
            "features": list(self.features),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "calibration": self.calibration or {"type": "none", "params": {}},
            "version": self.version,
        }

    def score_feature_map(self, features: T.Mapping[str, float]) -> float:
        """Return predicted candidate risk for a precomputed feature map."""
        linear = self.intercept
        for name, coefficient in zip(self.features, self.coefficients, strict=True):
            linear += coefficient * _float(features.get(name))
        return sigmoid(linear)

    def score_candidate(
        self,
        candidate: CandidateLike,
        metric: MetricLike,
        **context: T.Any,
    ) -> float:
        """Return predicted risk for one candidate and diagnostic context."""
        return self.score_feature_map(candidate_feature_map(candidate, metric, **context))


def load_runtime_resolver_scorer(path: str | Path) -> RuntimeResolverScorer:
    """Load a runtime resolver scorer artifact from disk."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runtime resolver scorer artifact must be a JSON object")
    return RuntimeResolverScorer.from_payload(payload, source_path=str(source))


def write_runtime_resolver_scorer(
    scorer: RuntimeResolverScorer,
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
    scorer: RuntimeResolverScorer,
    candidates: T.Sequence[CandidateLike],
    metrics: T.Mapping[str, MetricLike],
    **context: T.Any,
) -> dict[str, float]:
    """Score all candidates that have matching metric payloads."""
    return {
        str(candidate.name): float(
            scorer.score_candidate(candidate, metrics[str(candidate.name)], **context)
        )
        for candidate in candidates
        if str(candidate.name) in metrics
    }


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
    "candidate_feature_map",
    "candidate_scores",
    "feature_matrix",
    "load_runtime_resolver_scorer",
    "sigmoid",
    "write_runtime_resolver_scorer",
]
