#!/usr/bin/env python3
"""Training-data generation and model fitting for the stacked residual regressor (#223).

The trainer reuses the same cached candidate/sample contexts as the resolver
scorer training path. For each sample it:

1. Selects a safe base candidate (the regressor's ``base_candidate_policy``).
2. Builds a runtime-visible feature row via
   :func:`lib.landmarks.ensemble.runtime_features.stacked_regression_feature_map`.
3. Computes the correction target from the base candidate to the GT landmarks,
   expressed in the chosen ``output_mode`` and normalized by the base candidate's
   own landmark extent (so training and runtime share one normalization).

Targets and features are normalized identically to the runtime application, so a
model fit here can be applied directly by
:class:`lib.landmarks.ensemble.stacked_regressor.RuntimeStackedLandmarkRegressor`.

Splitting is by sample identity, never by candidate row, to avoid leakage.
"""

from __future__ import annotations

import hashlib
import logging
import math
import typing as T
from dataclasses import dataclass

import numpy as np

from lib.landmarks.ensemble.runtime_features import (
    forbidden_runtime_features,
    stacked_regression_feature_map,
)
from lib.landmarks.ensemble.stacked_regressor import (
    DEFAULT_STACKED_CANDIDATE_NAME,
    OUTPUT_MODE_GLOBAL_TRANSFORM,
    OUTPUT_MODE_LANDMARK_RESIDUAL_68,
    OUTPUT_MODE_REGION_RESIDUAL,
    STACKED_REGION_INDICES,
    STACKED_REGION_NAMES,
    RuntimeStackedLandmarkRegressor,
    StackedRegressorError,
    output_dim_for_mode,
)

logger = logging.getLogger(__name__)

ContextLike = T.Any

STACKED_REGRESSOR_ARTIFACT = "stacked_regressor.json"

#: Default L2 regularization for the standardized linear residual model.
DEFAULT_STACKED_L2 = 1.0

#: log-scale is clamped both at fit and apply time so a degenerate sample can't
#: produce an exploding similarity scale.
_LOG_SCALE_CLAMP = 1.0


class StackedRegressorTrainingError(StackedRegressorError):
    """Raised when training data is insufficient or inconsistent."""


@dataclass(frozen=True, slots=True)
class StackedTrainingExample:
    """One sample's runtime features plus correction target."""

    sample_id: str
    base_candidate: str
    features: dict[str, float]
    target: np.ndarray


def _base_extent_diagonal(landmarks: np.ndarray) -> float:
    mins = landmarks.min(axis=0)
    maxs = landmarks.max(axis=0)
    diag = float(math.hypot(float(maxs[0] - mins[0]), float(maxs[1] - mins[1])))
    return diag if diag > 1e-6 else 1.0


def global_transform_target(base: np.ndarray, gt: np.ndarray, diag: float) -> np.ndarray:
    """Return (tx_norm, ty_norm, log_scale, rotation) mapping ``base`` onto ``gt``.

    Solves the closed-form 2-D similarity ``gt' = a * base_centered + t`` in the
    complex plane, matching the transform applied at runtime (scale + rotation
    about the base centroid, then translation).
    """
    base_centroid = base.mean(axis=0)
    centered = base - base_centroid
    gt_centroid = gt.mean(axis=0)
    gt_centered = gt - gt_centroid

    p = centered[:, 0] + 1j * centered[:, 1]
    q = gt_centered[:, 0] + 1j * gt_centered[:, 1]
    denom = float(np.sum((p * np.conj(p)).real))
    a = complex(np.sum(q * np.conj(p)) / denom) if denom > 1e-12 else 1.0 + 0j

    scale = abs(a)
    log_scale = math.log(scale) if scale > 1e-6 else 0.0
    log_scale = max(-_LOG_SCALE_CLAMP, min(_LOG_SCALE_CLAMP, log_scale))
    rotation = math.atan2(a.imag, a.real)
    translation = gt_centroid - base_centroid
    target: np.ndarray = np.array(
        [
            float(translation[0]) / diag,
            float(translation[1]) / diag,
            log_scale,
            rotation,
        ],
        dtype="float64",
    )
    return target


def region_residual_target(base: np.ndarray, gt: np.ndarray, diag: float) -> np.ndarray:
    """Return per-region (dx, dy) centroid offsets from ``base`` to ``gt``."""
    target: np.ndarray = np.zeros(len(STACKED_REGION_NAMES) * 2, dtype="float64")
    for region_index, region in enumerate(STACKED_REGION_NAMES):
        indices = list(STACKED_REGION_INDICES[region])
        offset = gt[indices].mean(axis=0) - base[indices].mean(axis=0)
        target[region_index * 2] = float(offset[0]) / diag
        target[region_index * 2 + 1] = float(offset[1]) / diag
    return target


def landmark_residual_68_target(base: np.ndarray, gt: np.ndarray, diag: float) -> np.ndarray:
    """Return the flattened (136,) per-point residual from ``base`` to ``gt``."""
    target: np.ndarray = ((gt - base) / diag).reshape(-1).astype("float64")
    return target


def _target_for_mode(
    output_mode: str,
    base: np.ndarray,
    gt: np.ndarray,
    diag: float,
) -> np.ndarray:
    if output_mode == OUTPUT_MODE_GLOBAL_TRANSFORM:
        return global_transform_target(base, gt, diag)
    if output_mode == OUTPUT_MODE_REGION_RESIDUAL:
        return region_residual_target(base, gt, diag)
    if output_mode == OUTPUT_MODE_LANDMARK_RESIDUAL_68:
        return landmark_residual_68_target(base, gt, diag)
    raise StackedRegressorTrainingError(f"unsupported output_mode {output_mode!r}")


def select_base_candidate(context: ContextLike, policy: str) -> T.Any:
    """Pick the base candidate record for ``context`` matching the resolver order."""
    candidates = list(getattr(context, "candidates", ()))
    finite = {
        candidate.name: candidate
        for candidate in candidates
        if np.all(np.isfinite(np.asarray(candidate.landmarks, dtype="float64")))
    }
    if not finite:
        return None
    preferred: list[str] = []
    if policy:
        preferred.append(policy)
    preferred.append("static_weighted")
    preferred.extend(name for name in finite if name.startswith("static_weighted@"))
    for name in preferred:
        if name in finite:
            return finite[name]
    for candidate in candidates:
        if candidate.name in finite and getattr(candidate, "is_fusion", False):
            return finite[candidate.name]
    return next(iter(finite.values()), None)


def build_training_examples(
    contexts: T.Iterable[ContextLike],
    *,
    output_mode: str,
    base_candidate_policy: str,
) -> list[StackedTrainingExample]:
    """Build per-sample training examples from cached sample contexts."""
    output_dim_for_mode(output_mode)  # validate mode early
    examples: list[StackedTrainingExample] = []
    for context in contexts:
        truth = getattr(context, "truth_landmarks", None)
        if truth is None:
            continue
        gt = np.asarray(truth, dtype="float64")
        if gt.shape != (68, 2) or not np.all(np.isfinite(gt)):
            continue
        base = select_base_candidate(context, base_candidate_policy)
        if base is None:
            continue
        base_landmarks = np.asarray(base.landmarks, dtype="float64")
        if base_landmarks.shape != (68, 2) or not np.all(np.isfinite(base_landmarks)):
            continue
        diag = _base_extent_diagonal(base_landmarks)
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
        target = _target_for_mode(output_mode, base_landmarks, gt, diag)
        examples.append(
            StackedTrainingExample(
                sample_id=str(getattr(context, "sample_id", "")),
                base_candidate=str(base.name),
                features=features,
                target=target,
            )
        )
    return examples


def stacked_feature_order(examples: T.Sequence[StackedTrainingExample]) -> tuple[str, ...]:
    """Return the stable, sorted feature order across all examples.

    Stacked-regression feature names are not part of the scorer preferred order,
    so a deterministic alphabetical union keeps the artifact contract stable.
    """
    names: set[str] = set()
    for example in examples:
        names.update(example.features)
    leaked = forbidden_runtime_features(names)
    if leaked:
        raise StackedRegressorTrainingError(
            f"stacked regressor features leak offline-only fields: {sorted(leaked)}"
        )
    return tuple(sorted(names))


def fit_linear_residual_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a standardized ridge linear model, returning coef, intercept, mean, std."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    standardized = (x - mean) / std
    y_mean = y.mean(axis=0)
    y_centered = y - y_mean
    n_features = standardized.shape[1]
    gram = standardized.T @ standardized + float(l2) * np.eye(n_features)
    rhs = standardized.T @ y_centered
    coef = np.linalg.solve(gram, rhs)
    return coef, y_mean, mean, std


def _split_by_identity(
    sample_ids: T.Sequence[str],
    *,
    eval_fraction: float,
    split_seed: str | int,
) -> set[str]:
    """Return the held-out eval sample ids, split deterministically by identity."""
    unique = sorted(set(sample_ids))
    if not unique or eval_fraction <= 0:
        return set()
    seed_text = str(split_seed)

    def _rank(sample_id: str) -> float:
        digest = hashlib.sha256(f"{seed_text}:{sample_id}".encode()).hexdigest()
        return int(digest[:16], 16) / float(1 << 64)

    ranked = sorted(unique, key=_rank)
    n_eval = max(1, int(round(len(ranked) * min(1.0, eval_fraction))))
    return set(ranked[:n_eval])


def train_stacked_regressor(
    contexts: T.Iterable[ContextLike],
    *,
    output_mode: str = OUTPUT_MODE_GLOBAL_TRANSFORM,
    base_candidate_policy: str = "static_weighted",
    candidate_name: str = DEFAULT_STACKED_CANDIDATE_NAME,
    residual_clip_fraction: float = 0.05,
    l2: float = DEFAULT_STACKED_L2,
    eval_fraction: float = 0.2,
    split_seed: str | int = "stacked_residual_v1",
    allow_landmark_residual_68: bool = False,
) -> tuple[RuntimeStackedLandmarkRegressor, dict[str, T.Any]]:
    """Build examples, fit the model, and return the regressor plus metrics."""
    if output_mode == OUTPUT_MODE_LANDMARK_RESIDUAL_68 and not allow_landmark_residual_68:
        raise StackedRegressorTrainingError(
            "landmark_residual_68 output is gated; pass allow_landmark_residual_68=True "
            "only after constrained modes are evaluated and promoted"
        )
    examples = build_training_examples(
        contexts,
        output_mode=output_mode,
        base_candidate_policy=base_candidate_policy,
    )
    if not examples:
        raise StackedRegressorTrainingError("no usable training examples were produced")

    feature_names = stacked_feature_order(examples)
    eval_ids = _split_by_identity(
        [example.sample_id for example in examples],
        eval_fraction=eval_fraction,
        split_seed=split_seed,
    )
    train_examples = [ex for ex in examples if ex.sample_id not in eval_ids]
    eval_examples = [ex for ex in examples if ex.sample_id in eval_ids]
    if not train_examples:
        raise StackedRegressorTrainingError("training split is empty after identity split")

    def _matrix(rows: T.Sequence[StackedTrainingExample]) -> np.ndarray:
        matrix: np.ndarray = np.array(
            [[float(ex.features.get(name, 0.0)) for name in feature_names] for ex in rows],
            dtype="float64",
        )
        return matrix

    def _targets(rows: T.Sequence[StackedTrainingExample]) -> np.ndarray:
        targets: np.ndarray = np.array([ex.target for ex in rows], dtype="float64")
        return targets

    x_train = _matrix(train_examples)
    y_train = _targets(train_examples)
    coef, intercept, mean, std = fit_linear_residual_model(x_train, y_train, l2=l2)

    def _mse(rows: T.Sequence[StackedTrainingExample]) -> float:
        if not rows:
            return float("nan")
        x = _matrix(rows)
        y = _targets(rows)
        standardized = (x - mean) / std
        predicted = (standardized @ coef) + intercept
        return float(np.mean((predicted - y) ** 2))

    train_mse = _mse(train_examples)
    eval_mse = _mse(eval_examples)
    baseline_eval_mse = (
        float(np.mean(_targets(eval_examples) ** 2)) if eval_examples else float("nan")
    )

    training_metadata: dict[str, T.Any] = {
        "output_mode": output_mode,
        "base_candidate_policy": base_candidate_policy,
        "n_examples": len(examples),
        "n_train": len(train_examples),
        "n_eval": len(eval_examples),
        "train_mse": train_mse,
        "eval_mse": eval_mse,
        "baseline_eval_mse": baseline_eval_mse,
        "l2": float(l2),
        "eval_fraction": float(eval_fraction),
        "split_seed": str(split_seed),
        "eval_sample_ids": sorted(eval_ids),
        "feature_count": len(feature_names),
    }

    regressor = RuntimeStackedLandmarkRegressor(
        candidate_name=candidate_name,
        base_candidate_policy=base_candidate_policy,
        feature_names=feature_names,
        output_mode=output_mode,
        output_dim=output_dim_for_mode(output_mode),
        residual_clip_fraction=float(residual_clip_fraction),
        coef=coef,
        intercept=intercept,
        feature_mean=mean,
        feature_std=std,
        training_metadata=training_metadata,
    )
    metrics = dict(training_metadata)
    logger.info(
        "[StackedRegressor] trained output_mode=%s n_train=%d n_eval=%d train_mse=%.6g "
        "eval_mse=%.6g baseline_eval_mse=%.6g",
        output_mode,
        len(train_examples),
        len(eval_examples),
        train_mse,
        eval_mse,
        baseline_eval_mse,
    )
    return regressor, metrics


__all__ = [
    "DEFAULT_STACKED_L2",
    "STACKED_REGRESSOR_ARTIFACT",
    "StackedRegressorTrainingError",
    "StackedTrainingExample",
    "build_training_examples",
    "fit_linear_residual_model",
    "global_transform_target",
    "landmark_residual_68_target",
    "region_residual_target",
    "select_base_candidate",
    "stacked_feature_order",
    "train_stacked_regressor",
]
