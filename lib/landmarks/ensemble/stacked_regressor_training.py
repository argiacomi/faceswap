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
    DEFAULT_STACKED_EXPERT_KEY,
    EXPERT_POLICY_RUNTIME_BUCKET,
    OUTPUT_MODE_GLOBAL_TRANSFORM,
    OUTPUT_MODE_LANDMARK_RESIDUAL_68,
    OUTPUT_MODE_REGION_RESIDUAL,
    STACKED_REGION_INDICES,
    STACKED_REGION_NAMES,
    RuntimeStackedLandmarkRegressor,
    StackedRegressorError,
    output_dim_for_mode,
    runtime_bucket_expert_key,
)

logger = logging.getLogger(__name__)

ContextLike = T.Any

STACKED_REGRESSOR_ARTIFACT = "stacked_regressor.json"

#: Default L2 regularization for the standardized linear residual model.
DEFAULT_STACKED_L2 = 1.0

#: Minimum examples required to train a non-default runtime-bucket expert.
DEFAULT_STACKED_EXPERT_MIN_EXAMPLES = 128

SAMPLE_WEIGHT_POLICY_UNIFORM = "uniform"
SAMPLE_WEIGHT_POLICY_HARD_SLICE = "hard_slice"
SAMPLE_WEIGHT_POLICY_HARD_SLICE_PLUS_BASE_ERROR = "hard_slice_plus_base_error"
SAMPLE_WEIGHT_POLICIES: frozenset[str] = frozenset(
    {
        SAMPLE_WEIGHT_POLICY_UNIFORM,
        SAMPLE_WEIGHT_POLICY_HARD_SLICE,
        SAMPLE_WEIGHT_POLICY_HARD_SLICE_PLUS_BASE_ERROR,
    }
)

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
    runtime_bucket: str = ""
    sample_weight: float = 1.0


def _base_extent_diagonal(landmarks: np.ndarray) -> float:
    mins = landmarks.min(axis=0)
    maxs = landmarks.max(axis=0)
    diag = float(math.hypot(float(maxs[0] - mins[0]), float(maxs[1] - mins[1])))
    return diag if diag > 1e-6 else 1.0


def _bucket_sample_weight(bucket: str) -> float:
    """Return a conservative fixed weight for a runtime bucket.

    The goal is objective alignment, not gating. Hard slices get more influence
    during fitting because uniform residual MSE was dominated by common/easy
    samples while profile and large-yaw stayed net-negative by NME.
    """
    name = (bucket or "").lower()
    if "profile" in name:
        return 3.0
    if "large_yaw" in name or "yaw" in name:
        return 2.0
    if "roll" in name:
        return 2.0
    if "intermediate" in name:
        return 0.5
    if name in {"frontal", "no_pose", ""}:
        return 0.25
    return 1.0


def _base_nme_for_weight(base: np.ndarray, gt: np.ndarray, normalizer: T.Any) -> float | None:
    try:
        norm = float(normalizer)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(norm) or norm <= 0:
        return None
    distances = np.linalg.norm(base - gt, axis=1)
    return float(np.mean(distances) / norm)


def sample_weight_for_context(
    context: ContextLike,
    *,
    policy: str,
    base_landmarks: np.ndarray,
    gt: np.ndarray,
) -> float:
    """Return a positive sample weight for the requested training policy."""
    if policy == SAMPLE_WEIGHT_POLICY_UNIFORM:
        return 1.0
    if policy not in SAMPLE_WEIGHT_POLICIES:
        raise StackedRegressorTrainingError(
            f"unsupported sample_weight_policy {policy!r}; expected one of "
            f"{sorted(SAMPLE_WEIGHT_POLICIES)}"
        )

    bucket = str(getattr(context, "runtime_bucket", "") or "")
    weight = _bucket_sample_weight(bucket)

    if policy == SAMPLE_WEIGHT_POLICY_HARD_SLICE_PLUS_BASE_ERROR:
        base_nme = _base_nme_for_weight(base_landmarks, gt, getattr(context, "normalizer", None))
        if base_nme is not None:
            # Focus on samples where the base candidate has room to improve, while
            # avoiding one or two extreme outliers dominating the linear fit.
            error_scale = max(0.5, min(4.0, base_nme / 0.03))
            weight *= error_scale

    return float(max(weight, 1e-6))


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
    sample_weight_policy: str = SAMPLE_WEIGHT_POLICY_UNIFORM,
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
        runtime_bucket = str(getattr(context, "runtime_bucket", "") or "")
        features = stacked_regression_feature_map(
            base_landmarks=base_landmarks,
            model_landmarks=model_landmarks,
            reference_bbox=None,
            runtime_bucket=runtime_bucket,
            roll_estimate=getattr(context, "roll_estimate", None),
            yaw_estimate=getattr(context, "yaw_estimate", None),
            candidate_yaw_disagreement=getattr(context, "candidate_yaw_disagreement", None),
            max_disagreement_px=getattr(context, "max_disagreement_px", None),
            hard_case_tags=tuple(getattr(context, "hard_case_tags", ()) or ()),
            model_predictions_available=getattr(context, "model_predictions_available", None),
        )
        target = _target_for_mode(output_mode, base_landmarks, gt, diag)
        sample_weight = sample_weight_for_context(
            context,
            policy=sample_weight_policy,
            base_landmarks=base_landmarks,
            gt=gt,
        )
        examples.append(
            StackedTrainingExample(
                sample_id=str(getattr(context, "sample_id", "")),
                base_candidate=str(base.name),
                runtime_bucket=runtime_bucket,
                features=features,
                target=target,
                sample_weight=sample_weight,
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


def _normalized_sample_weights(
    sample_weights: np.ndarray | None,
    *,
    n_rows: int,
) -> np.ndarray | None:
    if sample_weights is None:
        return None
    weights = np.asarray(sample_weights, dtype="float64").reshape(-1)
    if weights.shape[0] != n_rows:
        raise StackedRegressorTrainingError(
            f"sample_weights length {weights.shape[0]} != row count {n_rows}"
        )
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise StackedRegressorTrainingError("sample_weights must be finite and positive")
    mean_weight = float(np.mean(weights))
    if mean_weight <= 0:
        raise StackedRegressorTrainingError("sample_weights mean must be positive")
    normalized: np.ndarray = T.cast(np.ndarray, weights / mean_weight)
    return normalized


def fit_linear_residual_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l2: float,
    sample_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a standardized ridge linear model, returning coef, intercept, mean, std."""
    weights = _normalized_sample_weights(sample_weights, n_rows=x.shape[0])
    if weights is None:
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

    mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x - mean) ** 2, axis=0, weights=weights)
    std = np.sqrt(variance)
    std = np.where(std < 1e-12, 1.0, std)
    standardized = (x - mean) / std

    y_mean = np.average(y, axis=0, weights=weights)
    y_centered = y - y_mean

    sqrt_weights = np.sqrt(weights)[:, None]
    weighted_x = standardized * sqrt_weights
    weighted_y = y_centered * sqrt_weights

    n_features = weighted_x.shape[1]
    gram = weighted_x.T @ weighted_x + float(l2) * np.eye(n_features)
    rhs = weighted_x.T @ weighted_y
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
    sample_weight_policy: str = SAMPLE_WEIGHT_POLICY_UNIFORM,
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
        sample_weight_policy=sample_weight_policy,
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

    def _weights(rows: T.Sequence[StackedTrainingExample]) -> np.ndarray:
        weights: np.ndarray = np.array([ex.sample_weight for ex in rows], dtype="float64")
        return weights

    x_train = _matrix(train_examples)
    y_train = _targets(train_examples)
    train_weights = (
        None if sample_weight_policy == SAMPLE_WEIGHT_POLICY_UNIFORM else _weights(train_examples)
    )
    coef, intercept, mean, std = fit_linear_residual_model(
        x_train,
        y_train,
        l2=l2,
        sample_weights=train_weights,
    )

    def _mse(rows: T.Sequence[StackedTrainingExample]) -> float:
        if not rows:
            return float("nan")
        x = _matrix(rows)
        y = _targets(rows)
        standardized = (x - mean) / std
        predicted = (standardized @ coef) + intercept
        return float(np.mean((predicted - y) ** 2))

    def _weighted_mse(rows: T.Sequence[StackedTrainingExample]) -> float:
        if not rows:
            return float("nan")
        x = _matrix(rows)
        y = _targets(rows)
        standardized = (x - mean) / std
        predicted = (standardized @ coef) + intercept
        per_row = np.mean((predicted - y) ** 2, axis=1)
        return float(np.average(per_row, weights=_weights(rows)))

    train_mse = _mse(train_examples)
    eval_mse = _mse(eval_examples)
    train_weighted_mse = _weighted_mse(train_examples)
    eval_weighted_mse = _weighted_mse(eval_examples)
    baseline_eval_mse = (
        float(np.mean(_targets(eval_examples) ** 2)) if eval_examples else float("nan")
    )
    weighted_baseline_eval_mse = (
        float(
            np.average(
                np.mean(_targets(eval_examples) ** 2, axis=1), weights=_weights(eval_examples)
            )
        )
        if eval_examples
        else float("nan")
    )
    train_weights_for_metrics = _weights(train_examples)
    eval_weights_for_metrics = (
        _weights(eval_examples) if eval_examples else np.array([], dtype="float64")
    )

    training_metadata: dict[str, T.Any] = {
        "output_mode": output_mode,
        "base_candidate_policy": base_candidate_policy,
        "n_examples": len(examples),
        "n_train": len(train_examples),
        "n_eval": len(eval_examples),
        "train_mse": train_mse,
        "eval_mse": eval_mse,
        "train_weighted_mse": train_weighted_mse,
        "eval_weighted_mse": eval_weighted_mse,
        "baseline_eval_mse": baseline_eval_mse,
        "weighted_baseline_eval_mse": weighted_baseline_eval_mse,
        "l2": float(l2),
        "sample_weight_policy": sample_weight_policy,
        "train_sample_weight_min": float(train_weights_for_metrics.min()),
        "train_sample_weight_max": float(train_weights_for_metrics.max()),
        "train_sample_weight_mean": float(train_weights_for_metrics.mean()),
        "eval_sample_weight_min": (
            float(eval_weights_for_metrics.min())
            if eval_weights_for_metrics.size
            else float("nan")
        ),
        "eval_sample_weight_max": (
            float(eval_weights_for_metrics.max())
            if eval_weights_for_metrics.size
            else float("nan")
        ),
        "eval_sample_weight_mean": (
            float(eval_weights_for_metrics.mean())
            if eval_weights_for_metrics.size
            else float("nan")
        ),
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
        "[StackedRegressor] trained output_mode=%s sample_weight_policy=%s "
        "n_train=%d n_eval=%d train_mse=%.6g eval_mse=%.6g baseline_eval_mse=%.6g "
        "train_weight_range=[%.3g, %.3g]",
        output_mode,
        sample_weight_policy,
        len(train_examples),
        len(eval_examples),
        train_mse,
        eval_mse,
        baseline_eval_mse,
        float(train_weights_for_metrics.min()),
        float(train_weights_for_metrics.max()),
    )
    return regressor, metrics


def _train_single_kwargs(
    *,
    output_mode: str,
    base_candidate_policy: str,
    candidate_name: str,
    residual_clip_fraction: float,
    l2: float,
    eval_fraction: float,
    split_seed: str | int,
    allow_landmark_residual_68: bool,
    sample_weight_policy: str | None,
) -> dict[str, T.Any]:
    kwargs: dict[str, T.Any] = {
        "output_mode": output_mode,
        "base_candidate_policy": base_candidate_policy,
        "candidate_name": candidate_name,
        "residual_clip_fraction": residual_clip_fraction,
        "l2": l2,
        "eval_fraction": eval_fraction,
        "split_seed": split_seed,
        "allow_landmark_residual_68": allow_landmark_residual_68,
    }
    if (
        sample_weight_policy is not None
        and "sample_weight_policy" in train_stacked_regressor.__code__.co_varnames
    ):
        kwargs["sample_weight_policy"] = sample_weight_policy
    return kwargs


def _group_contexts_by_runtime_expert(
    contexts: T.Iterable[ContextLike],
) -> dict[str, list[ContextLike]]:
    grouped: dict[str, list[ContextLike]] = {}
    for context in contexts:
        bucket = str(getattr(context, "runtime_bucket", "") or "")
        key = runtime_bucket_expert_key(bucket)
        grouped.setdefault(key, []).append(context)
    return grouped


def train_stacked_regressor_experts(
    contexts: T.Iterable[ContextLike],
    *,
    output_mode: str = OUTPUT_MODE_GLOBAL_TRANSFORM,
    base_candidate_policy: str = "static_weighted",
    candidate_name: str = DEFAULT_STACKED_CANDIDATE_NAME,
    residual_clip_fraction: float = 0.05,
    l2: float = DEFAULT_STACKED_L2,
    sample_weight_policy: str | None = None,
    eval_fraction: float = 0.2,
    split_seed: str | int = "stacked_residual_v1",
    allow_landmark_residual_68: bool = False,
    expert_min_examples: int = DEFAULT_STACKED_EXPERT_MIN_EXAMPLES,
) -> tuple[RuntimeStackedLandmarkRegressor, dict[str, T.Any]]:
    """Train one linear expert per runtime-bucket family in a single artifact.

    Routing is deterministic from runtime-visible bucket features. This changes
    the model class from one global linear fit to multiple pose-family linear fits;
    it does not add runtime accept/reject gates.
    """
    context_rows = list(contexts)
    if not context_rows:
        raise StackedRegressorTrainingError("no contexts supplied for expert training")

    grouped = _group_contexts_by_runtime_expert(context_rows)
    default_key = DEFAULT_STACKED_EXPERT_KEY
    default_contexts = grouped.get(default_key) or context_rows

    single_kwargs = _train_single_kwargs(
        output_mode=output_mode,
        base_candidate_policy=base_candidate_policy,
        candidate_name=candidate_name,
        residual_clip_fraction=residual_clip_fraction,
        l2=l2,
        eval_fraction=eval_fraction,
        split_seed=split_seed,
        allow_landmark_residual_68=allow_landmark_residual_68,
        sample_weight_policy=sample_weight_policy,
    )

    experts: dict[str, RuntimeStackedLandmarkRegressor] = {}
    expert_metrics: dict[str, dict[str, T.Any]] = {}

    default_model, default_metrics = train_stacked_regressor(default_contexts, **single_kwargs)
    experts[default_key] = default_model
    expert_metrics[default_key] = {
        **default_metrics,
        "n_contexts": len(default_contexts),
        "runtime_bucket_expert": default_key,
    }

    for key, key_contexts in sorted(grouped.items()):
        if key == default_key:
            continue
        if len(key_contexts) < expert_min_examples:
            logger.info(
                "[StackedRegressor] skipping expert=%s n_contexts=%d < min=%d; routes fall back to %s",
                key,
                len(key_contexts),
                expert_min_examples,
                default_key,
            )
            continue
        expert_model, metrics = train_stacked_regressor(
            key_contexts,
            **{
                **single_kwargs,
                "split_seed": f"{split_seed}:{key}",
            },
        )
        experts[key] = expert_model
        expert_metrics[key] = {
            **metrics,
            "n_contexts": len(key_contexts),
            "runtime_bucket_expert": key,
        }

    training_metadata: dict[str, T.Any] = {
        "expert_policy": EXPERT_POLICY_RUNTIME_BUCKET,
        "default_expert": default_key,
        "experts": expert_metrics,
        "expert_min_examples": int(expert_min_examples),
        "n_contexts": len(context_rows),
        "n_experts": len(experts),
        "output_mode": output_mode,
        "base_candidate_policy": base_candidate_policy,
        "residual_clip_fraction": float(residual_clip_fraction),
        "l2": float(l2),
        "sample_weight_policy": sample_weight_policy,
        "eval_fraction": float(eval_fraction),
        "split_seed": str(split_seed),
    }

    root = RuntimeStackedLandmarkRegressor(
        candidate_name=candidate_name,
        base_candidate_policy=base_candidate_policy,
        feature_names=default_model.feature_names,
        output_mode=default_model.output_mode,
        output_dim=default_model.output_dim,
        residual_clip_fraction=default_model.residual_clip_fraction,
        coef=default_model.coef,
        intercept=default_model.intercept,
        feature_mean=default_model.feature_mean,
        feature_std=default_model.feature_std,
        training_metadata=training_metadata,
        expert_policy=EXPERT_POLICY_RUNTIME_BUCKET,
        default_expert=default_key,
        experts=experts,
    )
    logger.info(
        "[StackedRegressor] trained runtime-bucket experts=%s default=%s n_contexts=%d",
        sorted(experts),
        default_key,
        len(context_rows),
    )
    return root, training_metadata


__all__ = [
    "DEFAULT_STACKED_L2",
    "train_stacked_regressor_experts",
    "DEFAULT_STACKED_EXPERT_MIN_EXAMPLES",
    "SAMPLE_WEIGHT_POLICIES",
    "SAMPLE_WEIGHT_POLICY_HARD_SLICE",
    "SAMPLE_WEIGHT_POLICY_HARD_SLICE_PLUS_BASE_ERROR",
    "SAMPLE_WEIGHT_POLICY_UNIFORM",
    "STACKED_REGRESSOR_ARTIFACT",
    "StackedRegressorTrainingError",
    "StackedTrainingExample",
    "build_training_examples",
    "fit_linear_residual_model",
    "global_transform_target",
    "landmark_residual_68_target",
    "region_residual_target",
    "sample_weight_for_context",
    "select_base_candidate",
    "stacked_feature_order",
    "train_stacked_regressor",
]
