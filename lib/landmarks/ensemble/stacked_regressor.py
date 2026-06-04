#!/usr/bin/env python3
"""Stacked residual landmark regressor artifact contract and runtime loader (#223).

The stacked regressor is a *candidate generator*, not a selector. It consumes
existing model predictions plus runtime-visible geometry features and produces a
small, clipped correction to a safe base candidate. The corrected shape is added
to the runtime resolver candidate pool as ``stacked_residual`` and then flows
through the existing metrics, geometry vetoes, learned scorer, and safety
fallbacks exactly like any other candidate.

Design constraints:

- Runtime-visible features only. No GT, NME, oracle, transform-regret, or
  label-only fields enter the feature vector. The single source of truth for the
  feature contract is :mod:`lib.landmarks.ensemble.runtime_features`.
- Constrained output. The first artifact corrects a global similarity transform
  (translation, scale, rotation); region-wise and full 68-point residual modes
  are supported but the 68-point mode is gated behind an explicit training flag.
- Bounded correction. Per-point residuals are clipped to a fraction of the face
  bbox diagonal so the regressor can never emit an unbounded landmark shift.

The serialized model is a small standardized linear model (``coef``/``intercept``
plus per-feature ``mean``/``std``). This keeps the artifact dependency-free,
deterministic, and trivially loadable on the extract path without pulling in a
heavyweight runtime such as LightGBM or Torch.
"""

from __future__ import annotations

import json
import logging
import math
import typing as T
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from lib.landmarks.ensemble.runtime_features import RUNTIME_FEATURE_CONTRACT_VERSION

logger = logging.getLogger(__name__)

#: Bumped whenever the artifact schema or feature semantics change in a way that
#: makes older artifacts unsafe to load.
STACKED_REGRESSOR_CONTRACT_VERSION = "stacked_residual_v1"

#: Default candidate name the regressor emits into the resolver pool.
DEFAULT_STACKED_CANDIDATE_NAME = "stacked_residual"

#: Only model payload the runtime loader knows how to evaluate.
MODEL_TYPE_NUMPY_LINEAR = "numpy_linear_residual_v1"

#: Supported correction parameterizations.
OUTPUT_MODE_GLOBAL_TRANSFORM = "global_transform"
OUTPUT_MODE_REGION_RESIDUAL = "region_residual"
OUTPUT_MODE_LANDMARK_RESIDUAL_68 = "landmark_residual_68"

OUTPUT_MODES: frozenset[str] = frozenset(
    {
        OUTPUT_MODE_GLOBAL_TRANSFORM,
        OUTPUT_MODE_REGION_RESIDUAL,
        OUTPUT_MODE_LANDMARK_RESIDUAL_68,
    }
)

#: 5-region partition covering all 68 canonical points. Used by
#: ``region_residual`` to broadcast a per-region ``(dx, dy)`` correction.
STACKED_REGION_INDICES: dict[str, tuple[int, ...]] = {
    "jaw": tuple(range(0, 17)),
    "brows": tuple(range(17, 27)),
    "nose": tuple(range(27, 36)),
    "eyes": tuple(range(36, 48)),
    "mouth": tuple(range(48, 68)),
}
STACKED_REGION_NAMES: tuple[str, ...] = tuple(STACKED_REGION_INDICES)

#: Output vector dimensionality per mode.
#: - global_transform: (tx, ty, log_scale, rotation_radians)
#: - region_residual: (dx, dy) per region
#: - landmark_residual_68: (dx, dy) per landmark
GLOBAL_TRANSFORM_OUTPUT_DIM = 4
REGION_RESIDUAL_OUTPUT_DIM = len(STACKED_REGION_NAMES) * 2
LANDMARK_RESIDUAL_68_OUTPUT_DIM = 68 * 2

OUTPUT_DIM_BY_MODE: dict[str, int] = {
    OUTPUT_MODE_GLOBAL_TRANSFORM: GLOBAL_TRANSFORM_OUTPUT_DIM,
    OUTPUT_MODE_REGION_RESIDUAL: REGION_RESIDUAL_OUTPUT_DIM,
    OUTPUT_MODE_LANDMARK_RESIDUAL_68: LANDMARK_RESIDUAL_68_OUTPUT_DIM,
}


class StackedRegressorError(RuntimeError):
    """Base error for stacked regressor artifact handling."""


class StackedRegressorInvalid(StackedRegressorError):
    """An artifact exists but its schema or payload is unusable."""


def output_dim_for_mode(output_mode: str) -> int:
    """Return the expected raw output dimensionality for ``output_mode``."""
    try:
        return OUTPUT_DIM_BY_MODE[output_mode]
    except KeyError as err:
        raise StackedRegressorInvalid(
            f"unsupported stacked regressor output_mode {output_mode!r}; "
            f"expected one of {sorted(OUTPUT_MODES)}"
        ) from err


@dataclass(frozen=True, slots=True)
class StackedResidualResult:
    """Corrected landmarks plus runtime-visible residual diagnostics."""

    landmarks: np.ndarray
    residual_norm_max: float
    residual_norm_mean: float
    clip_applied: bool

    def feature_values(self) -> dict[str, float]:
        """Return scorer-visible features describing the residual correction."""
        return {
            "stacked_residual_norm_max": float(self.residual_norm_max),
            "stacked_residual_norm_mean": float(self.residual_norm_mean),
            "stacked_clip_applied": 1.0 if self.clip_applied else 0.0,
        }


def _as_float_matrix(value: T.Any, *, name: str) -> np.ndarray:
    try:
        matrix: np.ndarray = np.asarray(value, dtype="float64")
    except (TypeError, ValueError) as err:
        raise StackedRegressorInvalid(f"stacked regressor {name} is not numeric") from err
    if not np.all(np.isfinite(matrix)):
        raise StackedRegressorInvalid(f"stacked regressor {name} has non-finite values")
    return matrix


def _bbox_diagonal_from_landmarks(landmarks: np.ndarray) -> float:
    """Return the diagonal of the landmark bounding box, with a safe floor."""
    mins = landmarks.min(axis=0)
    maxs = landmarks.max(axis=0)
    span = maxs - mins
    diag = float(math.hypot(float(span[0]), float(span[1])))
    return diag if diag > 1e-6 else 1.0


def _global_transform_residual(
    base: np.ndarray,
    raw_output: np.ndarray,
    bbox_diagonal: float,
) -> np.ndarray:
    """Convert a (tx, ty, log_scale, rotation) vector into a per-point residual."""
    tx, ty, log_scale, rotation = (float(value) for value in raw_output[:4])
    # log_scale is bounded by construction at training time, but guard against a
    # corrupt artifact producing a degenerate or exploding scale here too.
    scale = math.exp(max(-1.0, min(1.0, log_scale)))
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype="float64")
    centroid = base.mean(axis=0)
    centered = base - centroid
    transformed = (scale * (centered @ rotation_matrix.T)) + centroid
    translation = np.array([tx, ty], dtype="float64") * bbox_diagonal
    corrected = transformed + translation
    residual: np.ndarray = corrected - base
    return residual


def _region_residual(raw_output: np.ndarray, bbox_diagonal: float) -> np.ndarray:
    """Broadcast a per-region (dx, dy) vector into a (68, 2) residual."""
    residual: np.ndarray = np.zeros((68, 2), dtype="float64")
    for region_index, region in enumerate(STACKED_REGION_NAMES):
        dx = float(raw_output[region_index * 2])
        dy = float(raw_output[region_index * 2 + 1])
        for point in STACKED_REGION_INDICES[region]:
            residual[point, 0] = dx
            residual[point, 1] = dy
    scaled: np.ndarray = residual * bbox_diagonal
    return scaled


def _landmark_residual_68(raw_output: np.ndarray, bbox_diagonal: float) -> np.ndarray:
    """Reshape a flat 136-vector into a (68, 2) residual."""
    residual: np.ndarray = raw_output[:136].reshape(68, 2).astype("float64") * bbox_diagonal
    return residual


def apply_residual(
    base_landmarks: np.ndarray,
    raw_output: np.ndarray,
    *,
    output_mode: str,
    clip_fraction: float,
    bbox_diagonal: float | None = None,
) -> StackedResidualResult:
    """Apply a raw regressor output to ``base_landmarks`` as a clipped residual.

    ``raw_output`` carries normalized units (fractions of the bbox diagonal for
    translations/residuals, log-scale and radians for the global transform). The
    correction is converted to pixel space, clipped per-point to
    ``clip_fraction * bbox_diagonal``, and added to the base landmarks.
    """
    base = np.asarray(base_landmarks, dtype="float64")
    if base.shape != (68, 2):
        raise StackedRegressorInvalid(
            f"stacked regressor base candidate must be (68, 2); got {base.shape}"
        )
    diag = _bbox_diagonal_from_landmarks(base) if bbox_diagonal is None else float(bbox_diagonal)
    if diag <= 0:
        diag = 1.0

    if output_mode == OUTPUT_MODE_GLOBAL_TRANSFORM:
        residual = _global_transform_residual(base, raw_output, diag)
    elif output_mode == OUTPUT_MODE_REGION_RESIDUAL:
        residual = _region_residual(raw_output, diag)
    elif output_mode == OUTPUT_MODE_LANDMARK_RESIDUAL_68:
        residual = _landmark_residual_68(raw_output, diag)
    else:
        raise StackedRegressorInvalid(f"unsupported stacked regressor output_mode {output_mode!r}")

    clip_px = max(0.0, float(clip_fraction)) * diag
    magnitudes = np.linalg.norm(residual, axis=1)
    clip_applied = False
    if clip_px > 0:
        over = magnitudes > clip_px
        if bool(np.any(over)):
            clip_applied = True
            # Scale each over-limit point back onto the clip radius.
            scale = np.ones_like(magnitudes)
            scale[over] = clip_px / magnitudes[over]
            residual = residual * scale[:, None]
            magnitudes = np.linalg.norm(residual, axis=1)

    corrected = (base + residual).astype("float32")
    residual_norm = magnitudes / diag
    return StackedResidualResult(
        landmarks=corrected,
        residual_norm_max=float(residual_norm.max()) if residual_norm.size else 0.0,
        residual_norm_mean=float(residual_norm.mean()) if residual_norm.size else 0.0,
        clip_applied=clip_applied,
    )


@dataclass(frozen=True)
class RuntimeStackedLandmarkRegressor:
    """Loaded stacked residual regressor artifact usable at extract runtime."""

    candidate_name: str
    base_candidate_policy: str
    feature_names: tuple[str, ...]
    output_mode: str
    output_dim: int
    residual_clip_fraction: float
    coef: np.ndarray
    intercept: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    model_type: str = MODEL_TYPE_NUMPY_LINEAR
    contract_version: str = STACKED_REGRESSOR_CONTRACT_VERSION
    runtime_feature_contract_version: str = RUNTIME_FEATURE_CONTRACT_VERSION
    training_metadata: dict[str, T.Any] = field(default_factory=dict)
    source_path: str = ""

    def __post_init__(self) -> None:
        if self.model_type != MODEL_TYPE_NUMPY_LINEAR:
            raise StackedRegressorInvalid(
                f"unsupported stacked regressor model_type {self.model_type!r}; "
                f"expected {MODEL_TYPE_NUMPY_LINEAR!r}"
            )
        if self.contract_version != STACKED_REGRESSOR_CONTRACT_VERSION:
            raise StackedRegressorInvalid(
                f"stacked regressor contract_version {self.contract_version!r} is not "
                f"compatible with runtime {STACKED_REGRESSOR_CONTRACT_VERSION!r}"
            )
        if self.runtime_feature_contract_version != RUNTIME_FEATURE_CONTRACT_VERSION:
            raise StackedRegressorInvalid(
                "stacked regressor was trained against runtime feature contract "
                f"{self.runtime_feature_contract_version!r} but runtime is "
                f"{RUNTIME_FEATURE_CONTRACT_VERSION!r}"
            )
        if self.output_mode not in OUTPUT_MODES:
            raise StackedRegressorInvalid(
                f"unsupported stacked regressor output_mode {self.output_mode!r}; "
                f"expected one of {sorted(OUTPUT_MODES)}"
            )
        expected_dim = output_dim_for_mode(self.output_mode)
        if self.output_dim != expected_dim:
            raise StackedRegressorInvalid(
                f"stacked regressor output_dim={self.output_dim} does not match "
                f"output_mode={self.output_mode!r} (expected {expected_dim})"
            )
        if not self.feature_names:
            raise StackedRegressorInvalid("stacked regressor artifact has no feature_names")
        n_features = len(self.feature_names)
        if self.coef.shape != (n_features, self.output_dim):
            raise StackedRegressorInvalid(
                f"stacked regressor coef shape {self.coef.shape} does not match "
                f"(n_features={n_features}, output_dim={self.output_dim})"
            )
        if self.intercept.shape != (self.output_dim,):
            raise StackedRegressorInvalid(
                f"stacked regressor intercept shape {self.intercept.shape} does not "
                f"match output_dim={self.output_dim}"
            )
        if self.feature_mean.shape != (n_features,):
            raise StackedRegressorInvalid(
                f"stacked regressor feature_mean shape {self.feature_mean.shape} does "
                f"not match n_features={n_features}"
            )
        if self.feature_std.shape != (n_features,):
            raise StackedRegressorInvalid(
                f"stacked regressor feature_std shape {self.feature_std.shape} does "
                f"not match n_features={n_features}"
            )
        if self.residual_clip_fraction < 0:
            raise StackedRegressorInvalid("residual_clip_fraction must be non-negative")

    @classmethod
    def from_payload(
        cls,
        payload: T.Mapping[str, T.Any],
        *,
        source_path: str = "",
    ) -> RuntimeStackedLandmarkRegressor:
        """Build a regressor from a parsed artifact payload, validating the contract."""
        if not isinstance(payload, T.Mapping):
            raise StackedRegressorInvalid("stacked regressor artifact must be a JSON object")
        feature_names = tuple(str(item) for item in payload.get("feature_names", ()))
        output_mode = str(payload.get("output_mode", ""))
        output_dim = int(payload.get("output_dim", output_dim_for_mode(output_mode)))
        coef = _as_float_matrix(payload.get("coef"), name="coef")
        if coef.ndim != 2:
            raise StackedRegressorInvalid("stacked regressor coef must be 2-D")
        intercept = _as_float_matrix(payload.get("intercept", []), name="intercept").reshape(-1)
        n_features = len(feature_names)
        mean = payload.get("feature_mean")
        std = payload.get("feature_std")
        feature_mean = (
            _as_float_matrix(mean, name="feature_mean").reshape(-1)
            if mean is not None
            else np.zeros(n_features, dtype="float64")
        )
        feature_std = (
            _as_float_matrix(std, name="feature_std").reshape(-1)
            if std is not None
            else np.ones(n_features, dtype="float64")
        )
        # Guard against zero-variance columns producing NaNs at predict time.
        feature_std = np.where(np.abs(feature_std) < 1e-12, 1.0, feature_std)
        training_metadata = payload.get("training_metadata", {})
        return cls(
            candidate_name=str(payload.get("candidate_name", DEFAULT_STACKED_CANDIDATE_NAME)),
            base_candidate_policy=str(payload.get("base_candidate_policy", "static_weighted")),
            feature_names=feature_names,
            output_mode=output_mode,
            output_dim=output_dim,
            residual_clip_fraction=float(payload.get("residual_clip_fraction", 0.05)),
            coef=coef,
            intercept=intercept,
            feature_mean=feature_mean,
            feature_std=feature_std,
            model_type=str(payload.get("model_type", MODEL_TYPE_NUMPY_LINEAR)),
            contract_version=str(
                payload.get("contract_version", STACKED_REGRESSOR_CONTRACT_VERSION)
            ),
            runtime_feature_contract_version=str(
                payload.get("runtime_feature_contract_version", RUNTIME_FEATURE_CONTRACT_VERSION)
            ),
            training_metadata=dict(training_metadata)
            if isinstance(training_metadata, T.Mapping)
            else {},
            source_path=source_path,
        )

    def to_payload(self) -> dict[str, T.Any]:
        """Return a JSON-serializable artifact payload."""
        return {
            "candidate_name": self.candidate_name,
            "base_candidate_policy": self.base_candidate_policy,
            "feature_names": list(self.feature_names),
            "output_mode": self.output_mode,
            "output_dim": self.output_dim,
            "residual_clip_fraction": self.residual_clip_fraction,
            "coef": self.coef.tolist(),
            "intercept": self.intercept.tolist(),
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "model_type": self.model_type,
            "contract_version": self.contract_version,
            "runtime_feature_contract_version": self.runtime_feature_contract_version,
            "training_metadata": dict(self.training_metadata),
        }

    def feature_vector(self, features: T.Mapping[str, float]) -> np.ndarray:
        """Build the standardized model input row in the artifact feature order."""
        raw = np.array(
            [_feature_value(features.get(name)) for name in self.feature_names],
            dtype="float64",
        )
        standardized: np.ndarray = (raw - self.feature_mean) / self.feature_std
        return standardized

    def predict(self, features: T.Mapping[str, float]) -> np.ndarray:
        """Return the raw regressor output vector (normalized units)."""
        standardized = self.feature_vector(features)
        output: np.ndarray = (standardized @ self.coef) + self.intercept
        return output

    def corrected_landmarks(
        self,
        base_landmarks: np.ndarray,
        features: T.Mapping[str, float],
        *,
        bbox_diagonal: float | None = None,
    ) -> StackedResidualResult:
        """Predict and apply the clipped residual to ``base_landmarks``."""
        raw_output = self.predict(features)
        return apply_residual(
            base_landmarks,
            raw_output,
            output_mode=self.output_mode,
            clip_fraction=self.residual_clip_fraction,
            bbox_diagonal=bbox_diagonal,
        )


def _feature_value(value: T.Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


@lru_cache(maxsize=8)
def _load_stacked_regressor_cached(path: str) -> RuntimeStackedLandmarkRegressor:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise StackedRegressorInvalid(
            f"failed to read stacked regressor artifact at {source}: {err}"
        ) from err
    return RuntimeStackedLandmarkRegressor.from_payload(payload, source_path=str(source))


def load_stacked_regressor(path: str | Path) -> RuntimeStackedLandmarkRegressor:
    """Load and validate a stacked regressor artifact, memoized by resolved path."""
    return _load_stacked_regressor_cached(str(Path(path).resolve()))


def write_stacked_regressor(
    regressor: RuntimeStackedLandmarkRegressor,
    path: str | Path,
) -> Path:
    """Write a stacked regressor artifact to disk."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(regressor.to_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "DEFAULT_STACKED_CANDIDATE_NAME",
    "GLOBAL_TRANSFORM_OUTPUT_DIM",
    "LANDMARK_RESIDUAL_68_OUTPUT_DIM",
    "MODEL_TYPE_NUMPY_LINEAR",
    "OUTPUT_DIM_BY_MODE",
    "OUTPUT_MODES",
    "OUTPUT_MODE_GLOBAL_TRANSFORM",
    "OUTPUT_MODE_LANDMARK_RESIDUAL_68",
    "OUTPUT_MODE_REGION_RESIDUAL",
    "REGION_RESIDUAL_OUTPUT_DIM",
    "STACKED_REGION_INDICES",
    "STACKED_REGION_NAMES",
    "STACKED_REGRESSOR_CONTRACT_VERSION",
    "RuntimeStackedLandmarkRegressor",
    "StackedRegressorError",
    "StackedRegressorInvalid",
    "StackedResidualResult",
    "apply_residual",
    "load_stacked_regressor",
    "output_dim_for_mode",
    "write_stacked_regressor",
]
