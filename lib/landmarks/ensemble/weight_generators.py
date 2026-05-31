#!/usr/bin/env python3
"""Pluggable static landmark weight generators (issue #68).

Each generator turns an :class:`ErrorTable` of per-model per-landmark errors
into normalized per-landmark fusion weights. Generators are stateless objects
keyed by ``name`` so the static-weight pipeline, candidate search, and runtime
artifact validators can refer to a single canonical generator identity.

Outputs match the schema consumed by
:func:`lib.landmarks.ensemble.weights.save_weights`: ``weights[model]`` is a
length-68 list of non-negative values and the per-landmark column sums are
positive before normalization.
"""

from __future__ import annotations

import typing as T
from dataclasses import dataclass, field

import numpy as np

from lib.landmarks.ensemble.weights import LANDMARK_COUNT, normalize_static_weights

# Non-overlapping landmark → region mapping for region-aware generators. This
# is derived from ``lib.align.constants.LANDMARK_PARTS`` for the 2D_68 schema
# but flattened so every landmark belongs to exactly one region. ``chin`` (7-9)
# is intentionally absorbed into ``jaw`` (0-17) to keep the partition clean.
LANDMARK_REGION_RANGES: tuple[tuple[str, int, int], ...] = (
    ("jaw", 0, 17),
    ("right_eyebrow", 17, 22),
    ("left_eyebrow", 22, 27),
    ("nose", 27, 36),
    ("right_eye", 36, 42),
    ("left_eye", 42, 48),
    ("mouth_outer", 48, 60),
    ("mouth_inner", 60, 68),
)


def landmark_region_indices(landmark_count: int = LANDMARK_COUNT) -> dict[str, list[int]]:
    """Return a region name → landmark index list mapping for the partition."""
    mapping: dict[str, list[int]] = {}
    for name, start, end in LANDMARK_REGION_RANGES:
        if end > landmark_count:
            continue
        mapping[name] = list(range(start, end))
    return mapping


@dataclass(frozen=True)
class ErrorTable:
    """Per-sample per-landmark error stack for one or more models.

    ``errors[model]`` is a 2D array of shape ``(sample_count, landmark_count)``.
    A table built from precomputed means has ``sample_count == 1`` so generators
    that operate on the mean still work without seeing the underlying samples.
    """

    errors: T.Mapping[str, np.ndarray]
    models: tuple[str, ...]
    landmark_count: int = LANDMARK_COUNT

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("ErrorTable requires at least one model")
        if tuple(self.errors) != self.models:
            raise ValueError("errors keys must match models order")
        for name, matrix in self.errors.items():
            if matrix.ndim != 2 or matrix.shape[1] != self.landmark_count:
                raise ValueError(
                    f"errors[{name!r}] must have shape (samples, {self.landmark_count}), "
                    f"got {matrix.shape}"
                )
            if np.any(matrix < 0) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"errors[{name!r}] must contain finite non-negative values")

    @classmethod
    def from_samples(
        cls,
        errors: T.Mapping[str, T.Sequence[np.ndarray] | np.ndarray],
        *,
        landmark_count: int = LANDMARK_COUNT,
    ) -> ErrorTable:
        """Build an ErrorTable from per-sample per-landmark arrays."""
        stacked: dict[str, np.ndarray] = {}
        models: list[str] = []
        for name, value in errors.items():
            matrix = (
                np.asarray(value, dtype="float32")
                if isinstance(value, np.ndarray)
                else np.stack([np.asarray(item, dtype="float32") for item in value], axis=0)
            )
            if matrix.ndim == 1:
                matrix = matrix[None, :]
            stacked[name] = matrix
            models.append(name)
        return cls(errors=stacked, models=tuple(models), landmark_count=landmark_count)

    @classmethod
    def from_means(
        cls,
        mean_errors: T.Mapping[str, T.Sequence[float]],
        *,
        landmark_count: int = LANDMARK_COUNT,
    ) -> ErrorTable:
        """Build an ErrorTable from precomputed per-landmark mean errors."""
        return cls.from_samples(
            {
                name: np.asarray(values, dtype="float32")[None, :]
                for name, values in mean_errors.items()
            },
            landmark_count=landmark_count,
        )

    def mean_errors(self) -> dict[str, np.ndarray]:
        """Return per-model length-``landmark_count`` mean error arrays."""
        return {name: matrix.mean(axis=0) for name, matrix in self.errors.items()}

    def median_errors(self) -> dict[str, np.ndarray]:
        """Return per-model length-``landmark_count`` median error arrays."""
        return {name: np.median(matrix, axis=0) for name, matrix in self.errors.items()}


@dataclass(frozen=True)
class WeightFitResult:
    """One generator's normalized per-landmark weights and diagnostics."""

    name: str
    weights: dict[str, list[float]]
    diagnostics: dict[str, T.Any] = field(default_factory=dict)


@T.runtime_checkable
class WeightGenerator(T.Protocol):
    """Protocol for objects that turn errors into normalized weights."""

    name: str

    def fit(
        self,
        errors: ErrorTable,
        *,
        models: T.Sequence[str] | None = None,
    ) -> WeightFitResult: ...


def _resolve_models(errors: ErrorTable, models: T.Sequence[str] | None) -> tuple[str, ...]:
    if models is None:
        return errors.models
    chosen = tuple(models)
    missing = [name for name in chosen if name not in errors.errors]
    if missing:
        raise ValueError(f"errors missing required models: {missing}")
    return chosen


def _inverse_error_weights(matrix: np.ndarray, *, epsilon: float) -> np.ndarray:
    """Return per-landmark inverse-error weights for a ``(M, L)`` error matrix.

    Normalization is delegated to :func:`normalize_static_weights` via callers
    so this returns un-normalized inverse values.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be greater than zero")
    return 1.0 / np.maximum(matrix.astype("float32"), epsilon)  # type: ignore[no-any-return]


def _matrix_from_errors(
    errors: ErrorTable,
    models: T.Sequence[str],
    *,
    aggregator: T.Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Aggregate per-sample errors per model into a ``(M, L)`` matrix."""
    rows = [aggregator(errors.errors[name]) for name in models]
    matrix = np.stack(rows, axis=0).astype("float32")
    if matrix.shape != (len(models), errors.landmark_count):
        raise ValueError(
            f"aggregated errors must have shape {(len(models), errors.landmark_count)}, "
            f"got {matrix.shape}"
        )
    return matrix  # type: ignore[no-any-return]


def _normalize_to_result(
    name: str,
    models: T.Sequence[str],
    matrix: np.ndarray,
    *,
    diagnostics: dict[str, T.Any] | None = None,
    landmark_count: int = LANDMARK_COUNT,
) -> WeightFitResult:
    """Wrap a non-negative ``(M, L)`` matrix as a normalized WeightFitResult."""
    payload = {name: matrix[idx].tolist() for idx, name in enumerate(models)}
    normalized = normalize_static_weights(payload, landmark_count=landmark_count)
    return WeightFitResult(
        name=name,
        weights=normalized,
        diagnostics=diagnostics or {},
    )


@dataclass(frozen=True)
class EqualGenerator:
    """Equal weight ``1/M`` for every model and landmark."""

    name: str = "equal"

    def fit(
        self,
        errors: ErrorTable,
        *,
        models: T.Sequence[str] | None = None,
    ) -> WeightFitResult:
        chosen = _resolve_models(errors, models)
        matrix = np.ones((len(chosen), errors.landmark_count), dtype="float32")  # type: ignore[var-annotated]
        return _normalize_to_result(
            self.name,
            chosen,
            matrix,
            diagnostics={"models": list(chosen)},
            landmark_count=errors.landmark_count,
        )


@dataclass(frozen=True)
class InverseMeanErrorGenerator:
    """Inverse of per-landmark mean error. Parity with legacy weights_from_errors."""

    name: str = "inverse_mean_error"
    epsilon: float = 1e-6

    def fit(
        self,
        errors: ErrorTable,
        *,
        models: T.Sequence[str] | None = None,
    ) -> WeightFitResult:
        chosen = _resolve_models(errors, models)
        mean_matrix = _matrix_from_errors(errors, chosen, aggregator=lambda m: m.mean(axis=0))
        matrix = _inverse_error_weights(mean_matrix, epsilon=self.epsilon)
        return _normalize_to_result(
            self.name,
            chosen,
            matrix,
            diagnostics={
                "models": list(chosen),
                "epsilon": float(self.epsilon),
                "mean_errors": {
                    name: mean_matrix[idx].tolist() for idx, name in enumerate(chosen)
                },
            },
            landmark_count=errors.landmark_count,
        )


@dataclass(frozen=True)
class InverseMedianErrorGenerator:
    """Inverse of per-landmark median error. Less sensitive to outlier samples."""

    name: str = "inverse_median_error"
    epsilon: float = 1e-6

    def fit(
        self,
        errors: ErrorTable,
        *,
        models: T.Sequence[str] | None = None,
    ) -> WeightFitResult:
        chosen = _resolve_models(errors, models)
        median_matrix = _matrix_from_errors(
            errors, chosen, aggregator=lambda m: np.median(m, axis=0)
        )
        matrix = _inverse_error_weights(median_matrix, epsilon=self.epsilon)
        return _normalize_to_result(
            self.name,
            chosen,
            matrix,
            diagnostics={
                "models": list(chosen),
                "epsilon": float(self.epsilon),
                "median_errors": {
                    name: median_matrix[idx].tolist() for idx, name in enumerate(chosen)
                },
            },
            landmark_count=errors.landmark_count,
        )


@dataclass(frozen=True)
class RegionInverseErrorGenerator:
    """Inverse of per-region mean error broadcast over landmarks in that region."""

    name: str = "region_inverse_error"
    epsilon: float = 1e-6

    def fit(
        self,
        errors: ErrorTable,
        *,
        models: T.Sequence[str] | None = None,
    ) -> WeightFitResult:
        chosen = _resolve_models(errors, models)
        mean_matrix = _matrix_from_errors(errors, chosen, aggregator=lambda m: m.mean(axis=0))
        regions = landmark_region_indices(errors.landmark_count)
        region_matrix = _region_broadcast(mean_matrix, regions, errors.landmark_count)
        weights = _inverse_error_weights(region_matrix, epsilon=self.epsilon)
        return _normalize_to_result(
            self.name,
            chosen,
            weights,
            diagnostics={
                "models": list(chosen),
                "epsilon": float(self.epsilon),
                "regions": {name: indices for name, indices in regions.items()},
            },
            landmark_count=errors.landmark_count,
        )


def _region_broadcast(
    mean_matrix: np.ndarray,
    regions: dict[str, list[int]],
    landmark_count: int,
) -> np.ndarray:
    """Return a ``(M, L)`` matrix where each landmark holds its region's mean error."""
    output = np.copy(mean_matrix)
    for indices in regions.values():
        if not indices:
            continue
        region_means = mean_matrix[:, indices].mean(axis=1, keepdims=True)
        output[:, indices] = region_means
    return output[:, :landmark_count]  # type: ignore[no-any-return]


@dataclass(frozen=True)
class RegularizedInverseErrorGenerator:
    """Blend per-landmark, per-region, and global inverse-error weights.

    Default mixture matches the Epic suggestion:
    ``0.5 * per_landmark + 0.3 * per_region + 0.2 * global``. The components
    are normalized independently before mixing so each contributes a unit-sum
    column distribution; the final result is normalized again for safety.
    """

    name: str = "regularized_inverse_error"
    epsilon: float = 1e-6
    per_landmark_weight: float = 0.5
    per_region_weight: float = 0.3
    global_weight: float = 0.2

    def __post_init__(self) -> None:
        total = self.per_landmark_weight + self.per_region_weight + self.global_weight
        if total <= 0:
            raise ValueError("at least one component weight must be greater than zero")
        if min(self.per_landmark_weight, self.per_region_weight, self.global_weight) < 0:
            raise ValueError("component weights must be non-negative")

    def fit(
        self,
        errors: ErrorTable,
        *,
        models: T.Sequence[str] | None = None,
    ) -> WeightFitResult:
        chosen = _resolve_models(errors, models)
        mean_matrix = _matrix_from_errors(errors, chosen, aggregator=lambda m: m.mean(axis=0))

        per_landmark = _normalize_columns(
            _inverse_error_weights(mean_matrix, epsilon=self.epsilon)
        )
        regions = landmark_region_indices(errors.landmark_count)
        region_matrix = _region_broadcast(mean_matrix, regions, errors.landmark_count)
        per_region = _normalize_columns(
            _inverse_error_weights(region_matrix, epsilon=self.epsilon)
        )
        global_means = mean_matrix.mean(axis=1, keepdims=True)
        global_inverse = _inverse_error_weights(global_means, epsilon=self.epsilon)
        global_vector = (global_inverse / global_inverse.sum()).repeat(
            errors.landmark_count, axis=1
        )

        mixed = (
            self.per_landmark_weight * per_landmark
            + self.per_region_weight * per_region
            + self.global_weight * global_vector
        )
        return _normalize_to_result(
            self.name,
            chosen,
            mixed,
            diagnostics={
                "models": list(chosen),
                "epsilon": float(self.epsilon),
                "components": {
                    "per_landmark_weight": float(self.per_landmark_weight),
                    "per_region_weight": float(self.per_region_weight),
                    "global_weight": float(self.global_weight),
                },
                "regions": {name: indices for name, indices in regions.items()},
            },
            landmark_count=errors.landmark_count,
        )


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    """Normalize each landmark column to sum to one. Caller ensures positivity."""
    totals = matrix.sum(axis=0)
    if np.any(totals <= 0):
        raise ValueError("column sums must be positive before normalization")
    return matrix / totals[None, :]  # type: ignore[no-any-return]


@dataclass(frozen=True)
class WinnerTakeLandmarkGenerator:
    """Pick the lowest-mean-error model per landmark; ties resolved by first index."""

    name: str = "winner_take_landmark"

    def fit(
        self,
        errors: ErrorTable,
        *,
        models: T.Sequence[str] | None = None,
    ) -> WeightFitResult:
        chosen = _resolve_models(errors, models)
        mean_matrix = _matrix_from_errors(errors, chosen, aggregator=lambda m: m.mean(axis=0))
        winners = np.argmin(mean_matrix, axis=0)
        matrix = np.zeros_like(mean_matrix)
        matrix[winners, np.arange(mean_matrix.shape[1])] = 1.0
        return _normalize_to_result(
            self.name,
            chosen,
            matrix,
            diagnostics={
                "models": list(chosen),
                "winners": winners.tolist(),
            },
            landmark_count=errors.landmark_count,
        )


@dataclass(frozen=True)
class SoftmaxNegativeErrorGenerator:
    """Softmax over negated per-landmark mean error, scaled by ``temperature``.

    Higher temperature concentrates weight on the lower-error model; lower
    temperature spreads weight more evenly. Numerically stable max-subtraction
    is used so very large temperatures do not overflow.
    """

    name: str = "softmax_negative_error"
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be greater than zero")

    def fit(
        self,
        errors: ErrorTable,
        *,
        models: T.Sequence[str] | None = None,
    ) -> WeightFitResult:
        chosen = _resolve_models(errors, models)
        mean_matrix = _matrix_from_errors(errors, chosen, aggregator=lambda m: m.mean(axis=0))
        scaled = -mean_matrix * self.temperature
        shifted = scaled - scaled.max(axis=0, keepdims=True)
        exp = np.exp(shifted)
        totals = exp.sum(axis=0, keepdims=True)
        return _normalize_to_result(
            self.name,
            chosen,
            exp / totals,
            diagnostics={
                "models": list(chosen),
                "temperature": float(self.temperature),
            },
            landmark_count=errors.landmark_count,
        )


_DEFAULT_GENERATOR_INSTANCES: tuple[WeightGenerator, ...] = (  # type: ignore[assignment]
    EqualGenerator(),
    InverseMeanErrorGenerator(),
    InverseMedianErrorGenerator(),
    RegionInverseErrorGenerator(),
    RegularizedInverseErrorGenerator(),
    WinnerTakeLandmarkGenerator(),
    SoftmaxNegativeErrorGenerator(),
)

GENERATORS: dict[str, WeightGenerator] = {
    instance.name: instance for instance in _DEFAULT_GENERATOR_INSTANCES
}

GENERATOR_NAMES: tuple[str, ...] = tuple(GENERATORS)


def get_generator(name: str) -> WeightGenerator:
    """Return the registered generator for ``name`` or raise ValueError."""
    if name not in GENERATORS:
        raise ValueError(
            f"unknown weight generator {name!r}; supported generators: "
            + ", ".join(GENERATOR_NAMES)
        )
    return GENERATORS[name]


__all__ = [
    "ErrorTable",
    "EqualGenerator",
    "GENERATOR_NAMES",
    "GENERATORS",
    "InverseMeanErrorGenerator",
    "InverseMedianErrorGenerator",
    "LANDMARK_REGION_RANGES",
    "RegionInverseErrorGenerator",
    "RegularizedInverseErrorGenerator",
    "SoftmaxNegativeErrorGenerator",
    "WeightFitResult",
    "WeightGenerator",
    "WinnerTakeLandmarkGenerator",
    "get_generator",
    "landmark_region_indices",
]
