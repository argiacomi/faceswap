#!/usr/bin/env python3
"""Canonical landmark schema helpers for ensemble predictions."""

from __future__ import annotations

import typing as T

from dataclasses import dataclass, field

import numpy as np

CANONICAL_SCHEMA = "2d_68"


@dataclass(frozen=True)
class LandmarkSchema:
    """Description of a supported landmark layout."""

    name: str
    points: int
    dimensions: int

    @property
    def shape(self) -> tuple[int, int]:
        """Return the expected array shape for this schema."""
        return (self.points, self.dimensions)


SUPPORTED_SCHEMAS: dict[str, LandmarkSchema] = {
    "2d_4": LandmarkSchema("2d_4", 4, 2),
    "2d_51": LandmarkSchema("2d_51", 51, 2),
    "2d_68": LandmarkSchema("2d_68", 68, 2),
    "2d_98": LandmarkSchema("2d_98", 98, 2),
    "3d_26": LandmarkSchema("3d_26", 26, 3),
}

_SCHEMA_ALIASES = {
    "4": "2d_4",
    "4pt": "2d_4",
    "lm_2d_4": "2d_4",
    "51": "2d_51",
    "51pt": "2d_51",
    "lm_2d_51": "2d_51",
    "68": "2d_68",
    "68pt": "2d_68",
    "canonical": "2d_68",
    "lm_2d_68": "2d_68",
    "98": "2d_98",
    "98pt": "2d_98",
    "lm_2d_98": "2d_98",
    "26": "3d_26",
    "26pt3d": "3d_26",
    "lm_3d_26": "3d_26",
}


@dataclass(frozen=True, init=False)
class LandmarkPrediction:
    """A single adapter prediction with schema and coordinate metadata."""

    landmarks: np.ndarray
    confidence: np.ndarray | None = None
    model_name: str = ""
    source_landmark_count: int = 68
    coordinate_space: str = "frame"
    metadata: dict[str, T.Any] = field(default_factory=dict)
    schema: str = CANONICAL_SCHEMA

    def __init__(
        self,
        landmarks: T.Sequence[T.Sequence[float]] | np.ndarray | None = None,
        *,
        points: T.Sequence[T.Sequence[float]] | np.ndarray | None = None,
        schema: str | object | None = None,
        confidence: np.ndarray | None = None,
        model_name: str = "",
        source: str | None = None,
        source_landmark_count: int | None = None,
        coordinate_space: str = "frame",
        metadata: dict[str, T.Any] | None = None,
    ) -> None:
        """Create a landmark prediction.

        ``points`` and ``source`` are compatibility aliases for the first
        prototype. New code should prefer ``landmarks`` and ``model_name``.
        """
        if landmarks is None:
            if points is None:
                raise ValueError("landmarks are required")
            landmarks = points
        elif points is not None:
            raise ValueError("provide either landmarks or points, not both")

        raw = np.asarray(landmarks, dtype="float32")
        schema_name = (
            infer_schema(raw) if schema is None else canonicalize_schema(schema)
        )
        points_array = normalize_landmark_array(raw, schema=schema_name)
        name = model_name if source is None else source
        if source_landmark_count is None:
            source_landmark_count = points_array.shape[0]
        if source_landmark_count <= 0:
            raise ValueError("source_landmark_count must be greater than zero")
        if not coordinate_space.strip():
            raise ValueError("coordinate_space cannot be empty")

        object.__setattr__(self, "landmarks", points_array)
        object.__setattr__(self, "schema", schema_name)
        object.__setattr__(self, "model_name", name)
        object.__setattr__(self, "source_landmark_count", int(source_landmark_count))
        object.__setattr__(self, "coordinate_space", coordinate_space.strip())
        object.__setattr__(self, "metadata", {} if metadata is None else dict(metadata))
        if confidence is None:
            object.__setattr__(self, "confidence", None)
        else:
            conf = np.asarray(confidence, dtype="float32")
            if conf.shape != (points_array.shape[0],):
                raise ValueError(
                    "confidence must be a 1D array with one value per landmark point: "
                    f"expected {(points_array.shape[0],)}, got {conf.shape}"
                )
            if not np.all(np.isfinite(conf)):
                raise ValueError("confidence contains NaN or infinite values")
            object.__setattr__(self, "confidence", conf)

    @property
    def points(self) -> np.ndarray:
        """Compatibility alias for :attr:`landmarks`."""
        return self.landmarks

    @property
    def source(self) -> str:
        """Compatibility alias for :attr:`model_name`."""
        return self.model_name

    def canonical_68(self) -> LandmarkPrediction:
        """Return this prediction remapped to the 68-point 2D schema."""
        points = to_canonical_68(self.landmarks, source_schema=self.schema)
        confidence = None
        if self.confidence is not None and self.schema == CANONICAL_SCHEMA:
            confidence = self.confidence.copy()
        return LandmarkPrediction(
            landmarks=points,
            schema=CANONICAL_SCHEMA,
            confidence=confidence,
            model_name=self.model_name,
            source_landmark_count=self.source_landmark_count,
            coordinate_space=self.coordinate_space,
            metadata=self.metadata,
        )


def canonicalize_schema(schema: str | object) -> str:
    """Normalize schema names and Faceswap enum-like values to local schema names."""
    if hasattr(schema, "name"):
        raw = str(getattr(schema, "name"))
    else:
        raw = str(schema)
    key = raw.strip().lower().replace("-", "_")
    if key in SUPPORTED_SCHEMAS:
        return key
    if key in _SCHEMA_ALIASES:
        return _SCHEMA_ALIASES[key]
    raise ValueError(
        f"Unsupported landmark schema '{schema}'. "
        f"Supported schemas: {sorted(SUPPORTED_SCHEMAS)}"
    )


def infer_schema(points: np.ndarray) -> str:
    """Infer a supported schema from a landmark array shape."""
    if points.ndim != 2:
        raise ValueError(f"landmarks must be 2D, got shape {points.shape}")
    matches = [
        schema.name
        for schema in SUPPORTED_SCHEMAS.values()
        if points.shape == schema.shape
    ]
    if not matches:
        raise ValueError(f"Cannot infer landmark schema from shape {points.shape}")
    return matches[0]


def normalize_landmark_array(
    points: T.Sequence[T.Sequence[float]] | np.ndarray,
    *,
    schema: str | object | None = None,
    dtype: str | np.dtype = "float32",
) -> np.ndarray:
    """Coerce landmark data to a finite ``numpy`` array and validate its shape."""
    array = np.asarray(points, dtype=dtype)
    if array.ndim == 1:
        if array.size % 2 != 0:
            raise ValueError(
                f"flat landmark arrays must contain x/y pairs, got {array.size} values"
            )
        array = array.reshape((-1, 2))
    if array.ndim != 2:
        raise ValueError(f"landmarks must be a 2D array, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("landmarks contain NaN or infinite values")
    if schema is not None:
        schema_name = canonicalize_schema(schema)
        expected = SUPPORTED_SCHEMAS[schema_name].shape
        if array.shape != expected:
            raise ValueError(
                f"landmarks for schema '{schema_name}' must have shape {expected}, "
                f"got {array.shape}"
            )
    elif array.shape[1] not in (2, 3):
        raise ValueError(f"landmarks must have 2 or 3 dimensions, got {array.shape[1]}")
    return np.ascontiguousarray(array, dtype=dtype)


def to_canonical_68(
    points: T.Sequence[T.Sequence[float]] | np.ndarray,
    *,
    source_schema: str | object | None = None,
) -> np.ndarray:
    """Return a 68-point 2D landmark array."""
    array = normalize_landmark_array(points, schema=source_schema)
    schema = (
        infer_schema(array)
        if source_schema is None
        else canonicalize_schema(source_schema)
    )
    if schema == CANONICAL_SCHEMA:
        return array.astype("float32", copy=True)
    if schema == "2d_98":
        from lib.align.constants import MAP_2D_68, LandmarkType

        indexes = MAP_2D_68[LandmarkType.LM_2D_98]
        return array[indexes].astype("float32", copy=True)
    raise ValueError(f"Cannot map schema '{schema}' to canonical 68-point landmarks")


def normalize_landmarks(
    points: T.Sequence[T.Sequence[float]] | np.ndarray,
    *,
    source_schema: str | object | None = None,
    target_schema: str = CANONICAL_SCHEMA,
) -> np.ndarray:
    """Normalize landmark inputs to the requested public schema."""
    target = canonicalize_schema(target_schema)
    if target != CANONICAL_SCHEMA:
        raise ValueError(f"Unsupported target landmark schema '{target_schema}'")
    return to_canonical_68(points, source_schema=source_schema)
