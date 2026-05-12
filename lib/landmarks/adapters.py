#!/usr/bin/env python3
"""Adapter interfaces for landmark predictors."""

from __future__ import annotations

import typing as T

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from lib.landmarks.schema import (
    CANONICAL_SCHEMA,
    LandmarkPrediction,
    canonicalize_schema,
)


@dataclass(frozen=True)
class LandmarkAdapterConfig:
    """Static adapter metadata used by the ensemble layer."""

    name: str
    schema: str = CANONICAL_SCHEMA
    coordinate_space: str = "frame"
    weight: float = 1.0
    enabled: bool = True
    options: T.Mapping[str, T.Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ValueError("adapter name cannot be empty")
        if self.weight < 0:
            raise ValueError("adapter weight cannot be negative")
        if not self.coordinate_space.strip():
            raise ValueError("coordinate_space cannot be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "schema", canonicalize_schema(self.schema))
        object.__setattr__(self, "coordinate_space", self.coordinate_space.strip())
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


class LandmarkAdapter(ABC):
    """Base class for model-specific landmark predictor adapters."""

    config: LandmarkAdapterConfig

    def __init__(self, config: LandmarkAdapterConfig) -> None:
        self.config = config

    @abstractmethod
    def predict(
        self, image: np.ndarray, *, face: object | None = None
    ) -> LandmarkPrediction:
        """Predict landmarks for an image or detected face."""

    def normalize_prediction(
        self,
        points: T.Sequence[T.Sequence[float]] | np.ndarray,
        *,
        confidence: np.ndarray | None = None,
        metadata: dict[str, T.Any] | None = None,
    ) -> LandmarkPrediction:
        """Wrap raw adapter output in the shared prediction dataclass."""
        source_landmark_count = int(np.asarray(points).shape[0])
        return LandmarkPrediction(
            landmarks=np.asarray(points, dtype="float32"),
            schema=self.config.schema,
            confidence=confidence,
            model_name=self.config.name,
            source_landmark_count=source_landmark_count,
            coordinate_space=self.config.coordinate_space,
            metadata={} if metadata is None else metadata,
        )


class StaticLandmarkAdapter(LandmarkAdapter):
    """Small deterministic adapter useful for tests and fixtures."""

    def __init__(
        self,
        config: LandmarkAdapterConfig,
        points: T.Sequence[T.Sequence[float]] | np.ndarray,
    ) -> None:
        super().__init__(config)
        self._points = np.asarray(points, dtype="float32")

    def predict(
        self, image: np.ndarray, *, face: object | None = None
    ) -> LandmarkPrediction:
        """Return the configured static landmarks without touching model state."""
        del image, face
        return self.normalize_prediction(self._points.copy())
