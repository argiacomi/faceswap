#!/usr/bin/env python3
"""Adapter interfaces for landmark predictors."""

from __future__ import annotations

import typing as T
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from lib.landmarks.coordinates import CoordinateSpace, normalized_crop_to_frame
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
    coordinate_space: CoordinateSpace = "frame"
    weight: float = 1.0
    enabled: bool = True
    options: T.Mapping[str, T.Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ValueError("adapter name cannot be empty")
        if self.weight < 0:
            raise ValueError("adapter weight cannot be negative")
        coordinate_space = self.coordinate_space.strip()
        if coordinate_space not in ("normalized_crop", "frame"):
            raise ValueError(
                "coordinate_space must be either 'normalized_crop' or 'frame', "
                f"got {self.coordinate_space!r}"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "schema", canonicalize_schema(self.schema))
        object.__setattr__(self, "coordinate_space", coordinate_space)
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


class LandmarkAdapter(ABC):
    """Base class for model-specific landmark predictor adapters."""

    config: LandmarkAdapterConfig

    def __init__(self, config: LandmarkAdapterConfig) -> None:
        self.config = config

    @abstractmethod
    def predict(self, image: np.ndarray, *, face: object | None = None) -> LandmarkPrediction:
        """Predict landmarks for an image or detected face."""

    def predict_batch(
        self,
        images: np.ndarray,
        *,
        matrices: np.ndarray | None = None,
        faces: T.Sequence[object] | None = None,
    ) -> list[LandmarkPrediction]:
        """Predict a batch and normalize predictions to canonical 68 frame pixels."""
        if images.ndim < 1:
            raise ValueError("images must be a batched array")
        if faces is not None and len(faces) != images.shape[0]:
            raise ValueError("faces must contain one item per image")
        predictions = []
        for idx, image in enumerate(images):
            face = None if faces is None else faces[idx]
            matrix = None if matrices is None else matrices[idx]
            prediction = self.predict(image, face=face)
            predictions.append(self.to_frame_prediction(prediction, matrix=matrix))
        return predictions

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

    def to_frame_prediction(
        self,
        prediction: LandmarkPrediction,
        *,
        matrix: np.ndarray | None = None,
    ) -> LandmarkPrediction:
        """Return a canonical 68 prediction in original-frame pixel coordinates."""
        canonical = prediction.canonical_68()
        if canonical.coordinate_space == "frame":
            return canonical
        if canonical.coordinate_space != "normalized_crop":
            raise ValueError(
                f"Unsupported coordinate space '{canonical.coordinate_space}' "
                f"from adapter '{self.config.name}'"
            )
        if matrix is None:
            raise ValueError(
                f"Adapter '{self.config.name}' returned normalized crop landmarks "
                "but no crop-to-frame matrix was supplied"
            )
        metadata = dict(canonical.metadata)
        metadata["source_coordinate_space"] = canonical.coordinate_space
        return LandmarkPrediction(
            landmarks=normalized_crop_to_frame(canonical.points, matrix),
            schema=CANONICAL_SCHEMA,
            confidence=canonical.confidence,
            model_name=canonical.model_name,
            source_landmark_count=canonical.source_landmark_count,
            coordinate_space="frame",
            metadata=metadata,
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

    def predict(self, image: np.ndarray, *, face: object | None = None) -> LandmarkPrediction:
        """Return the configured static landmarks without touching model state."""
        del image, face
        return self.normalize_prediction(self._points.copy())

    def predict_batch(
        self,
        images: np.ndarray,
        *,
        matrices: np.ndarray | None = None,
        faces: T.Sequence[object] | None = None,
    ) -> list[LandmarkPrediction]:
        """Return the same configured landmarks for every batch item."""
        del faces
        predictions = []
        for idx in range(images.shape[0]):
            matrix = None if matrices is None else matrices[idx]
            prediction = self.normalize_prediction(self._points.copy())
            predictions.append(self.to_frame_prediction(prediction, matrix=matrix))
        return predictions


class FaceswapAlignerAdapter(LandmarkAdapter):
    """Adapter for existing Faceswap aligner plugin instances.

    The wrapped aligner is run on already-prepared ensemble crops. Its normalized
    crop-space output is converted to canonical 68 frame pixels using the supplied
    crop matrices before fusion.
    """

    def __init__(
        self,
        config: LandmarkAdapterConfig,
        plugin: object,
        *,
        input_is_rgb: bool = True,
        input_scale: tuple[int, int] = (0, 1),
    ) -> None:
        super().__init__(config)
        self.plugin = plugin
        self._input_is_rgb = input_is_rgb
        self._input_scale = input_scale

    def load_model(self) -> object:
        """Load the wrapped plugin model if it exposes the normal Faceswap hook."""
        if not hasattr(self.plugin, "load_model"):
            return self.plugin
        model = self.plugin.load_model()
        self.plugin.model = model
        return model

    def _format_images(self, images: np.ndarray) -> np.ndarray:
        """Convert ensemble crops into the wrapped plugin's image convention."""
        batch = images
        should_swap_channels = (
            self._input_is_rgb and not getattr(self.plugin, "is_rgb", True)
        ) or (not self._input_is_rgb and getattr(self.plugin, "is_rgb", False))
        if should_swap_channels:
            batch = batch[..., 2::-1]
        dtype = np.dtype(getattr(self.plugin, "dtype", batch.dtype))
        if batch.dtype != dtype:
            batch = batch.astype(dtype, copy=False)
        target_scale = getattr(self.plugin, "scale", self._input_scale)
        if self._input_scale != target_scale:
            in_low, in_high = self._input_scale
            out_low, out_high = target_scale
            batch = (batch.astype("float32") - in_low) / (in_high - in_low)
            batch = batch * (out_high - out_low) + out_low
            if dtype != np.dtype("float32"):
                batch = batch.astype(dtype, copy=False)
        return np.ascontiguousarray(batch)

    def predict(self, image: np.ndarray, *, face: object | None = None) -> LandmarkPrediction:
        """Predict one prepared crop with the wrapped Faceswap plugin."""
        del face
        return self.predict_batch(image[None])[0]

    def predict_batch(
        self,
        images: np.ndarray,
        *,
        matrices: np.ndarray | None = None,
        faces: T.Sequence[object] | None = None,
    ) -> list[LandmarkPrediction]:
        """Run the wrapped plugin and return canonical frame-space predictions."""
        del faces
        if images.ndim != 4:
            raise ValueError(f"images must have shape (N, H, W, C), got {images.shape}")
        raw = self.plugin.process(self._format_images(images))
        points = self.plugin.post_process(raw)
        points = np.asarray(points, dtype="float32")
        if points.ndim != 3 or points.shape[0] != images.shape[0]:
            raise ValueError(
                f"wrapped aligner '{self.config.name}' returned unexpected shape {points.shape}"
            )
        predictions = []
        for idx, landmark_points in enumerate(points):
            matrix = None if matrices is None else matrices[idx]
            prediction = self.normalize_prediction(
                landmark_points,
                metadata={"wrapped_plugin": type(self.plugin).__name__},
            )
            predictions.append(self.to_frame_prediction(prediction, matrix=matrix))
        return predictions
