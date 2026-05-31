#!/usr/bin/env python3
"""Adapter interfaces for landmark predictors."""

from __future__ import annotations

import importlib
import logging
import typing as T
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from lib.landmarks.coordinates import CoordinateSpace, normalized_crop_to_frame
from lib.landmarks.core.schema import (
    CANONICAL_SCHEMA,
    LandmarkPrediction,
    canonicalize_schema,
)

logger = logging.getLogger(__name__)

SUPPORTED_MODEL_PLUGINS: dict[str, tuple[str, str, str | None]] = {
    "fan": ("plugins.extract.align.fan", "FAN", "2d_68"),
    "hrnet": ("plugins.extract.align.hrnet", "HRNet", "2d_68"),
    "spiga": ("plugins.extract.align.spiga", "SPIGA", None),
    "orformer": ("plugins.extract.align.orformer", "ORFormer", None),
}


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

    def predict_landmarks_68(
        self,
        image: np.ndarray,
        *,
        matrix: np.ndarray | None = None,
        face: object | None = None,
    ) -> np.ndarray:
        """Return canonical ``(68, 2)`` landmarks in original-frame pixel space.

        ``matrix`` is required when an adapter reports normalized crop
        coordinates. It must map normalized crop coordinates to frame pixels.
        """
        prediction = self.predict(image, face=face)
        return self.to_frame_prediction(prediction, matrix=matrix).points

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
        self.plugin.model = model  # type: ignore[attr-defined]
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
        # Wrapped plugins may normalize in place, so do not share the Ensemble crop batch.
        return np.ascontiguousarray(batch).copy()  # type: ignore[no-any-return]

    def predict(self, image: np.ndarray, *, face: object | None = None) -> LandmarkPrediction:
        """Predict one prepared crop with the wrapped Faceswap plugin."""
        del face
        return self.predict_batch(image[None])[0]

    def predict_landmarks_68(
        self,
        image: np.ndarray,
        *,
        matrix: np.ndarray | None = None,
        face: object | None = None,
    ) -> np.ndarray:
        """Return wrapped plugin landmarks as canonical frame-space ``(68, 2)``."""
        del face
        matrices = None if matrix is None else np.asarray(matrix, dtype="float32")[None]
        return self.predict_batch(image[None], matrices=matrices)[0].points

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
        raw = self.plugin.process(self._format_images(images))  # type: ignore[attr-defined]
        points = self.plugin.post_process(raw)  # type: ignore[attr-defined]
        points = np.asarray(points, dtype="float32")
        if points.ndim != 3 or points.shape[0] != images.shape[0]:
            raise ValueError(
                f"wrapped aligner '{self.config.name}' returned unexpected shape {points.shape}"
            )
        plugin_metadata = getattr(self.plugin, "last_debug_metadata", [])
        predictions = []
        for idx, landmark_points in enumerate(points):
            matrix = None if matrices is None else matrices[idx]
            metadata = {"wrapped_plugin": type(self.plugin).__name__}
            if isinstance(plugin_metadata, list) and idx < len(plugin_metadata):
                metadata.update(plugin_metadata[idx])
            prediction = self.normalize_prediction(
                landmark_points,
                metadata=metadata,
            )
            predictions.append(self.to_frame_prediction(prediction, matrix=matrix))
        return predictions


def schema_from_plugin(plugin: object) -> str:
    """Infer a prediction schema from known model configuration attributes."""
    model_config = getattr(plugin, "_model_config", None)
    count = getattr(model_config, "num_landmarks", 68)
    return f"2d_{count}"


def _set_plugin_device(plugin: object, device: str | None) -> None:
    """Override a Faceswap torch plugin device before model loading."""
    if not device or device == "auto":
        return
    torch_infer = getattr(plugin, "_torch", None)
    if torch_infer is None:
        return
    try:
        import torch
    except ImportError as err:  # pragma: no cover - torch is present in normal Faceswap envs
        raise RuntimeError("device selection for landmark adapters requires torch") from err
    torch_device = torch.device(device)
    torch_infer.device = torch_device
    torch_infer._use_pinned = torch.cuda.is_available() and torch_device.type == "cuda"


def build_landmark_adapter(
    model_name: str,
    *,
    device: str | None = None,
    input_is_rgb: bool = False,
    input_scale: tuple[int, int] = (0, 255),
) -> LandmarkAdapter:
    """Build a supported landmark adapter by model name.

    The returned adapter wraps the existing Faceswap aligner plugin and emits
    canonical 68-point frame-space predictions when supplied crop-to-frame
    matrices.
    """
    name = model_name.strip().lower()
    if name not in SUPPORTED_MODEL_PLUGINS:
        raise ValueError(
            f"Unsupported landmark model '{model_name}'. Supported models: {sorted(SUPPORTED_MODEL_PLUGINS)}"  # noqa: E501
        )
    module_name, class_name, schema = SUPPORTED_MODEL_PLUGINS[name]
    module = importlib.import_module(module_name)
    plugin_cls = getattr(module, class_name)
    plugin = plugin_cls()
    _set_plugin_device(plugin, device)
    adapter_schema = schema or schema_from_plugin(plugin)
    logger.info("Built landmark adapter '%s' with schema %s", name, adapter_schema)
    return FaceswapAlignerAdapter(
        LandmarkAdapterConfig(
            name=name,
            schema=adapter_schema,
            coordinate_space="normalized_crop",
        ),
        plugin,
        input_is_rgb=input_is_rgb,
        input_scale=input_scale,
    )
