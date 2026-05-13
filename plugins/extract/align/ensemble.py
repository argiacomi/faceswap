#!/usr/bin/env python3
"""Landmark ensemble aligner plugin.

The ensemble runs model adapters on a shared aligner crop, converts every
prediction into canonical 68-point original-frame pixels, fuses in that common
space, then maps the fused result back to normalized crop coordinates for
Faceswap's normal aligner post-processing path.
"""

from __future__ import annotations

import importlib
import logging
import typing as T

import numpy as np

from lib.landmarks.adapters import (
    FaceswapAlignerAdapter,
    LandmarkAdapter,
    LandmarkAdapterConfig,
)
from lib.landmarks.coordinates import (
    frame_to_normalized_crop,
    roi_to_matrix,
)
from lib.landmarks.fusion import plain_average, weighted_average
from lib.landmarks.schema import LandmarkPrediction
from lib.utils import get_module_objects
from plugins.extract.base import ExtractPlugin

from . import ensemble_defaults as cfg

logger = logging.getLogger(__name__)


_PLUGIN_CLASSES = {
    "hrnet": ("plugins.extract.align.hrnet", "HRNet", "2d_68"),
    "spiga": ("plugins.extract.align.spiga", "SPIGA", None),
    "orformer": ("plugins.extract.align.orformer", "ORFormer", None),
}


class Ensemble(ExtractPlugin):
    """Faceswap aligner that fuses predictions from landmark adapters."""

    def __init__(
        self,
        adapters: T.Sequence[LandmarkAdapter] | None = None,
        *,
        crop_scale: float | None = None,
        reject_outliers: bool | None = None,
        outlier_threshold: float | None = None,
        min_models: int | None = None,
        strategy: str | None = None,
    ) -> None:
        super().__init__(
            input_size=256,
            batch_size=cfg.batch_size(),
            is_rgb=True,
            dtype="float32",
            scale=(0, 1),
        )
        self.realign_centering = "legacy"
        self._injected_adapters = list(adapters) if adapters is not None else None
        self._crop_scale = cfg.crop_scale() if crop_scale is None else crop_scale
        self._reject_outliers = (
            cfg.reject_outliers() if reject_outliers is None else reject_outliers
        )
        self._outlier_threshold = (
            cfg.outlier_threshold() if outlier_threshold is None else outlier_threshold
        )
        self._min_models = cfg.min_models() if min_models is None else min_models
        self._strategy = cfg.strategy() if strategy is None else strategy
        self._last_matrices: np.ndarray | None = None
        self.last_debug_metadata: list[dict[str, T.Any]] = []
        self.model: list[LandmarkAdapter]

    def load_model(self) -> list[LandmarkAdapter]:
        """Load configured adapters.

        Injected adapters are returned as-is for tests. Real adapters are only
        imported when their plugin modules exist in the local tree.
        """
        adapters = (
            list(self._injected_adapters)
            if self._injected_adapters is not None
            else self._build_configured_adapters()
        )
        loaded = [adapter for adapter in adapters if adapter.config.enabled]
        for adapter in loaded:
            if hasattr(adapter, "load_model"):
                adapter.load_model()  # type: ignore[attr-defined]
        if not loaded:
            raise ValueError("No enabled landmark ensemble adapters are available")
        logger.info(
            "Loaded landmark ensemble adapters: %s",
            ", ".join(adapter.config.name for adapter in loaded),
        )
        return loaded

    def _build_configured_adapters(self) -> list[LandmarkAdapter]:
        """Create adapters for configured aligner plugins that are importable."""
        adapters: list[LandmarkAdapter] = []
        for name in cfg.models():
            if name not in _PLUGIN_CLASSES:
                logger.warning("[Ensemble] Unknown adapter '%s'; skipping", name)
                continue
            module_name, class_name, schema = _PLUGIN_CLASSES[name]
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                logger.info(
                    "[Ensemble] Optional adapter '%s' is not installed; skipping",
                    name,
                )
                continue
            plugin_cls = getattr(module, class_name)
            plugin = plugin_cls()
            adapter_schema = schema or self._schema_from_plugin(plugin)
            adapters.append(
                FaceswapAlignerAdapter(
                    LandmarkAdapterConfig(
                        name=name,
                        schema=adapter_schema,
                        coordinate_space="normalized_crop",
                    ),
                    plugin,
                    input_is_rgb=self.is_rgb,
                    input_scale=self.scale,
                )
            )
        return adapters

    @staticmethod
    def _schema_from_plugin(plugin: object) -> str:
        """Infer an adapter schema from known model configuration attributes."""
        model_config = getattr(plugin, "_model_config", None)
        count = getattr(model_config, "num_landmarks", 68)
        return f"2d_{count}"

    def pre_process(self, batch: np.ndarray) -> np.ndarray:
        """Format detection boxes into a shared square ensemble crop."""
        heights = batch[:, 3] - batch[:, 1]
        widths = batch[:, 2] - batch[:, 0]
        ctr_x = np.rint((batch[:, 0] + batch[:, 2]) * 0.5).astype("int32")
        ctr_y = np.rint((batch[:, 1] + batch[:, 3]) * 0.5).astype("int32")
        side = np.maximum(widths, heights) * self._crop_scale
        half = np.rint(side * 0.5).astype("int32")

        retval = np.empty((batch.shape[0], 4), dtype=np.int32)
        retval[:, 0] = ctr_x - half
        retval[:, 1] = ctr_y - half
        retval[:, 2] = ctr_x + half
        retval[:, 3] = ctr_y + half
        self._last_matrices = roi_to_matrix(retval)
        return retval

    def _active_adapters(self) -> list[LandmarkAdapter]:
        """Return loaded or injected adapters."""
        model = getattr(self, "model", None)
        if model is not None:
            return [adapter for adapter in model if adapter.config.enabled]
        if self._injected_adapters is not None:
            return [adapter for adapter in self._injected_adapters if adapter.config.enabled]
        raise ValueError("Ensemble adapters have not been loaded")

    def _matrices_for_batch(self, batch_size: int) -> np.ndarray:
        """Return crop-to-frame matrices, falling back to identity for warmup calls."""
        if self._last_matrices is not None and self._last_matrices.shape[0] == batch_size:
            return self._last_matrices
        matrices = np.repeat(np.eye(3, dtype="float32")[None], batch_size, axis=0)
        return matrices

    def _collect_predictions(
        self, batch: np.ndarray, matrices: np.ndarray
    ) -> tuple[list[list[tuple[LandmarkAdapter, LandmarkPrediction]]], list[str]]:
        """Run adapters and bucket successful predictions by face index."""
        per_face: list[list[tuple[LandmarkAdapter, LandmarkPrediction]]] = [
            [] for _ in range(batch.shape[0])
        ]
        errors: list[str] = []
        for adapter in self._active_adapters():
            try:
                predictions = adapter.predict_batch(batch, matrices=matrices)
            except Exception as err:  # pylint:disable=broad-except
                logger.warning("[Ensemble] Adapter '%s' failed: %s", adapter.config.name, err)
                errors.append(f"{adapter.config.name}: {err}")
                continue
            if len(predictions) != batch.shape[0]:
                message = (
                    f"{adapter.config.name}: expected {batch.shape[0]} predictions, "
                    f"got {len(predictions)}"
                )
                logger.warning("[Ensemble] %s", message)
                errors.append(message)
                continue
            for idx, prediction in enumerate(predictions):
                per_face[idx].append((adapter, prediction))
        return per_face, errors

    def _fuse_face(
        self,
        predictions: list[tuple[LandmarkAdapter, LandmarkPrediction]],
        errors: list[str],
    ) -> np.ndarray:
        """Fuse one face's adapter predictions and return frame-space points."""
        if len(predictions) < self._min_models:
            raise ValueError(
                "Not enough successful landmark adapters for ensemble face: "
                f"required {self._min_models}, got {len(predictions)}"
            )
        adapters = [adapter for adapter, _prediction in predictions]
        items = [prediction for _adapter, prediction in predictions]
        weights = np.array([adapter.config.weight for adapter in adapters], dtype="float32")
        if self._strategy == "plain_average":
            fused = plain_average(
                items,
                reject_outliers=self._reject_outliers,
                outlier_threshold=self._outlier_threshold,
            )
        else:
            fused = weighted_average(
                items,
                weights=weights,
                reject_outliers=self._reject_outliers,
                outlier_threshold=self._outlier_threshold,
            )

        self.last_debug_metadata.append(
            {
                "sources": fused.sources,
                "weights": fused.weights.tolist(),
                "kept_indices": fused.kept_indices,
                "rejected_indices": fused.rejected_indices,
                "adapter_errors": tuple(errors),
                "strategy": fused.strategy,
            }
        )
        return fused.points

    def predict_landmarks_68(
        self,
        image: np.ndarray,
        *,
        matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return fused canonical ``(68, 2)`` landmarks in original-frame pixels.

        The input image is a prepared ensemble crop. ``matrix`` maps normalized
        crop coordinates for that crop into the original frame. If omitted, an
        identity matrix is used, matching warmup/test calls that already operate
        in frame space.
        """
        matrices = (
            np.eye(3, dtype="float32")[None]
            if matrix is None
            else np.asarray(matrix, dtype="float32")[None]
        )
        per_face, errors = self._collect_predictions(image[None], matrices)
        self.last_debug_metadata = []
        return self._fuse_face(per_face[0], errors)

    def process(self, batch: np.ndarray) -> np.ndarray:
        """Run adapter predictions, fuse in frame space and return normalized landmarks."""
        matrices = self._matrices_for_batch(batch.shape[0])
        per_face, errors = self._collect_predictions(batch, matrices)
        self.last_debug_metadata = []
        output = np.empty((batch.shape[0], 68, 2), dtype="float32")
        for idx, predictions in enumerate(per_face):
            output[idx] = frame_to_normalized_crop(
                self._fuse_face(predictions, errors),
                matrices[idx],
            )
        return output


__all__ = get_module_objects(__name__)
