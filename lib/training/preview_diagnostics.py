#!/usr/bin/env python3
"""Preview diagnostics metrics for training previews."""

from __future__ import annotations

import json
import logging
import os
import typing as T
from dataclasses import dataclass

import cv2
import numpy as np

from lib.logger import format_array, parse_class_init
from lib.training.data import get_label
from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Current and rolling values for a preview diagnostic metric."""

    current: float
    ema: float
    mean: float
    std: float
    count: int


@dataclass
class _RollingMetric:
    """Rolling statistics for a metric stream."""

    alpha: float
    count: int = 0
    ema: float = 0.0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: npt.NDArray[np.float32]) -> MetricSnapshot | None:
        """Update rolling statistics from a set of values."""
        values = values[np.isfinite(values)]
        if not values.size:
            return None

        current = float(values.mean(dtype=np.float64))
        if self.count == 0:
            self.ema = current
        else:
            self.ema = (self.alpha * current) + ((1.0 - self.alpha) * self.ema)

        for value in values.astype(np.float64, copy=False):
            self.count += 1
            delta = float(value) - self.mean
            self.mean += delta / self.count
            delta2 = float(value) - self.mean
            self.m2 += delta * delta2

        variance = self.m2 / (self.count - 1) if self.count > 1 else 0.0
        return MetricSnapshot(
            current=current,
            ema=self.ema,
            mean=self.mean,
            std=float(np.sqrt(variance)),
            count=self.count,
        )


class PreviewDiagnostics:
    """Compute and roll up training preview diagnostics metrics.

    Parameters
    ----------
    ema_alpha
        Exponential moving average alpha applied to each preview refresh.
    jsonl_path
        Optional JSONL artifact path to append structured preview diagnostics payloads to.
    """

    def __init__(self, ema_alpha: float = 0.2, jsonl_path: str | None = None) -> None:
        logger.debug(parse_class_init(locals()))
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(f"EMA alpha must be in the range (0.0, 1.0]. Got {ema_alpha}")
        self._ema_alpha = ema_alpha
        self._jsonl_path = jsonl_path
        self._metrics: dict[str, _RollingMetric] = {}

    @property
    def jsonl_path(self) -> str | None:
        """Optional JSONL artifact path."""
        return self._jsonl_path

    @classmethod
    def _center_crop(
        cls, image: npt.NDArray[np.float32], height: int, width: int
    ) -> npt.NDArray[np.float32]:
        """Crop preview arrays around their center to the requested dimensions."""
        src_height, src_width = image.shape[-3:-1]
        crop_height = min(src_height, height)
        crop_width = min(src_width, width)
        top = max((src_height - crop_height) // 2, 0)
        left = max((src_width - crop_width) // 2, 0)
        return image[..., top : top + crop_height, left : left + crop_width, :]

    @classmethod
    def _diagonal_predictions(
        cls, predictions: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Return source-to-self preview predictions in side order."""
        return np.stack(
            [predictions[idx, idx] for idx in range(predictions.shape[0])], axis=0
        ).astype(np.float32, copy=False)

    @classmethod
    def _weighted_mean(
        cls, values: npt.NDArray[np.float32], weights: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Return per-side/per-sample weighted means."""
        weight_sum = weights.sum(axis=(-2, -1))
        valid = weight_sum > 0.0
        weighted = (values * weights[..., None]).sum(axis=(-3, -2, -1))
        retval = np.full(weight_sum.shape, np.nan, dtype=np.float32)
        retval[valid] = weighted[valid] / (weight_sum[valid] * values.shape[-1])
        return retval

    @classmethod
    def _boundary_band(cls, masks: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Build a simple mask boundary band for each preview target."""
        bands = np.zeros_like(masks, dtype=np.float32)
        kernel_size = max(3, int(round(masks.shape[-1] * 0.03)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel: npt.NDArray[np.uint8] = np.ones((kernel_size, kernel_size), dtype=np.uint8)

        for idx in np.ndindex(masks.shape[:-2]):
            mask = (masks[idx] > 0.5).astype(np.uint8)
            if not mask.any() or mask.all():
                continue
            dilated = cv2.dilate(mask, kernel)
            eroded = cv2.erode(mask, kernel)
            bands[idx] = (dilated - eroded).astype(np.float32)
        return bands

    @classmethod
    def _detail_proxy(cls, image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Return a cheap per-side/per-sample image detail proxy."""
        vertical = np.abs(np.diff(image, axis=-3)).mean(axis=(-3, -2, -1))
        horizontal = np.abs(np.diff(image, axis=-2)).mean(axis=(-3, -2, -1))
        return ((vertical + horizontal) / 2.0).astype(np.float32, copy=False)

    @classmethod
    def _current_metrics(
        cls,
        predictions: npt.NDArray[np.float32],
        targets: npt.NDArray[np.float32],
    ) -> dict[str, npt.NDArray[np.float32]]:
        """Calculate current preview diagnostics metrics."""
        logger.debug(
            "[PreviewDiagnostics] predictions: %s, targets: %s",
            format_array(predictions),
            format_array(targets),
        )
        recon = cls._diagonal_predictions(predictions)
        target = cls._center_crop(targets, recon.shape[-3], recon.shape[-2])
        recon = cls._center_crop(recon, target.shape[-3], target.shape[-2])

        recon_rgb = recon[..., :3]
        target_rgb = target[..., :3]
        abs_err = np.abs(recon_rgb - target_rgb).astype(np.float32, copy=False)
        sq_err = np.square(recon_rgb - target_rgb).astype(np.float32, copy=False)

        metrics: dict[str, npt.NDArray[np.float32]] = {
            "reconstruction_mae": abs_err.mean(axis=(-3, -2, -1)),
            "reconstruction_mse": sq_err.mean(axis=(-3, -2, -1)),
            "detail_mae": (
                np.abs(np.diff(recon_rgb, axis=-3) - np.diff(target_rgb, axis=-3)).mean(
                    axis=(-3, -2, -1)
                )
                + np.abs(np.diff(recon_rgb, axis=-2) - np.diff(target_rgb, axis=-2)).mean(
                    axis=(-3, -2, -1)
                )
            ).astype(np.float32)
            / 2.0,
            "prediction_detail": cls._detail_proxy(recon_rgb),
            "target_detail": cls._detail_proxy(target_rgb),
        }

        side_mae = metrics["reconstruction_mae"].mean(axis=-1)
        if side_mae.size > 1:
            metrics["reconstruction_imbalance_mae"] = np.array(
                [side_mae.max() - side_mae.min()], dtype=np.float32
            )

        if target.shape[-1] > 3:
            masks = np.clip(target[..., 3], 0.0, 1.0).astype(np.float32, copy=False)
            metrics["masked_reconstruction_mae"] = cls._weighted_mean(abs_err, masks)

            boundary = cls._boundary_band(masks)
            if boundary.any():
                metrics["boundary_mae"] = cls._weighted_mean(abs_err, boundary)

        return metrics

    def _update_metric(self, name: str, values: npt.NDArray[np.float32]) -> MetricSnapshot | None:
        """Update or initialize a named rolling metric."""
        metric = self._metrics.setdefault(name, _RollingMetric(alpha=self._ema_alpha))
        return metric.update(values.reshape(-1).astype(np.float32, copy=False))

    def _update_metrics(
        self, metrics: dict[str, npt.NDArray[np.float32]]
    ) -> dict[str, MetricSnapshot]:
        """Update rolling metrics and return current snapshots."""
        snapshots: dict[str, MetricSnapshot] = {}
        for name, values in metrics.items():
            if values.ndim == 2:
                for side_idx, side_values in enumerate(values):
                    side_name = f"{name}_{get_label(side_idx, values.shape[0])}"
                    snapshot = self._update_metric(side_name, side_values)
                    if snapshot is not None:
                        snapshots[side_name] = snapshot
                snapshot = self._update_metric(name, values)
            else:
                snapshot = self._update_metric(name, values)
            if snapshot is not None:
                snapshots[name] = snapshot
        return snapshots

    @classmethod
    def _flatten(cls, snapshots: dict[str, MetricSnapshot]) -> dict[str, float]:
        """Flatten metric snapshots into TensorBoard scalar payloads."""
        retval: dict[str, float] = {}
        for name, snapshot in snapshots.items():
            retval[name] = snapshot.ema
            retval[f"{name}_current"] = snapshot.current
            retval[f"{name}_mean"] = snapshot.mean
            retval[f"{name}_std"] = snapshot.std
            retval[f"{name}_count"] = float(snapshot.count)
        return retval

    def _write_jsonl(self, iteration: int, snapshots: dict[str, MetricSnapshot]) -> None:
        """Append a structured diagnostics artifact line."""
        if self._jsonl_path is None:
            return
        os.makedirs(os.path.dirname(self._jsonl_path), exist_ok=True)
        payload = {
            "iteration": iteration,
            "metrics": {
                name: {
                    "current": snapshot.current,
                    "ema": snapshot.ema,
                    "mean": snapshot.mean,
                    "std": snapshot.std,
                    "count": snapshot.count,
                }
                for name, snapshot in snapshots.items()
            },
        }
        with open(self._jsonl_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(payload, sort_keys=True) + "\n")

    def update(
        self,
        predictions: npt.NDArray[np.float32],
        targets: npt.NDArray[np.float32],
        iteration: int,
    ) -> dict[str, float]:
        """Compute diagnostics for the current preview and update rolling metrics."""
        current = self._current_metrics(predictions, targets)
        snapshots = self._update_metrics(current)
        self._write_jsonl(iteration, snapshots)
        return self._flatten(snapshots)


__all__ = get_module_objects(__name__)
