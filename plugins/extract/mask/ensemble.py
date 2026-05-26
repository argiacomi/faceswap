#!/usr/bin/env python3
"""Confidence-weighted ensemble mask plugin."""

from __future__ import annotations

import logging
import typing as T

import cv2
import numpy as np

from lib.utils import FaceswapError, get_module_objects
from plugins.extract.base import FacePlugin
from plugins.plugin_loader import PluginLoader

from . import ensemble_defaults as cfg
from ._output import MaskPluginOutput

if T.TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)

_EPSILON = 1e-6


class Ensemble(FacePlugin):
    """Combine multiple semantic maskers into a confidence-weighted mask."""

    def __init__(self) -> None:
        super().__init__(
            input_size=512,
            batch_size=cfg.batch_size(),
            is_rgb=True,
            dtype="float32",
            scale=(0, 1),
            force_cpu=False,
            centering=T.cast(T.Literal["face", "head"], cfg.centering()),
        )
        self._source_names = _validate_source_names(cfg.source_models())
        self._strategy = cfg.strategy()
        self._sources: list[FacePlugin] = []
        self.storage_name: str = f"{self.storage_name}_{self.centering}"

    def load_model(self) -> list[FacePlugin]:
        """Load every configured source mask model."""
        sources: list[FacePlugin] = []
        for name in self._source_names:
            plugin = T.cast(FacePlugin, PluginLoader.get_extractor("mask", name))
            if not getattr(plugin, "supports_per_class_probs", False):
                raise FaceswapError(
                    f"Ensemble mask source '{name}' does not expose per-class probabilities. "
                    "Choose semantic parser sources such as bisenet-fp or segnext-fp."
                )
            plugin.model = plugin.load_model()
            sources.append(plugin)
        self._sources = sources
        logger.debug("Loaded ensemble sources: %s", [source.storage_name for source in sources])
        return sources

    def pre_process(self, batch: np.ndarray) -> np.ndarray:
        """Preserve the common RGB 0-1 aligned batch for each source model."""
        return batch

    def process(self, batch: np.ndarray) -> MaskPluginOutput:
        """Run source maskers and combine their selected semantic probabilities."""
        if not self._sources:
            loaded = self.load_model()
            self.model = loaded

        outputs = [self._run_source(source, batch) for source in self._sources]
        probability_maps = [
            self._selected_probability_map(source, output) for source, output in outputs
        ]
        confidence_maps = [
            self._selected_confidence_map(source, output) for source, output in outputs
        ]
        weights = _normalised_parser_weights(confidence_maps)

        if self._strategy == "confidence-weighted-union":
            combined = _confidence_weighted_union(probability_maps, weights)
        elif self._strategy == "confidence-weighted-intersection":
            combined = _confidence_weighted_intersection(probability_maps, weights)
        else:  # ConfigItem choices should prevent this, but keep a clear runtime guard.
            raise FaceswapError(f"Unsupported ensemble mask strategy: {self._strategy}")

        metadata = {
            "strategy": self._strategy,
            "sources": [source.storage_name for source, _ in outputs],
            "source_models": self._source_names,
            "weights": weights.tolist(),
        }
        return MaskPluginOutput(
            combined.astype("float32", copy=False),
            source_id=self.storage_name,
            per_class_probs=_binary_probs(combined),
            metadata=metadata,
        )

    def _run_source(
        self, source: FacePlugin, batch: np.ndarray
    ) -> tuple[FacePlugin, MaskPluginOutput]:
        """Run one source plugin on the common aligned input batch."""
        source_input = _resize_batch(batch, source.input_size)
        feed = source.pre_process(source_input)
        prediction = source.process(feed)
        output = source.post_process(prediction)
        if not isinstance(output, MaskPluginOutput):
            output = MaskPluginOutput(
                np.asarray(output, dtype=np.float32),
                source_id=source.storage_name,
                per_class_probs=None,
            )
        if output.per_class_probs is None:
            raise FaceswapError(
                f"Ensemble mask source '{source.storage_name}' does not expose per-class "
                "probabilities. Choose semantic parser sources such as bisenet-fp or segnext-fp."
            )
        if output.shape[1:3] != (self.input_size, self.input_size):
            output = _resize_output(output, self.input_size)
        return source, output

    @staticmethod
    def _selected_indices(source: FacePlugin, num_classes: int) -> tuple[int, ...]:
        """Return valid selected semantic classes for ``source``."""
        return tuple(
            sorted(
                {
                    int(index)
                    for index in getattr(source, "_segment_indices", ())
                    if 0 <= int(index) < num_classes
                }
            )
        )

    @classmethod
    def _selected_probability_map(
        cls, source: FacePlugin, output: MaskPluginOutput
    ) -> npt.NDArray[np.float32]:
        """Return per-pixel probability that source-selected classes are present."""
        probs = output.per_class_probs
        assert probs is not None
        indices = cls._selected_indices(source, probs.shape[-1])
        if not indices:
            return np.asarray(output, dtype=np.float32)
        selected = np.take(probs, indices, axis=-1)
        return np.clip(selected.sum(axis=-1), 0.0, 1.0).astype("float32")

    @classmethod
    def _selected_confidence_map(
        cls, source: FacePlugin, output: MaskPluginOutput
    ) -> npt.NDArray[np.float32]:
        """Return the best selected-class confidence per pixel."""
        probs = output.per_class_probs
        assert probs is not None
        indices = cls._selected_indices(source, probs.shape[-1])
        if not indices:
            return np.asarray(output, dtype=np.float32)
        return np.take(probs, indices, axis=-1).max(axis=-1).astype("float32")


def _validate_source_names(source_names: list[str]) -> list[str]:
    """Validate the configured source list."""
    if len(source_names) < 2:
        raise FaceswapError("Ensemble mask requires at least two source models.")
    normalised = [name.lower().replace("_", "-") for name in source_names]
    if "ensemble" in normalised:
        raise FaceswapError("Ensemble mask cannot include itself as a source model.")
    available = set(PluginLoader.get_available_extractors("mask"))
    invalid = [name for name in normalised if name not in available]
    if invalid:
        raise FaceswapError(
            f"Invalid ensemble mask source model(s): {invalid}. Select from {sorted(available)}"
        )
    return normalised


def _resize_batch(batch: np.ndarray, size: int) -> np.ndarray:
    """Resize a channels-last batch to ``size`` if needed."""
    if batch.shape[1:3] == (size, size):
        return batch
    resized = [cv2.resize(item, (size, size), interpolation=cv2.INTER_LINEAR) for item in batch]
    return T.cast(np.ndarray, np.asarray(resized, dtype=batch.dtype))


def _resize_output(output: MaskPluginOutput, size: int) -> MaskPluginOutput:
    """Resize a source output back into ensemble aligned space."""
    binary = _resize_batch(np.asarray(output, dtype=np.float32), size)
    probs = output.per_class_probs
    resized_probs = None
    if probs is not None:
        resized_probs = np.asarray(
            [cv2.resize(item, (size, size), interpolation=cv2.INTER_LINEAR) for item in probs],
            dtype=np.float32,
        )
    return MaskPluginOutput(
        binary.astype("float32", copy=False),
        source_id=output.source_id,
        per_class_probs=resized_probs,
        metadata=output.metadata,
    )


def _normalised_parser_weights(
    confidence_maps: list[npt.NDArray[np.float32]],
) -> npt.NDArray[np.float32]:
    """Return per-batch normalized source weights from selected-class confidence."""
    raw = np.stack(
        [np.maximum(conf.mean(axis=(1, 2)), 1e-3) for conf in confidence_maps],
        axis=0,
    ).astype("float32")
    total = np.maximum(raw.sum(axis=0, keepdims=True), _EPSILON)
    return raw / total


def _confidence_weighted_union(
    probability_maps: list[npt.NDArray[np.float32]],
    weights: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Blend agreement and model-unique regions using parser confidence weights."""
    stacked = np.stack(probability_maps, axis=0).astype("float32")
    agreement = stacked.min(axis=0)
    unique = np.clip(stacked - _max_other(stacked), 0.0, 1.0)
    weighted_unique = unique * (0.5 + 0.5 * weights[:, :, None, None])
    return np.clip(agreement + weighted_unique.sum(axis=0), 0.0, 1.0).astype("float32")


def _confidence_weighted_intersection(
    probability_maps: list[npt.NDArray[np.float32]],
    weights: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Keep mutually supported regions scaled by parser-confidence agreement."""
    stacked = np.stack(probability_maps, axis=0).astype("float32")
    agreement_scale = 1.0 - (weights.max(axis=0) - weights.min(axis=0))
    return np.clip(stacked.min(axis=0) * agreement_scale[:, None, None], 0.0, 1.0).astype(
        "float32"
    )


def _max_other(stacked: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Return max probability from all other sources for each source."""
    retval = np.empty_like(stacked)
    for index in range(stacked.shape[0]):
        others = np.delete(stacked, index, axis=0)
        retval[index] = others.max(axis=0)
    return retval


def _binary_probs(mask: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Expose the ensemble binary mask as background/foreground probabilities."""
    foreground = np.clip(mask, 0.0, 1.0).astype("float32")
    background = 1.0 - foreground
    return np.stack([background, foreground], axis=-1).astype("float32")


__all__ = get_module_objects(__name__)
