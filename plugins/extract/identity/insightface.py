#!/usr/bin/env python3
"""InsightFace identity-recognition plugin."""

from __future__ import annotations

import logging
import typing as T

import numpy as np

from lib.utils import FaceswapError, get_module_objects
from plugins.extract.base import FacePlugin

from . import insightface_defaults as cfg
from ._model_adapter import (
    LoadedIdentityModel,
    insightface_adapter,
    metadata,
    normalize_embeddings,
)

logger = logging.getLogger(__name__)


class InsightFace(FacePlugin):
    """InsightFace identity-recognition plugin with configurable model type.

    Embeddings are stored under the stable ``insightface`` identity key as
    contiguous ``float32`` vectors with shape ``(512,)`` per face. The chosen
    upstream model type is serialized in face metadata.
    """

    _SUPPORTED = ("antelopev2", "buffalo_l", "buffalo_sc")

    def __init__(self) -> None:
        self._model_type = T.cast(str, cfg.model_type())
        if self._model_type not in self._SUPPORTED:
            raise FaceswapError(
                f"Unsupported InsightFace model_type '{self._model_type}'. "
                f"Select from {list(self._SUPPORTED)}."
            )
        adapter = insightface_adapter(self._model_type)
        super().__init__(
            input_size=adapter.input_size,
            batch_size=cfg.batch_size(),
            is_rgb=False,
            dtype="float32",
            scale=(0, 255),
            force_cpu=cfg.cpu(),
            centering="face",
        )
        self._adapter = adapter
        self.model: LoadedIdentityModel
        logger.debug(
            "Initialized %s with model_type '%s'", self.__class__.__name__, self._model_type
        )

    @property
    def model_type(self) -> str:
        """The configured InsightFace model type."""
        return self._model_type

    @property
    def identity_metadata(self) -> dict[str, T.Any]:
        """Per-face provenance metadata to serialize with the embedding."""
        return metadata(self._adapter, storage_name=self.storage_name, model_type=self._model_type)

    def load_model(self) -> LoadedIdentityModel:
        """Load the configured InsightFace recognition backend."""
        return self._adapter.load()

    def process(self, batch: np.ndarray) -> np.ndarray:
        """Return InsightFace embeddings for aligned BGR face crops."""
        return self.model.embed(batch.astype(np.uint8))

    def post_process(self, batch: np.ndarray) -> np.ndarray:
        """Ensure stored InsightFace embeddings are stable float32 unit vectors."""
        return normalize_embeddings(batch)


__all__ = get_module_objects(__name__)
