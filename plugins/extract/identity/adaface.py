#!/usr/bin/env python3
"""AdaFace identity-recognition plugin."""

from __future__ import annotations

import logging
import typing as T

import numpy as np

from lib.utils import get_module_objects
from plugins.extract.base import FacePlugin

from . import adaface_defaults as cfg
from ._model_adapter import LoadedIdentityModel, adaface_adapter, metadata, normalize_embeddings

logger = logging.getLogger(__name__)


class AdaFace(FacePlugin):
    """AdaFace iResNet101 identity-recognition plugin.

    Embeddings are stored under the stable ``adaface`` identity key as
    contiguous ``float32`` vectors with shape ``(512,)`` per face.
    """

    def __init__(self) -> None:
        self._force_cpu = bool(cfg.cpu())
        adapter = adaface_adapter(force_cpu=self._force_cpu)
        super().__init__(
            input_size=adapter.input_size,
            batch_size=cfg.batch_size(),
            is_rgb=False,
            dtype=np.uint8,
            scale=(0, 255),
            force_cpu=self._force_cpu,
            centering="face",
        )
        self._adapter = adapter
        self.model: LoadedIdentityModel
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def identity_metadata(self) -> dict[str, T.Any]:
        """Per-face provenance metadata to serialize with the embedding."""
        return metadata(self._adapter, storage_name=self.storage_name)

    def load_model(self) -> LoadedIdentityModel:
        """Load the AdaFace recognition backend."""
        return self._adapter.load()

    def process(self, batch: np.ndarray) -> np.ndarray:
        """Return AdaFace embeddings for aligned BGR face crops."""
        return self.model.embed(batch)

    def post_process(self, batch: np.ndarray) -> np.ndarray:
        """Ensure stored AdaFace embeddings are stable float32 unit vectors."""
        return normalize_embeddings(batch)


__all__ = get_module_objects(__name__)
