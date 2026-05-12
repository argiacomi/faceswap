#!/usr/bin/env python3
"""Small in-memory prediction cache for landmark adapters."""

from __future__ import annotations

import hashlib
import typing as T
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from lib.landmarks.schema import LandmarkPrediction


@dataclass(frozen=True)
class CacheEntry:
    """Cached prediction payload."""

    key: str
    prediction: LandmarkPrediction


def make_prediction_key(
    image_id: str,
    adapter_name: str,
    *,
    face_id: str | int | None = None,
    version: str = "",
) -> str:
    """Create a stable, human-readable cache key."""
    parts = [str(image_id), str(adapter_name)]
    if face_id is not None:
        parts.append(str(face_id))
    if version:
        parts.append(str(version))
    return "::".join(parts)


def cache_key_for_array(
    image: np.ndarray,
    adapter_name: str,
    *,
    face_id: str | int | None = None,
    version: str = "",
) -> str:
    """Create a cache key from array metadata and bytes."""
    array = np.ascontiguousarray(image)
    digest = hashlib.blake2b(array.view(np.uint8), digest_size=16)
    image_id = f"{array.shape}:{array.dtype}:{digest.hexdigest()}"
    return make_prediction_key(
        image_id,
        adapter_name,
        face_id=face_id,
        version=version,
    )


class PredictionCache:
    """Simple LRU cache keyed by adapter/image identifiers."""

    def __init__(self, max_size: int = 256) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be greater than zero")
        self.max_size = max_size
        self._entries: OrderedDict[str, LandmarkPrediction] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> LandmarkPrediction | None:
        """Return a cached prediction and mark it as recently used."""
        prediction = self._entries.get(key)
        if prediction is None:
            return None
        self._entries.move_to_end(key)
        return prediction

    def put(self, key: str, prediction: LandmarkPrediction) -> None:
        """Insert or update a prediction, evicting the least-recently-used entry."""
        self._entries[key] = prediction
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_size:
            self._entries.popitem(last=False)

    def get_or_predict(
        self,
        key: str,
        predict: T.Callable[[], LandmarkPrediction],
    ) -> LandmarkPrediction:
        """Return a cached prediction or compute and store one."""
        cached = self.get(key)
        if cached is not None:
            return cached
        prediction = predict()
        self.put(key, prediction)
        return prediction

    def clear(self) -> None:
        """Clear all cache entries."""
        self._entries.clear()
