#!/usr/bin/env python3
"""Disk prediction cache for landmark evaluation."""

from __future__ import annotations

import hashlib
import json
import typing as T
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from lib.landmarks.schema import LandmarkPrediction, normalize_landmarks


@dataclass(frozen=True)
class PredictionCacheMetadata:
    """Metadata stored beside cached model predictions."""

    model_name: str
    checkpoint: str
    schema: str
    coordinate_space: str
    config_hash: str
    source_landmark_count: int


def config_hash(config: T.Mapping[str, T.Any] | str) -> str:
    """Return a stable hash for model/cache configuration."""
    payload = (
        config
        if isinstance(config, str)
        else json.dumps(config, sort_keys=True, separators=(",", ":"))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DiskPredictionCache:
    """Store predictions as ``sample_id/model.npy`` with ``metadata.json``."""

    def __init__(self, root: str | Path = "outputs/landmark_predictions") -> None:
        self.root = Path(root)

    def sample_dir(self, sample_id: str) -> Path:
        """Return the directory for a sample."""
        return self.root / sample_id

    def prediction_path(self, sample_id: str, model_name: str) -> Path:
        """Return the prediction array path."""
        return self.sample_dir(sample_id) / f"{model_name}.npy"

    def metadata_path(self, sample_id: str) -> Path:
        """Return the metadata path for a sample."""
        return self.sample_dir(sample_id) / "metadata.json"

    def load_metadata(self, sample_id: str) -> dict[str, dict[str, T.Any]]:
        """Load metadata for a sample."""
        path = self.metadata_path(sample_id)
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def write(
        self,
        sample_id: str,
        prediction: LandmarkPrediction,
        *,
        checkpoint: str = "",
        config: T.Mapping[str, T.Any] | str = "",
    ) -> Path:
        """Write one prediction and update sample metadata."""
        sample_dir = self.sample_dir(sample_id)
        sample_dir.mkdir(parents=True, exist_ok=True)
        model_name = prediction.model_name
        path = self.prediction_path(sample_id, model_name)
        np.save(str(path), prediction.landmarks.astype("float32", copy=False))
        metadata = self.load_metadata(sample_id)
        entry = PredictionCacheMetadata(
            model_name=model_name,
            checkpoint=checkpoint,
            schema=prediction.schema,
            coordinate_space=prediction.coordinate_space,
            config_hash=config_hash(config),
            source_landmark_count=prediction.source_landmark_count,
        )
        metadata[model_name] = asdict(entry)
        self.metadata_path(sample_id).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path

    def read(
        self,
        sample_id: str,
        model_name: str,
        *,
        expected_config_hash: str | None = None,
    ) -> LandmarkPrediction:
        """Read a prediction, optionally rejecting stale model configs."""
        metadata = self.load_metadata(sample_id)
        entry = metadata.get(model_name)
        if entry is None:
            raise FileNotFoundError(f"missing metadata for {sample_id}/{model_name}")
        if expected_config_hash is not None and entry.get("config_hash") != expected_config_hash:
            raise ValueError(f"cached prediction for {sample_id}/{model_name} is stale")
        path = self.prediction_path(sample_id, model_name)
        if not path.is_file():
            raise FileNotFoundError(path)
        landmarks = np.load(str(path)).astype("float32")
        return LandmarkPrediction(
            landmarks=normalize_landmarks(landmarks, source_schema=entry.get("schema")),
            schema="2d_68",
            model_name=model_name,
            source_landmark_count=int(entry.get("source_landmark_count", landmarks.shape[0])),
            coordinate_space=str(entry.get("coordinate_space", "frame")),
            metadata={"checkpoint": entry.get("checkpoint", ""), "sample_id": sample_id},
        )

    def available_models(self, sample_id: str) -> tuple[str, ...]:
        """Return cached model names for a sample."""
        return tuple(sorted(self.load_metadata(sample_id)))

    def sample_ids(self) -> tuple[str, ...]:
        """Return all sample ids with metadata."""
        if not self.root.is_dir():
            return ()
        return tuple(
            sorted(path.name for path in self.root.iterdir() if (path / "metadata.json").is_file())
        )
