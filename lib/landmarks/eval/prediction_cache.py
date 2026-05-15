#!/usr/bin/env python3
"""Disk prediction cache for landmark evaluation."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import typing as T
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from lib.landmarks.schema import LandmarkPrediction, normalize_landmarks

_SAMPLE_DIR_PREFIX = "sample-"
_WINDOWS_UNSAFE_CHARS = set('<>:"/\\|?*')
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{idx}" for idx in range(1, 10)),
        *(f"LPT{idx}" for idx in range(1, 10)),
    }
)


@dataclass(frozen=True)
class PredictionCacheMetadata:
    """Metadata stored beside cached model predictions."""

    model_name: str
    checkpoint: str
    schema: str
    coordinate_space: str
    config_hash: str
    source_landmark_count: int
    prediction_hash: str


def _is_safe_sample_dir_name(sample_id: str) -> bool:
    """Return whether a sample ID can be used directly as one path segment."""
    if not sample_id or sample_id in {".", ".."}:
        return False
    if sample_id.startswith(_SAMPLE_DIR_PREFIX):
        return False
    if sample_id[-1] in {" ", "."}:
        return False
    if any(ord(char) < 32 or char in _WINDOWS_UNSAFE_CHARS for char in sample_id):
        return False
    stem = sample_id.split(".", 1)[0].upper()
    return stem not in _WINDOWS_RESERVED_NAMES


def _encode_sample_id(sample_id: str) -> str:
    """Return a filesystem-safe, reversible cache directory name."""
    if _is_safe_sample_dir_name(sample_id):
        return sample_id
    token = base64.urlsafe_b64encode(sample_id.encode("utf-8")).decode("ascii")
    return _SAMPLE_DIR_PREFIX + token.rstrip("=")


def _decode_sample_dir_name(name: str) -> str:
    """Return the sample id for an encoded cache directory name."""
    if not name.startswith(_SAMPLE_DIR_PREFIX):
        return name
    token = name[len(_SAMPLE_DIR_PREFIX) :]
    token += "=" * (-len(token) % 4)
    try:
        return base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return name


def config_hash(config: T.Mapping[str, T.Any] | str) -> str:
    """Return a stable hash for model/cache configuration."""
    payload = (
        config
        if isinstance(config, str)
        else json.dumps(config, sort_keys=True, separators=(",", ":"))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prediction_hash(points: np.ndarray) -> str:
    """Return a stable hash for a prediction array."""
    array = np.ascontiguousarray(points.astype("float32", copy=False))
    return hashlib.sha256(array.tobytes()).hexdigest()


class DiskPredictionCache:
    """Store predictions in filesystem-safe sample directories."""

    def __init__(self, root: str | Path = "outputs/landmark_predictions") -> None:
        self.root = Path(root)

    def sample_dir(self, sample_id: str) -> Path:
        """Return the directory for a sample."""
        return self.root / _encode_sample_id(sample_id)

    def _legacy_sample_dir(self, sample_id: str) -> Path:
        """Return the pre-encoding directory for a sample."""
        return self.root / sample_id

    def _existing_sample_dir(self, sample_id: str) -> Path:
        """Return an existing sample directory, preferring encoded cache paths."""
        sample_dir = self.sample_dir(sample_id)
        if sample_dir.exists():
            return sample_dir
        legacy_sample_dir = self._legacy_sample_dir(sample_id)
        if legacy_sample_dir.exists():
            return legacy_sample_dir
        return sample_dir

    def prediction_path(self, sample_id: str, model_name: str) -> Path:
        """Return the prediction array path."""
        return self._existing_sample_dir(sample_id) / f"{model_name}.npy"

    def metadata_path(self, sample_id: str) -> Path:
        """Return the metadata path for a sample."""
        return self._existing_sample_dir(sample_id) / "metadata.json"

    def load_metadata(self, sample_id: str) -> dict[str, dict[str, T.Any]]:
        """Load metadata for a sample."""
        path = self.metadata_path(sample_id)
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def metadata_entry(
        self,
        prediction: LandmarkPrediction,
        *,
        checkpoint: str = "",
        config: T.Mapping[str, T.Any] | str = "",
    ) -> PredictionCacheMetadata:
        """Build the metadata entry for a prediction."""
        return PredictionCacheMetadata(
            model_name=prediction.model_name,
            checkpoint=checkpoint,
            schema=prediction.schema,
            coordinate_space=prediction.coordinate_space,
            config_hash=config_hash(config),
            source_landmark_count=prediction.source_landmark_count,
            prediction_hash=prediction_hash(prediction.landmarks),
        )

    def is_fresh(
        self,
        sample_id: str,
        prediction: LandmarkPrediction,
        *,
        checkpoint: str = "",
        config: T.Mapping[str, T.Any] | str = "",
    ) -> bool:
        """Return whether a cached prediction already matches metadata."""
        model_name = prediction.model_name
        path = self.prediction_path(sample_id, model_name)
        if not path.is_file():
            return False
        metadata = self.load_metadata(sample_id)
        expected = asdict(self.metadata_entry(prediction, checkpoint=checkpoint, config=config))
        return metadata.get(model_name) == expected

    def write(
        self,
        sample_id: str,
        prediction: LandmarkPrediction,
        *,
        checkpoint: str = "",
        config: T.Mapping[str, T.Any] | str = "",
        refresh: bool = False,
    ) -> Path:
        """Write one prediction and update sample metadata."""
        sample_dir = self._existing_sample_dir(sample_id)
        sample_dir.mkdir(parents=True, exist_ok=True)
        model_name = prediction.model_name
        path = self.prediction_path(sample_id, model_name)
        if not refresh and self.is_fresh(
            sample_id, prediction, checkpoint=checkpoint, config=config
        ):
            return path
        np.save(str(path), prediction.landmarks.astype("float32", copy=False))
        metadata = self.load_metadata(sample_id)
        entry = self.metadata_entry(prediction, checkpoint=checkpoint, config=config)
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
            sorted(
                {
                    _decode_sample_dir_name(path.name)
                    for path in self.root.iterdir()
                    if (path / "metadata.json").is_file()
                }
            )
        )
