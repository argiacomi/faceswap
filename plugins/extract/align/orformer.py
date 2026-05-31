#!/usr/bin/env python3
"""ORFormer facial landmarks extractor for faceswap.py.

Code adapted and modified from:
https://github.com/ben0919/ORFormer

Upstream provenance:
- Repository commit pinned for vendored model code:
  7e77569783b677f00a71f0caa45d8663d6113167
- Official public weights are linked from the upstream README as Google Drive ``weights.zip``.
- The upstream repository does not contain an explicit LICENSE file. This private-fork plugin
  documents that licensing gap and validates the public archive/checkpoints by SHA256.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import typing as T
import zipfile
from dataclasses import dataclass

import numpy as np

from lib.utils import FaceswapError, GetModelFromUrl, get_module_objects
from plugins.extract.base import ExtractPlugin

from . import orformer_defaults as cfg
from ._orformer.model import ORFormerFaceswapModel, w300_edge_info, wflw_edge_info

_WEIGHTS_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1ebTTbJb9Hsp2bLiUsFgm7_PvtdtsPgPH&export=download&confirm=t"
)
_WEIGHTS_ARCHIVE = "orformer_weights_20240704.zip"
_WEIGHTS_ARCHIVE_SHA256 = "00aa35f078512124a199a126df25279fa946803a41764e02e84843842e517ef6"


@dataclass(frozen=True)
class ORFormerCheckpoint:
    """A checkpoint file inside the official ORFormer weights archive."""

    archive_member: str
    filename: str
    sha256: str


@dataclass(frozen=True)
class ORFormerModelConfig:
    """Configuration for paired ORFormer/HGNet checkpoints."""

    num_landmarks: int
    num_edges: int
    edge_info: T.Callable[[], list[tuple[bool, list[int]]]]
    hgnet: ORFormerCheckpoint
    orformer: ORFormerCheckpoint


_MODEL_CONFIG = {
    "300w": ORFormerModelConfig(
        num_landmarks=68,
        num_edges=13,
        edge_info=w300_edge_info,
        hgnet=ORFormerCheckpoint(
            archive_member="weights/HGNet/300W/best_model.pt",
            filename="orformer_hgnet_300w_20240704.pt",
            sha256="d661072e7444104f98b428e961180ea08a379c06eec255e975e3512b6929ffb1",
        ),
        orformer=ORFormerCheckpoint(
            archive_member="weights/ORFormer/300W/best_model.pt",
            filename="orformer_vqvae_300w_20240704.pt",
            sha256="147b16fca2e7636dd2be695c46e5862d1aaa066c4477d64394dbe0e283a4ee1c",
        ),
    ),
    "wflw": ORFormerModelConfig(
        num_landmarks=98,
        num_edges=15,
        edge_info=wflw_edge_info,
        hgnet=ORFormerCheckpoint(
            archive_member="weights/HGNet/WFLW/best_model.pt",
            filename="orformer_hgnet_wflw_20240704.pt",
            sha256="77b1646c825abbe516b7f3cb6484c404c16cc9910187358f5681b06f77dee585",
        ),
        orformer=ORFormerCheckpoint(
            archive_member="weights/ORFormer/WFLW/best_model.pt",
            filename="orformer_vqvae_wflw_20240704.pt",
            sha256="6229cea05411f0c1fe48e1cd1c1cdb1053a03b66a900d7baaebb0d2bffb59bfd",
        ),
    ),
}


class ORFormer(ExtractPlugin):
    """ORFormer face alignment plugin."""

    def __init__(self) -> None:
        super().__init__(
            input_size=256,
            batch_size=cfg.batch_size(),
            is_rgb=True,
            dtype="float32",
            scale=(0, 1),
        )
        self._target_dist = cfg.crop_scale()
        model_name = cfg.model()
        if model_name not in _MODEL_CONFIG:
            raise FaceswapError(
                f"Unsupported ORFormer model: {model_name!r}. Select from {list(_MODEL_CONFIG)}."
            )
        self._model_config = _MODEL_CONFIG[model_name]
        self.model: ORFormerFaceswapModel
        self.realign_centering = "legacy"
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def load_model(self) -> ORFormerFaceswapModel:
        """Load the paired ORFormer and HGNet models.

        Returns
        -------
        ORFormerFaceswapModel
            The loaded ORFormer/HGNet wrapper.
        """
        paths = _get_checkpoint_paths(self._model_config)
        model = T.cast(
            ORFormerFaceswapModel,
            self.load_torch_model(
                ORFormerFaceswapModel(
                    num_points=self._model_config.num_landmarks,
                    num_edges=self._model_config.num_edges,
                    edge_info=self._model_config.edge_info(),
                    orformer_weights_path=paths["orformer"],
                ),
                paths["hgnet"],
            ),
        )
        return model

    def pre_process(self, batch: np.ndarray) -> np.ndarray:
        """Format face detection boxes into ORFormer prediction crops."""
        heights = batch[:, 3] - batch[:, 1]
        widths = batch[:, 2] - batch[:, 0]
        ctr_x = np.rint((batch[:, 0] + batch[:, 2]) * 0.5).astype("int32")
        ctr_y = np.rint((batch[:, 1] + batch[:, 3]) * 0.5).astype("int32")
        side = np.maximum(widths, heights) * self._target_dist
        half = np.rint(side * 0.5).astype("int32")

        retval = np.empty((batch.shape[0], 4), dtype=np.int32)
        retval[:, 0] = ctr_x - half
        retval[:, 1] = ctr_y - half
        retval[:, 2] = ctr_x + half
        retval[:, 3] = ctr_y + half
        return retval  # type: ignore[no-any-return]

    def process(self, batch: np.ndarray) -> np.ndarray:
        """Predict face landmarks."""
        batch = batch.copy()
        batch -= self._mean
        batch /= self._std
        feed = np.ascontiguousarray(batch.transpose(0, 3, 1, 2))
        return self.from_torch(feed)

    def post_process(self, batch: np.ndarray) -> np.ndarray:
        """Return normalized 68/98 point landmarks in Faceswap's aligner contract."""
        return batch.astype("float32", copy=False)


def _get_checkpoint_paths(model_config: ORFormerModelConfig) -> dict[str, str]:
    """Download the official archive if required and extract validated checkpoint files."""
    archive_path = GetModelFromUrl(
        _WEIGHTS_ARCHIVE, _WEIGHTS_URL, _WEIGHTS_ARCHIVE_SHA256
    ).model_path
    cache_dir = os.path.dirname(archive_path)
    # Open the zip once for both checkpoint extractions; the archive is large enough
    # that two separate opens (header parse + central directory read) is wasted work.
    pending: list[tuple[str, ORFormerCheckpoint]] = []
    resolved: dict[str, str] = {}
    for key, checkpoint in (("hgnet", model_config.hgnet), ("orformer", model_config.orformer)):
        output_path = os.path.join(cache_dir, checkpoint.filename)
        if _checkpoint_is_valid(output_path, checkpoint.sha256):
            resolved[key] = output_path
        else:
            pending.append((key, checkpoint))

    if pending:
        with zipfile.ZipFile(archive_path) as archive:
            for key, checkpoint in pending:
                resolved[key] = _extract_checkpoint(archive, cache_dir, checkpoint)
    return resolved


def _checkpoint_is_valid(path: str, expected_sha256: str) -> bool:
    """Return True when a previously-extracted checkpoint matches the expected hash.

    Maintains a ``<path>.sha256.ok`` sentinel keyed on file size + mtime so repeated
    plugin loads in the same install don't re-hash multi-hundred-megabyte files.
    """
    if not os.path.exists(path):
        return False
    sentinel_path = f"{path}.sha256.ok"
    try:
        stat_result = os.stat(path)
    except OSError:
        return False
    sentinel_token = f"{expected_sha256}:{stat_result.st_size}:{int(stat_result.st_mtime_ns)}"
    if os.path.exists(sentinel_path):
        try:
            with open(sentinel_path, encoding="utf-8") as handle:
                if handle.read().strip() == sentinel_token:
                    return True
        except OSError:
            pass
    if _hash_file(path) != expected_sha256:
        return False
    try:
        with open(sentinel_path, "w", encoding="utf-8") as handle:
            handle.write(sentinel_token)
    except OSError:
        pass
    return True


def _extract_checkpoint(
    archive: zipfile.ZipFile, cache_dir: str, checkpoint: ORFormerCheckpoint
) -> str:
    """Extract and validate one checkpoint from the official weights archive."""
    output_path = os.path.join(cache_dir, checkpoint.filename)
    # Per-process tempfile to prevent two concurrent extractors from clobbering each
    # other's partial file. os.replace is atomic; whichever process finishes last wins
    # and leaves a complete, validated checkpoint behind.
    fd, partial_path = tempfile.mkstemp(
        prefix=f"{checkpoint.filename}.", suffix=".part", dir=cache_dir
    )
    os.close(fd)
    success = False
    try:
        with archive.open(checkpoint.archive_member) as source, open(partial_path, "wb") as output:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                output.write(chunk)
        actual = _hash_file(partial_path)
        if actual != checkpoint.sha256:
            raise RuntimeError(
                f"Extracted ORFormer checkpoint hash mismatch for {checkpoint.archive_member}. "
                f"Expected {checkpoint.sha256}, got {actual}."
            )
        os.replace(partial_path, output_path)
        success = True
    finally:
        if not success and os.path.exists(partial_path):
            os.remove(partial_path)
    # Prime the validation sentinel so the next plugin load short-circuits the rehash.
    _checkpoint_is_valid(output_path, checkpoint.sha256)
    return output_path


def _hash_file(filename: str) -> str:
    """Return the SHA256 hash for ``filename``."""
    sha = hashlib.sha256()
    with open(filename, "rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


__all__ = get_module_objects(__name__)
