#!/usr/bin/env python3
"""SegNeXt Face-Parsing mask plugin.

Architecture from https://github.com/e4s2022/SegNeXt-FaceParser (Apache 2.0).
Implemented as a peer to the existing BiSeNet-FP plugin so users can pick either
parser when generating per-component face masks.

Faceswap pins three validated 19-class CelebAMask-HQ checkpoints from
``e4s2022/SegNeXt-FaceParser``:

* ``small``: MSCAN-S
* ``base``: MSCAN-B
* ``large``: MSCAN-L
"""

from __future__ import annotations

import logging
import os
import typing as T
from pathlib import Path

import numpy as np
import torch

from lib.google_drive import download_google_drive_file
from lib.utils import PROJECT_ROOT, FaceswapError, GetModelFromUrl, get_module_objects
from plugins.extract.base import FacePlugin

from . import segnext_fp_defaults as cfg
from ._output import MaskPluginOutput, softmax_last_axis
from ._segnext_fp.model import (
    BASE_CONFIG,
    LARGE_CONFIG,
    SMALL_CONFIG,
    MSCANConfig,
    SegNeXtFaceParser,
    filter_state_dict,
)

logger = logging.getLogger(__name__)
# pylint:disable=duplicate-code

_E4S_DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/12jOIkj3lZhn4sJ5rN8Y4TUAxiEX0hr79?usp=share_link"
)
_E4S_DRIVE_URL = (
    "https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
)

_E4S_SMALL_FILE_ID = "1FJDN1edNpUyx8Bv5Eo7JutAvbL9zFfn4"
_E4S_BASE_FILE_ID = "1YL4VuCBhhl-sjI3oPZOJhxTf9rIJRxD5"
_E4S_LARGE_FILE_ID = "1cYbwLSJDX0nnooR9JPQkA5qYHH3b9d3B"

_EXPECTED_NUM_CLASSES = 19
_CLASSIFIER_WEIGHT_KEY = "decode_head.conv_seg.weight"
_CLASSIFIER_BIAS_KEY = "decode_head.conv_seg.bias"


class _SegNeXtCheckpoint(T.NamedTuple):
    """One published SegNeXt-FaceParser checkpoint."""

    filename: str
    url: str
    sha256: str
    config: MSCANConfig
    num_classes: int = _EXPECTED_NUM_CLASSES
    google_drive_file_id: str | None = None


# e4s2022 publishes CelebAMask-HQ SegNeXt checkpoints for MSCAN-S/B/L.
# See: https://github.com/e4s2022/SegNeXt-FaceParser
#
# NOTE:
# The public README links a Google Drive folder, but Faceswap's downloader expects
# individual file ids. Fill these ids and hashes from the validated files in that folder.
_CHECKPOINTS = {
    "small": _SegNeXtCheckpoint(
        filename="segnext.small.512x512.celebamaskhq.160k.pth",
        url=_E4S_DRIVE_URL.format(file_id=_E4S_SMALL_FILE_ID),
        sha256="0f87738b6b6f5dca82cc63298d3d625f81915a9c1ed3d0a359b8866b2b76b321",
        config=SMALL_CONFIG,
        google_drive_file_id=_E4S_SMALL_FILE_ID,
    ),
    "base": _SegNeXtCheckpoint(
        filename="segnext.base.512x512.celebamaskhq.160k.pth",
        url=_E4S_DRIVE_URL.format(file_id=_E4S_BASE_FILE_ID),
        sha256="996e6339a28f5ef1a4c1b5ce324c0c8203c71ee6abb259b0a255d387af504cee",
        config=BASE_CONFIG,
        google_drive_file_id=_E4S_BASE_FILE_ID,
    ),
    "large": _SegNeXtCheckpoint(
        filename="segnext.large.512x512.celebamaskhq.160k.pth",
        url=_E4S_DRIVE_URL.format(file_id=_E4S_LARGE_FILE_ID),
        sha256="b15d77541e101bf6b117e3b00139f9f14d86017ac7564e3bb81d1a3f5703ead1",
        config=LARGE_CONFIG,
        google_drive_file_id=_E4S_LARGE_FILE_ID,
    ),
}

# CelebAMask-HQ class indices (from
# https://github.com/e4s2022/SegNeXt-FaceParser/blob/main/mmseg/datasets/celebamaskhq.py).
# Kept here as the source of truth for the SegNeXt label space; ``_get_segment_indices``
# below maps the user-facing component flags onto this layout.
#   0: background  1: skin   2: nose    3: eye_g   4: l_eye   5: r_eye   6: l_brow
#   7: r_brow      8: l_ear  9: r_ear  10: mouth  11: u_lip  12: l_lip  13: hair
#  14: hat        15: ear_r 16: neck_l 17: neck   18: cloth
_CLASS_FACE_INTERIOR = (1, 2, 4, 5, 6, 7, 11, 12)  # skin, nose, eyes, brows, lips
_CLASS_MOUTH = (10,)
_CLASS_GLASSES = (3,)
_CLASS_EARS = (8, 9, 15)  # left/right ears + earrings
_CLASS_HAIR = (13,)


class SegNeXtFP(FacePlugin):
    """SegNeXt face-parser mask plugin - drop-in alternative to BiSeNet-FP."""

    supports_per_class_probs = True
    """Whether this plugin exposes per-class probabilities in its post-process output."""

    def __init__(self) -> None:
        super().__init__(
            input_size=512,
            batch_size=cfg.batch_size(),
            is_rgb=True,
            dtype="float32",
            scale=(0, 1),
            force_cpu=cfg.cpu(),
            centering="head" if cfg.include_hair() else "face",
        )
        self.model: SegNeXtFaceParser
        self._checkpoint = _CHECKPOINTS[cfg.model()]
        self._segment_indices = self._get_segment_indices()
        # Match BiSeNet-FP's per-centering storage namespacing so downstream consumers can
        # treat the two plugins symmetrically.
        self.storage_name: str = f"{self.storage_name}_{self.centering}"

        # Upstream SegNeXt face parser uses ImageNet normalization in 0-1 RGB space.
        self._mean = np.array((0.485, 0.456, 0.406), dtype="float32")
        self._std = np.array((0.229, 0.224, 0.225), dtype="float32")

    def _get_segment_indices(self) -> list[int]:
        """Resolve the user-configured face components into CelebAMask-HQ class indices.

        Returns
        -------
        Sorted, de-duplicated list of class indices that make up the binary face mask.
        """
        retval: set[int] = set(_CLASS_FACE_INTERIOR)
        if cfg.include_mouth():
            retval.update(_CLASS_MOUTH)
        if cfg.include_glasses():
            retval.update(_CLASS_GLASSES)
        if cfg.include_ears():
            retval.update(_CLASS_EARS)
        if cfg.include_hair():
            retval.update(_CLASS_HAIR)
        result = sorted(retval)
        logger.debug("Selected segment indices: %s", result)
        return result

    def load_model(self) -> SegNeXtFaceParser:
        """Initialize the SegNeXt Face Parsing model and load the pinned checkpoint."""
        upstream_path = _resolve_checkpoint_path(self._checkpoint)
        # The upstream mmseg checkpoint bundles optimizer state and a ``meta`` dict
        # containing numpy scalars, which torch.load rejects under ``weights_only=True``
        # (the secure default used by the Faceswap base loader). On first use we
        # sanitize the checkpoint into a tensor-only state-dict file alongside the
        # original - pinned by checkpoint SHA256 so a re-download invalidates it.
        weights = _sanitize_checkpoint(
            upstream_path,
            self._checkpoint.sha256,
            expected_num_classes=self._checkpoint.num_classes,
        )
        model = SegNeXtFaceParser(
            self._checkpoint.config, num_classes=self._checkpoint.num_classes
        )
        return T.cast(
            SegNeXtFaceParser,
            self.load_torch_model(model, weights, return_indices=None),
        )

    def pre_process(self, batch: np.ndarray) -> np.ndarray:
        """Normalize and channel-swap the input batch for the model."""
        return T.cast(np.ndarray, ((batch - self._mean) / self._std).transpose(0, 3, 1, 2))

    def process(self, batch: np.ndarray) -> np.ndarray:
        """Run the model and return per-class scores.

        Returns
        -------
        ``(N, H, W, 19)`` raw class logits in CelebAMask-HQ class order. Downstream
        consumers that need probabilities can apply a softmax along the last axis.
        Symmetric with the BiSeNet-FP process() contract.
        """
        return T.cast(np.ndarray, self.from_torch(batch).transpose(0, 2, 3, 1))

    def post_process(self, batch: np.ndarray) -> MaskPluginOutput:
        """Reduce per-class logits to a binary face mask.

        Parameters
        ----------
        batch
            ``(N, H, W, 19)`` raw class logits from :meth:`process`.

        Returns
        -------
        :class:`~plugins.extract.mask._output.MaskPluginOutput` whose array data
        is the ``(N, H, W)`` float32 binary mask and whose
        :attr:`~MaskPluginOutput.per_class_probs` carries the softmaxed class
        probability distribution.
        """
        pred = batch.argmax(-1).astype("uint8")
        binary = np.isin(pred, self._segment_indices).astype("float32")
        return MaskPluginOutput(
            binary,
            source_id=self.storage_name,
            per_class_probs=softmax_last_axis(batch),
        )


def _resolve_checkpoint_path(checkpoint: _SegNeXtCheckpoint) -> str:
    """Return the local cached path for ``checkpoint``."""
    if checkpoint.google_drive_file_id is not None:
        cache_path = Path(PROJECT_ROOT) / ".fs_cache" / checkpoint.filename
        return str(
            download_google_drive_file(
                checkpoint.google_drive_file_id,
                cache_path,
                expected_sha256=checkpoint.sha256,
            )
        )

    return T.cast(
        str,
        GetModelFromUrl(checkpoint.filename, checkpoint.url, checkpoint.sha256).model_path,
    )


def _sanitize_checkpoint(upstream_path: str, sha256: str, expected_num_classes: int) -> str:
    """Return a path to a tensor-only state-dict copy of an mmseg checkpoint.

    Parameters
    ----------
    upstream_path
        Path to the SHA-validated upstream ``.pth`` file as downloaded by
        :class:`GetModelFromUrl`.
    sha256
        Expected SHA256 of the upstream file. Embedded in the cleaned filename so a
        re-download (which only happens if the cached upstream is corrupted) forces a
        re-sanitize of the cleaned copy as well.
    expected_num_classes
        The required CelebAMask-HQ class count for the classifier head.

    Returns
    -------
    Path to a torch-saved file containing only the ``state_dict`` tensors, safe to load
    with ``torch.load(..., weights_only=True)``.

    Notes
    -----
    Concurrent faceswap processes that miss the cache simultaneously will both run the
    sanitize path and ``os.replace`` over each other. ``os.replace`` is atomic on POSIX
    and on macOS (the only platforms faceswap targets), so the worst case is the
    redundant work of one process; the on-disk file is never half-written.
    """
    base, ext = os.path.splitext(upstream_path)
    cleaned_path = f"{base}.fs-clean.{sha256[:12]}{ext}"
    if os.path.exists(cleaned_path):
        try:
            _validate_clean_checkpoint(cleaned_path, expected_num_classes)
            return cleaned_path
        except Exception:  # pylint:disable=broad-except
            os.remove(cleaned_path)

    state_dict = _load_state_dict(upstream_path, weights_only=False)
    cleaned = filter_state_dict(state_dict)
    _validate_checkpoint_classes(cleaned, expected_num_classes)
    tmp_path = f"{cleaned_path}.part"
    torch.save(cleaned, tmp_path)
    os.replace(tmp_path, cleaned_path)
    try:
        _validate_clean_checkpoint(cleaned_path, expected_num_classes)
    except Exception as err:  # pylint:disable=broad-except
        # Should never happen - the file we just wrote refused to load on the secure
        # path or failed the class-count check. Surface loudly rather than silently
        # consume garbage on next run.
        if os.path.exists(cleaned_path):
            os.remove(cleaned_path)
        if isinstance(err, FaceswapError):
            raise
        raise RuntimeError(
            f"Sanitized checkpoint failed weights_only round-trip: {cleaned_path}"
        ) from err
    return cleaned_path


def _load_state_dict(path: str, *, weights_only: bool) -> T.Mapping[str, torch.Tensor]:
    """Load and unwrap a checkpoint into a plain state dict."""
    raw = torch.load(path, map_location="cpu", weights_only=weights_only)
    state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    if not isinstance(state_dict, dict):
        raise FaceswapError(f"SegNeXt-FP checkpoint is not a valid state dict: {path}")
    return T.cast(T.Mapping[str, torch.Tensor], state_dict)


def _validate_clean_checkpoint(path: str, expected_num_classes: int) -> None:
    """Validate that a sanitized checkpoint loads securely and matches 19-class parsing."""
    state_dict = _load_state_dict(path, weights_only=True)
    _validate_checkpoint_classes(state_dict, expected_num_classes)


def _validate_checkpoint_classes(
    state_dict: T.Mapping[str, torch.Tensor], expected_num_classes: int
) -> None:
    """Reject checkpoints whose classifier head is not the expected CelebAMask-HQ layout."""
    if _CLASSIFIER_WEIGHT_KEY not in state_dict or _CLASSIFIER_BIAS_KEY not in state_dict:
        raise FaceswapError(
            "SegNeXt-FP checkpoint is missing decode_head.conv_seg weights. "
            "The selected model/checkpoint pair is incompatible."
        )

    weight = state_dict[_CLASSIFIER_WEIGHT_KEY]
    bias = state_dict[_CLASSIFIER_BIAS_KEY]
    weight_classes = int(weight.shape[0])
    bias_classes = int(bias.shape[0])
    if weight_classes != bias_classes:
        raise FaceswapError(
            "SegNeXt-FP checkpoint class-count mismatch: expected "
            f"{expected_num_classes} CelebAMask-HQ classes, got inconsistent "
            f"classifier shapes (weight={weight_classes}, bias={bias_classes}). "
            "The selected model/checkpoint pair is incompatible."
        )
    if weight_classes != expected_num_classes:
        raise FaceswapError(
            "SegNeXt-FP checkpoint class-count mismatch: expected "
            f"{expected_num_classes} CelebAMask-HQ classes, got "
            f"{weight_classes}. The selected model/checkpoint pair is "
            "incompatible."
        )


__all__ = get_module_objects(__name__)
