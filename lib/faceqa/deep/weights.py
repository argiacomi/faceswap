#!/usr/bin/env python3
"""Locate, download, and load the DECA encoder weights.

The DECA encoder weights (``deca_model.tar``) and the FLAME model are
published by the DECA project. FaceQA only consumes the encoder coefficients,
so the FLAME mesh decoder is not loaded here. The default path downloads the
official ``deca_model.tar`` into the standard faceswap model cache via
``GetModelFromUrl``.
"""

from __future__ import annotations

import logging
import os
import typing as T

from lib.faceqa.deep.deca_encoder import TorchDecaEncoder
from lib.utils import PROJECT_ROOT, FaceswapError, GetModelFromUrl, get_module_objects

logger = logging.getLogger(__name__)

#: Expected file name of the upstream DECA checkpoint.
DECA_WEIGHTS_FILENAME = "deca_model.tar"
#: Official DECA Google Drive file id for ``deca_model.tar``.
DECA_WEIGHTS_FILE_ID = "1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje"
#: Direct download URL used by :class:`lib.utils.GetModelFromUrl`.
DECA_WEIGHTS_URL = (
    "https://drive.usercontent.google.com/download?"
    f"id={DECA_WEIGHTS_FILE_ID}&export=download&confirm=t"
)
#: SHA256 of the official ``deca_model.tar`` checkpoint.
DECA_WEIGHTS_SHA256 = "e714ed293054cba5eea9c96bd3b6b57880074cd84b3fd00d606cbaf0bee7c5c2"

#: Key under which upstream DECA stores the FLAME-parameter encoder.
_DECA_ENCODER_KEY = "E_flame"


def default_weights_path() -> str:
    """Return the default cache path for ``deca_model.tar``."""
    return os.path.join(PROJECT_ROOT, ".fs_cache", DECA_WEIGHTS_FILENAME)


def resolve_weights_path() -> str:
    """Return the cached DECA weights path, downloading when needed.

    Uses the standard faceswap model cache and SHA256-validates the official
    Google Drive checkpoint, matching other external FaceSwap models.
    """
    model_path = GetModelFromUrl(
        DECA_WEIGHTS_FILENAME,
        DECA_WEIGHTS_URL,
        DECA_WEIGHTS_SHA256,
    ).model_path
    return T.cast(str, model_path)


def remap_deca_state_dict(raw: dict[str, T.Any]) -> dict[str, T.Any]:
    """Map an upstream DECA encoder checkpoint onto the faceswap module keys.

    Upstream DECA's ``E_flame`` encoder names its ResNet backbone under
    ``encoder.*`` and its projection head under ``layers.*``; the faceswap
    module uses ``backbone.*`` and ``head.*`` respectively. Keys that match
    neither prefix are passed through unchanged so a differing upstream layout
    still surfaces as ``unexpected_keys`` during the (non-strict) load rather
    than being silently dropped.
    """
    remapped: dict[str, T.Any] = {}
    for key, value in raw.items():
        if key.startswith("encoder."):
            remapped[f"backbone.{key[len('encoder.') :]}"] = value
        elif key.startswith("layers."):
            remapped[f"head.{key[len('layers.') :]}"] = value
        else:
            remapped[key] = value
    return remapped


def _extract_encoder_state(checkpoint: T.Any) -> dict[str, T.Any]:
    """Return the encoder sub-state-dict from a loaded DECA checkpoint.

    Handles both the upstream layout (a dict containing an ``E_flame`` entry)
    and a bare encoder state dict.
    """
    if isinstance(checkpoint, dict) and _DECA_ENCODER_KEY in checkpoint:
        encoder_state = checkpoint[_DECA_ENCODER_KEY]
    else:
        encoder_state = checkpoint
    if not isinstance(encoder_state, dict):
        raise FaceswapError(
            "DECA checkpoint did not contain a recognizable encoder state dict "
            f"(expected a dict, optionally under '{_DECA_ENCODER_KEY}')."
        )
    return encoder_state


def load_deca_encoder(*, device: str = "cpu") -> TorchDecaEncoder:
    """Load the DECA encoder from cached research weights.

    Parameters
    ----------
    device
        Torch device string for inference (``"cpu"`` keeps the default
        FaceQA workflow CPU-safe).

    Raises
    ------
    FaceswapError
        If the weights file cannot be downloaded, fails SHA256 validation, or
        cannot be parsed into an encoder state dict.
    """
    path = resolve_weights_path()

    import torch

    logger.info("Loading DECA encoder weights from '%s'.", path)
    # The official checkpoint is a pickled dict of state dicts, so the
    # full-pickle path is required. The cached file is SHA256-validated by the
    # model downloader before this load.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    encoder_state = _extract_encoder_state(checkpoint)
    remapped = remap_deca_state_dict(encoder_state)
    return TorchDecaEncoder.from_state_dict(remapped, device=device, strict=False)


__all__ = get_module_objects(__name__)
