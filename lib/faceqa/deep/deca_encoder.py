#!/usr/bin/env python3
"""DECA encoder: aligned face crop -> FLAME pose / expression / lighting.

This wraps the encoder half of DECA (Detailed Expression Capture and
Animation, Feng et al. 2021). For FaceQA's deep audit we only need the
*coefficients* DECA regresses - FLAME pose, expression, and spherical-harmonic
lighting - so the FLAME mesh decoder and the PyTorch3D rasterizer (only needed
for rendering) are deliberately not part of this integration.

The encoder is a ResNet-50 backbone followed by DECA's two-layer projection
head that outputs a single ``DECA_PARAM_DIM``-wide parameter vector. The
vector is sliced into the canonical FLAME parameter groups (see
``DECA_PARAM_LAYOUT``).

Weights
-------
The pretrained encoder weights (``deca_model.tar``) and the FLAME model are
research-licensed by MPI and are NOT redistributed with faceswap. They are
loaded at runtime from the local cache - see :mod:`lib.faceqa.deep.weights`.

Validation note
---------------
The real ``deca_model.tar`` path refuses missing/unexpected state-dict keys so
unvalidated partial loads cannot feed readiness or pruning. The pure-numpy
:func:`decode_parameters` slicing and the :class:`DecaEncoder` protocol are
fully unit-tested with a synthetic encoder; the torch backbone below is the
real-weights path that runs on the user's hardware.
"""

from __future__ import annotations

import logging
import typing as T
from dataclasses import dataclass

import numpy as np

from lib.utils import FaceswapError, get_module_objects

logger = logging.getLogger(__name__)

# Canonical DECA / FLAME parameter layout (decoded in this order from the flat
# encoder output). Counts match upstream DECA's default config
# (n_shape=100, n_tex=50, n_exp=50, n_pose=6, n_cam=3, n_light=27).
DECA_PARAM_LAYOUT: tuple[tuple[str, int], ...] = (
    ("shape", 100),
    ("tex", 50),
    ("exp", 50),
    ("pose", 6),
    ("cam", 3),
    ("light", 27),
)
DECA_PARAM_DIM: int = sum(count for _, count in DECA_PARAM_LAYOUT)  # 236

# DECA consumes 224x224 RGB crops scaled to [0, 1].
DECA_INPUT_SIZE: int = 224

# SH lighting is 9 bands x 3 channels.
_LIGHT_BANDS = 9
_LIGHT_CHANNELS = 3


@dataclass(frozen=True)
class DecaCoefficients:
    """Decoded DECA coefficient groups for a batch of faces.

    Each array is ``(n_faces, group_dim)``. Only the groups the FaceQA deep
    audit consumes are surfaced; ``shape`` / ``tex`` / ``cam`` are decoded but
    not retained here (identity/texture/camera are out of scope for coverage).
    """

    expression: np.ndarray  # (n, 50)
    pose: np.ndarray  # (n, 6) axis-angle: global rot (0:3) + jaw (3:6)
    light: np.ndarray  # (n, 27) SH coefficients (9 bands x 3 channels)

    def __len__(self) -> int:
        return int(self.expression.shape[0])


def _slice_offsets() -> dict[str, tuple[int, int]]:
    """Return ``{group: (start, stop)}`` offsets for the flat param vector."""
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, count in DECA_PARAM_LAYOUT:
        offsets[name] = (cursor, cursor + count)
        cursor += count
    return offsets


_OFFSETS = _slice_offsets()


def decode_parameters(parameters: np.ndarray) -> DecaCoefficients:
    """Slice a flat DECA parameter matrix into FLAME coefficient groups.

    Parameters
    ----------
    parameters
        ``(n_faces, DECA_PARAM_DIM)`` (or a single ``(DECA_PARAM_DIM,)``
        vector) of raw encoder outputs.

    Returns
    -------
    DecaCoefficients
        The expression / pose / lighting groups.

    Raises
    ------
    ValueError
        If the trailing dimension is not :data:`DECA_PARAM_DIM`.
    """
    matrix = np.asarray(parameters, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != DECA_PARAM_DIM:
        raise ValueError(
            f"Expected DECA parameters of shape (n, {DECA_PARAM_DIM}); got {matrix.shape}."
        )
    exp_start, exp_stop = _OFFSETS["exp"]
    pose_start, pose_stop = _OFFSETS["pose"]
    light_start, light_stop = _OFFSETS["light"]
    return DecaCoefficients(
        expression=matrix[:, exp_start:exp_stop].copy(),
        pose=matrix[:, pose_start:pose_stop].copy(),
        light=matrix[:, light_start:light_stop].copy(),
    )


@T.runtime_checkable
class DecaEncoder(T.Protocol):
    """Structural interface for anything that produces DECA parameters.

    The deep audit depends only on this protocol so a lightweight synthetic
    encoder can drive the full pipeline in tests without torch or the
    research-licensed weights.
    """

    def encode(self, crops: np.ndarray) -> np.ndarray:
        """Return ``(n_faces, DECA_PARAM_DIM)`` parameters for ``crops``.

        ``crops`` is ``(n_faces, H, W, 3)`` RGB uint8.
        """
        ...


class TorchDecaEncoder:
    """Real DECA encoder backed by torch (ResNet-50 + DECA projection head).

    Construct via :meth:`from_state_dict`. The forward path matches upstream
    DECA: 224x224 RGB crops scaled to ``[0, 1]`` through a ResNet-50 backbone
    and a ``2048 -> 1024 -> DECA_PARAM_DIM`` head.
    """

    def __init__(
        self,
        module: T.Any,
        device: str = "cpu",
        *,
        missing_keys_count: int = 0,
        unexpected_keys_count: int = 0,
        matched_key_ratio: float | None = None,
        device_auto_selected: bool = False,
    ) -> None:
        self._module = module
        self._device = device
        self.device = device
        self.device_auto_selected = device_auto_selected
        self.missing_keys_count = missing_keys_count
        self.unexpected_keys_count = unexpected_keys_count
        self.matched_key_ratio = matched_key_ratio

    @classmethod
    def build_module(cls) -> T.Any:
        """Return an untrained DECA encoder ``nn.Module`` (no ImageNet load)."""
        from torch import nn
        from torchvision.models import resnet50

        backbone = resnet50(weights=None)
        # Drop the ImageNet classifier; keep features up to the 2048-d pool.
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        class _DecaEncoderModule(nn.Module):
            """ResNet-50 features + DECA's two-layer parameter head."""

            def __init__(self) -> None:
                super().__init__()
                self.backbone = backbone
                self.head = nn.Sequential(
                    nn.Linear(feature_dim, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, DECA_PARAM_DIM),
                )

            def forward(self, images: T.Any) -> T.Any:  # noqa: D401
                features = self.backbone(images)
                return self.head(features)

        module = _DecaEncoderModule()
        module.eval()
        return module

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, T.Any],
        *,
        device: str = "cpu",
        strict: bool = False,
        allow_partial: bool = False,
        device_auto_selected: bool = False,
    ) -> TorchDecaEncoder:
        """Build an encoder and load ``state_dict`` (already key-remapped).

        The default real-weights path rejects partial loads. ``strict=False``
        is still used to collect missing/unexpected-key diagnostics before
        failing with a FaceSwap error that can be surfaced to the CLI.
        """
        import torch  # noqa: F401

        module = cls.build_module()
        result = module.load_state_dict(state_dict, strict=strict)
        missing = list(getattr(result, "missing_keys", []) or [])
        unexpected = list(getattr(result, "unexpected_keys", []) or [])
        expected_keys = set(module.state_dict())
        matched_key_ratio = (
            0.0
            if not expected_keys
            else (len(expected_keys) - len(missing)) / float(len(expected_keys))
        )
        logger.info(
            "DECA encoder weights loaded: %d missing, %d unexpected keys, %.3f matched.",
            len(missing),
            len(unexpected),
            matched_key_ratio,
        )
        if missing:
            logger.warning("DECA encoder missing keys (first 5): %s", missing[:5])
        if unexpected:
            logger.warning("DECA encoder unexpected keys (first 5): %s", unexpected[:5])
        if (missing or unexpected) and not allow_partial:
            raise FaceswapError(
                "DECA encoder weights did not fully match the expected FaceQA module "
                f"({len(missing)} missing keys, {len(unexpected)} unexpected keys, "
                f"matched_key_ratio={matched_key_ratio:.3f}). "
                "Refusing to use unvalidated DECA outputs for readiness or pruning."
            )
        module.to(device)
        module.eval()
        return cls(
            module,
            device=device,
            missing_keys_count=len(missing),
            unexpected_keys_count=len(unexpected),
            matched_key_ratio=matched_key_ratio,
            device_auto_selected=device_auto_selected,
        )

    @staticmethod
    def _preprocess(crops: np.ndarray) -> T.Any:
        """Return a ``(n, 3, 224, 224)`` float tensor scaled to ``[0, 1]``.

        Accepts ``(n, H, W, 3)`` RGB uint8; resizes with the same area / linear
        policy faceswap uses elsewhere and scales to ``[0, 1]`` (DECA's
        normalization).
        """
        import cv2
        import torch

        batch = np.asarray(crops)
        if batch.ndim != 4 or batch.shape[-1] != 3:
            raise ValueError(f"Expected crops of shape (n, H, W, 3); got {batch.shape}.")
        resized = np.empty((batch.shape[0], DECA_INPUT_SIZE, DECA_INPUT_SIZE, 3), dtype=np.float32)
        for index, crop in enumerate(batch):
            if crop.shape[0] != DECA_INPUT_SIZE or crop.shape[1] != DECA_INPUT_SIZE:
                interp = cv2.INTER_AREA if crop.shape[0] > DECA_INPUT_SIZE else cv2.INTER_LINEAR
                crop = cv2.resize(crop, (DECA_INPUT_SIZE, DECA_INPUT_SIZE), interpolation=interp)
            resized[index] = crop.astype(np.float32) / 255.0
        # NHWC -> NCHW
        tensor = torch.from_numpy(np.ascontiguousarray(resized.transpose(0, 3, 1, 2)))
        return tensor

    def encode(self, crops: np.ndarray) -> np.ndarray:
        """Return ``(n_faces, DECA_PARAM_DIM)`` parameters for ``crops``."""
        import torch

        if len(crops) == 0:
            return np.empty((0, DECA_PARAM_DIM), dtype=np.float32)
        tensor = self._preprocess(crops).to(self._device)
        with torch.no_grad():
            output = self._module(tensor)
        return T.cast(np.ndarray, output.detach().cpu().numpy().astype(np.float32))


__all__ = get_module_objects(__name__)
