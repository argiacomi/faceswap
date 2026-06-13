#! /usr/env/bin/python3
"""Advanced, optional training-time loss components for Faceswap.

Implements the remaining "Group 2" training-loss improvements that are not covered by the
generic perceptual / mask-aware loss plumbing in :mod:`lib.training.loss`:

- :class:`IdentityLoss`: a frozen face-recognition embedding loss.
- :class:`RegionWeightedPerceptualLoss`: an explicit region-weighted wrapper around the
  existing perceptual losses.
- :class:`BoundaryLoss`: a mask-edge band reconstruction loss for improving seam quality.
- :func:`occlusion_exclusion_weight`: helper to down-weight occluded regions in the
  reconstruction loss.

Every component is opt-in and produces no effect when disabled. The components operate on
``(N, C, H, W)`` ``float32`` tensors in the ``0.0 - 1.0`` range, matching the per-output
tensors handled by :class:`lib.training.loss.LossCollator`.
"""

from __future__ import annotations

import logging
import typing as T

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from lib.logger import parse_class_init
from lib.model.losses import get_loss_function
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)

_EPS = 1e-8


def _is_spatial(func: nn.Module, channels: int = 3, size: int = 32) -> bool:
    """Determine whether a loss function returns a spatial ``(N, C, H, W)`` output.

    Parameters
    ----------
    func
        The loss function to probe
    channels
        The number of channels to use for the dummy input
    size
        The spatial size to use for the dummy input

    Returns
    -------
    ``True`` if the loss returns a 4D spatial output, ``False`` if it returns a per-item
    ``(N, )`` scalar
    """
    dummy_a = torch.rand((1, channels, size, size), dtype=torch.float32)
    dummy_b = torch.rand((1, channels, size, size), dtype=torch.float32)
    with torch.no_grad():
        out = func(dummy_a, dummy_b)
    if out.ndim not in (1, 4):
        raise RuntimeError(
            "Loss functions should return either spatial output per item (N, C, H, W) "
            f"(4 dims) or scalar per item (N, ) (1 dim). Got {out.ndim} dims"
        )
    return bool(out.ndim == 4)


def _reduce_spatial(loss_map: torch.Tensor) -> torch.Tensor:
    """Reduce a spatial loss map to a per-item ``(N, )`` scalar.

    Parameters
    ----------
    loss_map
        The spatial loss map in ``(N, C, H, W)`` order

    Returns
    -------
    The per-item mean loss
    """
    return loss_map.mean(dim=tuple(range(1, loss_map.ndim)))


def _masked_region_mean(loss_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Obtain the per-item mean of a spatial loss map within a binary/soft mask.

    Items where the mask is empty return ``0`` rather than ``NaN``.

    Parameters
    ----------
    loss_map
        The spatial loss map in ``(N, C, H, W)`` order
    mask
        The mask to apply in ``(N, 1, H, W)`` order

    Returns
    -------
    The per-item masked mean loss
    """
    dims = tuple(range(1, loss_map.ndim))
    weighted = (loss_map * mask).sum(dim=dims)
    norm = mask.expand_as(loss_map).sum(dim=dims).clamp_min(_EPS)
    return weighted / norm


def compute_boundary_band(
    mask: torch.Tensor,
    band_pixels: int,
    mode: T.Literal["inner", "outer", "both"],
) -> torch.Tensor:
    """Compute a boundary band mask around the edge of a face mask.

    The band is derived from morphological dilation / erosion of the (thresholded) input
    mask using max-pooling operations.

    Parameters
    ----------
    mask
        The face mask in ``(N, 1, H, W)`` order in the ``0.0 - 1.0`` range
    band_pixels
        The half-width of the band in pixels
    mode
        Which side of the mask edge the band should cover

    Returns
    -------
    The boundary band mask in ``(N, 1, H, W)`` order in the ``0.0 - 1.0`` range
    """
    band_pixels = max(1, int(band_pixels))
    kernel = band_pixels * 2 + 1
    binary = (mask >= 0.5).to(mask.dtype)
    dilated = F.max_pool2d(binary, kernel_size=kernel, stride=1, padding=band_pixels)
    eroded = -F.max_pool2d(-binary, kernel_size=kernel, stride=1, padding=band_pixels)
    if mode == "inner":
        band = binary - eroded
    elif mode == "outer":
        band = dilated - binary
    else:
        band = dilated - eroded
    return band.clamp(0.0, 1.0)


def occlusion_exclusion_weight(mask_occlusion: torch.Tensor, strength: float) -> torch.Tensor:
    """Compute a per-pixel reconstruction weight that down-weights occluded regions.

    Parameters
    ----------
    mask_occlusion
        The occlusion mask in ``(N, 1, H, W)`` order in the ``0.0 - 1.0`` range, where high
        values indicate occlusion
    strength
        How strongly to exclude occluded regions. ``1.0`` fully ignores occluded pixels,
        ``0.0`` is a no-op

    Returns
    -------
    The per-pixel weight in ``(N, 1, H, W)`` order in the ``0.0 - 1.0`` range
    """
    strength = float(min(max(strength, 0.0), 1.0))
    return (1.0 - strength * mask_occlusion.clamp(0.0, 1.0)).clamp(0.0, 1.0)


class BoundaryLoss(nn.Module):
    """Mask-edge band reconstruction loss.

    Applies a standard reconstruction loss to a band around the edge of the face mask to
    improve the quality of the seam between the swapped face and the background.

    Parameters
    ----------
    loss_function
        The name of the loss function to use for the band
    band_pixels
        The half-width of the boundary band in pixels
    mode
        Which side of the mask edge the band should cover
    color_order
        The color order that the model is training in
    """

    def __init__(
        self,
        loss_function: str,
        band_pixels: int,
        mode: T.Literal["inner", "outer", "both"],
        color_order: T.Literal["bgr", "rgb"],
    ) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._band_pixels = band_pixels
        self._mode = mode
        self._function = get_loss_function(loss_function, color_order)
        self._spatial = _is_spatial(self._function)
        logger.debug("Initialized %s (spatial=%s)", self.__class__.__name__, self._spatial)

    def forward(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, mask_face: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the boundary loss for a batch.

        Parameters
        ----------
        y_true
            The ground truth images in ``(N, C, H, W)`` order
        y_pred
            The model predictions in ``(N, C, H, W)`` order
        mask_face
            The face mask in ``(N, 1, H, W)`` order

        Returns
        -------
        The per-item boundary loss in ``(N, )`` order
        """
        band = compute_boundary_band(mask_face, self._band_pixels, self._mode)
        if self._spatial:
            loss_map = self._function(y_true, y_pred)
            return _masked_region_mean(loss_map, band)
        return T.cast(torch.Tensor, self._function(y_true * band, y_pred * band))


class RegionWeightedPerceptualLoss(nn.Module):
    """Explicit region-weighted wrapper around a perceptual loss.

    Applies independent weight multipliers to the face, eye, mouth and skin regions of a
    perceptual loss, allowing perceptual detail to be prioritized intentionally rather than
    relying on the global eye / mouth multipliers.

    Parameters
    ----------
    loss_function
        The name of the perceptual loss function to wrap
    color_order
        The color order that the model is training in
    face_weight
        The weight multiplier for the face mask region
    eye_weight
        The weight multiplier for the eye region
    mouth_weight
        The weight multiplier for the mouth region
    skin_weight
        The weight multiplier for the skin region (face excluding eyes and mouth)
    """

    def __init__(
        self,
        loss_function: str,
        color_order: T.Literal["bgr", "rgb"],
        face_weight: float,
        eye_weight: float,
        mouth_weight: float,
        skin_weight: float,
    ) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._function = get_loss_function(loss_function, color_order)
        self._spatial = _is_spatial(self._function)
        self._face_weight = face_weight
        self._eye_weight = eye_weight
        self._mouth_weight = mouth_weight
        self._skin_weight = skin_weight
        logger.debug("Initialized %s (spatial=%s)", self.__class__.__name__, self._spatial)

    @classmethod
    def _skin_mask(
        cls,
        mask_face: torch.Tensor,
        mask_eye: torch.Tensor | None,
        mask_mouth: torch.Tensor | None,
    ) -> torch.Tensor:
        """Derive a skin mask from the face mask, excluding the eye and mouth regions.

        Parameters
        ----------
        mask_face
            The face mask in ``(N, 1, H, W)`` order
        mask_eye
            The eye mask in ``(N, 1, H, W)`` order if available
        mask_mouth
            The mouth mask in ``(N, 1, H, W)`` order if available

        Returns
        -------
        The skin mask in ``(N, 1, H, W)`` order
        """
        skin = mask_face.clone()
        if mask_eye is not None:
            skin = skin - mask_eye
        if mask_mouth is not None:
            skin = skin - mask_mouth
        return skin.clamp(0.0, 1.0)

    def _spatial_loss(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mask_face: torch.Tensor | None,
        mask_eye: torch.Tensor | None,
        mask_mouth: torch.Tensor | None,
    ) -> torch.Tensor:
        """Calculate region-weighted loss for a spatial perceptual loss.

        Parameters
        ----------
        y_true
            The ground truth images in ``(N, C, H, W)`` order
        y_pred
            The model predictions in ``(N, C, H, W)`` order
        mask_face
            The face mask in ``(N, 1, H, W)`` order if available
        mask_eye
            The eye mask in ``(N, 1, H, W)`` order if available
        mask_mouth
            The mouth mask in ``(N, 1, H, W)`` order if available

        Returns
        -------
        The per-item region-weighted loss in ``(N, )`` order
        """
        loss_map = self._function(y_true, y_pred)
        weight_map = torch.ones_like(loss_map)
        if mask_face is not None:
            weight_map = weight_map * (1.0 + (self._face_weight - 1.0) * mask_face)
            skin = self._skin_mask(mask_face, mask_eye, mask_mouth)
            weight_map = weight_map * (1.0 + (self._skin_weight - 1.0) * skin)
        if mask_eye is not None:
            weight_map = weight_map * (1.0 + (self._eye_weight - 1.0) * mask_eye)
        if mask_mouth is not None:
            weight_map = weight_map * (1.0 + (self._mouth_weight - 1.0) * mask_mouth)
        return _reduce_spatial(loss_map * weight_map)

    def _non_spatial_loss(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mask_face: torch.Tensor | None,
        mask_eye: torch.Tensor | None,
        mask_mouth: torch.Tensor | None,
    ) -> torch.Tensor:
        """Calculate region-weighted loss for a non-spatial perceptual loss.

        Each region is masked into the inputs and the loss is run independently, then summed
        with its region weight.

        Parameters
        ----------
        y_true
            The ground truth images in ``(N, C, H, W)`` order
        y_pred
            The model predictions in ``(N, C, H, W)`` order
        mask_face
            The face mask in ``(N, 1, H, W)`` order if available
        mask_eye
            The eye mask in ``(N, 1, H, W)`` order if available
        mask_mouth
            The mouth mask in ``(N, 1, H, W)`` order if available

        Returns
        -------
        The per-item region-weighted loss in ``(N, )`` order
        """
        if mask_face is None:
            return T.cast(torch.Tensor, self._function(y_true, y_pred)) * self._face_weight

        regions: list[tuple[torch.Tensor, float]] = [(mask_face, self._face_weight)]
        skin = self._skin_mask(mask_face, mask_eye, mask_mouth)
        regions.append((skin, self._skin_weight))
        if mask_eye is not None:
            regions.append((mask_eye, self._eye_weight))
        if mask_mouth is not None:
            regions.append((mask_mouth, self._mouth_weight))

        losses = torch.stack(
            [self._function(y_true * mask, y_pred * mask) * weight for mask, weight in regions]
        )
        return losses.sum(dim=0)

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mask_face: torch.Tensor | None,
        mask_eye: torch.Tensor | None,
        mask_mouth: torch.Tensor | None,
    ) -> torch.Tensor:
        """Calculate the region-weighted perceptual loss for a batch.

        Parameters
        ----------
        y_true
            The ground truth images in ``(N, C, H, W)`` order
        y_pred
            The model predictions in ``(N, C, H, W)`` order
        mask_face
            The face mask in ``(N, 1, H, W)`` order if available
        mask_eye
            The eye mask in ``(N, 1, H, W)`` order if available
        mask_mouth
            The mouth mask in ``(N, 1, H, W)`` order if available

        Returns
        -------
        The per-item region-weighted perceptual loss in ``(N, )`` order
        """
        if self._spatial:
            return self._spatial_loss(y_true, y_pred, mask_face, mask_eye, mask_mouth)
        return self._non_spatial_loss(y_true, y_pred, mask_face, mask_eye, mask_mouth)


class IdentityLoss(nn.Module):
    """Frozen face-recognition identity embedding loss.

    Compares the identity embeddings of the ground truth face and the model's reconstruction
    using a frozen, pre-trained recognition network. The recognition model is never trained
    and is deliberately kept out of the module tree so that its weights are not serialized
    into the Faceswap model checkpoint.

    Parameters
    ----------
    recognizer
        The frozen recognition module. It must accept a ``(N, 3, input_size, input_size)``
        RGB tensor normalized to the ``-1.0 - 1.0`` range and return ``(N, D)`` embeddings
    input_size
        The spatial input size expected by the recognition model
    color_order
        The color order that the model is training in
    crop
        How to crop the face before feeding the recognition model
    """

    def __init__(
        self,
        recognizer: nn.Module,
        input_size: int,
        color_order: T.Literal["bgr", "rgb"],
        crop: T.Literal["face", "mask_bbox"] = "face",
    ) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        recognizer.eval()
        for param in recognizer.parameters():
            param.requires_grad_(False)
        # Store the recognizer inside a list so that ``nn.Module`` does not register it as a
        # submodule. This keeps its (frozen) weights out of the Faceswap model ``state_dict``
        # and therefore out of saved checkpoints.
        self._recognizer = [recognizer]
        self._input_size = input_size
        self._color_order = color_order
        self._crop = crop
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def recognizer(self) -> nn.Module:
        """The wrapped frozen recognition module."""
        return self._recognizer[0]

    @staticmethod
    def _crop_to_mask_bbox(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Crop each image in a batch to the bounding box of its mask.

        Parameters
        ----------
        image
            The batch of images in ``(N, C, H, W)`` order
        mask
            The batch of masks in ``(N, 1, H, W)`` order

        Returns
        -------
        The batch of images cropped and resized back to the original spatial size
        """
        size = image.shape[-1]
        binary = mask[:, 0] >= 0.5
        height, width = binary.shape[1], binary.shape[2]
        # Compute every bounding box in a single vectorized pass so that the variable-size
        # crops below only require one device-to-host transfer. Doing the index extraction
        # per item (with ``int(...)`` conversions) forces a GPU sync on every bound, which
        # serializes the training step.
        rows = torch.any(binary, dim=2)  # (N, H)
        cols = torch.any(binary, dim=1)  # (N, W)
        has_any = rows.any(dim=1) & cols.any(dim=1)  # (N, )
        row_ids = torch.arange(height, device=binary.device)
        col_ids = torch.arange(width, device=binary.device)
        top = torch.where(rows, row_ids, height).amin(dim=1)
        bottom = torch.where(rows, row_ids, -1).amax(dim=1) + 1
        left = torch.where(cols, col_ids, width).amin(dim=1)
        right = torch.where(cols, col_ids, -1).amax(dim=1) + 1
        bounds = torch.stack([top, bottom, left, right], dim=1).tolist()
        flags = has_any.tolist()
        cropped: list[torch.Tensor] = []
        for idx in range(image.shape[0]):
            if not flags[idx]:
                cropped.append(image[idx])
                continue
            top_i, bottom_i, left_i, right_i = bounds[idx]
            patch = image[idx : idx + 1, :, top_i:bottom_i, left_i:right_i]
            cropped.append(
                F.interpolate(patch, size=(size, size), mode="bilinear", align_corners=False)[0]
            )
        return torch.stack(cropped)

    def _preprocess(self, image: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Prepare a batch of faces for the recognition model.

        Parameters
        ----------
        image
            The batch of faces in ``(N, C, H, W)`` order in the ``0.0 - 1.0`` range
        mask
            The face mask in ``(N, 1, H, W)`` order if available

        Returns
        -------
        The preprocessed batch in ``(N, 3, input_size, input_size)`` RGB order normalized to
        the ``-1.0 - 1.0`` range
        """
        if self._color_order == "bgr":
            image = image.flip(1)
        if self._crop == "mask_bbox" and mask is not None:
            image = self._crop_to_mask_bbox(image, mask)
        if image.shape[-1] != self._input_size:
            image = F.interpolate(
                image,
                size=(self._input_size, self._input_size),
                mode="bilinear",
                align_corners=False,
            )
        return image * 2.0 - 1.0

    def _embed(self, image: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Obtain L2-normalized identity embeddings for a batch of faces.

        Parameters
        ----------
        image
            The batch of faces in ``(N, C, H, W)`` order in the ``0.0 - 1.0`` range
        mask
            The face mask in ``(N, 1, H, W)`` order if available

        Returns
        -------
        The normalized embeddings in ``(N, D)`` order
        """
        prepared = self._preprocess(image, mask)
        embeddings = self.recognizer(prepared)
        if embeddings.ndim > 2:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
        return F.normalize(embeddings, dim=1, eps=_EPS)

    def forward(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, mask_face: torch.Tensor | None
    ) -> torch.Tensor:
        """Calculate the identity loss for a batch.

        Parameters
        ----------
        y_true
            The ground truth images in ``(N, C, H, W)`` order
        y_pred
            The model predictions in ``(N, C, H, W)`` order
        mask_face
            The face mask in ``(N, 1, H, W)`` order if available

        Returns
        -------
        The per-item identity distance (``1 - cosine similarity``) in ``(N, )`` order
        """
        recognizer = self.recognizer
        param = next(recognizer.parameters(), None)
        if param is not None and param.device != y_pred.device:
            recognizer.to(y_pred.device)
        with torch.no_grad():
            target = self._embed(y_true, mask_face)
        predicted = self._embed(y_pred, mask_face)
        cosine = (predicted * target).sum(dim=1)
        return 1.0 - cosine


__all__ = get_module_objects(__name__)
