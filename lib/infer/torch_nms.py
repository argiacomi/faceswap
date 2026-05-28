#!/usr/bin/env python3
"""Shared Torch non-maximum suppression helpers for extract inference paths."""

from __future__ import annotations

import logging

import torch

from lib.utils import get_module_objects

logger = logging.getLogger(__name__)

try:
    from torchvision.ops import nms as _torchvision_nms
except (ImportError, RuntimeError):
    _torchvision_nms = None


def _custom_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Fallback NMS implementation that works without TorchVision.

    Uses the standard continuous-coordinate IoU formula (no +1 term) to match
    :func:`torchvision.ops.nms` so all torch_nms paths agree.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    x_1 = boxes[:, 0]
    y_1 = boxes[:, 1]
    x_2 = boxes[:, 2]
    y_2 = boxes[:, 3]
    areas = (x_2 - x_1) * (y_2 - y_1)
    order = torch.argsort(scores, descending=True)
    keep: list[int] = []

    while order.numel() > 0:
        idx = int(order[0].item())
        keep.append(idx)
        if order.numel() == 1:
            break

        remaining = order[1:]
        xx_1 = torch.maximum(x_1[idx], x_1.index_select(0, remaining))
        yy_1 = torch.maximum(y_1[idx], y_1.index_select(0, remaining))
        xx_2 = torch.minimum(x_2[idx], x_2.index_select(0, remaining))
        yy_2 = torch.minimum(y_2[idx], y_2.index_select(0, remaining))

        width = torch.clamp(xx_2 - xx_1, min=0.0)
        height = torch.clamp(yy_2 - yy_1, min=0.0)
        intersection = width * height
        overlap = intersection / (areas[idx] + areas.index_select(0, remaining) - intersection)
        order = remaining[overlap <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def torch_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Run NMS through TorchVision when available, otherwise use the local fallback."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Expected boxes with shape (N, 4), got {tuple(boxes.shape)}")
    if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
        raise ValueError(
            f"Expected scores with shape ({boxes.shape[0]},), got {tuple(scores.shape)}"
        )
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    if _torchvision_nms is not None:
        try:
            return _torchvision_nms(boxes, scores, iou_threshold)
        except (RuntimeError, NotImplementedError) as err:
            logger.debug(
                "TorchVision NMS unavailable for %s, using custom fallback: %s",
                boxes.device,
                err,
            )
    return _custom_nms(boxes, scores, iou_threshold)


__all__ = get_module_objects(__name__)
