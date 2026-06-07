#!/usr/bin/env python3
"""CD-ViT landmarks extractor for faceswap.py.

Model architecture ported from:
https://github.com/argiacomi/AccurateFacialLandmarkDetection
"""

from __future__ import annotations

import logging
import math
import os
import typing as T
from collections import namedtuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lib.align.aligned_utils import bbox_to_square_roi
from lib.utils import PROJECT_ROOT, get_module_objects
from plugins.extract.base import ExtractPlugin

from . import cd_vit_defaults as cfg

logger = logging.getLogger(__name__)

_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, ".fs_cache", "cd_vit_v1.pth")
_INPUT_SIZE = 256
_HEATMAP_SIZE = 32
_MAX_DEPTH = 256
_NSTACK = 8
_NUM_LANDMARKS = 68


class Cd_Vit(ExtractPlugin):
    """CD-ViT face alignment plugin."""

    def __init__(self) -> None:
        super().__init__(
            input_size=_INPUT_SIZE,
            batch_size=cfg.batch_size(),
            is_rgb=True,
            dtype="float32",
            scale=(-1, 1),
        )
        self.realign_centering = "legacy"
        self._crop_scale = cfg.crop_scale()
        self.model: CDViTStage

    def load_model(self) -> CDViTStage:
        """Load the CD-ViT aligner model."""
        if not os.path.isfile(_WEIGHTS_PATH):
            raise FileNotFoundError(f"CD-ViT weights not found: {_WEIGHTS_PATH}")
        self._validate_checkpoint(_WEIGHTS_PATH)
        return T.cast(CDViTStage, self.load_torch_model(CDViTStage(), _WEIGHTS_PATH))

    @staticmethod
    def _canonical_key(key: T.Any) -> str:
        """Normalize common wrapper prefixes from checkpoint keys."""
        retval = str(key)
        for prefix in ("module.", "model."):
            if retval.startswith(prefix):
                retval = retval[len(prefix) :]
        return retval

    @staticmethod
    def _validate_checkpoint(weights_path: str) -> None:
        """Reject checkpoint families that do not match this staged CD-ViT port."""
        state = torch.load(weights_path, map_location="cpu")
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError("CD-ViT checkpoint does not contain a state dictionary")

        keys = [Cd_Vit._canonical_key(key) for key in state]
        prefixes = sorted({key.split(".", 1)[0] for key in keys})
        logger.debug("CD-ViT checkpoint prefixes: %s", prefixes)
        if any(key.startswith("unet.") for key in keys):
            raise ValueError("This checkpoint is UNet-backed, not staged CD-ViT.")
        if not any(key.startswith("stages.") for key in keys):
            raise ValueError("This checkpoint does not look like staged CD-ViT.")

    def pre_process(self, batch: np.ndarray) -> np.ndarray:
        """Format detector bounding boxes into CD-ViT input crops."""
        retval: np.ndarray = bbox_to_square_roi(batch, self._crop_scale)
        return retval

    def process(self, batch: np.ndarray) -> np.ndarray:
        """Predict normalized 68-point landmarks."""
        inputs = np.ascontiguousarray(batch.transpose(0, 3, 1, 2))
        retval: np.ndarray = self.from_torch(inputs)
        return retval


class WSConv2d(nn.Module):
    """Weight-scaled convolution used by the CD-ViT backbone."""

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        us_ws: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channel, out_channel, kernel_size, stride, padding, dilation, bias=False
        )
        self.scale = math.sqrt(2.0) / math.sqrt(in_channel * kernel_size * kernel_size)
        if not us_ws:
            self.scale = 1.0
        self.bias = bias
        if self.bias:
            self.bias_data = nn.Parameter(torch.zeros((1, out_channel, 1, 1)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the convolution and optional explicit bias."""
        retval = self.conv(inputs) * self.scale
        if self.bias:
            retval = retval + self.bias_data
        return retval


class SEModule(nn.Module):
    """Squeeze-and-excitation module used by the backbone bottlenecks."""

    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = WSConv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = WSConv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply channel attention."""
        module_input = inputs
        out = self.avg_pool(inputs)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return module_input * out


class Bottleneck(namedtuple("Block", ["in_channel", "depth", "stride"])):
    """A CD-ViT backbone block descriptor."""


def get_block(in_channel: int, depth: int, num_units: int, stride: int = 2) -> list[Bottleneck]:
    """Return a repeated bottleneck block definition."""
    return [Bottleneck(in_channel, depth, stride)] + [
        Bottleneck(depth, depth, 1) for _ in range(num_units - 1)
    ]


class BottleneckIRSE(nn.Module):
    """IR-SE bottleneck used by ``HeadingNet``."""

    def __init__(self, in_channel: int, depth: int, stride: int = 1) -> None:
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                WSConv2d(in_channel, depth, 1, stride, bias=False),
                nn.BatchNorm2d(depth),
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            WSConv2d(in_channel, depth, 3, 1, 1, bias=False),
            nn.PReLU(depth),
            WSConv2d(depth, depth, 3, stride, 1, bias=False),
            nn.BatchNorm2d(depth),
            SEModule(depth, 16),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.res_layer(inputs) + self.shortcut_layer(inputs)


class HeadingNet(nn.Module):
    """CD-ViT convolutional feature backbone."""

    def __init__(self, channels: tuple[int, ...] = (32, 64, 256), in_channel: int = 3) -> None:
        super().__init__()
        blocks = [get_block(in_channel=in_channel, depth=channels[0], num_units=3)]
        for idx in range(len(channels) - 1):
            blocks.append(
                get_block(in_channel=channels[idx], depth=channels[idx + 1], num_units=3)
            )
        units = [
            BottleneckIRSE(block.in_channel, block.depth, block.stride)
            for group in blocks
            for block in group
        ]
        self.body = nn.Sequential(*units)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.body(inputs)


class DoubleConv(nn.Module):
    """Two convolutions with a residual shortcut."""

    def __init__(self, in_channel: int, out_channel: int, mid_channel: int | None = None) -> None:
        super().__init__()
        if mid_channel is None:
            mid_channel = out_channel
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.short_cut = (
            nn.Identity()
            if in_channel == out_channel
            else nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.relu(self.double_conv(inputs) + self.short_cut(inputs))


class AddCoordsTh(nn.Module):
    """Append normalized coordinate channels to an image tensor."""

    def __init__(self, x_dim: int, y_dim: int, with_r: bool, with_boundary: bool) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r
        self.with_boundary = with_boundary

    def forward(
        self, input_tensor: torch.Tensor, heatmap: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Append x/y, and optional radius/boundary, channels."""
        batch_size = input_tensor.shape[0]
        xx_ones = torch.ones([1, self.y_dim], dtype=torch.int32).to(input_tensor).unsqueeze(-1)
        xx_range = (
            torch.arange(self.x_dim, dtype=torch.int32).unsqueeze(0).to(input_tensor).unsqueeze(1)
        )
        xx_channel = torch.matmul(xx_ones.float(), xx_range.float()).unsqueeze(-1)

        yy_ones = torch.ones([1, self.x_dim], dtype=torch.int32).to(input_tensor).unsqueeze(1)
        yy_range = (
            torch.arange(self.y_dim, dtype=torch.int32).unsqueeze(0).to(input_tensor).unsqueeze(-1)
        )
        yy_channel = torch.matmul(yy_range.float(), yy_ones.float()).unsqueeze(-1)

        xx_channel = xx_channel.permute(0, 3, 2, 1)
        yy_channel = yy_channel.permute(0, 3, 2, 1)
        xx_channel = xx_channel / (self.x_dim - 1) * 2 - 1
        yy_channel = yy_channel / (self.y_dim - 1) * 2 - 1
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)

        channels = [input_tensor, xx_channel, yy_channel]
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
            channels.append(rr / torch.max(rr))
        if self.with_boundary and heatmap is not None:
            boundary_channel = torch.clamp(heatmap[:, -1:, :, :], 0.0, 1.0)
            zero_tensor = torch.zeros_like(xx_channel).to(xx_channel)
            channels.extend(
                [
                    torch.where(boundary_channel > 0.05, xx_channel, zero_tensor),
                    torch.where(boundary_channel > 0.05, yy_channel, zero_tensor),
                ]
            )
        return torch.cat(channels, dim=1)


class CoordConvTh(nn.Module):
    """CoordConv layer."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        with_r: bool,
        with_boundary: bool,
        in_channels: int,
        out_channels: int,
        first_one: bool = False,
        relu: bool = False,
        bn: bool = False,
        *args: T.Any,
        **kwargs: T.Any,
    ) -> None:
        super().__init__()
        self.addcoords = AddCoordsTh(
            x_dim=x_dim, y_dim=y_dim, with_r=with_r, with_boundary=with_boundary
        )
        coord_channels = in_channels + 2 + int(with_r)
        if with_boundary and not first_one:
            coord_channels += 2
        self.conv = nn.Conv2d(coord_channels, out_channels, *args, **kwargs)
        self.relu = nn.ReLU() if relu else None
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.with_boundary = with_boundary
        self.first_one = first_one

    def forward(
        self, input_tensor: torch.Tensor, heatmap: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass."""
        assert (self.with_boundary and not self.first_one) == (heatmap is not None)
        retval = self.conv(self.addcoords(input_tensor, heatmap))
        if self.bn is not None:
            retval = self.bn(retval)
        if self.relu is not None:
            retval = self.relu(retval)
        return retval


class SelfAttentionBlock2(nn.Module):
    """Local channel attention branch."""

    def __init__(self, channels: int, win_size: int = 2, out_channel: int | None = None) -> None:
        super().__init__()
        self.win_size = win_size
        self.out_channel = channels if out_channel is None else out_channel
        multiplier = win_size * win_size
        width = self.out_channel * multiplier
        self.mha = nn.MultiheadAttention(width, 4, batch_first=True)
        self.ln = nn.LayerNorm([width])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([width]),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.patch_embed = nn.Conv2d(channels, width, win_size, win_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch, _channels, height, width = inputs.shape
        out = (
            self.patch_embed(inputs)
            .permute((0, 2, 3, 1))
            .reshape((batch, (height * width) // (self.win_size * self.win_size), -1))
        )
        attention_value, _ = self.mha(self.ln(out), self.ln(out), self.ln(out))
        attention_value = attention_value + out
        attention_value = self.ff_self(attention_value) + attention_value
        return (
            attention_value.reshape(
                (
                    batch,
                    height // self.win_size,
                    width // self.win_size,
                    self.win_size,
                    self.win_size,
                    self.out_channel,
                )
            )
            .permute((0, 5, 1, 3, 2, 4))
            .reshape((batch, self.out_channel, height, width))
            .contiguous()
        )


class SelfAttention2Block(nn.Module):
    """Local spatial attention branch."""

    def __init__(
        self, img_size: int, in_channel: int, out_channel: int, win_size: int = 2
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.win_size = win_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        width = img_size * img_size // (win_size * win_size)
        self.mha = nn.MultiheadAttention(width, 4, batch_first=True)
        self.ln = nn.LayerNorm([width])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([width]),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.patch_embed = nn.Conv2d(
            in_channel, out_channel * win_size * win_size, win_size, win_size
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch, _channels, height, width = inputs.shape
        out = self.patch_embed(inputs).reshape(
            (
                batch,
                self.out_channel * self.win_size * self.win_size,
                height * width // (self.win_size * self.win_size),
            )
        )
        attention_value, _ = self.mha(self.ln(out), self.ln(out), self.ln(out))
        attention_value = attention_value + out
        attention_value = self.ff_self(attention_value) + attention_value
        return (
            attention_value.reshape(
                (
                    batch,
                    self.win_size,
                    self.win_size,
                    self.out_channel,
                    height // self.win_size,
                    width // self.win_size,
                )
            )
            .permute((0, 3, 4, 1, 5, 2))
            .reshape((batch, self.out_channel, height, width))
        )


class SA2SA1_2(nn.Module):
    """Combined CD-ViT attention block used by the checkpoint."""

    def __init__(
        self, img_size: int = _HEATMAP_SIZE, channel_size: int = _MAX_DEPTH, win_size: int = 2
    ) -> None:
        super().__init__()
        self.sa1 = SelfAttentionBlock2(
            channels=channel_size, win_size=win_size, out_channel=channel_size // 2
        )
        self.sa2 = SelfAttention2Block(
            img_size=img_size,
            in_channel=channel_size,
            out_channel=channel_size // 2,
            win_size=win_size,
        )
        self.conv = DoubleConv(channel_size, channel_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.conv(torch.cat([self.sa2(inputs), self.sa1(inputs)], dim=1))


class CDViTStage(nn.Module):
    """CD-ViT staged attention landmark network."""

    def __init__(self) -> None:
        super().__init__()
        self.pre = HeadingNet((32, 64, _MAX_DEPTH))
        self.stages = nn.ModuleList(self._stage_block() for _ in range(_NSTACK))
        self.output_layers = nn.ModuleList(
            nn.Conv2d(_MAX_DEPTH, _NUM_LANDMARKS, 1) for _ in range(_NSTACK)
        )
        self.merge = nn.ModuleList(
            DoubleConv(_MAX_DEPTH * 2, _MAX_DEPTH, _MAX_DEPTH) for _ in range(_NSTACK - 1)
        )
        row, col = self.make_grid(_HEATMAP_SIZE)
        self.register_buffer("xx_loc", col, False)
        self.register_buffer("yy_loc", row, False)

    @staticmethod
    def _stage_block() -> nn.Sequential:
        return nn.Sequential(
            CoordConvTh(
                _HEATMAP_SIZE,
                _HEATMAP_SIZE,
                True,
                False,
                _MAX_DEPTH,
                _MAX_DEPTH,
                kernel_size=3,
                padding=1,
            ),
            SA2SA1_2(_HEATMAP_SIZE, _MAX_DEPTH),
            CoordConvTh(
                _HEATMAP_SIZE,
                _HEATMAP_SIZE,
                True,
                False,
                _MAX_DEPTH,
                _MAX_DEPTH,
                kernel_size=3,
                padding=1,
            ),
            SA2SA1_2(_HEATMAP_SIZE, _MAX_DEPTH),
        )

    @staticmethod
    def make_grid(size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Create normalized heatmap coordinate grids."""
        row, col = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        denom = size - 1.0
        row = row / denom
        col = col / denom
        return row.reshape((1, 1, size, size)), col.reshape((1, 1, size, size))

    def get_coord(self, heatmap: torch.Tensor) -> torch.Tensor:
        """Convert heatmaps to normalized landmark coordinates."""
        batch, channels, height, width = heatmap.shape
        heatmap = F.softmax(heatmap.reshape((batch, channels, -1)), dim=-1).reshape(
            (batch, channels, height, width)
        )
        xx = (heatmap * self.xx_loc).sum([2, 3])
        yy = (heatmap * self.yy_loc).sum([2, 3])
        return torch.stack([xx, yy], dim=2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return final-stage normalized landmark coordinates."""
        feat = self.pre(inputs)
        pre_hm = None
        coord = None
        for idx, stage in enumerate(self.stages):
            if idx == 0:
                merge = feat
            else:
                assert pre_hm is not None
                merge = self.merge[idx - 1](torch.cat([feat, pre_hm], dim=1))
            stage_out = stage(merge)
            coord = self.get_coord(self.output_layers[idx](stage_out))
            pre_hm = stage_out
        return T.cast(torch.Tensor, coord)


__all__ = get_module_objects(__name__)
