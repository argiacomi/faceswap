#!/usr/bin/env python3
"""Minimal ORFormer/HGNet inference graph for Faceswap landmark extraction."""

from __future__ import annotations

import logging
import typing as T

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


logger = logging.getLogger(__name__)

EdgeInfo = list[tuple[bool, list[int]]]


def _make_grid(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(
        torch.arange(height).float() / (height - 1) * 2 - 1,
        torch.arange(width).float() / (width - 1) * 2 - 1,
        indexing="ij",
    )
    return yy, xx


def _coords_from_heatmap(
    heatmap: torch.Tensor, grid_yy: torch.Tensor, grid_xx: torch.Tensor
) -> torch.Tensor:
    """Return soft-argmax coordinates in [-1, +1] from heatmaps using cached grids."""
    heatmap_sum = torch.clamp(heatmap.sum([2, 3]), min=1e-6)
    yy_coord = (grid_yy * heatmap).sum([2, 3]) / heatmap_sum
    xx_coord = (grid_xx * heatmap).sum([2, 3]) / heatmap_sum
    return torch.stack([xx_coord, yy_coord], dim=-1)


class Activation(nn.Module):
    """Small activation helper matching upstream ORFormer."""

    def __init__(self, kind: str = "relu", channel: int | None = None) -> None:
        super().__init__()
        norm_str, act_str = kind.split("+") if "+" in kind else ("none", kind)
        self.norm_fn: T.Callable[[torch.Tensor], torch.Tensor] | nn.Module | None = {
            "in": F.instance_norm,
            "bn": nn.BatchNorm2d(channel),
            "bn_noaffine": nn.BatchNorm2d(channel, affine=False, track_running_stats=True),
            "none": None,
        }[norm_str]
        self.act_fn: T.Callable[[torch.Tensor], torch.Tensor] | nn.Module | None = {
            "relu": F.relu,
            "softplus": nn.Softplus(),
            "exp": torch.exp,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "none": None,
        }[act_str]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.norm_fn is not None:
            inputs = self.norm_fn(inputs)
        if self.act_fn is not None:
            inputs = self.act_fn(inputs)
        return inputs


class ConvBlock(nn.Module):
    """Convolution block used throughout HGNet."""

    def __init__(
        self,
        inp_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        bn: bool = False,
        relu: bool = True,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding, bias=True)
        self.relu = nn.ReLU() if relu else None
        self.bn = nn.BatchNorm2d(out_dim) if bn else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv(inputs)
        if self.bn is not None:
            outputs = self.bn(outputs)
        if self.relu is not None:
            outputs = self.relu(outputs)
        return outputs


class AddCoordsTh(nn.Module):
    """Append normalized x/y coordinate channels, with optional radius/boundary channels."""

    def __init__(
        self, x_dim: int, y_dim: int, with_r: bool = False, with_boundary: bool = False
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r
        self.with_boundary = with_boundary

    def forward(
        self, input_tensor: torch.Tensor, heatmap: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size = input_tensor.shape[0]
        xx_ones = torch.ones([1, self.y_dim], dtype=torch.int32).to(input_tensor).unsqueeze(-1)
        xx_range = torch.arange(self.x_dim, dtype=torch.int32).unsqueeze(0).to(input_tensor)
        xx_range = xx_range.unsqueeze(1)
        xx_channel = torch.matmul(xx_ones.float(), xx_range.float()).unsqueeze(-1)

        yy_ones = torch.ones([1, self.x_dim], dtype=torch.int32).to(input_tensor).unsqueeze(1)
        yy_range = torch.arange(self.y_dim, dtype=torch.int32).unsqueeze(0).to(input_tensor)
        yy_range = yy_range.unsqueeze(-1)
        yy_channel = torch.matmul(yy_range.float(), yy_ones.float()).unsqueeze(-1)

        xx_channel = xx_channel.permute(0, 3, 2, 1) / (self.x_dim - 1) * 2 - 1
        yy_channel = yy_channel.permute(0, 3, 2, 1) / (self.y_dim - 1) * 2 - 1
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)

        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
            ret = torch.cat([ret, rr / torch.max(rr)], dim=1)
        if self.with_boundary and heatmap is not None:
            boundary_channel = torch.clamp(heatmap[:, -1:, :, :], 0.0, 1.0)
            zero_tensor = torch.zeros_like(xx_channel).to(xx_channel)
            ret = torch.cat(
                [
                    ret,
                    torch.where(boundary_channel > 0.05, xx_channel, zero_tensor),
                    torch.where(boundary_channel > 0.05, yy_channel, zero_tensor),
                ],
                dim=1,
            )
        return ret


class CoordConvTh(nn.Module):
    """CoordConv layer as used by upstream HGNet."""

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
        **kwargs: T.Any,
    ) -> None:
        super().__init__()
        self.addcoords = AddCoordsTh(
            x_dim=x_dim, y_dim=y_dim, with_r=with_r, with_boundary=with_boundary
        )
        coord_channels = 2 + (1 if with_r else 0) + (2 if with_boundary and not first_one else 0)
        self.conv = nn.Conv2d(in_channels + coord_channels, out_channels, **kwargs)
        self.relu = nn.ReLU() if relu else None
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.with_boundary = with_boundary
        self.first_one = first_one

    def forward(
        self, input_tensor: torch.Tensor, heatmap: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert (self.with_boundary and not self.first_one) == (heatmap is not None)
        ret = self.conv(self.addcoords(input_tensor, heatmap))
        if self.bn is not None:
            ret = self.bn(ret)
        if self.relu is not None:
            ret = self.relu(ret)
        return ret


class ResBlock(nn.Module):
    """Residual block used by HGNet."""

    def __init__(self, inp_dim: int, out_dim: int, mid_dim: int | None = None) -> None:
        super().__init__()
        if mid_dim is None:
            mid_dim = out_dim // 2
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = ConvBlock(inp_dim, mid_dim, 1, relu=False)
        self.bn2 = nn.BatchNorm2d(mid_dim)
        self.conv2 = ConvBlock(mid_dim, mid_dim, 3, relu=False)
        self.bn3 = nn.BatchNorm2d(mid_dim)
        self.conv3 = ConvBlock(mid_dim, out_dim, 1, relu=False)
        self.skip_layer = ConvBlock(inp_dim, out_dim, 1, relu=False)
        self.need_skip = inp_dim != out_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.skip_layer(inputs) if self.need_skip else inputs
        outputs = self.conv1(self.relu(self.bn1(inputs)))
        outputs = self.conv2(self.relu(self.bn2(outputs)))
        outputs = self.conv3(self.relu(self.bn3(outputs)))
        return outputs + residual


class Hourglass(nn.Module):
    """Recursive hourglass block."""

    def __init__(
        self,
        levels: int,
        features: int,
        increase: int = 0,
        add_coord: bool = False,
        first_one: bool = False,
    ) -> None:
        super().__init__()
        next_features = features + increase
        self.coordconv = (
            CoordConvTh(
                x_dim=64,
                y_dim=64,
                with_r=True,
                with_boundary=True,
                relu=False,
                bn=False,
                in_channels=features,
                out_channels=features,
                first_one=first_one,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            if add_coord
            else None
        )
        self.up1 = ResBlock(features, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.low1 = ResBlock(features, next_features)
        self.low2 = (
            Hourglass(levels - 1, next_features, increase=increase)
            if levels > 1
            else ResBlock(next_features, next_features)
        )
        self.low3 = ResBlock(next_features, features)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, inputs: torch.Tensor, heatmap: torch.Tensor | None = None) -> torch.Tensor:
        if self.coordconv is not None:
            inputs = self.coordconv(inputs, heatmap)
        up1 = self.up1(inputs)
        low1 = self.low1(self.pool1(inputs))
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        return up1 + self.up2(low3)


class E2HTransform(nn.Module):
    """Transform edge maps into point heatmap weights."""

    def __init__(self, edge_info: EdgeInfo, num_points: int, num_edges: int) -> None:
        super().__init__()
        e2h_matrix = np.zeros([num_points, num_edges])
        for edge_id, (_is_closed, indices) in enumerate(edge_info):
            for point_id in indices:
                e2h_matrix[point_id, edge_id] = 1
        matrix = torch.from_numpy(e2h_matrix).float()
        self.register_buffer("weight", matrix.view(matrix.size(0), matrix.size(1), 1, 1))
        bias = ((matrix @ torch.ones(matrix.size(1)).to(matrix)) < 0.5).to(matrix)
        self.register_buffer("bias", bias)

    def forward(self, edgemaps: torch.Tensor) -> torch.Tensor:
        return F.conv2d(edgemaps, weight=self.weight, bias=self.bias)


class IntegrationStackedHGNet(nn.Module):
    """HGNet landmark model integrated with ORFormer reference heatmaps."""

    def __init__(
        self,
        num_points: int,
        num_edges: int,
        edge_info: EdgeInfo,
        nstack: int = 4,
        nlevels: int = 4,
        in_channel: int = 256,
    ) -> None:
        super().__init__()
        self.nstack = nstack
        self.num_heats = num_points
        self.num_edges = num_edges
        self.num_points = num_points

        conv_block = CoordConvTh(
            x_dim=256,
            y_dim=256,
            with_r=True,
            with_boundary=False,
            relu=True,
            bn=True,
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
        )
        self.e2h_transform = E2HTransform(edge_info, self.num_points, self.num_edges)
        self.pre = nn.Sequential(
            conv_block,
            ResBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(128, 128),
            ResBlock(128, in_channel),
        )
        self.conv = nn.Conv2d(in_channel + len(edge_info), in_channel, kernel_size=1)
        self.hgs = nn.ModuleList(
            [
                Hourglass(
                    levels=nlevels,
                    features=in_channel,
                    add_coord=True,
                    first_one=(idx == 0),
                )
                for idx in range(nstack)
            ]
        )
        self.features = nn.ModuleList(
            [
                nn.Sequential(
                    ResBlock(in_channel, in_channel),
                    ConvBlock(in_channel, in_channel, 1, bn=True, relu=True),
                )
                for _idx in range(nstack)
            ]
        )
        self.out_heatmaps = nn.ModuleList(
            [
                ConvBlock(in_channel, self.num_heats, 1, relu=False, bn=False)
                for _idx in range(nstack)
            ]
        )
        self.out_edgemaps = nn.ModuleList(
            [
                ConvBlock(in_channel, self.num_edges, 1, relu=False, bn=False)
                for _idx in range(nstack)
            ]
        )
        self.out_pointmaps = nn.ModuleList(
            [
                ConvBlock(in_channel, self.num_points, 1, relu=False, bn=False)
                for _idx in range(nstack)
            ]
        )
        self.merge_features = nn.ModuleList(
            [
                ConvBlock(in_channel, in_channel, 1, relu=False, bn=False)
                for _idx in range(nstack - 1)
            ]
        )
        self.merge_heatmaps = nn.ModuleList(
            [
                ConvBlock(self.num_heats, in_channel, 1, relu=False, bn=False)
                for _idx in range(nstack - 1)
            ]
        )
        self.merge_edgemaps = nn.ModuleList(
            [
                ConvBlock(self.num_edges, in_channel, 1, relu=False, bn=False)
                for _idx in range(nstack - 1)
            ]
        )
        self.merge_pointmaps = nn.ModuleList(
            [
                ConvBlock(self.num_points, in_channel, 1, relu=False, bn=False)
                for _idx in range(nstack - 1)
            ]
        )
        self.heatmap_act = Activation("in+relu", self.num_heats)
        self.edgemap_act = Activation("sigmoid", self.num_edges)
        self.pointmap_act = Activation("sigmoid", self.num_points)

        # Heatmap grid is fixed at 64x64 for a 256 input; cache as buffers so the soft-argmax
        # coordinate tensors are not reallocated on every forward.
        grid_yy, grid_xx = _make_grid(64, 64)
        self.register_buffer("_grid_yy", grid_yy.view(1, 1, 64, 64), persistent=False)
        self.register_buffer("_grid_xx", grid_xx.view(1, 1, 64, 64), persistent=False)

    def forward(
        self, inputs: torch.Tensor, reference_heatmaps: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.pre(inputs)
        if reference_heatmaps is not None:
            x = self.conv(torch.cat([x, reference_heatmaps], dim=1))

        heatmaps = None
        landmarks = None
        for idx in range(self.nstack):
            hg = self.hgs[idx](x, heatmap=heatmaps)
            feature = self.features[idx](hg)
            heatmaps = self.heatmap_act(self.out_heatmaps[idx](feature))
            edgemaps = self.edgemap_act(self.out_edgemaps[idx](feature))
            pointmaps = self.pointmap_act(self.out_pointmaps[idx](feature))
            attention_mask = self.e2h_transform(edgemaps) * pointmaps
            landmarks = _coords_from_heatmap(
                attention_mask * heatmaps, self._grid_yy, self._grid_xx
            )
            if idx < self.nstack - 1:
                x = (
                    x
                    + self.merge_features[idx](feature)
                    + self.merge_heatmaps[idx](heatmaps)
                    + self.merge_edgemaps[idx](edgemaps)
                    + self.merge_pointmaps[idx](pointmaps)
                )

        assert landmarks is not None
        return landmarks


class ResidualLayer(nn.Module):
    """Residual layer for VQ-VAE."""

    def __init__(self, in_dim: int, h_dim: int, res_h_dim: int) -> None:
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, 3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, 1, stride=1, bias=False),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.res_block(inputs)


class ResidualStack(nn.Module):
    """Stack of VQ-VAE residual layers."""

    def __init__(self, in_dim: int, h_dim: int, res_h_dim: int, n_res_layers: int) -> None:
        super().__init__()
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim) for _idx in range(n_res_layers)]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer in self.stack:
            inputs = layer(inputs)
        return F.relu(inputs)


class Encoder(nn.Module):
    """VQ-VAE encoder."""

    def __init__(self, in_dim: int, h_dim: int, n_res_layers: int, res_h_dim: int) -> None:
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, 3, stride=1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv_stack(inputs)


class Decoder(nn.Module):
    """VQ-VAE decoder that emits ORFormer edge heatmaps."""

    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        n_res_layers: int,
        res_h_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(in_dim, h_dim, 3, stride=1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim // 2, output_dim, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.inverse_conv_stack(inputs)


def _posemb_sincos_2d(
    height: int,
    width: int,
    dim: int,
    temperature: int = 10000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    y_coord, x_coord = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y_coord = y_coord.flatten()[:, None] * omega[None, :]
    x_coord = x_coord.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x_coord.sin(), x_coord.cos(), y_coord.sin(), y_coord.cos()), dim=1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    """Transformer feed-forward block."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class ORAttention(nn.Module):
    """Self-attention plus ORFormer messenger-token attention."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.to_ORq = nn.Linear(dim, inner_dim, bias=False)

    def _heads(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, tokens, _channels = inputs.shape
        return inputs.view(batch, tokens, self.heads, self.dim_head).permute(0, 2, 1, 3)

    def _merge_heads(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, _heads, tokens, _dim = inputs.shape
        return inputs.permute(0, 2, 1, 3).reshape(batch, tokens, self.heads * self.dim_head)

    def forward(
        self,
        inputs: torch.Tensor,
        or_query: torch.Tensor,
        alpha: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.norm(inputs)
        qkv = self.to_qkv(inputs).chunk(3, dim=-1)
        q_tensor, k_tensor, v_tensor = (self._heads(item) for item in qkv)
        dots = torch.matmul(q_tensor, k_tensor.transpose(-1, -2)) * self.scale
        out = self._merge_heads(torch.matmul(self.attend(dots), v_tensor))

        or_query = self.norm(or_query)
        or_q = self._heads(self.to_ORq(or_query))
        or_dots = torch.matmul(or_q, k_tensor.transpose(-1, -2)) * self.scale
        tokens = or_dots.shape[2]
        # TorchInductor on MPS is brittle with a boolean ``torch.eye`` generated inside the
        # compiled graph. Keep the mask numeric so the diagonal suppression stays broadcastable
        # without introducing the problematic bool pointwise kernel.
        mask = 1.0 - torch.eye(tokens, device=or_dots.device, dtype=or_dots.dtype)
        or_dots = or_dots * mask
        if alpha is not None:
            repeat_alpha = alpha.unsqueeze(1).repeat(1, or_dots.shape[1], 1, 1)
            or_dots = or_dots * (1 - repeat_alpha.permute(0, 1, 3, 2))
        or_out = self._merge_heads(torch.matmul(self.attend(or_dots), v_tensor))
        return self.to_out(out), self.to_out(or_out)


class ORTransformer(nn.Module):
    """ORFormer transformer stack."""

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ORAttention(dim, heads=heads, dim_head=dim_head),
                        FeedForward(dim, mlp_dim),
                        nn.Linear(dim, 1),
                    ]
                )
                for _idx in range(depth)
            ]
        )

    def forward(
        self, inputs: torch.Tensor, or_query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha = None
        attention_weights = None
        for attn, ff_layer, occlusion_head in self.layers:
            x_attn, or_query_attn = attn(inputs, or_query, alpha)
            inputs = ff_layer(x_attn + inputs) + x_attn + inputs
            or_x = ff_layer(or_query_attn) + or_query_attn
            norm_x = self.norm(inputs)
            norm_or_x = self.norm(or_x)
            alpha = occlusion_head(torch.square(norm_x - norm_or_x)).sigmoid()
            attention_weights = alpha
        assert attention_weights is not None
        return norm_x, norm_or_x, attention_weights


class PatchRearrange(nn.Module):
    """Parameter-free replacement for upstream einops Rearrange patch extraction."""

    def __init__(self, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Equivalent to einops `Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)')` from upstream.
        batch, channels, height, width = inputs.shape
        patch = self.patch_size
        patches = inputs.reshape(batch, channels, height // patch, patch, width // patch, patch)
        return patches.permute(0, 2, 4, 3, 5, 1).reshape(batch, -1, patch * patch * channels)


class ORFormerTokens(nn.Module):
    """ORFormer token predictor without external einops dependency."""

    def __init__(
        self,
        image_size: int = 16,
        patch_size: int = 1,
        num_classes: int = 2048,
        dim: int = 256,
        depth: int = 3,
        heads: int = 8,
        mlp_dim: int = 512,
        channels: int = 256,
        dim_head: int = 64,
    ) -> None:
        super().__init__()
        assert image_size % patch_size == 0
        patch_dim = channels * patch_size * patch_size
        num_patches = (image_size // patch_size) ** 2
        self.to_patch_embedding = nn.Sequential(
            PatchRearrange(patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.register_buffer(
            "pos_embedding",
            _posemb_sincos_2d(image_size // patch_size, image_size // patch_size, dim),
            persistent=False,
        )
        self.ORquery = nn.Parameter(torch.randn(1, num_patches, dim))
        self.transformer = ORTransformer(dim, depth, heads, dim_head, mlp_dim)
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = inputs.shape[0]
        x_tensor = self.to_patch_embedding(inputs)
        x_tensor = x_tensor + self.pos_embedding.to(inputs.device, dtype=x_tensor.dtype)
        target = self.ORquery.repeat(batch, 1, 1).to(inputs.device)
        x_tensor, or_x, alpha = self.transformer(x_tensor, target)
        return self.linear_head(x_tensor), self.linear_head(or_x), alpha, alpha


class VectorQuantizer(nn.Module):
    """ORFormer vector quantizer inference path."""

    def __init__(self, n_e: int, e_dim: int, c_dim: int, beta: float) -> None:
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.c_dim = c_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.n_e, self.c_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z_tensor: torch.Tensor, vit: ORFormerTokens) -> torch.Tensor:
        predicted, or_predicted, or_portion, _attention = vit(z_tensor)
        z_tensor = z_tensor.permute(0, 2, 3, 1).contiguous()
        min_indices = predicted.argmax(dim=-1).reshape(-1, 1)
        min_encodings = torch.zeros(min_indices.shape[0], self.n_e, device=z_tensor.device)
        min_encodings.scatter_(1, min_indices, 1)
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z_tensor.shape)

        or_indices = or_predicted.argmax(dim=-1).reshape(-1, 1)
        or_encodings = torch.zeros(or_indices.shape[0], self.n_e, device=z_tensor.device)
        or_encodings.scatter_(1, or_indices, 1)
        or_z_q = torch.matmul(or_encodings, self.embedding.weight).view(z_tensor.shape)

        or_portion = or_portion.view(z_tensor.shape[0], z_tensor.shape[1], z_tensor.shape[2], 1)
        merged = or_portion * or_z_q + (1 - or_portion) * z_q
        return merged.permute(0, 3, 1, 2).contiguous()


class VQVAE(nn.Module):
    """VQ-VAE plus ORFormer that produces edge heatmaps."""

    def __init__(self, output_dim: int, vit: ORFormerTokens) -> None:
        super().__init__()
        self.encoder = Encoder(3, 128, 2, 32)
        self.pre_quantization_conv = nn.Conv2d(128, 256, kernel_size=1, stride=1)
        self.vector_quantization = VectorQuantizer(2048, 256, 256, 0.25)
        self.decoder = Decoder(256, 128, 2, 32, output_dim)
        self.vit = vit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        z_e = self.pre_quantization_conv(self.encoder(inputs))
        z_q = self.vector_quantization(z_e, self.vit)
        return self.decoder(z_q)


def _validate_orformer_checkpoint(result: T.Any) -> None:
    """Raise if the VQ-VAE checkpoint has unapproved missing or unexpected keys.

    Allowlisted mismatches are training-only artefacts that are expected and safe
    to ignore at inference time.  Any key outside the allowlists indicates real
    checkpoint drift and must not be silently skipped.
    """
    missing = set(result.missing_keys)
    unexpected = set(result.unexpected_keys)
    bad_missing = missing - _ALLOWED_ORFORMER_MISSING
    bad_unexpected = unexpected - _ALLOWED_ORFORMER_UNEXPECTED
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "Unexpected ORFormer checkpoint mismatch. "
            f"missing={sorted(bad_missing)}, "
            f"unexpected={sorted(bad_unexpected)}. "
            "Update _ALLOWED_ORFORMER_MISSING / _ALLOWED_ORFORMER_UNEXPECTED in "
            "plugins/extract/align/_orformer/model.py after confirming keys are "
            "training-only and not required for inference."
        )
    if missing or unexpected:
        logger.warning(
            "ORFormer checkpoint loaded with allowlisted mismatch: missing=%s unexpected=%s",
            sorted(missing),
            sorted(unexpected),
        )


# Confirmed empty after loading both official checkpoints (WFLW 77b1646c, 300W 147b16fc).
# Both load with zero missing and zero unexpected keys under strict=False.
# Any key added here in future must be confirmed training-only and documented with a comment.
_ALLOWED_ORFORMER_MISSING: frozenset[str] = frozenset()

# Confirmed empty after loading both official checkpoints (WFLW 77b1646c, 300W 147b16fc).
# Both load with zero missing and zero unexpected keys under strict=False.
# Any key added here in future must be confirmed training-only and documented with a comment.
_ALLOWED_ORFORMER_UNEXPECTED: frozenset[str] = frozenset()


class ORFormerFaceswapModel(nn.Module):
    """Single-task wrapper: return only 68/98 normalized landmark coordinates."""

    def __init__(
        self,
        num_points: int,
        num_edges: int,
        edge_info: EdgeInfo,
        orformer_weights_path: str,
    ) -> None:
        super().__init__()
        self.hgnet = IntegrationStackedHGNet(
            num_points=num_points, num_edges=num_edges, edge_info=edge_info
        )
        self.orformer = VQVAE(output_dim=num_edges, vit=ORFormerTokens())
        orformer_state = torch.load(orformer_weights_path, map_location="cpu", weights_only=True)
        result = self.orformer.load_state_dict(orformer_state, strict=False)
        _validate_orformer_checkpoint(result)

    def load_state_dict(
        self,
        state_dict: T.Mapping[str, T.Any],
        strict: bool = True,
        assign: bool = False,
    ) -> T.Any:
        """Load HGNet weights from Faceswap's loader. ORFormer weights are loaded at init."""
        return self.hgnet.load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(
        self, inputs: torch.Tensor, reference_input: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run the ORFormer inference pipeline.

        Parameters
        ----------
        inputs:
            Normalised face crop, shape (B, 3, 256, 256).
        reference_input:
            Optional pre-computed 64×64 reference input (B, 3, 64, 64).  When
            provided it is used directly instead of downsampling *inputs* with
            PyTorch bilinear interpolation.  Pass an OpenCV-resized tensor here
            for upstream parity testing; leave as ``None`` for normal inference.
        """
        if reference_input is None:
            # Production path: PyTorch bilinear downsample after normalisation.
            reference_input = F.interpolate(
                inputs, size=(64, 64), mode="bilinear", align_corners=False
            )
        reference_heatmaps = self.orformer(reference_input)
        landmarks = self.hgnet(inputs, reference_heatmaps=reference_heatmaps)
        # Soft-argmax coordinates are in [-1, +1] over a 64-pixel grid whose cell centers span
        # `arange(64) / 63 * 2 - 1`. Map to pixel-index space [0, 63/64] so downstream consumers
        # see normalised landmarks consistent with the rest of Faceswap's aligners.
        return (landmarks + 1.0) * (0.5 * (63.0 / 64.0))


def wflw_edge_info() -> EdgeInfo:
    """Return WFLW-98 edge definitions."""
    return [
        (
            False,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
            ],
        ),
        (False, [33, 34, 35, 36, 37]),
        (False, [38, 39, 40, 41, 33]),
        (False, [42, 43, 44, 45, 46]),
        (False, [46, 47, 48, 49, 50]),
        (False, [51, 52, 53, 54]),
        (False, [55, 56, 57, 58, 59]),
        (False, [60, 61, 62, 63, 64]),
        (False, [64, 65, 66, 67, 60]),
        (False, [68, 69, 70, 71, 72]),
        (False, [72, 73, 74, 75, 68]),
        (False, [76, 77, 78, 79, 80, 81, 82]),
        (False, [82, 83, 84, 85, 86, 87, 76]),
        (False, [88, 89, 90, 91, 92]),
        (False, [92, 93, 94, 95, 88]),
    ]


def w300_edge_info() -> EdgeInfo:
    """Return 300W-68 edge definitions."""
    return [
        (False, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
        (False, [17, 18, 19, 20, 21]),
        (False, [22, 23, 24, 25, 26]),
        (False, [27, 28, 29, 30]),
        (False, [31, 32, 33, 34, 35]),
        (False, [36, 37, 38, 39]),
        (False, [39, 40, 41, 36]),
        (False, [42, 43, 44, 45]),
        (False, [45, 46, 47, 42]),
        (False, [48, 49, 50, 51, 52, 53, 54]),
        (False, [54, 55, 56, 57, 58, 59, 48]),
        (False, [60, 61, 62, 63, 64]),
        (False, [64, 65, 66, 67, 60]),
    ]
