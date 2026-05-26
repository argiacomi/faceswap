#!/usr/bin/env python3
"""Stand-alone PyTorch implementation of the SegNeXt face-parser.

This module reimplements the MSCAN backbone and the LightHam (``Hamburger``) decoder
from https://github.com/e4s2022/SegNeXt-FaceParser without requiring mmseg/mmcv as
runtime dependencies. Submodule names mirror the upstream mmseg layout so the official
CelebAMask-HQ checkpoints load directly via ``load_state_dict`` after stripping the
mmseg ``state_dict``/``meta`` wrapper (handled in the Faceswap base extract loader).

License (Apache 2.0): inherited from the upstream SegNeXt and SegNeXt-FaceParser repos.
"""

from __future__ import annotations

import typing as T
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MSCANConfig:
    """Architecture parameters for one SegNeXt face-parser variant."""

    embed_dims: tuple[int, int, int, int]
    depths: tuple[int, int, int, int]
    mlp_ratios: tuple[int, int, int, int]
    drop_path_rate: float
    decoder_in_channels: tuple[int, int, int]
    decoder_channels: int
    ham_channels: int
    md_r: int


# Settings reproduced from local_configs/segnext/{small,base}/segnext.*.celebamaskhq.160k.py
# in the upstream SegNeXt-FaceParser repo.
SMALL_CONFIG = MSCANConfig(
    embed_dims=(64, 128, 320, 512),
    depths=(2, 2, 4, 2),
    mlp_ratios=(8, 8, 4, 4),
    drop_path_rate=0.1,
    decoder_in_channels=(128, 320, 512),
    decoder_channels=256,
    ham_channels=256,
    md_r=16,
)
BASE_CONFIG = MSCANConfig(
    embed_dims=(64, 128, 320, 512),
    depths=(3, 3, 12, 3),
    mlp_ratios=(8, 8, 4, 4),
    drop_path_rate=0.1,
    decoder_in_channels=(128, 320, 512),
    decoder_channels=512,
    ham_channels=512,
    md_r=16,
)


class _DWConv(nn.Module):
    """Depthwise 3x3 convolution used inside the MLP block of MSCAN."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dwconv(x)


class _Mlp(nn.Module):
    """MSCAN feed-forward block: 1x1 conv -> depthwise 3x3 -> GELU -> 1x1 conv."""

    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = _DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, in_features, 1)
        # Dropout buffers exist in the upstream module but with p=0 they are no-ops at
        # inference, so we omit them to keep state_dicts compatible (no params/buffers).

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class _AttentionModule(nn.Module):
    """Multi-scale convolutional attention with three depthwise strip-conv branches."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x
        attn = self.conv0(x)
        attn_0 = self.conv0_2(self.conv0_1(attn))
        attn_1 = self.conv1_2(self.conv1_1(attn))
        attn_2 = self.conv2_2(self.conv2_1(attn))
        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        return attn * u


class _SpatialAttention(nn.Module):
    """Pre/post 1x1 projections wrapping ``AttentionModule`` with a residual connection."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = _AttentionModule(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut


class _Block(nn.Module):
    """One MSCAN encoder block (spatial attention + feed-forward, with layer scale)."""

    def __init__(self, dim: int, mlp_ratio: int) -> None:
        super().__init__()
        # SyncBN in the upstream config collapses to plain BN on a single device. State dict
        # keys (weight/bias/running_mean/running_var/num_batches_tracked) are identical.
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = _SpatialAttention(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = _Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(torch.full((dim,), layer_scale_init_value))
        self.layer_scale_2 = nn.Parameter(torch.full((dim,), layer_scale_init_value))

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        b, _, c = x.shape
        # (B, N, C) -> (B, C, H, W)
        feat = x.permute(0, 2, 1).view(b, c, height, width)
        feat = feat + self.layer_scale_1.view(1, -1, 1, 1) * self.attn(self.norm1(feat))
        feat = feat + self.layer_scale_2.view(1, -1, 1, 1) * self.mlp(self.norm2(feat))
        return feat.view(b, c, -1).permute(0, 2, 1)


class _StemConv(nn.Module):
    """First-stage patch embedding: two stride-2 convolutions with BN + GELU between."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, h, w = x.shape
        return x.flatten(2).transpose(1, 2), h, w


class _OverlapPatchEmbed(nn.Module):
    """Stages 2-4 patch embedding: 3x3 stride-2 conv followed by BN."""

    def __init__(self, patch_size: int, stride: int, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, h, w = x.shape
        x = self.norm(x)
        return x.flatten(2).transpose(1, 2), h, w


class _MSCAN(nn.Module):
    """Multi-Scale Convolutional Attention Network backbone.

    Submodule names (``patch_embedX``, ``blockX``, ``normX``) match the upstream
    mmseg implementation so the published checkpoints load directly.
    """

    def __init__(self, config: MSCANConfig) -> None:
        super().__init__()
        self.depths = config.depths
        self.num_stages = 4

        for i in range(self.num_stages):
            if i == 0:
                patch_embed: nn.Module = _StemConv(3, config.embed_dims[0])
            else:
                patch_embed = _OverlapPatchEmbed(
                    patch_size=3,
                    stride=2,
                    in_chans=config.embed_dims[i - 1],
                    embed_dim=config.embed_dims[i],
                )
            block = nn.ModuleList(
                [
                    _Block(dim=config.embed_dims[i], mlp_ratio=config.mlp_ratios[i])
                    for _ in range(config.depths[i])
                ]
            )
            norm = nn.LayerNorm(config.embed_dims[i])
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs: list[torch.Tensor] = []
        b = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block: nn.ModuleList = getattr(self, f"block{i + 1}")
            norm: nn.LayerNorm = getattr(self, f"norm{i + 1}")
            x, height, width = patch_embed(x)
            for blk in block:
                x = blk(x, height, width)
            x = norm(x)
            x = x.reshape(b, height, width, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs


class _ConvModule(nn.Module):
    """Conv + GroupNorm(32) + ReLU, mirroring mmcv ``ConvModule`` state-dict layout.

    mmcv's ``build_norm_layer`` assigns the norm submodule a fixed attribute name based
    on its type - ``gn`` for ``GroupNorm`` and ``bn`` for any BatchNorm variant. We mirror
    that here so the upstream ``decode_head.{squeeze,align,...}.gn.{weight,bias}`` keys
    line up with this module's parameters.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        use_norm: bool = True,
        use_act: bool = True,
    ) -> None:
        super().__init__()
        # mmcv ``ConvModule`` sets ``bias=auto``, which becomes ``False`` whenever a
        # normalization layer is attached. We keep the same convention for state dict
        # symmetry - the upstream checkpoints have no conv bias when norm is on.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=not use_norm)
        if use_norm:
            self.gn = nn.GroupNorm(32, out_channels)
        if use_act:
            self.activate = nn.ReLU(inplace=False)
        self._has_norm = use_norm
        self._has_act = use_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self._has_norm:
            x = self.gn(x)
        if self._has_act:
            x = self.activate(x)
        return x


class _NMF2D(nn.Module):
    """Non-negative matrix factorization unit used by the Hamburger decoder.

    The upstream block re-initializes random bases on every forward in eval mode (the
    ``RAND_INIT=True`` default) and runs ``EVAL_STEPS`` multiplicative-update iterations
    of NMF. There are no learned parameters, so checkpoint loading is unaffected.

    Faceswap deviates from upstream in one way: the per-forward random init is seeded
    from a fixed ``torch.Generator`` at the start of every call, so two runs of extract
    on the same input produce byte-identical masks. Numerically equivalent to upstream
    on average - the algorithm is unchanged, only the entropy source is pinned.
    """

    _BASES_SEED = 0xC0FFEE

    def __init__(self, md_r: int) -> None:
        super().__init__()
        self.r_rank = md_r
        self.eval_steps = 7
        self.inv_t = 1  # NMF2D overrides the base default (100) to 1.
        self._generator: torch.Generator | None = None
        self._gen_device: torch.device | None = None

    def _build_bases(self, batch: int, device: torch.device, dim: int) -> torch.Tensor:
        # Lazily allocate a per-device Generator the first time we see this device,
        # then reseed it every call so output is reproducible per (batch, device).
        if self._generator is None or self._gen_device != device:
            self._generator = torch.Generator(device=device)
            self._gen_device = device
        self._generator.manual_seed(self._BASES_SEED)
        bases = torch.rand((batch, dim, self.r_rank), device=device, generator=self._generator)
        return F.normalize(bases, dim=1)

    @staticmethod
    def _multiplicative_update(
        x: torch.Tensor, bases: torch.Tensor, coef: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)
        numerator = torch.bmm(x, coef)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        bases = bases * numerator / (denominator + 1e-6)
        return bases, coef

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Upstream supports an MD_S spatial-group factor; the SegNeXt config never sets
        # it, so it is hardcoded to 1 here and the equivalent (b * 1, c, h*w) reshape
        # collapses to the same view.
        n = h * w
        x_flat = x.view(b, c, n)
        bases = self._build_bases(b, x.device, c)
        coef = torch.bmm(x_flat.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)
        for _ in range(self.eval_steps):
            bases, coef = self._multiplicative_update(x_flat, bases, coef)
        # Final coefficient refinement (one extra MU step matching upstream ``compute_coef``).
        numerator = torch.bmm(x_flat.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)
        out = torch.bmm(bases, coef.transpose(1, 2))
        return out.view(b, c, h, w)


class _Hamburger(nn.Module):
    """Bread + ham + bread block: 1x1 conv, NMF refinement, 1x1 conv with GroupNorm, ReLU."""

    def __init__(self, ham_channels: int, md_r: int) -> None:
        super().__init__()
        self.ham_in = _ConvModule(
            ham_channels, ham_channels, kernel_size=1, use_norm=False, use_act=False
        )
        self.ham = _NMF2D(md_r)
        self.ham_out = _ConvModule(
            ham_channels, ham_channels, kernel_size=1, use_norm=True, use_act=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=False)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        return F.relu(x + enjoy, inplace=False)


class _LightHamHead(nn.Module):
    """LightHamHead decoder: concatenate selected feature levels, squeeze, NMF, classify."""

    def __init__(self, config: MSCANConfig, num_classes: int) -> None:
        super().__init__()
        self.in_channels = config.decoder_in_channels
        self.channels = config.decoder_channels
        self.ham_channels = config.ham_channels
        self.squeeze = _ConvModule(
            sum(config.decoder_in_channels), config.ham_channels, kernel_size=1
        )
        self.hamburger = _Hamburger(config.ham_channels, config.md_r)
        self.align = _ConvModule(config.ham_channels, config.decoder_channels, kernel_size=1)
        # ``conv_seg`` produces per-class logits at 1/8 input resolution.
        self.conv_seg = nn.Conv2d(config.decoder_channels, num_classes, kernel_size=1)
        # A dropout layer with p=0.1 exists in the upstream module. With no learnable
        # parameters and inference always running ``eval()`` it is a pass-through, so we
        # omit it to keep the state_dict compatible (no extra keys).

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        # Upstream config selects ``in_index=[1, 2, 3]`` - stages 2/3/4 from MSCAN.
        selected = features[1:]
        target_size = selected[0].shape[2:]
        upsampled = [
            F.interpolate(level, size=target_size, mode="bilinear", align_corners=False)
            for level in selected
        ]
        x = torch.cat(upsampled, dim=1)
        x = self.squeeze(x)
        x = self.hamburger(x)
        x = self.align(x)
        return self.conv_seg(x)


class SegNeXtFaceParser(nn.Module):
    """SegNeXt face-parser: MSCAN backbone with a LightHam decode head.

    Parameters
    ----------
    config
        The MSCAN architecture variant (see ``SMALL_CONFIG`` / ``BASE_CONFIG``).
    num_classes
        Number of CelebAMask-HQ segmentation classes (19 for the public checkpoints).

    Notes
    -----
    State-dict layout matches ``mmseg.models.segmentors.EncoderDecoder`` with prefixes
    ``backbone.*`` and ``decode_head.*``. The Faceswap base loader handles unwrapping
    ``{"meta": ..., "state_dict": ...}`` checkpoints before calling
    ``load_state_dict``.
    """

    def __init__(self, config: MSCANConfig, num_classes: int = 19) -> None:
        super().__init__()
        self.backbone = _MSCAN(config)
        self.decode_head = _LightHamHead(config, num_classes=num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-class logits at the input spatial resolution.

        The decode head outputs at 1/8 of the input size; we upsample to match the
        Faceswap mask plugin contract (``(N, num_classes, H, W)`` in input space).
        """
        h, w = x.shape[-2:]
        features = self.backbone(x)
        logits = self.decode_head(features)
        return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)


def filter_state_dict(
    state_dict: T.Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Drop auxiliary keys from an mmseg checkpoint that do not exist on this model.

    The upstream mmseg ``BaseDecodeHead`` registers a ``dropout`` module with no
    parameters, so no keys are produced for it; however ``conv_seg`` is created with
    ``num_classes`` set in the config, which for the upstream "base" variant config is
    erroneously left at 150 - it is overridden at training time to 19. Anything that does
    not correspond to a parameter in this module is silently dropped here so a future
    re-trained checkpoint with extra heads still loads.
    """
    return {
        k: v
        for k, v in state_dict.items()
        if not k.startswith(("auxiliary_head.", "decode_head.loss_decode."))
    }


__all__ = [
    "MSCANConfig",
    "SMALL_CONFIG",
    "BASE_CONFIG",
    "SegNeXtFaceParser",
    "filter_state_dict",
]
