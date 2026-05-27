#!/usr/bin/env python3
"""Defaults for the ArcFace identity recognition plugin."""

# pylint:disable=duplicate-code
from lib.config import ConfigItem

HELPTEXT = (
    "ArcFace iResNet101 identity recognition (Deng et al., CVPR 2019).\n"
    "Produces 512-dim L2-normalized embeddings. Weights are loaded from the "
    "minchul/cvlface_arcface_ir101_webface4m Hugging Face repository. "
    "Check upstream terms before use."
)


batch_size = ConfigItem(
    datatype=int,
    default=8,
    group="settings",
    info="The batch size to use for ArcFace identity embedding extraction.",
    rounding=1,
    min_max=(1, 128),
)

cpu = ConfigItem(
    datatype=bool,
    default=False,
    group="settings",
    info="Force CPU inference for this plugin. ArcFace uses PyTorch/Transformers CVLFace "
    "weights and otherwise prefers the best available Torch backend.",
)
