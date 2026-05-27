#!/usr/bin/env python3
"""Defaults for the AdaFace identity recognition plugin."""

# pylint:disable=duplicate-code
from lib.config import ConfigItem

HELPTEXT = (
    "AdaFace iResNet101 identity recognition (Kim et al., CVPR 2022).\n"
    "Produces 512-dim L2-normalized embeddings. Weights are loaded from the "
    "minchul/cvlface_adaface_ir101_webface12m Hugging Face repository. "
    "Model code and weights are MIT licensed."
)


batch_size = ConfigItem(
    datatype=int,
    default=8,
    group="settings",
    info="The batch size to use for AdaFace identity embedding extraction.",
    rounding=1,
    min_max=(1, 128),
)

cpu = ConfigItem(
    datatype=bool,
    default=False,
    group="settings",
    info="Force CPU inference for this plugin. AdaFace uses PyTorch/Transformers CVLFace "
    "weights and otherwise prefers the best available Torch backend.",
)
