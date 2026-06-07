#!/usr/bin/env python3
"""Default options for the CD-ViT aligner plugin."""

from lib.config import ConfigItem

HELPTEXT = (
    "CD-ViT Aligner options.\n"
    "A 68-point staged attention landmark model ported from "
    "argiacomi/AccurateFacialLandmarkDetection."
)


batch_size = ConfigItem(
    datatype=int,
    default=16,
    group="settings",
    info=(
        "The batch size to use. CD-ViT is relatively large, so lower this value if GPU "
        "or system memory is limited."
    ),
    rounding=1,
    min_max=(1, 128),
)

crop_scale = ConfigItem(
    datatype=float,
    default=1.5,
    group="settings",
    info=(
        "Square crop scale relative to the detected face box's longest side. The default "
        "matches the CD-ViT training crop padding."
    ),
    rounding=2,
    min_max=(1.0, 3.0),
)
