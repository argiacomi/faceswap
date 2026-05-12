#!/usr/bin/env python3
"""Default options for the landmark ensemble aligner plugin."""

from lib.config import ConfigItem

HELPTEXT = (
    "Landmark ensemble aligner options.\n"
    "Runs available aligner adapters, converts predictions to canonical 68-point frame "
    "coordinates for fusion, then returns Faceswap-compatible normalized landmarks."
)


batch_size = ConfigItem(
    datatype=int,
    default=8,
    group="settings",
    info="The batch size to use for the ensemble wrapper.",
    rounding=1,
    min_max=(1, 128),
)

models = ConfigItem(
    datatype=list,
    default=["hrnet", "spiga", "orformer"],
    group="settings",
    info="Aligner adapters to try when the ensemble plugin is loaded.",
    choices=["hrnet", "spiga", "orformer"],
)

crop_scale = ConfigItem(
    datatype=float,
    default=1.6,
    group="settings",
    info="Square crop scale relative to the detected face box's longest side.",
    rounding=2,
    min_max=(1.0, 3.0),
)

strategy = ConfigItem(
    datatype=str,
    default="static_weighted",
    group="settings",
    info="Fusion strategy for combining adapter predictions.",
    choices=["plain_average", "static_weighted"],
)

reject_outliers = ConfigItem(
    datatype=bool,
    default=True,
    group="settings",
    info="Reject adapter predictions that are far from the ensemble median before fusion.",
)

outlier_threshold = ConfigItem(
    datatype=float,
    default=3.5,
    group="settings",
    info="Robust z-score threshold used when outlier rejection is enabled.",
    rounding=2,
    min_max=(0.5, 20.0),
)

min_models = ConfigItem(
    datatype=int,
    default=1,
    group="settings",
    info="Minimum successful adapter predictions required for each face.",
    rounding=1,
    min_max=(1, 3),
)
