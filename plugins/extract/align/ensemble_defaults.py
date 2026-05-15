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
    info=(
        "Fusion strategy for combining adapter predictions. Threshold-aware strategies "
        "(``static_weighted_hard_drop``, ``static_weighted_downweight``) consume "
        "``outlier_threshold``; other strategies ignore it."
    ),
    choices=[
        "plain_average",
        "static_weighted",
        "static_weighted_hard_drop",
        "static_weighted_downweight",
        "weighted_median",
    ],
)

reject_outliers = ConfigItem(
    datatype=bool,
    default=True,
    group="settings",
    info=(
        "Deprecated compatibility flag retained so existing configs keep their "
        "historical hard-drop behavior. When ``strategy=static_weighted`` and this is "
        "true, the plugin translates the run to ``static_weighted_hard_drop`` and logs "
        "a deprecation notice. Ignored for every other strategy. New configs should "
        "set ``strategy`` explicitly and leave this disabled."
    ),
)

outlier_threshold = ConfigItem(
    datatype=float,
    default=3.5,
    group="settings",
    info=(
        "Robust z-score threshold used by threshold-aware strategies. Ignored when the "
        "selected strategy does not consume a threshold."
    ),
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
