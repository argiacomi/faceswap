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

setup_path = ConfigItem(
    datatype=str,
    default="best_setup.json",
    group="landmark_ensemble",
    info=(
        "Optional path to a promoted ``best_setup.json`` (#71). When set, the plugin "
        "loads the setup and uses its strategy / outlier threshold / per-landmark "
        "weights for fusion."
    ),
)

weights_path = ConfigItem(
    datatype=str,
    default="best_weights.json",
    group="landmark_ensemble",
    info=(
        "Static per-landmark weight JSON for production ensemble fusion. This is used by "
        "runtime resolver configs and by tooling that promotes static weights separately "
        "from a full setup artifact."
    ),
)

setup_mode = ConfigItem(
    datatype=str,
    default="strict",
    group="landmark_ensemble",
    info=(
        "How to consume ``setup_path``. ``off`` ignores it. ``strict`` hard-fails on "
        "any incompatible artifact. ``fallback`` logs a warning and falls back to the "
        "configured ``strategy`` when the artifact is unusable."
    ),
    choices=["off", "strict", "fallback"],
)

resolver_policy = ConfigItem(
    datatype=str,
    default="roll_aware_veto",
    group="landmark_ensemble",
    info=(
        "Production runtime resolver policy. ``roll_aware_veto`` applies conservative "
        "roll/geometry safety checks and then chooses from configured candidate priority."
    ),
    choices=["roll_aware_veto"],
)

use_alignment_resolver = ConfigItem(
    datatype=bool,
    default=True,
    group="landmark_ensemble",
    info=(
        "Enable the geometry-risk alignment resolver. When on, per-face fusion is routed "
        "through ``lib.landmarks.ensemble.alignment_resolver`` which uses candidate "
        "disagreement, bbox-aspect, and validity flags to pick between the general "
        "strategy, hard-case strategy, or fallback."
    ),
)

hard_case_strategy = ConfigItem(
    datatype=str,
    default="static_weighted_downweight",
    group="landmark_ensemble",
    info="Primary strategy the resolver uses for hard or high-risk faces.",
    choices=[
        "plain_average",
        "static_weighted",
        "static_weighted_hard_drop",
        "static_weighted_downweight",
        "weighted_median",
    ],
)

secondary_hard_case_strategy = ConfigItem(
    datatype=str,
    default="static_weighted_hard_drop",
    group="landmark_ensemble",
    info=(
        "Secondary hard-case strategy retained in runtime debug/config metadata. The current "
        "resolver uses the primary hard-case strategy first and falls through to configured "
        "fallback when needed."
    ),
    choices=[
        "plain_average",
        "static_weighted",
        "static_weighted_hard_drop",
        "static_weighted_downweight",
        "weighted_median",
    ],
)

resolver_hard_case_strategy = ConfigItem(
    datatype=str,
    default="static_weighted_downweight",
    group="landmark_ensemble",
    info=(
        "Backward-compatible alias for ``hard_case_strategy``. New configs should use "
        "``hard_case_strategy``."
    ),
    choices=[
        "plain_average",
        "static_weighted",
        "static_weighted_hard_drop",
        "static_weighted_downweight",
        "weighted_median",
    ],
)

resolver_high_disagreement_px = ConfigItem(
    datatype=float,
    default=12.0,
    group="landmark_ensemble",
    info="Inter-model disagreement (px) above which the resolver routes to the hard-case path.",
    rounding=1,
    min_max=(1.0, 100.0),
)

fallback_model = ConfigItem(
    datatype=str,
    default="orformer",
    group="landmark_ensemble",
    info="Single-model fallback to prefer when the resolver cannot safely use fusion.",
    choices=["hrnet", "spiga", "orformer"],
)

fallback_strategy = ConfigItem(
    datatype=str,
    default="plain_average",
    group="landmark_ensemble",
    info=(
        "Strategy used when ``setup_mode=fallback`` and the promoted setup fails to "
        "load. Set to ``adapter_config`` to fall back to the ``strategy`` field above; "
        "any other value names a canonical strategy directly."
    ),
    choices=[
        "plain_average",
        "static_weighted",
        "static_weighted_hard_drop",
        "static_weighted_downweight",
        "weighted_median",
        "adapter_config",
    ],
)

strict = ConfigItem(
    datatype=bool,
    default=True,
    group="landmark_ensemble",
    info=(
        "Fail fast when configured setup/weights/adapters are unavailable. Disable to allow "
        "fallback behavior when enough single-model predictions remain available."
    ),
)

roll_veto_degrees = ConfigItem(
    datatype=float,
    default=15.0,
    group="landmark_ensemble",
    info="Reject fusion candidates whose roll estimate differs from consensus by more than this.",
    min_max=(0.0, 90.0),
    rounding=1,
)

hard_roll_degrees = ConfigItem(
    datatype=float,
    default=30.0,
    group="landmark_ensemble",
    info="Faces with absolute consensus roll above this are treated as hard-case candidates.",
    min_max=(0.0, 90.0),
    rounding=1,
)
