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
    default=16,
    group="settings",
    info="The batch size to use for the ensemble wrapper.",
    rounding=1,
    min_max=(1, 128),
)

models = ConfigItem(
    datatype=list,
    default=["fan", "hrnet", "spiga", "orformer"],
    group="settings",
    info="Aligner adapters to try when the ensemble plugin is loaded.",
    choices=["fan", "hrnet", "spiga", "orformer"],
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

# NOTE: setup_path, weights_path, setup_mode, and resolver_scorer_path used
# to be user-visible config knobs. They are now deployment artifacts the
# landmark resolver pipeline installs into a known location
# (.fs_cache/landmark_ensemble/current/ by default, overridable via the
# FACESWAP_LANDMARK_ENSEMBLE_ARTIFACTS env var). The runtime reads them from
# the installed bundle in plugins/extract/align/ensemble.py, so they no
# longer belong in the user-editable config schema. Removing them prevents
# extract.ini from carrying a path that no longer matches the artifact on
# disk — the historical cause of resolver_policy ↔ scorer_model_type
# mismatches.

resolver_policy = ConfigItem(
    datatype=str,
    default="roll_aware_veto",
    group="landmark_ensemble",
    info=(
        "Production runtime resolver policy. ``roll_aware_veto`` applies conservative "
        "roll/geometry safety checks and then chooses from configured candidate priority. "
        "``learned_quality_v3`` scores geometry-valid candidates with the installed "
        "scorer artifact and chooses "
        "the lowest predicted risk/cost."
    ),
    choices=[
        "roll_aware_veto",
        "learned_quality_v3",
    ],
)

use_alignment_resolver = ConfigItem(
    datatype=bool,
    default=False,
    group="landmark_ensemble",
    info=(
        "Enable the production runtime resolver. When on, per-face fusion is routed "
        "through ``lib.landmarks.ensemble.runtime_resolver`` which builds single-model "
        "and fusion candidates, estimates pose/geometry buckets, applies v7 bucket "
        "priority and vetoes, and emits debug metadata."
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

hard_disagreement_px = ConfigItem(
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
        "Strategy used when the promoted setup fails strict validation and falls back to "
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
    default=False,
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
