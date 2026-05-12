#!/usr/bin/env python3
"""The default options for the faceswap ORFormer Alignments plugin.

Defaults files should be named `<plugin_name>_defaults.py`
"""

from lib.config import ConfigItem


HELPTEXT = (
    "ORFormer Aligner options.\n"
    "Occlusion-robust transformer landmark detector using the official WACV 2025 ORFormer "
    "public weights. The upstream repository does not publish an explicit code or weights "
    "license; use in private forks is documented in the plugin wrapper."
)

model = ConfigItem(
    datatype=str,
    default="wflw",
    group="settings",
    info="The ORFormer model to use. WFLW outputs 98 landmarks and is the better general "
    "purpose choice for difficult or occluded faces. 300W outputs the standard 68-point "
    "landmark set.\n\n"
    "Note: COFW (29 landmarks) is not offered because Faceswap's aligner contract "
    "accepts only 68 or 98 landmarks. MERL-RAV is not an upstream ORFormer checkpoint "
    "option.",
    choices=["wflw", "300w"],
    gui_radio=True,
)

crop_scale = ConfigItem(
    datatype=float,
    default=1.20,
    group="settings",
    info="Expansion factor applied to the detector bounding box before cropping the face "
    "patch fed to ORFormer. Upstream ORFormer uses 1.20 for WFLW/300W training, but "
    "upstream derives its bbox from tight ground-truth landmark extrema, not a face "
    "detector box.\n\n"
    "Sensitivity benchmark on 30 curated lapa samples (WFLW model, true NME vs "
    "reference landmarks): 1.20 is best overall (NME 0.396) and best for close-up and "
    "hair-visible faces. Performance degrades monotonically above 1.20 — reaching 0.416 "
    "at 1.60. For clean frontal faces 1.30 is marginally better (0.420 vs 0.422), but "
    "the difference is within noise. Decision: keep default 1.20.",
    rounding=2,
    min_max=(1.0, 2.0),
)

batch_size = ConfigItem(
    datatype=int,
    default=4,
    group="settings",
    info="The batch size to use. ORFormer runs an integrated HGNet plus transformer heatmap "
    "generator, so reduce this if you encounter out-of-memory errors.",
    rounding=1,
    min_max=(1, 64),
)
