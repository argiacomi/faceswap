#!/usr/bin/env python3
"""The default options for the faceswap SPIGA Alignments plugin.

Defaults files should be named `<plugin_name>_defaults.py`

Any qualifying items placed into this file will automatically get added to the relevant config
.ini files within the faceswap/config folder and added to the relevant GUI settings page.

The following variable should be defined:

    Parameters
    ----------
    HELPTEXT: str
        A string describing what this plugin does

Further plugin configuration options are assigned using:
>>> <config_item> = ConfigItem(...)

where <config_item> is the name of the configuration option to be added (lower-case, alpha-numeric
+ underscore only) and ConfigItem(...) is the [`~lib.config.objects.ConfigItem`] data for the
option.

See the docstring/ReadtheDocs documentation required parameters for the ConfigItem object.
Items will be grouped together as per their `group` parameter, but otherwise will be processed in
the order that they are added to this module.
from lib.config import ConfigItem
"""

from lib.config import ConfigItem


HELPTEXT = (
    "SPIGA Aligner options.\n"
    "Shape-preserving landmark detector using the upstream SPIGA 300W-68 and WFLW-98 PyTorch "
    "models."
)

model = ConfigItem(
    datatype=str,
    default="wflw",
    group="settings",
    info="The SPIGA model to use. WFLW outputs 98 landmarks and is the best general-purpose "
    "choice for difficult, occluded, or profile faces. 300W and MERL-RAV output the "
    "standard 68-point landmark set; MERL-RAV is trained on real-world face video and "
    "may generalise better to in-the-wild footage.",
    choices=["wflw", "300w", "merlrav"],
    gui_radio=True,
)

crop_scale = ConfigItem(
    datatype=float,
    default=1.60,
    group="settings",
    info="Expansion factor applied to the detector bounding box before cropping the face patch "
    "fed to SPIGA. 1.60 matches the upstream SPIGA default.\n\n"
    "Sensitivity benchmark on 30 curated lapa samples (WFLW model, NME vs reference "
    "landmarks): performance improves steadily from 1.20 (NME 0.454) through 1.40 (0.402) "
    "then plateaus — 1.50 (0.396) and 1.60 (0.397) are essentially tied. Per category: "
    "1.50 is marginally better for close-up and hair-visible faces; 1.60 is marginally "
    "better for clean frontal faces. The difference is ~0.001 NME, within noise for small "
    "datasets.\n\n"
    "Decision: keep default 1.60 for upstream parity. If you process mostly close-up or "
    "profile faces, 1.50 may give slightly tighter landmarks. Going below 1.40 measurably "
    "hurts accuracy.",
    rounding=2,
    min_max=(1.0, 2.0),
)

batch_size = ConfigItem(
    datatype=int,
    default=4,
    group="settings",
    info="The batch size to use. SPIGA is memory intensive, so reduce this if you encounter "
    "out-of-memory errors.",
    rounding=1,
    min_max=(1, 64),
)
