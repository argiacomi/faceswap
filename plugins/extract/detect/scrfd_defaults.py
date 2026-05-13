#!/usr/bin/env python3
"""The default options for the faceswap SCRFD Detect plugin.

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
    "InsightFace SCRFD Detector options.\n"
    "PyTorch face detector based on InsightFace SCRFD-10GF and SCRFD-34GF models."
)

cpu = ConfigItem(
    datatype=bool,
    default=False,
    group="settings",
    info="Enable CPU mode here to force PyTorch to use the CPU for this detector.",
)

model = ConfigItem(
    datatype=str,
    default="10g",
    group="settings",
    info="The SCRFD model to use. 10G is the default high quality detector. 34G is heavier and "
    "more accurate, especially for difficult faces.",
    choices=["10g", "34g"],
    gui_radio=True,
)

confidence = ConfigItem(
    datatype=int,
    default=50,
    group="settings",
    info="The confidence level at which the detector has successfully found a face.\n"
    "Higher levels will be more discriminating, lower levels will have more false "
    "positives.",
    rounding=5,
    min_max=(25, 100),
)

batch_size = ConfigItem(
    datatype=int,
    default=4,
    group="settings",
    info="The batch size to use. To a point, higher batch sizes equal better performance, "
    "but setting it too high can harm performance.",
    rounding=1,
    min_max=(1, 128),
)

scrfd_postprocess = ConfigItem(
    datatype=str,
    default="auto",
    group="settings",
    info="Select the post-processing path for SCRFD detections. 'auto' prefers the Torch path "
    "when an accelerated device is active and falls back to NumPy otherwise. 'torch' "
    "forces the Torch decode/threshold/NMS path for debugging. 'numpy' preserves the "
    "legacy NumPy implementation.",
    choices=["auto", "torch", "numpy"],
    gui_radio=True,
)
