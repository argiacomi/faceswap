#!/usr/bin/env python3
"""Defaults for the InsightFace identity recognition plugin."""

# pylint:disable=duplicate-code
from lib.config import ConfigItem

HELPTEXT = (
    "InsightFace identity recognition.\n"
    "Exposes one plugin with a configurable model type. Supported packs are antelopev2, "
    "buffalo_l, and buffalo_sc. These upstream InsightFace recognition packs are generally "
    "distributed for research/non-commercial use; check upstream terms before use."
)


batch_size = ConfigItem(
    datatype=int,
    default=16,
    group="settings",
    info="The batch size to use for InsightFace identity embedding extraction.",
    rounding=1,
    min_max=(1, 256),
)

cpu = ConfigItem(
    datatype=bool,
    default=False,
    group="settings",
    info="Force CPU inference for this plugin. InsightFace loads on CPU by default for "
    "cross-platform compatibility.",
)

model_type = ConfigItem(
    datatype=str,
    default="buffalo_l",
    group="settings",
    info="The InsightFace recognition pack to use."
    "\n\tantelopev2 - ArcFace R100 pack."
    "\n\tbuffalo_l - ArcFace R100 pack; good general-purpose default."
    "\n\tbuffalo_sc - MobileFaceNet pack for lower-resource systems.",
    choices=["antelopev2", "buffalo_l", "buffalo_sc"],
    gui_radio=True,
)
