#!/usr/bin/env python3
"""The default options for the faceswap ensemble mask plugin."""

from __future__ import annotations

from lib.config import ConfigItem
from plugins.plugin_loader import PluginLoader

HELPTEXT = (
    "Ensemble mask options.\n"
    "Combines two or more semantic mask models using confidence-weighted probability maps."
)


def _source_choices() -> list[str]:
    """Return currently discoverable mask plugins that can be configured as sources."""
    return [
        plugin for plugin in PluginLoader.get_available_extractors("mask") if plugin != "ensemble"
    ]


source_models = ConfigItem(
    datatype=list,
    default=["bisenet-fp", "segnext-fp"],
    group="settings",
    info="Two or more mask models to combine. Source models must expose per-class probability "
    "maps. Models without probability maps will be rejected at runtime with a clear error.",
    choices=_source_choices(),
)

strategy = ConfigItem(
    datatype=str,
    default="confidence-weighted-union",
    group="settings",
    info="How to combine source masks.\n"
    "\n\tconfidence-weighted-union - Preserve mutually supported regions and keep "
    "model-unique regions according to that model's confidence."
    "\n\tconfidence-weighted-intersection - Keep only mutually supported regions, scaled by "
    "model-confidence agreement.",
    choices=["confidence-weighted-union", "confidence-weighted-intersection"],
    gui_radio=True,
)

centering = ConfigItem(
    datatype=str,
    default="face",
    group="settings",
    info="The centering to use for the ensemble input and final stored mask.\n"
    "\n\tface - Store as a face-centered ensemble mask."
    "\n\thead - Store as a head-centered ensemble mask.",
    choices=["face", "head"],
    gui_radio=True,
)

batch_size = ConfigItem(
    datatype=int,
    default=4,
    group="settings",
    info="The batch size to use for ensemble inference. Each configured source model runs for "
    "every batch, so reduce this if you encounter out-of-memory errors.",
    rounding=1,
    min_max=(1, 64),
)
