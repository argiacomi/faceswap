#!/usr/bin/env python3
"""Shared aligner plugin helpers for the manual tool."""

from __future__ import annotations

from plugins.plugin_loader import PluginLoader

_DISPLAY_NAMES = {
    "cv2-dnn": "cv2-dnn",
    "fan": "FAN",
    "hrnet": "HRNet",
    "orformer": "ORFormer",
    "spiga": "SPIGA",
}


def available_aligners() -> list[str]:
    """Return all aligner plugins using display labels suitable for the manual UI."""
    return [
        _DISPLAY_NAMES.get(name, name.title())
        for name in PluginLoader.get_available_extractors("align")
    ]


def default_aligner() -> str:
    """Return the preferred manual tool aligner, falling back to the first available plugin."""
    aligners = available_aligners()
    return "HRNet" if "HRNet" in aligners else aligners[0]
