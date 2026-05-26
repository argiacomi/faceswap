#!/usr/bin/env python3
"""Compatibility wrapper for Qt Manual Tool frame-viewer overlays."""

from __future__ import annotations

from .frame_viewer.overlays import ManualFrameOverlay
from .frame_viewer.viewport import FrameViewport

__all__ = ["FrameViewport", "ManualFrameOverlay"]
