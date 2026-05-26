#!/usr/bin/env python3
"""Compatibility wrapper for Qt Manual Tool face-viewer thumbnails."""

from __future__ import annotations

from .face_viewer.thumbnails import (
    FaceThumbnailPanel,
    ManualThumbnailPanel,
    _decode_jpeg_to_qimage,
)

__all__ = ["FaceThumbnailPanel", "ManualThumbnailPanel", "_decode_jpeg_to_qimage"]
