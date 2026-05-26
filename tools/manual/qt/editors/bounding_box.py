#!/usr/bin/env python3
"""Bounding Box editor mode helpers."""

from __future__ import annotations

EDITOR_MODE = "BoundingBox"


def is_active(editor_mode: str) -> bool:
    """Return whether ``editor_mode`` selects this editor."""
    return editor_mode == EDITOR_MODE
