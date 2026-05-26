#!/usr/bin/env python3
"""Public Qt Manual Tool root-window import surface."""

from __future__ import annotations

from .window_impl import ManualToolWindow

# Keep the public class surface anchored at tools.manual.qt.window while the
# implementation lives in the smaller, movable window_impl module.
ManualToolWindow.__module__ = __name__

__all__ = ["ManualToolWindow"]
