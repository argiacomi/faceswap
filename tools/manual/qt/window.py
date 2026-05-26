#!/usr/bin/env python3
"""Public Qt Manual Tool root-window import surface."""

from __future__ import annotations

from .window_impl import ManualToolWindow as _ManualToolWindowBase
from .window_state import WindowStateMixin


class ManualToolWindow(WindowStateMixin, _ManualToolWindowBase):
    """Qt Manual Tool root window assembled from focused mixins."""


__all__ = ["ManualToolWindow"]
