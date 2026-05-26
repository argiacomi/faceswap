#!/usr/bin/env python3
"""Qt Manual Tool root-window composition."""

from __future__ import annotations

from .state import WindowStateMixin
from .window_impl import ManualToolWindow as _ManualToolWindowBase


class ManualToolWindow(WindowStateMixin, _ManualToolWindowBase):
    """Qt Manual Tool root window assembled from focused behavior mixins."""


__all__ = ["ManualToolWindow"]
