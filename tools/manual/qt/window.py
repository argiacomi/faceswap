#!/usr/bin/env python3
"""Qt Manual Tool root-window composition."""

from __future__ import annotations

from .controls import ControlsMixin
from .face_browser import FaceBrowserMixin
from .layout import LayoutMixin
from .lifecycle import LifecycleMixin
from .navigation import NavigationMixin
from .persistence import PersistenceMixin
from .startup import StartupMixin
from .state import WindowStateMixin
from .window_impl import ManualToolWindow as _ManualToolWindowBase


class ManualToolWindow(
    LifecycleMixin,
    PersistenceMixin,
    StartupMixin,
    NavigationMixin,
    FaceBrowserMixin,
    ControlsMixin,
    LayoutMixin,
    WindowStateMixin,
    _ManualToolWindowBase,
):
    """Qt Manual Tool root window assembled from focused behavior mixins."""


__all__ = ["ManualToolWindow"]
