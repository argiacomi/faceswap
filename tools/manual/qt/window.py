#!/usr/bin/env python3
"""Qt Manual Tool root-window composition."""

from __future__ import annotations

from .controls import ControlsMixin
from .core import ManualToolWindow as _ManualToolWindowBase
from .face_browser import FaceBrowserMixin
from .layout import LayoutMixin
from .lifecycle import LifecycleMixin
from .navigation import NavigationMixin
from .persistence import PersistenceMixin
from .startup import StartupMixin
from .state import WindowStateMixin


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

    def _maybe_run_aligner(self, face_index: int, *, aligner_name: str | None = None) -> None:
        """Run post-commit BBox alignment when the Auto-run control allows it."""
        if self._editor_state.editor_mode != "BoundingBox":
            return
        if not self._editor_state.aligner_auto_run:
            return
        self.rerun_aligner_for_face(int(face_index), aligner_name=aligner_name)


__all__ = ["ManualToolWindow"]
