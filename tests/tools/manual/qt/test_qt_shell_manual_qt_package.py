#!/usr/bin/env python3
"""Qt Manual Tool package-boundary smoke tests."""

from __future__ import annotations

import importlib
from pathlib import Path

from tools.manual import qt


def test_qt_manual_tool_public_exports_live_under_tools_manual_qt() -> None:
    """Public Manual Tool Qt classes are exported from the Manual Tool package."""
    assert qt.ManualToolWindow.__module__ == "tools.manual.qt.window"
    assert qt.ManualFrameView.__module__ == "tools.manual.qt.frame_view"
    assert qt.ManualFrameOverlay.__module__ == "tools.manual.qt.overlays"
    assert qt.CrossFrameFaceGridPanel.__module__ == "tools.manual.qt.face_grid"
    assert qt.FaceGridThumbnailRenderer.__module__ == "tools.manual.qt.face_grid_renderer"
    assert qt.ManualTransportBar.__module__ == "tools.manual.qt.transport"
    assert qt.ManualStartupWorker.__module__ == "tools.manual.qt.workers"


def test_qt_manual_tool_focused_modules_import_cleanly() -> None:
    """Focused Qt Manual Tool modules import without relying on the old lib path."""
    module_names = (
        "tools.manual.qt.actions",
        "tools.manual.qt.editors.bounding_box",
        "tools.manual.qt.editors.extract_box",
        "tools.manual.qt.editors.landmarks",
        "tools.manual.qt.editors.mask",
        "tools.manual.qt.face_grid",
        "tools.manual.qt.face_grid_renderer",
        "tools.manual.qt.filter_controls",
        "tools.manual.qt.frame_view",
        "tools.manual.qt.overlays",
        "tools.manual.qt.thumbnails",
        "tools.manual.qt.transport",
        "tools.manual.qt.video",
        "tools.manual.qt.window",
        "tools.manual.qt.workers",
    )
    for module_name in module_names:
        assert importlib.import_module(module_name).__name__ == module_name


def test_legacy_lib_qt_shell_manual_tool_module_is_removed() -> None:
    """The old lib.gui.qt_shell Manual Tool implementation is not a shim."""
    assert not Path("lib/gui/qt_shell/manual_tool.py").exists()
