#!/usr/bin/env python3
"""Qt Manual Tool package-boundary smoke tests."""

from __future__ import annotations

import importlib
from pathlib import Path

from tools.manual import qt


def test_qt_manual_tool_public_exports_live_under_tools_manual_qt() -> None:
    """Public Manual Tool Qt classes are exported from the Manual Tool package."""
    assert qt.ManualToolWindow.__module__ == "tools.manual.qt.window"
    assert qt.ManualFrameView.__module__ == "tools.manual.qt.frame_viewer.frame_view"
    assert qt.ManualFrameOverlay.__module__ == "tools.manual.qt.overlays"
    assert qt.CrossFrameFaceGridPanel.__module__ == "tools.manual.qt.face_viewer.frame"
    assert qt.FaceGridThumbnailRenderer.__module__ == "tools.manual.qt.face_viewer.viewport"
    assert qt.FaceThumbnailPanel.__module__ == "tools.manual.qt.face_viewer.thumbnails"
    assert qt.ManualThumbnailPanel.__module__ == "tools.manual.qt.face_viewer.thumbnails"
    assert qt.ManualTransportBar.__module__ == "tools.manual.qt.transport"
    assert qt.ManualStartupWorker.__module__ == "tools.manual.qt.workers"


def test_qt_manual_tool_focused_modules_import_cleanly() -> None:
    """Focused Qt Manual Tool modules import without relying on the old lib path."""
    module_names = (
        "tools.manual.qt.actions",
        "tools.manual.qt.face_viewer",
        "tools.manual.qt.face_viewer.frame",
        "tools.manual.qt.face_viewer.thumbnails",
        "tools.manual.qt.face_viewer.viewport",
        "tools.manual.qt.filter_controls",
        "tools.manual.qt.frame_viewer.editor.bounding_box",
        "tools.manual.qt.frame_viewer.editor.drag",
        "tools.manual.qt.frame_viewer.editor.extract_box",
        "tools.manual.qt.frame_viewer.editor.landmarks",
        "tools.manual.qt.frame_viewer.editor.mask",
        "tools.manual.qt.frame_viewer.frame_view",
        "tools.manual.qt.overlays",
        "tools.manual.qt.transport",
        "tools.manual.qt.video",
        "tools.manual.qt.window",
        "tools.manual.qt.workers",
    )
    for module_name in module_names:
        assert importlib.import_module(module_name).__name__ == module_name


def test_qt_manual_tool_face_viewer_logic_stays_in_face_viewer_modules() -> None:
    """Face-viewer modules own face-grid widgets, renderers and thumbnails."""
    from tools.manual.qt.face_viewer import frame, thumbnails, viewport

    assert qt.CrossFrameFaceGridPanel is frame.CrossFrameFaceGridPanel
    assert qt.FaceGridThumbnailRenderer is viewport.FaceGridThumbnailRenderer
    assert qt.FaceThumbnailPanel is thumbnails.FaceThumbnailPanel
    assert qt.ManualThumbnailPanel is thumbnails.ManualThumbnailPanel
    assert qt.FaceGridEntry is viewport.FaceGridEntry
    assert qt.FaceGridRenderRequest is viewport.FaceGridRenderRequest


def test_qt_manual_tool_editor_logic_stays_in_editor_modules() -> None:
    """Frame/window modules expose seams while editor modules own editor behavior."""
    from tools.manual.qt.frame_viewer import frame_view
    from tools.manual.qt.frame_viewer.editor import (
        bounding_box,
        drag,
        extract_box,
        landmarks,
        mask,
    )

    assert "_begin_edit_drag" not in frame_view.ManualFrameView.__dict__
    assert "_begin_edit_drag" in bounding_box.BoundingBoxFrameEditorMixin.__dict__
    assert "_resize_bbox" in bounding_box.BoundingBoxFrameEditorMixin.__dict__
    assert "_begin_extract_drag" in extract_box.ExtractBoxFrameEditorMixin.__dict__
    assert "_begin_landmark_drag" in landmarks.LandmarkFrameEditorMixin.__dict__
    assert "_begin_mask_drag" in mask.MaskFrameEditorMixin.__dict__
    assert "_update_edit_drag" in drag.FrameEditDragMixin.__dict__
    assert "_on_face_move_requested" not in qt.ManualToolWindow.__dict__
    assert "_on_face_move_requested" in bounding_box.BoundingBoxWindowEditorMixin.__dict__


def test_legacy_lib_qt_shell_manual_tool_module_is_removed() -> None:
    """The old lib.gui.qt_shell Manual Tool implementation is not a shim."""
    assert not Path("lib/gui/qt_shell/manual_tool.py").exists()
