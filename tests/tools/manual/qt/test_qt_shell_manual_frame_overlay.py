#!/usr/bin/env python3
"""Qt Manual Tool frame-overlay parity tests."""

from __future__ import annotations

from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage, QPainter

from tools.manual.qt.frame_viewer.overlays import ManualFrameOverlay
from tools.manual.qt.frame_viewer.viewport import FrameViewport
from tools.manual.session import ManualEditableAlignments


def _overlay_for(mode: str, annotation: str = "") -> ManualFrameOverlay:
    overlay = ManualFrameOverlay(
        ManualEditableAlignments(),
        frame_index_provider=lambda: 0,
    )
    overlay.install_visibility_providers(
        editor_mode_provider=lambda: mode,
        annotation_mode_provider=lambda: annotation,
    )
    return overlay


def test_qt_manual_frame_overlay_uses_legacy_annotation_colors() -> None:
    """Frame overlay defaults match the legacy Tk annotation colors."""
    overlay = _overlay_for("View")

    assert overlay._color("bbox").name() == "#0000ff"
    assert overlay._color("extract").name() == "#00ff00"
    assert overlay._color("landmark").name() == "#ff00ff"
    assert overlay._color("mesh").name() == "#00ffff"
    assert overlay._color("mask").name() == "#ff0000"


def test_qt_manual_frame_overlay_matches_legacy_mode_display_matrix() -> None:
    """F1-F5 overlay defaults match the legacy editor display rules."""
    assert _overlay_for("View").annotation_visibility() == {
        "bbox": True,
        "extract": True,
        "handles": False,
        "landmarks": True,
        "mesh": True,
        "mask": False,
    }
    assert _overlay_for("BoundingBox").annotation_visibility() == {
        "bbox": True,
        "extract": False,
        "handles": True,
        "landmarks": False,
        "mesh": True,
        "mask": False,
    }
    assert _overlay_for("ExtractBox").annotation_visibility() == {
        "bbox": False,
        "extract": True,
        "handles": True,
        "landmarks": False,
        "mesh": True,
        "mask": False,
    }
    assert _overlay_for("Landmarks").annotation_visibility() == {
        "bbox": False,
        "extract": True,
        "handles": False,
        "landmarks": True,
        "mesh": True,
        "mask": False,
    }
    assert _overlay_for("Mask").annotation_visibility() == {
        "bbox": False,
        "extract": False,
        "handles": False,
        "landmarks": False,
        "mesh": False,
        "mask": True,
    }


def test_qt_manual_frame_overlay_uses_landmark_parts_for_mesh() -> None:
    """Mesh groups use LANDMARK_PARTS rather than one disconnected point list."""
    landmarks = tuple((float(index), float(index % 17)) for index in range(68))

    groups = ManualFrameOverlay.landmark_part_groups(landmarks)

    assert groups["polygon"]
    assert groups["line"]
    assert all(len(group) > 1 for group in groups["polygon"] + groups["line"])


def test_bbox_mode_draws_handles_for_all_visible_faces() -> None:
    """BBox mode exposes anchors on every visible face, not only the active one."""
    editable = ManualEditableAlignments()
    editable.add_face(0, (10.0, 10.0, 20.0, 20.0))
    editable.add_face(0, (40.0, 10.0, 20.0, 20.0))
    overlay = ManualFrameOverlay(editable, frame_index_provider=lambda: 0)
    overlay.set_active(0)
    overlay.install_visibility_providers(
        editor_mode_provider=lambda: "BoundingBox",
        annotation_mode_provider=lambda: "",
    )
    calls: list[QRectF] = []
    overlay._draw_handles = lambda _painter, rect: calls.append(QRectF(rect))  # type: ignore[assignment]  # type:ignore[method-assign] # noqa:SLF001
    image = QImage(80, 40, QImage.Format_ARGB32)
    painter = QPainter(image)
    try:
        overlay(
            painter,
            FrameViewport(source_size=(80, 40), target_rect=QRectF(0, 0, 80, 40), zoom=1.0),
        )
    finally:
        painter.end()

    assert len(calls) == 2
