#!/usr/bin/env python3
"""Native Qt Manual Tool overlay + editing integration tests."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QColor, QPixmap

from lib.gui.qt_shell.manual_tool import (
    FrameViewport,
    ManualFrameOverlay,
    ManualToolWindow,
)
from tools.manual.session import ManualEditableAlignments, ManualSession


def _session_with_frames(folder: Path, count: int = 3) -> ManualSession:
    """Write ``count`` small PNG fixtures and return a session."""
    for index in range(count):
        path = folder / f"frame_{index:03d}.png"
        pixmap = QPixmap(64, 48)
        pixmap.fill(QColor("#3366ff"))
        assert pixmap.save(str(path), "PNG")
    return ManualSession.create(frames=str(folder))


def test_frame_viewport_source_to_widget_translates_origin() -> None:
    """source_to_widget maps (0,0) to the target rectangle origin."""
    viewport = FrameViewport(
        source_size=(100, 50),
        target_rect=QRectF(10.0, 20.0, 200.0, 100.0),
        zoom=1.0,
    )
    assert viewport.source_to_widget(0.0, 0.0) == QPointF(10.0, 20.0)
    assert viewport.source_to_widget(100.0, 50.0) == QPointF(210.0, 120.0)


def test_frame_viewport_source_rect_to_widget_scales() -> None:
    """source_rect_to_widget produces a properly scaled QRectF."""
    viewport = FrameViewport(
        source_size=(100, 50),
        target_rect=QRectF(0.0, 0.0, 200.0, 100.0),
        zoom=1.0,
    )
    rect = viewport.source_rect_to_widget(25.0, 10.0, 50.0, 20.0)
    assert rect.x() == 50.0
    assert rect.y() == 20.0
    assert rect.width() == 100.0
    assert rect.height() == 40.0


def test_manual_frame_overlay_set_active_round_trips() -> None:
    """set_active updates active_face without touching the model."""
    model = ManualEditableAlignments()
    overlay = ManualFrameOverlay(model, frame_index_provider=lambda: 0)
    assert overlay.active_face is None
    overlay.set_active(2)
    assert overlay.active_face == 2
    overlay.set_active(None)
    assert overlay.active_face is None


def test_manual_frame_overlay_annotation_visibility_matrix() -> None:
    """Overlay visibility follows editor mode plus annotation toggles."""
    model = ManualEditableAlignments()
    state = {"editor_mode": "View", "annotation_mode": "None"}
    overlay = ManualFrameOverlay(model, frame_index_provider=lambda: 0)
    overlay.install_visibility_providers(
        editor_mode_provider=lambda: state["editor_mode"],
        annotation_mode_provider=lambda: state["annotation_mode"],
    )

    assert overlay.annotation_visibility() == {
        "bbox": False,
        "handles": False,
        "landmarks": False,
        "mask": False,
    }

    state["editor_mode"] = "BoundingBox"
    assert overlay.annotation_visibility() == {
        "bbox": True,
        "handles": True,
        "landmarks": False,
        "mask": False,
    }

    state["editor_mode"] = "Landmarks"
    assert overlay.annotation_visibility() == {
        "bbox": True,
        "handles": False,
        "landmarks": True,
        "mask": False,
    }

    state["editor_mode"] = "View"
    state["annotation_mode"] = "Mask"
    assert overlay.annotation_visibility() == {
        "bbox": True,
        "handles": False,
        "landmarks": False,
        "mask": True,
    }


def test_manual_tool_window_registers_overlay(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """ManualToolWindow installs its overlay on the frame view."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    overlays = window.frame_view._overlays  # noqa: SLF001 - intentional inspection
    assert window.frame_overlay in overlays


def test_manual_tool_overlay_colors_are_editor_configurable(  # type:ignore[no-untyped-def]
    qtbot, tmp_path: Path
) -> None:
    """Per-editor overlay color overrides feed the frame overlay painter."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)

    window.editor_state.set("editor_mode", "Landmarks")
    window.set_editor_overlay_color("Landmarks", "landmark", "#112233")

    assert window.editor_overlay_color("Landmarks", "landmark") == QColor("#112233")
    assert window.frame_overlay._color("landmark") == QColor("#112233")  # noqa: SLF001


def test_add_face_action_appends_to_editable_model(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Triggering add_face puts a new face into the editable model."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    assert window.editable_alignments.face_count(0) == 0

    window.add_face_at_center()

    assert window.editable_alignments.face_count(0) == 1
    assert window.editor_state.edited is True
    assert window.editor_state.unsaved is True
    assert window.editor_state.face_index == 0


def test_delete_face_action_mutates_editable_model(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Triggering delete_active_face removes the face and clamps face_index."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.editable_alignments.add_face(0, (10.0, 10.0, 20.0, 20.0))
    window.editable_alignments.add_face(0, (40.0, 10.0, 20.0, 20.0))
    window.editor_state.set("face_index", 1)

    window.delete_active_face()

    assert window.editable_alignments.face_count(0) == 1
    assert window.editor_state.face_index == 0
    assert window.editor_state.unsaved is True


def test_undo_action_reverts_edit(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """undo_edit reverses the last mutation and toggles redo availability."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)

    window.add_face_at_center()
    assert window.editable_alignments.face_count(0) == 1

    window.undo_edit()
    assert window.editable_alignments.face_count(0) == 0
    assert window.actions_by_key["redo_edit"].isEnabled() is True

    window.redo_edit()
    assert window.editable_alignments.face_count(0) == 1


def test_undo_redo_actions_track_availability(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """The undo/redo actions are disabled when their stacks are empty."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    actions = window.actions_by_key

    assert actions["undo_edit"].isEnabled() is False
    assert actions["redo_edit"].isEnabled() is False

    window.editable_alignments.add_face(0, (10.0, 10.0, 20.0, 20.0))
    assert actions["undo_edit"].isEnabled() is True
    assert actions["redo_edit"].isEnabled() is False

    window.editable_alignments.undo()
    assert actions["undo_edit"].isEnabled() is False
    assert actions["redo_edit"].isEnabled() is True


def test_frame_click_selects_face_under_pointer(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """clicked_at -> _on_frame_clicked hit-tests the editable model."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    # Two bounding boxes; clicking inside the second selects face_index 1.
    window.editable_alignments.add_face(0, (0.0, 0.0, 30.0, 30.0))
    window.editable_alignments.add_face(0, (40.0, 0.0, 20.0, 20.0))

    window._on_frame_clicked(QPointF(45.0, 10.0))  # noqa: SLF001
    # Editor state should reflect the hit-tested face index from the model.
    assert window.editor_state.face_index == 1


def test_nudge_active_face_translates_bbox(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """nudge_active_face moves the active face's bbox and marks the session dirty."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.editable_alignments.add_face(0, (10.0, 10.0, 20.0, 20.0))
    window.editor_state.set("face_index", 0)

    assert window.nudge_active_face(5.0, -3.0) is True
    face = window.editable_alignments.faces(0)[0]
    assert face.bbox == (15.0, 7.0, 20.0, 20.0)
    assert window.editor_state.unsaved is True


def test_revert_clears_editable_stack(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """revert_current_frame undoes all queued edits."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.add_face_at_center()
    window.add_face_at_center()
    assert window.editable_alignments.face_count(0) == 2

    window.revert_current_frame()
    assert window.editable_alignments.face_count(0) == 0
    assert window.editor_state.unsaved is False


def test_face_panel_clear_resets_editor_state_face_index(  # type:ignore[no-untyped-def]
    qtbot,
    tmp_path: Path,
) -> None:
    """Emptying the face panel propagates -1 into editor_state so stale active state clears."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.editor_state.set("face_index", 3)
    assert window.editor_state.face_index == 3

    window.face_panel.set_faces(())

    assert window.editor_state.face_index == -1


def test_revert_only_reverts_current_frame_edits(  # type:ignore[no-untyped-def]
    qtbot,
    tmp_path: Path,
) -> None:
    """revert_current_frame must not touch edits made on other frames."""
    session = _session_with_frames(tmp_path, count=3)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.add_face_at_center()  # frame 0
    window._next_frame()  # noqa: SLF001 - move to frame 1.
    window.add_face_at_center()  # frame 1
    window._next_frame()  # noqa: SLF001 - back to frame 2 via wrap? confirm.

    # Move back to frame 1 and revert only it.
    window._previous_frame()  # noqa: SLF001 - row 1

    window.revert_current_frame()
    assert window.editable_alignments.face_count(0) == 1, "frame 0 must be preserved"
    assert window.editable_alignments.face_count(1) == 0
    assert window.editor_state.unsaved is True  # frame 0 still dirty


def test_delete_last_face_clears_active_face(  # type:ignore[no-untyped-def]
    qtbot,
    tmp_path: Path,
) -> None:
    """Deleting the only face leaves face_index = -1, not 0."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.editable_alignments.add_face(0, (10.0, 10.0, 30.0, 30.0))
    window.editor_state.set("face_index", 0)

    window.delete_active_face()

    assert window.editable_alignments.face_count(0) == 0
    assert window.editor_state.face_index == -1


def test_face_panel_reflects_editable_model_pre_save(  # type:ignore[no-untyped-def]
    qtbot,
    tmp_path: Path,
) -> None:
    """The face panel renders editable-model faces before they are persisted."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)

    window.editable_alignments.add_face(0, (10.0, 10.0, 30.0, 30.0))
    window.editable_alignments.add_face(0, (50.0, 10.0, 20.0, 20.0))

    # No persistence yet, but the panel should already show 2 entries.
    assert len(window.face_panel.faces) == 2
    assert [face.face_index for face in window.face_panel.faces] == [0, 1]


def test_overlay_active_face_tracks_editor_state(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Updating editor_state.face_index syncs the overlay highlight."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.editable_alignments.add_face(0, (10.0, 10.0, 20.0, 20.0))

    window.editor_state.set("face_index", 7)
    assert window.frame_overlay.active_face == 7
    window.editor_state.set("face_index", 0)
    assert window.frame_overlay.active_face == 0


def test_manual_frame_overlay_handle_at_hits_eight_handles() -> None:
    """All eight resize handles are addressable around a bbox."""
    rect = QRectF(100.0, 50.0, 200.0, 100.0)  # x,y,w,h
    # Anchor points (corner + midpoints) should all hit.
    hits = {
        ManualFrameOverlay.handle_at(rect, QPointF(100.0, 50.0)),  # nw
        ManualFrameOverlay.handle_at(rect, QPointF(200.0, 50.0)),  # n
        ManualFrameOverlay.handle_at(rect, QPointF(300.0, 50.0)),  # ne
        ManualFrameOverlay.handle_at(rect, QPointF(300.0, 100.0)),  # e
        ManualFrameOverlay.handle_at(rect, QPointF(300.0, 150.0)),  # se
        ManualFrameOverlay.handle_at(rect, QPointF(200.0, 150.0)),  # s
        ManualFrameOverlay.handle_at(rect, QPointF(100.0, 150.0)),  # sw
        ManualFrameOverlay.handle_at(rect, QPointF(100.0, 100.0)),  # w
    }
    assert hits == {"nw", "n", "ne", "e", "se", "s", "sw", "w"}
    # A point well inside the bbox but away from any handle returns None.
    assert ManualFrameOverlay.handle_at(rect, QPointF(200.0, 100.0)) is None
