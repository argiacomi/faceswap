#!/usr/bin/env python3
"""Native Qt Manual Tool action surface tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QPixmap

from tools.manual.frame_filter import FILTER_MODES
from tools.manual.qt import MANUAL_ACTIONS, ManualFrameView, ManualToolWindow
from tools.manual.session import FaceThumbnail, ManualSession


def _session_with_frames(folder: Path, count: int = 3) -> ManualSession:
    """Write ``count`` small PNG fixtures and return a session."""
    for index in range(count):
        path = folder / f"frame_{index:03d}.png"
        pixmap = QPixmap(48, 32)
        pixmap.fill(QColor("#3366ff"))
        assert pixmap.save(str(path), "PNG")
    return ManualSession.create(frames=str(folder))


def _face_fixture(face_index: int) -> FaceThumbnail:
    return FaceThumbnail(
        frame_index=0,
        frame_name="frame_000.png",
        face_index=face_index,
        thumbnail_jpeg=b"",
    )


def _source_to_widget(view: ManualFrameView, sx: float, sy: float) -> QPointF:
    target = view._target_rect()  # noqa: SLF001 - test projection helper
    src_w, src_h = view.source_size
    return QPointF(
        target.x() + target.width() * (sx / src_w),
        target.y() + target.height() * (sy / src_h),
    )


def test_manual_action_registry_keys_are_unique() -> None:
    """Each action in the registry has a unique key."""
    keys = [action.key for action in MANUAL_ACTIONS]
    assert sorted(keys) == sorted(set(keys))


def test_manual_action_registry_covers_required_editor_commands() -> None:
    """The registry exposes the full editor command surface from the legacy tool."""
    required = {
        "save",
        "revert_frame",
        "first_frame",
        "previous_frame",
        "next_frame",
        "last_frame",
        "play_pause",
        "copy_prev_face",
        "copy_next_face",
        "delete_face",
        "cycle_filter",
        "cycle_annotation",
        "toggle_mask_annotation",
        "set_view_mode",
        "set_boundingbox_mode",
        "set_extractbox_mode",
        "set_landmarks_mode",
        "set_mask_mode",
        "zoom_in",
        "zoom_out",
        "reset_view",
        "legacy_tool",
    }
    keys = {action.key for action in MANUAL_ACTIONS}
    assert required.issubset(keys), required - keys


def test_manual_action_shortcuts_match_legacy_bindings() -> None:
    """Key legacy shortcuts (Z/X navigation, Ctrl+S, F1-F5, Delete) are bound."""
    by_key = {action.key: action for action in MANUAL_ACTIONS}
    assert "Ctrl+S" in by_key["save"].shortcut
    assert "Z" in by_key["previous_frame"].shortcut
    assert "X" in by_key["next_frame"].shortcut
    assert "Home" in by_key["first_frame"].shortcut
    assert "End" in by_key["last_frame"].shortcut
    assert "Space" in by_key["play_pause"].shortcut
    assert "C" in by_key["copy_prev_face"].shortcut
    assert "V" in by_key["copy_next_face"].shortcut
    assert "Delete" in by_key["delete_face"].shortcut
    assert "F1" in by_key["set_view_mode"].shortcut
    assert "F2" in by_key["set_boundingbox_mode"].shortcut
    assert "F5" in by_key["set_mask_mode"].shortcut
    assert "F9" in by_key["cycle_annotation"].shortcut
    assert "F10" in by_key["toggle_mask_annotation"].shortcut


def test_window_registers_actions_with_shortcuts(qtbot, tmp_path: Path) -> None:
    """Building the window populates the action registry with shortcuts."""
    from PySide6.QtGui import QKeySequence

    session = _session_with_frames(tmp_path, count=3)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    registry = window.actions_by_key
    for spec in MANUAL_ACTIONS:
        assert spec.key in registry, f"Action {spec.key} missing from registry"
        if spec.shortcut:
            shortcuts = {seq.toString() for seq in registry[spec.key].shortcuts()}
            for expected in spec.shortcut:
                normalized = QKeySequence(expected).toString()
                assert normalized in shortcuts, (spec.key, expected, shortcuts)


def test_save_action_disabled_until_dirty(qtbot, tmp_path: Path) -> None:
    """Save starts disabled and enables once the editor state is dirty."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    save_action = window.actions_by_key["save"]
    assert save_action.isEnabled() is False
    window.mark_dirty(True)
    assert save_action.isEnabled() is True


def test_save_success_clears_dirty_state(
    qtbot,
    tmp_path: Path,
) -> None:
    """save() persists edits, clears dirty state and drops the undo history."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.editable_alignments.add_face(0, (10.0, 10.0, 30.0, 30.0))
    assert window.editor_state.unsaved is True
    assert window.editable_alignments.can_undo is True

    assert window.save() is True
    qtbot.waitUntil(lambda: window._save_worker is None, timeout=5000)

    assert window.editor_state.unsaved is False
    assert window.editor_state.edited is False
    assert window.editable_alignments.can_undo is False
    assert window.editable_alignments.can_redo is False
    assert (tmp_path / "alignments.fsa").exists()


def test_navigation_actions_track_thumbnail_position(qtbot, tmp_path: Path) -> None:
    """First/Previous disable at row 0; Next/Last disable at the last row."""
    session = _session_with_frames(tmp_path, count=3)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    actions = window.actions_by_key

    # First row → cannot go back further.
    assert actions["first_frame"].isEnabled() is False
    assert actions["previous_frame"].isEnabled() is False
    assert actions["next_frame"].isEnabled() is True
    assert actions["last_frame"].isEnabled() is True

    window._next_frame()  # noqa: SLF001
    assert actions["first_frame"].isEnabled() is True
    assert actions["previous_frame"].isEnabled() is True
    assert actions["next_frame"].isEnabled() is True

    window.goto_last_frame()
    assert actions["next_frame"].isEnabled() is False
    assert actions["last_frame"].isEnabled() is False
    assert actions["previous_frame"].isEnabled() is True


def test_delete_face_action_requires_face_selection(qtbot, tmp_path: Path) -> None:
    """Delete is disabled when the editable model is empty, enabled when a face exists."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    delete_action = window.actions_by_key["delete_face"]
    assert delete_action.isEnabled() is False

    window.editable_alignments.add_face(0, (10.0, 10.0, 30.0, 30.0))
    assert delete_action.isEnabled() is True


@pytest.mark.parametrize("mode", ("BoundingBox", "ExtractBox"))
def test_delete_key_path_deletes_hovered_frame_face_in_edit_modes(
    qtbot,
    tmp_path: Path,
    mode: str,
) -> None:
    """Delete targets the hovered frame face in BBox and Extract Box modes."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitUntil(lambda: window._frame_view.source_size != (0, 0), timeout=2000)
    first = window.editable_alignments.add_face(0, (5.0, 5.0, 20.0, 20.0))
    second = window.editable_alignments.add_face(0, (25.0, 5.0, 20.0, 20.0))
    window.editor_state.set("face_index", first)
    window.editor_state.set("editor_mode", mode)

    window._frame_view._update_hover_cursor(_source_to_widget(window._frame_view, 30.0, 10.0))
    assert window._frame_view.hovered_face_index == second

    window.actions_by_key["delete_face"].trigger()

    faces = window.editable_alignments.faces(0)
    assert len(faces) == 1
    assert faces[0].bbox == (5.0, 5.0, 20.0, 20.0)


def test_editor_mode_actions_toggle_other_modes(qtbot, tmp_path: Path) -> None:
    """Activating an editor mode disables its own action and enables the others."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    actions = window.actions_by_key

    # Initial editor mode is "View" → set_view_mode disabled, others enabled.
    assert window.editor_state.editor_mode == "View"
    assert actions["set_view_mode"].isEnabled() is False
    assert actions["set_landmarks_mode"].isEnabled() is True

    window.set_editor_landmarks()
    assert window.editor_state.editor_mode == "Landmarks"
    assert actions["set_view_mode"].isEnabled() is True
    assert actions["set_landmarks_mode"].isEnabled() is False


def test_cycle_filter_advances_state(qtbot, tmp_path: Path) -> None:
    """cycle_filter_mode walks the legacy rotation order."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    seen: list[str] = []
    window.editor_state.subscribe("filter_mode", seen.append)

    for _ in range(len(FILTER_MODES) + 1):
        window.cycle_filter_mode()

    assert len(seen) == len(FILTER_MODES) + 1
    # First call leaves the rotation at "Has Face(s)" (next after "All Frames").
    assert seen[0] == "Has Face(s)"
    # After a full rotation through all modes, we land back at "Has Face(s)".
    assert seen[-1] == "Has Face(s)"


def test_copy_prev_face_copies_into_editable_model(
    qtbot,
    tmp_path: Path,
) -> None:
    """copy_prev_face replaces the current frame faces with the previous frame's."""
    session = _session_with_frames(tmp_path, count=3)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.editable_alignments.add_face(0, (5.0, 5.0, 30.0, 30.0))
    window._next_frame()  # noqa: SLF001 - move to frame 1 so prev = frame 0.

    assert window.copy_prev_face() is True

    faces = window.editable_alignments.faces(1)
    assert len(faces) == 1
    assert faces[0].bbox == (5.0, 5.0, 30.0, 30.0)
    assert window.editor_state.unsaved is True
    assert window.editor_state.edited is True
    assert window.actions_by_key["save"].isEnabled() is True


def test_copy_prev_face_uses_nearest_previous_frame_with_faces(
    qtbot,
    tmp_path: Path,
) -> None:
    """Copy previous skips empty adjacent frames like the legacy Manual Tool."""
    session = _session_with_frames(tmp_path, count=4)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.editable_alignments.add_face(0, (5.0, 5.0, 30.0, 30.0))
    window._thumbnail_panel.setCurrentRow(2)

    assert window.copy_prev_face() is True

    faces = window.editable_alignments.faces(2)
    assert len(faces) == 1
    assert faces[0].bbox == (5.0, 5.0, 30.0, 30.0)


def test_copy_prev_face_noop_when_source_empty(
    qtbot,
    tmp_path: Path,
) -> None:
    """copy_prev_face does not mark the session dirty when prev frame has no faces."""
    session = _session_with_frames(tmp_path, count=3)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window._next_frame()  # noqa: SLF001 - prev frame (0) is empty.

    assert window.copy_prev_face() is False
    assert window.editor_state.unsaved is False
    assert window.editor_state.edited is False
    assert window.editable_alignments.face_count(1) == 0


def test_revert_clears_dirty(qtbot, tmp_path: Path) -> None:
    """Revert clears edited + unsaved on the shared editor state."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.mark_dirty(True)
    window.editor_state.set("edited", True)

    window.revert_current_frame()
    assert window.editor_state.unsaved is False
    assert window.editor_state.edited is False


def test_action_triggered_signal_fires_on_trigger(qtbot, tmp_path: Path) -> None:
    """Triggering an action emits action_triggered with the action key."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    captured: list[str] = []
    window.action_triggered.connect(captured.append)

    window.actions_by_key["set_landmarks_mode"].trigger()
    window.actions_by_key["cycle_annotation"].trigger()
    window.actions_by_key["toggle_mask_annotation"].trigger()

    assert "set_landmarks_mode" in captured
    assert "cycle_annotation" in captured
    assert "toggle_mask_annotation" in captured


def test_legacy_action_requires_legacy_args(qtbot, tmp_path: Path) -> None:
    """The legacy fallback action is disabled when no legacy command is configured."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session, legacy_args=None)
    qtbot.addWidget(window)
    assert window.actions_by_key["legacy_tool"].isEnabled() is False

    window_with_legacy = ManualToolWindow(session, legacy_args=["python", "tools.py", "manual"])
    qtbot.addWidget(window_with_legacy)
    assert window_with_legacy.actions_by_key["legacy_tool"].isEnabled() is True


@pytest.mark.parametrize(
    ("action_key", "expected_mode"),
    [
        ("set_view_mode", "View"),
        ("set_boundingbox_mode", "BoundingBox"),
        ("set_extractbox_mode", "ExtractBox"),
        ("set_landmarks_mode", "Landmarks"),
        ("set_mask_mode", "Mask"),
    ],
)
def test_editor_mode_actions_set_editor_state(
    qtbot,
    tmp_path: Path,
    action_key: str,
    expected_mode: str,
) -> None:
    """Each editor-mode action updates editor_state.editor_mode."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    # Move away from default View so all five params produce a change.
    window.editor_state.set("editor_mode", "Other")
    window.actions_by_key[action_key].trigger()
    assert window.editor_state.editor_mode == expected_mode
