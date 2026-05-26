#!/usr/bin/env python3
"""Filter-aware navigation regressions for the Qt Manual Tool (#107).

These tests exercise the GUI-facing contract on top of the GUI-neutral
``tools.manual.frame_filter`` unit tests:

* active filter modes drive the ManualToolWindow filtered frame list;
* first/previous/next/last/playback use the filtered list, not raw rows;
* empty filters disable transport/navigation safely and surface status text;
* the Misaligned threshold control is visible only for that filter and
  re-runs the filtered model when moved;
* editable face-count changes refresh the active filtered list.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtGui import QColor, QPixmap

from tools.manual.qt import ManualToolWindow
from tools.manual.session import ManualSession


def _session_with_frames(folder: Path, count: int = 5) -> ManualSession:
    """Create a small image-folder Manual session with ``count`` frames."""
    for index in range(count):
        path = folder / f"frame_{index:03d}.png"
        pixmap = QPixmap(80, 60)
        pixmap.fill(QColor("#446699"))
        assert pixmap.save(str(path), "PNG")
    return ManualSession.create(frames=str(folder))


def _make_window(qtbot, folder: Path, count: int = 5) -> ManualToolWindow:  # type:ignore[no-untyped-def]
    """Return a shown ManualToolWindow with startup fully drained."""
    window = ManualToolWindow(_session_with_frames(folder, count=count))
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(lambda: window._startup_worker is None, timeout=5000)
    return window


def _seed_face(window: ManualToolWindow, frame_index: int) -> int:
    """Seed one editable face on ``frame_index``."""
    return window.editable_alignments.add_face(
        frame_index,
        (10.0, 10.0, 20.0, 20.0),
        landmarks=(),
    )


def test_filter_mode_drives_navigation_transport_and_playback(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Navigation actions walk matching frames from the active filtered list."""
    window = _make_window(qtbot, tmp_path, count=5)
    _seed_face(window, 1)
    _seed_face(window, 3)
    _seed_face(window, 3)
    _seed_face(window, 4)

    # Start from a raw row that the filter will reject; enabling Has Face(s)
    # should clamp to the first matching frame and shrink the transport range.
    window._thumbnail_panel.setCurrentRow(0)
    window.editor_state.set("filter_mode", "Has Face(s)")

    assert window.filtered_frame_indices() == (1, 3, 4)
    assert window._thumbnail_panel.currentRow() == 1
    assert window._transport_bar.slider.maximum() == 2
    assert window._transport_bar.counter_label.text() == "Frame: 1 / 3"

    window._next_frame()
    assert window._thumbnail_panel.currentRow() == 3
    assert window._transport_bar.counter_label.text() == "Frame: 2 / 3"

    window.goto_last_frame()
    assert window._thumbnail_panel.currentRow() == 4
    assert window._transport_bar.counter_label.text() == "Frame: 3 / 3"

    window.goto_first_frame()
    assert window._thumbnail_panel.currentRow() == 1
    window.toggle_play()
    assert window.editor_state.is_playing is True
    window._advance_during_playback()
    assert window._thumbnail_panel.currentRow() == 3
    window._advance_during_playback()
    assert window._thumbnail_panel.currentRow() == 4
    window._advance_during_playback()
    assert window.editor_state.is_playing is False


def test_empty_filter_disables_transport_and_reports_status(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """An empty filter result is safe and disables filtered navigation."""
    window = _make_window(qtbot, tmp_path, count=3)

    window.editor_state.set("filter_mode", "Has Face(s)")

    assert window.filtered_frame_indices() == ()
    assert window._transport_bar.slider.isEnabled() is False
    assert window._transport_bar.jump_entry.isEnabled() is False
    assert window.actions_by_key["play_pause"].isEnabled() is False
    assert window.actions_by_key["next_frame"].isEnabled() is False
    assert window._filter_label.text() == "Filter: Has Face(s) (0 match)"

    window.goto_first_frame()
    assert window.statusBar().currentMessage() == "No frames match filter: Has Face(s)"
    window.toggle_play()
    assert window.statusBar().currentMessage() == "No frames match filter: Has Face(s)"


def test_misaligned_threshold_control_refreshes_filtered_results(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type:ignore[no-untyped-def]
    """The Misaligned threshold slider is visible and re-runs filtering."""
    from tools.manual import frame_filter

    window = _make_window(qtbot, tmp_path, count=4)
    _seed_face(window, 2)

    def _predicate_for_model(_model, threshold_raw: int):  # type:ignore[no-untyped-def]
        return lambda frame_index: frame_index == 2 and threshold_raw <= 10

    monkeypatch.setattr(frame_filter, "misaligned_predicate_for_model", _predicate_for_model)

    assert window._filter_threshold_slider.isVisible() is False
    window.editor_state.set("filter_mode", "Misaligned Faces")

    assert window._filter_threshold_slider.isVisible() is True
    assert window._filter_threshold_label.isVisible() is True
    assert window._filter_threshold_value.isVisible() is True
    assert window._filter_threshold_slider.minimum() == frame_filter.MISALIGNED_THRESHOLD_MIN
    assert window._filter_threshold_slider.maximum() == frame_filter.MISALIGNED_THRESHOLD_MAX
    assert window.filtered_frame_indices() == (2,)
    assert window._filter_label.text() == "Filter: Misaligned Faces (1 match)"

    window._filter_threshold_slider.setValue(20)
    assert window.editor_state.filter_distance == 20
    assert window.filtered_frame_indices() == ()
    assert window._filter_threshold_value.text() == "20"
    assert window._filter_label.text() == "Filter: Misaligned Faces (0 match)"

    window._filter_threshold_slider.setValue(5)
    assert window.editor_state.filter_distance == 5
    assert window.filtered_frame_indices() == (2,)
    assert window._filter_threshold_value.text() == "5"


def test_face_count_edit_refreshes_active_filter_results(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Adding a face should immediately update an active count-based filter."""
    window = _make_window(qtbot, tmp_path, count=3)
    window.editor_state.set("filter_mode", "Has Face(s)")
    assert window.filtered_frame_indices() == ()

    _seed_face(window, 0)

    assert window.filtered_frame_indices() == (0,)
    assert window._transport_bar.slider.isEnabled() is True
    assert window._filter_label.text() == "Filter: Has Face(s) (1 match)"


def test_non_current_face_count_edit_refreshes_active_filter_results(
    qtbot, tmp_path: Path
) -> None:  # type:ignore[no-untyped-def]
    """Adding a face on another frame refreshes filters before the current-frame guard."""
    window = _make_window(qtbot, tmp_path, count=4)
    window._thumbnail_panel.setCurrentRow(0)
    window.editor_state.set("filter_mode", "Has Face(s)")
    assert window.filtered_frame_indices() == ()

    _seed_face(window, 2)

    assert window.filtered_frame_indices() == (2,)
    assert window._thumbnail_panel.currentRow() == 2
    assert window._transport_bar.slider.isEnabled() is True
    assert window._transport_bar.counter_label.text() == "Frame: 1 / 1"
    assert window._filter_label.text() == "Filter: Has Face(s) (1 match)"
