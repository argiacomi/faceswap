#!/usr/bin/env python3
"""Frame-navigation polish tests for the Qt Manual Tool (#106).

Covers the acceptance criteria of issue #106:

* Transport slider exposes the active frame range and moves with navigation.
* Jump-to-frame entry box handles empty / invalid / out-of-range input safely.
* Play/Pause auto-advances with a Qt timer and stops at the filtered range end.
* Manual navigation stops playback.
* Play icon/text reflect playing vs paused.
* Status text reports current / total filtered position.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QPixmap

from lib.gui.qt_shell.manual_tool import ManualToolWindow, ManualTransportBar
from tools.manual.session import ManualSession


def _session_with_frames(folder: Path, count: int = 5) -> ManualSession:
    for index in range(count):
        path = folder / f"frame_{index:03d}.png"
        pixmap = QPixmap(64, 48)
        pixmap.fill(QColor("#446699"))
        assert pixmap.save(str(path), "PNG")
    return ManualSession.create(frames=str(folder))


def _make_window(qtbot, folder: Path, count: int = 5) -> ManualToolWindow:  # type:ignore[no-untyped-def]
    session = _session_with_frames(folder, count=count)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    return window


# ---------------------------------------------------------------------------
# ManualTransportBar
# ---------------------------------------------------------------------------


def test_transport_bar_set_total_configures_range(qtbot) -> None:  # type:ignore[no-untyped-def]
    """``set_total`` syncs slider range, validator range, and the counter."""
    bar = ManualTransportBar()
    qtbot.addWidget(bar)
    bar.set_total(0)
    assert bar.slider.isEnabled() is False
    assert bar.counter_label.text() == "Frame: – / 0"

    bar.set_total(10)
    assert bar.slider.isEnabled() is True
    assert bar.slider.minimum() == 0
    assert bar.slider.maximum() == 9
    assert bar.jump_entry.isEnabled() is True


def test_transport_bar_set_position_is_signal_silent(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Programmatic ``set_position`` does not emit ``position_changed``."""
    bar = ManualTransportBar()
    qtbot.addWidget(bar)
    bar.set_total(10)
    received: list[int] = []
    bar.position_changed.connect(received.append)
    bar.set_position(4)
    assert bar.slider.value() == 4
    assert received == []


def test_transport_bar_user_slider_drag_emits(qtbot) -> None:  # type:ignore[no-untyped-def]
    """A user-driven slider value emits ``position_changed``."""
    bar = ManualTransportBar()
    qtbot.addWidget(bar)
    bar.set_total(10)
    received: list[int] = []
    bar.position_changed.connect(received.append)
    bar.slider.setValue(3)
    assert received == [3]


def test_transport_bar_jump_entry_handles_empty_and_invalid(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Empty / non-numeric input must not emit and must restore the slider value."""
    bar = ManualTransportBar()
    qtbot.addWidget(bar)
    bar.set_total(10)
    bar.set_position(5)
    received: list[int] = []
    bar.position_changed.connect(received.append)
    bar.jump_entry.setText("")
    bar._on_jump_submit()  # noqa: SLF001 - exercising the slot directly
    assert received == []
    assert bar.jump_entry.text() == "5"

    bar.jump_entry.setText("oops")
    bar._on_jump_submit()  # noqa: SLF001
    assert received == []
    assert bar.jump_entry.text() == "5"


def test_transport_bar_jump_entry_clamps_out_of_range(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Jump entries past the last frame clamp to ``total - 1``."""
    bar = ManualTransportBar()
    qtbot.addWidget(bar)
    bar.set_total(10)
    bar.set_position(2)
    received: list[int] = []
    bar.position_changed.connect(received.append)
    bar.jump_entry.setText("999")
    bar._on_jump_submit()  # noqa: SLF001
    assert received == [9]
    assert bar.jump_entry.text() == "9"


# ---------------------------------------------------------------------------
# ManualToolWindow navigation
# ---------------------------------------------------------------------------


def test_transport_total_matches_frame_count(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """The transport bar's total is set to the discovered frame count."""
    window = _make_window(qtbot, tmp_path, count=4)
    assert window._transport_bar.slider.maximum() == 3
    assert window._transport_bar.counter_label.text().endswith("/ 4")


def test_transport_position_changed_drives_thumbnail_row(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """A user-driven slider/jump value navigates the thumbnail panel."""
    window = _make_window(qtbot, tmp_path, count=5)
    window._transport_bar.slider.setValue(3)
    assert window._thumbnail_panel.currentRow() == 3


def test_navigation_keeps_transport_in_sync(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Programmatic next/prev/first/last keeps slider + counter in sync."""
    window = _make_window(qtbot, tmp_path, count=5)
    window._next_frame()
    assert window._transport_bar.slider.value() == 1
    window._next_frame()
    window._next_frame()
    window.goto_last_frame()
    assert window._transport_bar.slider.value() == 4
    assert window._transport_bar.counter_label.text() == "Frame: 5 / 5"
    window.goto_first_frame()
    assert window._transport_bar.slider.value() == 0


def test_play_pause_starts_and_stops_timer(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """``toggle_play`` toggles the timer and the editor-state flag."""
    window = _make_window(qtbot, tmp_path, count=3)
    assert window._play_timer.isActive() is False

    window.toggle_play()
    assert window._editor_state.is_playing is True
    assert window._play_timer.isActive() is True

    window.toggle_play()
    assert window._editor_state.is_playing is False
    assert window._play_timer.isActive() is False


def test_play_loop_advances_and_stops_at_last_frame(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Auto-advance walks the panel and auto-stops at the end."""
    window = _make_window(qtbot, tmp_path, count=3)
    window.toggle_play()
    # Drive the timer manually instead of relying on wall-clock ticks.
    window._advance_during_playback()
    assert window._thumbnail_panel.currentRow() == 1
    window._advance_during_playback()
    assert window._thumbnail_panel.currentRow() == 2
    # Next call should stop playback because we're at the last frame.
    window._advance_during_playback()
    assert window._editor_state.is_playing is False
    assert window._play_timer.isActive() is False
    # Row stays at the last frame.
    assert window._thumbnail_panel.currentRow() == 2


def test_play_from_last_frame_rewinds_to_start(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Pressing Play at the end rewinds to frame 0 before starting playback."""
    window = _make_window(qtbot, tmp_path, count=3)
    window.goto_last_frame()
    window.toggle_play()
    assert window._editor_state.is_playing is True
    assert window._thumbnail_panel.currentRow() == 0


def test_manual_navigation_stops_playback(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Manually walking frames during playback halts the auto-advance timer."""
    window = _make_window(qtbot, tmp_path, count=4)
    window.toggle_play()
    assert window._play_timer.isActive() is True
    window._next_frame()
    assert window._editor_state.is_playing is False
    assert window._play_timer.isActive() is False


def test_play_action_icon_reflects_playing_state(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Play action text flips between Play / Pause based on playback."""
    window = _make_window(qtbot, tmp_path, count=3)
    action = window.actions_by_key["play_pause"]
    assert action.text() == "Play"
    window.toggle_play()
    assert action.text() == "Pause"
    window.toggle_play()
    assert action.text() == "Play"


def test_status_label_shows_filtered_position(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """The transport counter reports current / total frame position."""
    window = _make_window(qtbot, tmp_path, count=4)
    window._next_frame()
    assert window._transport_bar.counter_label.text() == "Frame: 2 / 4"


def test_empty_filter_leaves_transport_disabled(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """A session with no frames keeps the transport disabled instead of crashing."""
    bar = ManualTransportBar()
    qtbot.addWidget(bar)
    bar.set_total(0)
    # The widget must not throw or emit when nothing is loaded.
    received: list[int] = []
    bar.position_changed.connect(received.append)
    bar.set_position(0)
    bar.jump_entry.setText("")
    bar._on_jump_submit()  # noqa: SLF001
    assert received == []
    assert bar.slider.isEnabled() is False
