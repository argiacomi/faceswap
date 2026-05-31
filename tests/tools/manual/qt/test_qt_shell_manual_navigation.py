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

from tools.manual.qt import ManualToolWindow, ManualTransportBar
from tools.manual.session import ManualSession


def _session_with_frames(folder: Path, count: int = 5) -> ManualSession:
    for index in range(count):
        path = folder / f"frame_{index:03d}.png"
        pixmap = QPixmap(64, 48)
        pixmap.fill(QColor("#446699"))
        assert pixmap.save(str(path), "PNG")
    return ManualSession.create(frames=str(folder))


def _make_window(qtbot, folder: Path, count: int = 5) -> ManualToolWindow:
    session = _session_with_frames(folder, count=count)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    return window


# ---------------------------------------------------------------------------
# ManualTransportBar
# ---------------------------------------------------------------------------


def test_transport_bar_set_total_configures_range(qtbot) -> None:
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


def test_transport_bar_set_position_is_signal_silent(qtbot) -> None:
    """Programmatic ``set_position`` does not emit ``position_changed``."""
    bar = ManualTransportBar()
    qtbot.addWidget(bar)
    bar.set_total(10)
    received: list[int] = []
    bar.position_changed.connect(received.append)
    bar.set_position(4)
    assert bar.slider.value() == 4
    assert received == []


def test_transport_bar_user_slider_drag_emits(qtbot) -> None:
    """A user-driven slider value emits ``position_changed``."""
    bar = ManualTransportBar()
    qtbot.addWidget(bar)
    bar.set_total(10)
    received: list[int] = []
    bar.position_changed.connect(received.append)
    bar.slider.setValue(3)
    assert received == [3]


def test_transport_bar_jump_entry_handles_empty_and_invalid(qtbot) -> None:
    """Empty / non-numeric input must not emit and must restore the *1-based* slider value."""
    bar = ManualTransportBar()
    qtbot.addWidget(bar)
    bar.set_total(10)
    bar.set_position(5)
    received: list[int] = []
    bar.position_changed.connect(received.append)
    bar.jump_entry.setText("")
    bar._on_jump_submit()  # noqa: SLF001 - exercising the slot directly
    assert received == []
    # Slider position 5 is 1-based frame "6" — restored as such.
    assert bar.jump_entry.text() == "6"

    bar.jump_entry.setText("oops")
    bar._on_jump_submit()  # noqa: SLF001
    assert received == []
    assert bar.jump_entry.text() == "6"


def test_transport_bar_jump_entry_clamps_out_of_range(qtbot) -> None:
    """Out-of-range 1-based input clamps to ``1..total`` and emits the 0-based index."""
    bar = ManualTransportBar()
    qtbot.addWidget(bar)
    bar.set_total(10)
    bar.set_position(2)
    received: list[int] = []
    bar.position_changed.connect(received.append)
    bar.jump_entry.setText("999")
    bar._on_jump_submit()  # noqa: SLF001
    # Entry displays the clamped 1-based frame (10 of 10); emitted index is the
    # underlying 0-based transport position (9).
    assert received == [9]
    assert bar.jump_entry.text() == "10"

    received.clear()
    bar.jump_entry.setText("1")
    bar._on_jump_submit()  # noqa: SLF001
    assert received == [0]
    assert bar.jump_entry.text() == "1"


def test_transport_bar_jump_entry_uses_one_based_numbering(qtbot) -> None:
    """Typing ``5`` selects frame 5 of N (0-based index 4)."""
    bar = ManualTransportBar()
    qtbot.addWidget(bar)
    bar.set_total(10)
    received: list[int] = []
    bar.position_changed.connect(received.append)
    bar.jump_entry.setText("5")
    bar._on_jump_submit()  # noqa: SLF001
    assert received == [4]
    assert bar.jump_entry.text() == "5"
    # Counter agrees.
    bar.set_position(4)
    assert bar.counter_label.text() == "Frame: 5 / 10"


# ---------------------------------------------------------------------------
# ManualToolWindow navigation
# ---------------------------------------------------------------------------


def test_transport_total_matches_frame_count(qtbot, tmp_path: Path) -> None:
    """The transport bar's total is set to the discovered frame count."""
    window = _make_window(qtbot, tmp_path, count=4)
    assert window._transport_bar.slider.maximum() == 3
    assert window._transport_bar.counter_label.text().endswith("/ 4")


def test_transport_position_changed_drives_thumbnail_row(qtbot, tmp_path: Path) -> None:
    """A user-driven slider/jump value navigates the thumbnail panel."""
    window = _make_window(qtbot, tmp_path, count=5)
    window._transport_bar.slider.setValue(3)
    assert window._thumbnail_panel.currentRow() == 3


def test_navigation_keeps_transport_in_sync(qtbot, tmp_path: Path) -> None:
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


def test_play_pause_starts_and_stops_timer(qtbot, tmp_path: Path) -> None:
    """``toggle_play`` toggles the timer and the editor-state flag."""
    window = _make_window(qtbot, tmp_path, count=3)
    assert window._play_timer.isActive() is False

    window.toggle_play()
    assert window._editor_state.is_playing is True
    assert window._play_timer.isActive() is True

    window.toggle_play()
    assert window._editor_state.is_playing is False
    assert window._play_timer.isActive() is False


def test_play_loop_advances_and_stops_at_last_frame(qtbot, tmp_path: Path) -> None:
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


def test_play_from_last_frame_rewinds_to_start(qtbot, tmp_path: Path) -> None:
    """Pressing Play at the end rewinds to frame 0 before starting playback."""
    window = _make_window(qtbot, tmp_path, count=3)
    window.goto_last_frame()
    window.toggle_play()
    assert window._editor_state.is_playing is True
    assert window._thumbnail_panel.currentRow() == 0


def test_manual_navigation_stops_playback(qtbot, tmp_path: Path) -> None:
    """Manually walking frames during playback halts the auto-advance timer."""
    window = _make_window(qtbot, tmp_path, count=4)
    window.toggle_play()
    assert window._play_timer.isActive() is True
    window._next_frame()
    assert window._editor_state.is_playing is False
    assert window._play_timer.isActive() is False


def test_play_action_icon_reflects_playing_state(qtbot, tmp_path: Path) -> None:
    """Play action text flips between Play / Pause based on playback."""
    window = _make_window(qtbot, tmp_path, count=3)
    action = window.actions_by_key["play_pause"]
    assert action.text() == "Play"
    assert action.toolTip() == "Play playback (Space)"
    assert action.icon().isNull() is False
    window.toggle_play()
    assert action.text() == "Pause"
    assert action.toolTip() == "Pause playback (Space)"
    assert action.icon().isNull() is False
    window.toggle_play()
    assert action.text() == "Play"
    assert action.toolTip() == "Play playback (Space)"


def test_status_label_shows_filtered_position(qtbot, tmp_path: Path) -> None:
    """The transport counter reports current / total frame position."""
    window = _make_window(qtbot, tmp_path, count=4)
    window._next_frame()
    assert window._transport_bar.counter_label.text() == "Frame: 2 / 4"


def test_empty_filter_leaves_transport_disabled(qtbot, tmp_path: Path) -> None:
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


# ---------------------------------------------------------------------------
# #113 — arrow-key shortcut collision between navigation and frame-view nudge
# ---------------------------------------------------------------------------


def test_previous_next_frame_shortcuts_drop_left_right(qtbot, tmp_path: Path) -> None:
    """``previous_frame`` / ``next_frame`` no longer bind ``Left``/``Right``.

    The frame-view-scoped nudge actions own arrow keys when the frame view
    has focus; binding them to window-level navigation as well caused both
    actions to fire on every arrow press.
    """
    window = _make_window(qtbot, tmp_path, count=3)
    prev_action = window.actions_by_key["previous_frame"]
    next_action = window.actions_by_key["next_frame"]

    prev_shortcuts = {seq.toString() for seq in prev_action.shortcuts()}
    next_shortcuts = {seq.toString() for seq in next_action.shortcuts()}

    # Z / X remain — they're the documented legacy navigation keys.
    assert "Z" in prev_shortcuts
    assert "X" in next_shortcuts
    # Left / Right are NOT bound to navigation any more.
    assert "Left" not in prev_shortcuts
    assert "Right" not in next_shortcuts


def test_z_and_x_still_trigger_navigation(qtbot, tmp_path: Path) -> None:
    """The Z / X legacy navigation shortcuts continue to fire prev/next."""
    window = _make_window(qtbot, tmp_path, count=4)
    window._thumbnail_panel.setCurrentRow(2)
    window.actions_by_key["previous_frame"].trigger()
    assert window._thumbnail_panel.currentRow() == 1
    window.actions_by_key["next_frame"].trigger()
    assert window._thumbnail_panel.currentRow() == 2


# ---------------------------------------------------------------------------
# #119 task 3 — Return in the jump entry must not emit twice
# ---------------------------------------------------------------------------


def test_return_in_jump_entry_emits_position_change_once(qtbot) -> None:
    """Pressing Return in the jump entry triggers exactly one navigation request.

    Before the fix the bar wired both ``editingFinished`` *and*
    ``returnPressed`` to ``_on_jump_submit``, so one Return key press
    produced two emissions and could double-step navigation.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    bar = ManualTransportBar()
    qtbot.addWidget(bar)
    bar.show()
    qtbot.waitExposed(bar)
    bar.set_total(10)
    bar.set_position(2)
    received: list[int] = []
    bar.position_changed.connect(received.append)

    bar.jump_entry.setFocus()
    bar.jump_entry.selectAll()
    QTest.keyClicks(bar.jump_entry, "7")
    QTest.keyClick(bar.jump_entry, Qt.Key_Return)  # type: ignore[attr-defined]

    # Exactly one emission, carrying the 0-based equivalent of 1-based "7".
    assert received == [6]
