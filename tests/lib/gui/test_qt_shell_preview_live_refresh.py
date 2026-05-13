#!/usr/bin/env python3
"""Qt Preview panel live-refresh tests."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel

from lib.gui.qt_shell.preview_panel import PreviewPanel


def _touch_image(folder: Path, filename: str) -> Path:
    """Create a placeholder image file for preview discovery tests."""
    path = folder / filename
    path.write_bytes(b"not-a-real-image")
    return path


def _status(panel: PreviewPanel) -> QLabel:
    """Return the preview status label."""
    label = panel.findChild(QLabel, "qt-shell-preview-status")
    assert label is not None
    return label


def test_preview_live_refresh_starts_only_with_source(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Live refresh should only start when a preview source is configured."""
    panel = PreviewPanel()
    qtbot.addWidget(panel)

    panel.start_live_refresh(interval_ms=10)

    assert panel.is_live_refreshing is False

    panel.configure_output(tmp_path)
    panel.start_live_refresh(interval_ms=10)

    assert panel.is_live_refreshing is True
    assert panel._refresh_timer.interval() == 250  # pylint:disable=protected-access


def test_preview_live_refresh_status_marks_live_mode(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Live refresh status should show that the preview is polling."""
    _touch_image(tmp_path, "first.png")
    panel = PreviewPanel()
    qtbot.addWidget(panel)

    panel.configure_output(tmp_path)
    panel.start_live_refresh()
    assert panel.refresh_preview() is True

    assert _status(panel).text() == "Loaded 1 preview image (live)"


def test_preview_refresh_preserves_selected_image(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Refreshing should preserve the selected preview image when it still exists."""
    _touch_image(tmp_path, "first.png")
    second = _touch_image(tmp_path, "second.png")
    panel = PreviewPanel()
    qtbot.addWidget(panel)
    assert panel.load_output(tmp_path) is True
    panel._image_list.setCurrentRow(1)  # pylint:disable=protected-access

    _touch_image(tmp_path, "third.png")
    assert panel.refresh_preview() is True

    assert panel._image_list.currentItem().data(0x0100) == str(second)  # pylint:disable=protected-access


def test_preview_refresh_selects_first_when_previous_selection_disappears(  # type:ignore[no-untyped-def]
    qtbot,
    tmp_path: Path,
) -> None:
    """Refreshing should fall back to the first image when the prior selection is gone."""
    first = _touch_image(tmp_path, "first.png")
    second = _touch_image(tmp_path, "second.png")
    panel = PreviewPanel()
    qtbot.addWidget(panel)
    assert panel.load_output(tmp_path) is True
    panel._image_list.setCurrentRow(1)  # pylint:disable=protected-access

    second.unlink()
    assert panel.refresh_preview() is True

    assert panel._image_list.currentItem().data(0x0100) == str(first)  # pylint:disable=protected-access


def test_clear_preview_stops_live_refresh(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Clearing preview output should stop polling and reset status."""
    panel = PreviewPanel()
    qtbot.addWidget(panel)
    panel.configure_output(tmp_path)
    panel.start_live_refresh()
    assert panel.is_live_refreshing is True

    panel.clear_preview()

    assert panel.is_live_refreshing is False
    assert _status(panel).text() == "No preview images loaded"
