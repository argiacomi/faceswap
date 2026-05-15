#!/usr/bin/env python3
"""Qt Preview panel live-refresh tests."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel

from lib.gui.qt_shell.preview_panel import PreviewPanel
from lib.gui.services.preview_output_service import PreviewOutputService


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


def test_training_preview_live_refresh_status_marks_live_mode(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Train preview polling should use Tk's fixed training preview cache image."""
    _touch_image(tmp_path, PreviewOutputService.TRAINING_PREVIEW)
    _touch_image(tmp_path, "ignored.png")
    panel = PreviewPanel()
    qtbot.addWidget(panel)

    panel.configure_training_preview(tmp_path)
    panel.start_live_refresh()
    assert panel.refresh_preview() is True

    assert panel.service.mode == "train"
    assert len(panel.service.images) == 1
    assert panel.service.images[0].path.name == PreviewOutputService.TRAINING_PREVIEW
    assert _status(panel).text() == "Loaded 1 training preview image (live)"


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
    assert panel.service.source is None
    assert _status(panel).text() == "No preview images loaded"


def test_cleanup_preview_stops_timer_and_clears_ui_and_service(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Terminal preview cleanup should clear timer, service state and UI state together."""
    panel = PreviewPanel()
    qtbot.addWidget(panel)
    panel.configure_training_preview(tmp_path)
    panel.start_live_refresh()
    assert panel.is_live_refreshing is True

    panel.cleanup_preview(message="Preview stopped")

    assert panel.is_live_refreshing is False
    assert panel.service.source is None
    assert panel.service.images == ()
    assert _status(panel).text() == "Preview stopped"


def test_main_window_starts_and_stops_preview_polling_for_extract(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """MainWindow should own preview polling for preview-capable jobs."""
    monkeypatch.setenv("HOME", str(tmp_path))

    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec
    from lib.gui.qt_shell.main_window import MainWindow

    schema = CommandSchema(
        (
            CommandSpec(
                "faceswap",
                "extract",
                (OptionSpec("Output", "-o"),),
            ),
        )
    )
    window = MainWindow(schema)
    qtbot.addWidget(window)
    window._runner.configure_runtime_context = lambda _context: None  # type:ignore[method-assign] # pylint:disable=protected-access
    window._runner.start = lambda _args, command=None: None  # type:ignore[method-assign] # pylint:disable=protected-access
    window._command_panel.set_command("extract", {"-o": str(tmp_path)})  # pylint:disable=protected-access

    window._run_command()  # pylint:disable=protected-access

    assert window._preview_panel_widget is not None  # pylint:disable=protected-access
    assert window._preview_panel_widget.is_live_refreshing is True  # pylint:disable=protected-access

    window._job_finished(0)  # pylint:disable=protected-access

    assert window._preview_panel_widget.is_live_refreshing is False  # pylint:disable=protected-access
