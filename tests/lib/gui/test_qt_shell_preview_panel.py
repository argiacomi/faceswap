#!/usr/bin/env python3
"""Qt Preview output panel tests."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QLabel, QListWidget, QPushButton, QTabWidget

from lib.gui.services.command_context import CommandExecutionContext
from lib.gui.services.preview_output_service import PreviewOutputService


def _touch(path: Path) -> Path:
    """Create an empty preview file and return it."""
    path.write_bytes(b"")
    return path


def _image(path: Path) -> Path:
    """Create a small valid preview image and return it."""
    image = QImage(24, 16, QImage.Format.Format_RGB32)
    image.fill(0x123456)
    assert image.save(str(path))
    return path


def _panel():  # type:ignore[no-untyped-def]
    """Return a PreviewPanel."""
    from lib.gui.qt_shell.preview_panel import PreviewPanel

    return PreviewPanel()


def _button(panel, name: str) -> QPushButton:  # type:ignore[no-untyped-def]
    """Return a PreviewPanel button by object name suffix."""
    button = panel.findChild(QPushButton, f"qt-shell-preview-{name}")
    assert button is not None
    return button


def _label(panel, name: str) -> QLabel:  # type:ignore[no-untyped-def]
    """Return a PreviewPanel label by object name suffix."""
    label = panel.findChild(QLabel, f"qt-shell-preview-{name}")
    assert label is not None
    return label


def _list(panel) -> QListWidget:  # type:ignore[no-untyped-def]
    """Return the PreviewPanel image list."""
    image_list = panel.findChild(QListWidget, "qt-shell-preview-list")
    assert image_list is not None
    return image_list


def test_preview_panel_initial_state(qtbot) -> None:  # type:ignore[no-untyped-def]
    """PreviewPanel should start empty with only Open enabled."""
    panel = _panel()
    qtbot.addWidget(panel)

    assert _label(panel, "source").text() == "No preview source configured"
    assert _label(panel, "status").text() == "No preview images loaded"
    assert _list(panel).count() == 0
    assert _button(panel, "open").isEnabled() is True
    assert _button(panel, "refresh").isEnabled() is False
    assert _button(panel, "clear").isEnabled() is False
    assert _button(panel, "zoom-in").isEnabled() is False
    assert _button(panel, "zoom-out").isEnabled() is False
    assert _button(panel, "reset-view").isEnabled() is False
    assert _button(panel, "train-update").isVisible() is False
    assert _button(panel, "train-mask").isVisible() is False


def test_preview_panel_configures_pending_output_path(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """PreviewPanel should show pending paths before output files exist."""
    panel = _panel()
    qtbot.addWidget(panel)
    source = tmp_path / "pending-output"

    panel.configure_output(source)

    assert panel.service.source == source
    assert _list(panel).count() == 0
    assert _label(panel, "source").text() == f"Preview source: {source}"
    assert _label(panel, "status").text() == f"Waiting for preview output: {source}"
    assert _button(panel, "refresh").isEnabled() is True
    assert _button(panel, "clear").isEnabled() is True


def test_preview_panel_refresh_loads_later_images(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Refresh should load images created after the source was configured."""
    panel = _panel()
    qtbot.addWidget(panel)
    panel.configure_output(tmp_path)
    first = _touch(tmp_path / "a.png")
    second = _touch(tmp_path / "b.jpg")

    refreshed = panel.refresh_preview()
    image_list = _list(panel)

    assert refreshed is True
    assert image_list.count() == 2
    assert image_list.item(0).text() == first.name
    assert image_list.item(1).text() == second.name
    assert panel.service.images[0].path == first
    assert _label(panel, "status").text() == "Loaded 2 preview images"


def test_preview_panel_loads_single_file(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """PreviewPanel should load a single image path."""
    panel = _panel()
    qtbot.addWidget(panel)
    source = _touch(tmp_path / "preview.webp")

    loaded = panel.load_output(source)

    assert loaded is True
    assert _list(panel).count() == 1
    assert _list(panel).item(0).text() == "preview.webp"
    assert _label(panel, "status").text() == "Loaded 1 preview image"


def test_preview_panel_load_failure_displays_error(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Load failures should stay in-panel and not raise dialogs."""
    panel = _panel()
    qtbot.addWidget(panel)

    loaded = panel.load_output(tmp_path / "missing.png")

    assert loaded is False
    assert "does not exist" in _label(panel, "status").text()
    assert _list(panel).count() == 0
    assert _button(panel, "refresh").isEnabled() is True
    assert _button(panel, "clear").isEnabled() is True


def test_preview_panel_apply_context_uses_preview_output_path(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """CommandExecutionContext should configure preview output source."""
    panel = _panel()
    qtbot.addWidget(panel)
    context = CommandExecutionContext(preview_output_path=str(tmp_path))

    applied = panel.apply_context(context)

    assert applied is True
    assert panel.service.source == tmp_path
    assert panel.service.mode == "output"
    assert _label(panel, "source").text() == f"Preview source: {tmp_path}"


def test_preview_panel_apply_context_uses_batch_mode(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Batch contexts should expose the newest child folder and full image list."""
    old_batch = tmp_path / "old"
    new_batch = tmp_path / "new"
    old_batch.mkdir()
    new_batch.mkdir()
    _touch(old_batch / "old.png")
    first = _touch(new_batch / "a.png")
    second = _touch(new_batch / "b.png")
    panel = _panel()
    qtbot.addWidget(panel)
    context = CommandExecutionContext(preview_output_path=str(tmp_path), batch_mode=True)

    applied = panel.apply_context(context)

    assert applied is True
    assert panel.service.mode == "batch"
    assert panel.service.resolved_source == new_batch
    assert [panel.service.images[index].path for index in range(2)] == [first, second]
    assert _list(panel).count() == 2
    assert "current batch" in _label(panel, "source").text()


def test_preview_panel_apply_context_uses_training_preview_cache(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Train contexts should configure Tk-compatible training preview discovery."""
    panel = _panel()
    qtbot.addWidget(panel)
    context = CommandExecutionContext(model_name="original", model_folder="/models")

    applied = panel.apply_context(context)

    assert applied is True
    assert panel.service.mode == "train"
    assert panel.service.source is not None
    assert panel.service.source.name == "preview"
    assert _label(panel, "source").text().startswith("Training preview source:")
    assert _button(panel, "train-update").isVisible() is True
    assert _button(panel, "train-mask").isVisible() is True


def test_preview_panel_training_preview_loads_only_gui_training_image(
    qtbot,
    tmp_path: Path,
) -> None:  # type:ignore[no-untyped-def]
    """Train preview should mirror Tk PreviewTrain's fixed cache filename behavior."""
    _touch(tmp_path / "ignored.png")
    preview = _image(tmp_path / PreviewOutputService.TRAINING_PREVIEW)
    panel = _panel()
    qtbot.addWidget(panel)

    panel.configure_training_preview(tmp_path)

    assert panel.service.mode == "train"
    assert len(panel.service.images) == 1
    assert panel.service.images[0].path == preview
    assert _list(panel).count() == 1
    assert _list(panel).item(0).text() == PreviewOutputService.TRAINING_PREVIEW
    assert _label(panel, "status").text() == "Loaded 1 training preview image"


def test_preview_panel_training_buttons_create_tk_trigger_files(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:  # type:ignore[no-untyped-def]
    """Train refresh/mask controls should mirror Tk PreviewTrain trigger files."""
    monkeypatch.setattr("lib.gui.qt_shell.preview_panel.PATH_CACHE", str(tmp_path))
    panel = _panel()
    qtbot.addWidget(panel)
    panel.configure_training_preview(tmp_path / "preview")

    _button(panel, "train-update").click()
    _button(panel, "train-mask").click()

    assert (tmp_path / ".preview_trigger").is_file()
    assert (tmp_path / ".preview_mask_toggle").is_file()


def test_preview_panel_apply_context_ignores_missing_preview_path(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Command contexts without preview output should be ignored."""
    panel = _panel()
    qtbot.addWidget(panel)

    applied = panel.apply_context(CommandExecutionContext())

    assert applied is False
    assert panel.service.source is None
    assert _label(panel, "source").text() == "No preview source configured"


def test_preview_panel_clear_resets_state(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Clear should reset source, image list and buttons."""
    panel = _panel()
    qtbot.addWidget(panel)
    assert panel.load_output(_touch(tmp_path / "preview.png")) is True

    panel.clear_preview()

    assert panel.service.source is None
    assert _list(panel).count() == 0
    assert _label(panel, "source").text() == "No preview source configured"
    assert _label(panel, "status").text() == "No preview images loaded"
    assert _button(panel, "refresh").isEnabled() is False
    assert _button(panel, "clear").isEnabled() is False


def test_preview_panel_cleanup_stops_timer_and_clears_service_first(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Terminal cleanup should stop polling, clear service state, then reset UI."""
    panel = _panel()
    qtbot.addWidget(panel)
    panel.configure_output(tmp_path)
    panel.start_live_refresh(interval_ms=10)
    assert panel.is_live_refreshing is True

    panel.cleanup_preview(message="Preview cleared after stop")

    assert panel.is_live_refreshing is False
    assert panel.service.source is None
    assert panel.service.images == ()
    assert _list(panel).count() == 0
    assert _label(panel, "source").text() == "No preview source configured"
    assert _label(panel, "status").text() == "Preview cleared after stop"


def test_preview_panel_zoom_buttons_update_view(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Preview image should support basic zoom reset controls."""
    panel = _panel()
    qtbot.addWidget(panel)
    assert panel.load_output(_image(tmp_path / "preview.png")) is True

    _button(panel, "zoom-in").click()

    assert panel.zoom > 1.0
    assert _button(panel, "zoom-out").isEnabled() is True
    assert _button(panel, "reset-view").isEnabled() is True

    _button(panel, "reset-view").click()

    assert panel.zoom == 1.0
    assert panel.pan == (0.0, 0.0)


def test_main_window_uses_real_preview_panel(qtbot, monkeypatch, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """MainWindow should install a real PreviewPanel in the Preview tab."""
    monkeypatch.setenv("HOME", str(tmp_path))

    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec
    from lib.gui.qt_shell.main_window import MainWindow
    from lib.gui.qt_shell.preview_panel import PreviewPanel

    schema = CommandSchema((CommandSpec("faceswap", "extract", (OptionSpec("Output", "-o"),)),))
    window = MainWindow(schema)
    qtbot.addWidget(window)
    tabs = window.findChild(QTabWidget, "qt-shell-display-tabs")
    preview = window.findChild(PreviewPanel, "qt-shell-preview-panel")
    assert tabs is not None
    assert preview is not None

    window._apply_preview_context(  # pylint:disable=protected-access
        CommandExecutionContext(preview_output_path=str(tmp_path))
    )

    assert preview.service.source == tmp_path
    assert tabs.tabText(1) == "Preview"
