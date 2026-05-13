#!/usr/bin/env python3
"""Qt recent-file and close-project polish tests."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QMessageBox

from lib.gui.models.project import ProjectFile
from lib.gui.services.project_session_service import PROJECT_KIND, TASK_KIND


def _main_window(qtbot, monkeypatch, tmp_path: Path):  # type:ignore[no-untyped-def]
    """Return a MainWindow with a deterministic schema."""
    monkeypatch.setenv("HOME", str(tmp_path))

    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec
    from lib.gui.qt_shell.main_window import MainWindow

    schema = CommandSchema(
        (
            CommandSpec("faceswap", "extract", (OptionSpec("Input", "-i"),)),
            CommandSpec("faceswap", "train", (OptionSpec("Model", "-m"),)),
        )
    )
    window = MainWindow(schema)
    qtbot.addWidget(window)
    return window


def test_close_project_can_be_cancelled_by_dirty_prompt(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Close Project should preserve state when unsaved-change discard is declined."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    filename = str(tmp_path / "dirty.fsw")
    window._project_filename = filename  # pylint:disable=protected-access
    window._set_modified(True)  # pylint:disable=protected-access
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.No,
    )

    closed = window._close_project()  # pylint:disable=protected-access

    assert closed is False
    assert window._project_filename == filename  # pylint:disable=protected-access
    assert window._modified is True  # pylint:disable=protected-access


def test_close_project_discards_after_confirmation(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Close Project should reset project state after confirmation."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    window._project_filename = str(tmp_path / "dirty.fsw")  # pylint:disable=protected-access
    window._file_kind = TASK_KIND  # pylint:disable=protected-access
    window._project = ProjectFile(tab_name="train", tasks={"train": {"-m": "/models"}})  # pylint:disable=protected-access
    window._set_modified(True)  # pylint:disable=protected-access
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )

    closed = window._close_project()  # pylint:disable=protected-access

    assert closed is True
    assert window._project_filename is None  # pylint:disable=protected-access
    assert window._file_kind == PROJECT_KIND  # pylint:disable=protected-access
    assert window._project == ProjectFile()  # pylint:disable=protected-access
    assert window._modified is False  # pylint:disable=protected-access
    assert window.windowTitle() == "Faceswap Qt Shell Prototype - Untitled"


def test_missing_recent_file_is_pruned_when_opened(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Opening a missing recent file should remove it and keep the current project."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    missing = str(tmp_path / "missing.fst")
    current = str(tmp_path / "current.fsw")
    window._project_filename = current  # pylint:disable=protected-access
    window._recent_files.add(missing, TASK_KIND)  # pylint:disable=protected-access

    opened = window._open_session_file(missing, TASK_KIND)  # pylint:disable=protected-access

    assert opened is False
    assert window._project_filename == current  # pylint:disable=protected-access
    assert window._recent_files.load() == []  # pylint:disable=protected-access
    assert window.statusBar().currentMessage() == "Recent file no longer exists"


def test_refresh_recent_menu_prunes_missing_files(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Refreshing the recent menu should remove missing targets."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    existing = tmp_path / "existing.fsw"
    missing = tmp_path / "missing.fst"
    existing.write_text("{}", encoding="utf-8")
    window._recent_files.add(str(missing), TASK_KIND)  # pylint:disable=protected-access
    window._recent_files.add(str(existing), PROJECT_KIND)  # pylint:disable=protected-access

    window._refresh_recent_menu()  # pylint:disable=protected-access

    assert [(item.filename, item.kind) for item in window._recent_files.load()] == [  # pylint:disable=protected-access
        (str(existing), PROJECT_KIND)
    ]
    assert window._recent_menu is not None  # pylint:disable=protected-access
    assert window._recent_menu.actions()[0].text() == "Project: existing.fsw"  # pylint:disable=protected-access


def test_clear_recent_files_resets_menu(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Clear Recent Files should empty the recent store and update the menu."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    existing = tmp_path / "existing.fsw"
    existing.write_text("{}", encoding="utf-8")
    window._recent_files.add(str(existing), PROJECT_KIND)  # pylint:disable=protected-access
    window._refresh_recent_menu()  # pylint:disable=protected-access

    window._clear_recent_files()  # pylint:disable=protected-access

    assert window._recent_files.load() == []  # pylint:disable=protected-access
    assert window._recent_menu is not None  # pylint:disable=protected-access
    assert window._recent_menu.actions()[0].text() == "No recent files"  # pylint:disable=protected-access
    assert window.statusBar().currentMessage() == "Recent files cleared"
