#!/usr/bin/env python3
"""Qt project/task/session lifecycle tests."""

from __future__ import annotations

from pathlib import Path

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


def test_window_title_tracks_dirty_state(qtbot, monkeypatch, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Window title should include active filename and dirty marker."""
    window = _main_window(qtbot, monkeypatch, tmp_path)

    assert window.windowTitle() == "Faceswap Qt Shell Prototype - Untitled"

    window._project_filename = str(tmp_path / "example.fsw")  # pylint:disable=protected-access
    window._set_modified(True)  # pylint:disable=protected-access

    assert window.windowTitle() == "Faceswap Qt Shell Prototype - example.fsw*"

    window._set_modified(False)  # pylint:disable=protected-access

    assert window.windowTitle() == "Faceswap Qt Shell Prototype - example.fsw"


def test_save_project_persists_all_project_tasks(qtbot, monkeypatch, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Saving a project should preserve existing tasks and current command values."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    filename = str(tmp_path / "project.fsw")
    window._project = ProjectFile(tab_name="train", tasks={"train": {"-m": "/models"}})  # pylint:disable=protected-access
    window._command_panel.set_command("extract", {"-i": "/input"})  # pylint:disable=protected-access

    window._save_project(filename)  # pylint:disable=protected-access
    loaded = window._project_store.load(filename)  # pylint:disable=protected-access

    assert loaded.tab_name == "extract"
    assert loaded.tasks == {"train": {"-m": "/models"}, "extract": {"-i": "/input"}}
    assert window._project_filename == filename  # pylint:disable=protected-access
    assert window._file_kind == PROJECT_KIND  # pylint:disable=protected-access
    assert window._modified is False  # pylint:disable=protected-access
    assert window.windowTitle() == "Faceswap Qt Shell Prototype - project.fsw"


def test_save_task_persists_only_current_command(qtbot, monkeypatch, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Saving a task should not replace the active project and should contain one task."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    filename = str(tmp_path / "task.fst")
    window._project_filename = str(tmp_path / "project.fsw")  # pylint:disable=protected-access
    window._file_kind = PROJECT_KIND  # pylint:disable=protected-access
    window._project = ProjectFile(tab_name="train", tasks={"train": {"-m": "/models"}})  # pylint:disable=protected-access
    window._command_panel.set_command("extract", {"-i": "/input"})  # pylint:disable=protected-access

    window._save_task(filename)  # pylint:disable=protected-access
    loaded = window._project_store.load(filename)  # pylint:disable=protected-access

    assert loaded.tab_name == "extract"
    assert loaded.tasks == {"extract": {"-i": "/input"}}
    assert window._project_filename == str(tmp_path / "project.fsw")  # pylint:disable=protected-access
    assert window._file_kind == PROJECT_KIND  # pylint:disable=protected-access


def test_open_task_routes_kind_and_applies_command(qtbot, monkeypatch, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Opening a task should set task kind and apply the selected command values."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    filename = str(tmp_path / "task.fst")
    window._project_store.save(  # pylint:disable=protected-access
        filename,
        ProjectFile(tab_name="train", tasks={"train": {"-m": "/models"}}),
    )

    window._open_session_file(filename, TASK_KIND)  # pylint:disable=protected-access
    _, command, values = window._command_panel.command_spec()  # pylint:disable=protected-access

    assert command == "train"
    assert values["-m"] == "/models"
    assert window._project_filename == filename  # pylint:disable=protected-access
    assert window._file_kind == TASK_KIND  # pylint:disable=protected-access
    assert window._modified is False  # pylint:disable=protected-access
    assert window.windowTitle() == "Faceswap Qt Shell Prototype - task.fst"


def test_recent_menu_routes_saved_files(qtbot, monkeypatch, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Recent menu actions should reopen their target file and kind."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    first = str(tmp_path / "first.fsw")
    second = str(tmp_path / "second.fst")
    window._project_store.save(  # pylint:disable=protected-access
        first,
        ProjectFile(tab_name="extract", tasks={"extract": {"-i": "first"}}),
    )
    window._project_store.save(  # pylint:disable=protected-access
        second,
        ProjectFile(tab_name="train", tasks={"train": {"-m": "second"}}),
    )
    window._recent_files.add(first, PROJECT_KIND)  # pylint:disable=protected-access
    window._recent_files.add(second, TASK_KIND)  # pylint:disable=protected-access
    window._refresh_recent_menu()  # pylint:disable=protected-access
    recent_menu = window._recent_menu  # pylint:disable=protected-access
    assert recent_menu is not None

    recent_menu.actions()[0].trigger()

    assert window._project_filename == second  # pylint:disable=protected-access
    assert window._file_kind == TASK_KIND  # pylint:disable=protected-access
    _, command, values = window._command_panel.command_spec()  # pylint:disable=protected-access
    assert command == "train"
    assert values["-m"] == "second"


def test_restore_last_session_opens_cached_file(qtbot, monkeypatch, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Restore Last Session should open the cached project/task file."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    filename = str(tmp_path / "last.fsw")
    window._project_store.save(  # pylint:disable=protected-access
        filename,
        ProjectFile(tab_name="extract", tasks={"extract": {"-i": "last"}}),
    )
    window._last_session.save(filename, PROJECT_KIND)  # pylint:disable=protected-access

    window._restore_last_session()  # pylint:disable=protected-access

    assert window._project_filename == filename  # pylint:disable=protected-access
    assert window._file_kind == PROJECT_KIND  # pylint:disable=protected-access
    _, command, values = window._command_panel.command_spec()  # pylint:disable=protected-access
    assert command == "extract"
    assert values["-i"] == "last"


def test_generate_marks_project_dirty(qtbot, monkeypatch, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Generate should snapshot current command state and mark the window dirty."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    window._project_filename = str(tmp_path / "dirty.fsw")  # pylint:disable=protected-access
    window._set_modified(False)  # pylint:disable=protected-access
    window._command_panel.set_command("extract", {"-i": "/input"})  # pylint:disable=protected-access

    window._generate_command()  # pylint:disable=protected-access

    assert window._modified is True  # pylint:disable=protected-access
    assert window._project.tasks["extract"]["-i"] == "/input"  # pylint:disable=protected-access
    assert window.windowTitle() == "Faceswap Qt Shell Prototype - dirty.fsw*"
