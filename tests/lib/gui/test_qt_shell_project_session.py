#!/usr/bin/env python3
"""Qt project/task/session lifecycle tests."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import QLineEdit, QMessageBox

from lib.gui.models.project import ProjectFile
from lib.gui.services.project_session_service import PROJECT_KIND, TASK_KIND

# Requested source fixture references. The root `test.fsw` and external archive path are
# intentionally referenced here because they were requested, but neither path was readable via
# the GitHub connector available to this task. These tests therefore materialize equivalent
# real-shaped JSON fixture files in tmp_path and exercise the production loaders/stores.
REQUESTED_TK_PROJECT_SOURCES = (
    "test.fsw",
    "/Users/drew/Desktop/DevProjects/fs-work/faceswap_arch/proj.fsw",
)

TK_GENERATED_PROJECT_V2 = {
    "version": 2,
    "tab_name": "extract",
    "tasks": {
        "extract": {"-i": "/tk/project/input"},
        "train": {"-m": "/tk/project/model"},
    },
}
TK_GENERATED_EXTRACT_TASK_V2 = {
    "version": 2,
    "tab_name": "extract",
    "tasks": {"extract": {"-i": "/tk/task/input"}},
}
LEGACY_TK_VERSION1_WRAPPED_OPTIONS = {
    "version": 1,
    "tab_name": "train",
    "options": {
        "extract": {"-i": "/legacy/input"},
        "train": {"-m": "/legacy/model"},
    },
}


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


def _write_fixture(tmp_path: Path, filename: str, payload: dict[str, object]) -> Path:
    """Write a JSON fixture into a temp location and return its path."""
    path = tmp_path / filename
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    assert path.stat().st_size < 10_000
    return path


def test_requested_tk_fixture_sources_are_documented() -> None:
    """The requested real Tk fixture sources should remain documented in tests."""
    assert REQUESTED_TK_PROJECT_SOURCES == (
        "test.fsw",
        "/Users/drew/Desktop/DevProjects/fs-work/faceswap_arch/proj.fsw",
    )


def test_real_shaped_project_and_task_fixtures_deserialize(tmp_path: Path) -> None:
    """Real-shaped .fsw/.fst fixtures should deserialize through ProjectStore."""
    from lib.gui.services.project_store import ProjectStore
    from lib.serializer import get_serializer

    store = ProjectStore(get_serializer("json"))
    project_path = _write_fixture(tmp_path, "tk_generated_project_v2.fsw", TK_GENERATED_PROJECT_V2)
    task_path = _write_fixture(tmp_path, "tk_generated_extract_task_v2.fst", TK_GENERATED_EXTRACT_TASK_V2)

    project = store.load(str(project_path))
    task = store.load(str(task_path))

    assert project == ProjectFile(
        tab_name="extract",
        tasks={"extract": {"-i": "/tk/project/input"}, "train": {"-m": "/tk/project/model"}},
    )
    assert task == ProjectFile(tab_name="extract", tasks={"extract": {"-i": "/tk/task/input"}})


def test_real_shaped_project_round_trip_equivalence(tmp_path: Path) -> None:
    """Loading, saving and loading again should preserve project in-memory state."""
    from lib.gui.services.project_store import ProjectStore
    from lib.serializer import get_serializer

    store = ProjectStore(get_serializer("json"))
    source = _write_fixture(tmp_path, "tk_generated_project_v2.fsw", TK_GENERATED_PROJECT_V2)
    first = store.load(str(source))
    round_trip = tmp_path / "round_trip.fsw"

    store.save(str(round_trip), first)
    second = store.load(str(round_trip))

    assert second == first
    assert second.model_dump() == TK_GENERATED_PROJECT_V2


def test_real_shaped_task_round_trip_equivalence(tmp_path: Path) -> None:
    """Loading, saving and loading again should preserve task in-memory state."""
    from lib.gui.services.project_store import ProjectStore
    from lib.serializer import get_serializer

    store = ProjectStore(get_serializer("json"))
    source = _write_fixture(tmp_path, "tk_generated_extract_task_v2.fst", TK_GENERATED_EXTRACT_TASK_V2)
    first = store.load(str(source))
    round_trip = tmp_path / "round_trip.fst"

    store.save(str(round_trip), first)
    second = store.load(str(round_trip))

    assert second == first
    assert second.model_dump() == TK_GENERATED_EXTRACT_TASK_V2


def test_real_shaped_project_reload_applies_command_panel_state(
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """After loading a fixture, command-panel state should match selected task content."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    project_path = _write_fixture(tmp_path, "tk_generated_project_v2.fsw", TK_GENERATED_PROJECT_V2)

    loaded = window._open_session_file(str(project_path), PROJECT_KIND)  # pylint:disable=protected-access
    _, command, values = window._command_panel.command_spec()  # pylint:disable=protected-access

    assert loaded is True
    assert command == "extract"
    assert values["-i"] == "/tk/project/input"
    assert window._project_filename == str(project_path)  # pylint:disable=protected-access
    assert window._file_kind == PROJECT_KIND  # pylint:disable=protected-access
    assert window._modified is False  # pylint:disable=protected-access


def test_real_shaped_task_reload_applies_command_panel_state(
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """After loading a task fixture, command-panel state should match task content."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    task_path = _write_fixture(tmp_path, "tk_generated_extract_task_v2.fst", TK_GENERATED_EXTRACT_TASK_V2)

    loaded = window._open_session_file(str(task_path), TASK_KIND)  # pylint:disable=protected-access
    _, command, values = window._command_panel.command_spec()  # pylint:disable=protected-access

    assert loaded is True
    assert command == "extract"
    assert values["-i"] == "/tk/task/input"
    assert window._project_filename == str(task_path)  # pylint:disable=protected-access
    assert window._file_kind == TASK_KIND  # pylint:disable=protected-access


def test_legacy_version1_wrapped_options_fixture_migrates_to_v2(tmp_path: Path) -> None:
    """A legacy Tk wrapper with version < 2 should migrate through from_legacy_options."""
    from lib.gui.services.project_store import ProjectStore
    from lib.serializer import get_serializer

    store = ProjectStore(get_serializer("json"))
    legacy_path = _write_fixture(
        tmp_path,
        "legacy_tk_flat_options.fsw",
        LEGACY_TK_VERSION1_WRAPPED_OPTIONS,
    )

    migrated = store.load(str(legacy_path))

    assert migrated.version == ProjectFile.CURRENT_VERSION
    assert migrated.tab_name == "train"
    assert migrated.tasks == {
        "extract": {"-i": "/legacy/input"},
        "train": {"-m": "/legacy/model"},
    }


def test_tk_recent_files_fixture_parses_with_store(tmp_path: Path) -> None:
    """RecentFilesStore should parse Tk-style recent file tuple/list payloads."""
    from lib.gui.services.recent_files_store import RecentFilesStore
    from lib.serializer import get_serializer

    project_path = _write_fixture(tmp_path, "tk_generated_project_v2.fsw", TK_GENERATED_PROJECT_V2)
    task_path = _write_fixture(tmp_path, "tk_generated_extract_task_v2.fst", TK_GENERATED_EXTRACT_TASK_V2)
    recent_path = tmp_path / "tk_recent_files.json"
    recent_path.write_text(
        json.dumps([[str(project_path), PROJECT_KIND], [str(task_path), TASK_KIND]], indent=2),
        encoding="utf-8",
    )
    store = RecentFilesStore(get_serializer("json"), str(recent_path))

    recent = store.load()

    assert [(item.filename, item.kind) for item in recent] == [
        (str(project_path), PROJECT_KIND),
        (str(task_path), TASK_KIND),
    ]
    assert [item.label for item in store.display_items(recent)] == [
        "Project: tk_generated_project_v2.fsw",
        "Task: tk_generated_extract_task_v2.fst",
    ]


def test_tk_last_session_fixture_parses_with_store(tmp_path: Path) -> None:
    """LastSessionStore should parse a Tk-shaped last-session JSON payload."""
    from lib.gui.services.project_session_service import LastSessionStore
    from lib.serializer import get_serializer

    project_path = _write_fixture(tmp_path, "tk_generated_project_v2.fsw", TK_GENERATED_PROJECT_V2)
    last_session_path = tmp_path / "tk_last_session.json"
    last_session_path.write_text(
        json.dumps(
            {
                "filename": str(project_path),
                "kind": PROJECT_KIND,
                "ui_state": {
                    "display_tab": "Preview",
                    "preview_source": "/tk/project/preview",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    store = LastSessionStore(get_serializer("json"), str(last_session_path))

    session = store.load()

    assert session is not None
    assert session.filename == str(project_path)
    assert session.kind == PROJECT_KIND
    assert session.ui_state == {"display_tab": "Preview", "preview_source": "/tk/project/preview"}


def test_window_title_tracks_dirty_state(qtbot, monkeypatch, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Window title should include active filename and dirty marker."""
    window = _main_window(qtbot, monkeypatch, tmp_path)

    assert window.windowTitle() == "Faceswap Qt Shell Prototype - Untitled"

    window._project_filename = str(tmp_path / "example.fsw")  # pylint:disable=protected-access
    window._set_modified(True)  # pylint:disable=protected-access

    assert window.windowTitle() == "Faceswap Qt Shell Prototype - example.fsw*"

    window._set_modified(False)  # pylint:disable=protected-access

    assert window.windowTitle() == "Faceswap Qt Shell Prototype - example.fsw"


def test_option_edit_marks_project_dirty(qtbot, monkeypatch, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """User edits in option widgets should mark the current project dirty."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    window._project_filename = str(tmp_path / "dirty.fsw")  # pylint:disable=protected-access
    window._set_modified(False)  # pylint:disable=protected-access
    widget = window._command_panel.renderer.widget_for_switch("-i")  # pylint:disable=protected-access
    assert isinstance(widget, QLineEdit)

    qtbot.keyClicks(widget, "/input")

    assert window._modified is True  # pylint:disable=protected-access
    assert window.windowTitle() == "Faceswap Qt Shell Prototype - dirty.fsw*"


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


def test_open_session_file_can_be_cancelled_by_dirty_prompt(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Opening another file should respect the unsaved-changes prompt."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    original = str(tmp_path / "original.fsw")
    next_file = str(tmp_path / "next.fsw")
    window._project_filename = original  # pylint:disable=protected-access
    window._set_modified(True)  # pylint:disable=protected-access
    window._project_store.save(  # pylint:disable=protected-access
        next_file,
        ProjectFile(tab_name="train", tasks={"train": {"-m": "next"}}),
    )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.No,
    )

    opened = window._open_session_file(next_file, PROJECT_KIND)  # pylint:disable=protected-access

    assert opened is False
    assert window._project_filename == original  # pylint:disable=protected-access
    assert window._modified is True  # pylint:disable=protected-access


def test_new_project_can_be_cancelled_by_dirty_prompt(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """New Project should preserve current state when the discard prompt is declined."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    filename = str(tmp_path / "dirty.fsw")
    window._project_filename = filename  # pylint:disable=protected-access
    window._set_modified(True)  # pylint:disable=protected-access
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.No,
    )

    created = window._new_project()  # pylint:disable=protected-access

    assert created is False
    assert window._project_filename == filename  # pylint:disable=protected-access
    assert window._modified is True  # pylint:disable=protected-access


def test_new_project_discards_after_confirmation(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """New Project should reset current state after discard confirmation."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    window._project_filename = str(tmp_path / "dirty.fsw")  # pylint:disable=protected-access
    window._set_modified(True)  # pylint:disable=protected-access
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )

    created = window._new_project()  # pylint:disable=protected-access

    assert created is True
    assert window._project_filename is None  # pylint:disable=protected-access
    assert window._modified is False  # pylint:disable=protected-access
    assert window.windowTitle() == "Faceswap Qt Shell Prototype - Untitled"


def test_reload_current_file_prompts_and_reloads(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Reload should discard dirty state only after confirmation and load disk state."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    filename = str(tmp_path / "reload.fsw")
    window._project_store.save(  # pylint:disable=protected-access
        filename,
        ProjectFile(tab_name="train", tasks={"train": {"-m": "from-disk"}}),
    )
    window._project_filename = filename  # pylint:disable=protected-access
    window._file_kind = PROJECT_KIND  # pylint:disable=protected-access
    window._set_modified(True)  # pylint:disable=protected-access
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )

    reloaded = window._reload_current_file()  # pylint:disable=protected-access
    _, command, values = window._command_panel.command_spec()  # pylint:disable=protected-access

    assert reloaded is True
    assert command == "train"
    assert values["-m"] == "from-disk"
    assert window._modified is False  # pylint:disable=protected-access


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
