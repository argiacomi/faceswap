#!/usr/bin/env python3
"""Real Tk-generated `.fsw`/`.fst` fixture compatibility tests.

Exercises the versioned `ProjectFile` loader, `ProjectStore`, and
`RecentFilesStore` against real on-disk shapes written by the legacy Tk GUI.

Fixtures live in ``tests/lib/gui/data/`` -- see ``data/README.md`` for the
source of each file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.gui.models.project import ProjectFile
from lib.gui.services.project_session_service import (
    PROJECT_KIND,
    TASK_KIND,
    LastSession,
    LastSessionStore,
    ProjectSessionService,
)
from lib.gui.services.project_store import ProjectStore
from lib.gui.services.recent_files_store import RecentFile, RecentFilesStore
from lib.serializer import get_serializer

FIXTURES = Path(__file__).parent / "data"

TK_V1_PROJECT = FIXTURES / "tk_generated_project_v1.fsw"
TK_V1_PROJECT_ALT = FIXTURES / "tk_generated_project_v1_alt.fsw"
TK_V1_TASK = FIXTURES / "tk_generated_task_v1.fst"
QT_V2_TASK = FIXTURES / "qt_generated_task_v2.fst"
TK_RECENT_FILES = FIXTURES / "tk_recent_files.json"


def _serializer():
    return get_serializer("json")


@pytest.fixture(name="store")
def fixture_store() -> ProjectStore:
    """Return a `ProjectStore` wired with the JSON serializer."""
    return ProjectStore(_serializer())


def test_fixtures_exist_and_are_small() -> None:
    """All checked-in fixtures should exist and stay well under a megabyte."""
    for path in (
        TK_V1_PROJECT,
        TK_V1_PROJECT_ALT,
        TK_V1_TASK,
        QT_V2_TASK,
        TK_RECENT_FILES,
    ):
        assert path.exists(), f"missing fixture: {path}"
        assert path.stat().st_size < 64 * 1024, f"fixture too large: {path}"


@pytest.mark.parametrize(
    ("fixture", "expected_tab", "expected_command"),
    [
        (TK_V1_PROJECT, "extract", "extract"),
        (TK_V1_PROJECT_ALT, "train", "train"),
        (TK_V1_TASK, "extract", "extract"),
        (QT_V2_TASK, "extract", "extract"),
    ],
)
def test_project_store_loads_fixture(
    store: ProjectStore,
    fixture: Path,
    expected_tab: str,
    expected_command: str,
) -> None:
    """Each fixture should deserialize to a `ProjectFile` at the current version."""
    loaded = store.load(str(fixture))

    assert isinstance(loaded, ProjectFile)
    assert loaded.version == ProjectFile.CURRENT_VERSION
    assert loaded.tab_name == expected_tab
    assert expected_command in loaded.tasks
    # Every task payload must be a plain dict, never lifted from the legacy
    # flat shape (i.e. no leaked `tab_name` key inside a task).
    for command, values in loaded.tasks.items():
        assert isinstance(values, dict), f"task '{command}' is not a mapping"
        assert "tab_name" not in values


def test_legacy_v1_project_migrates_to_current_version(store: ProjectStore) -> None:
    """A legacy flat `.fsw` should migrate cleanly to the current schema."""
    loaded = store.load(str(TK_V1_PROJECT))

    # The flat legacy payload has a top-level `tab_name` and one dict per
    # command.  The migrated model should hoist `tab_name` and place all
    # command dicts under `tasks` without dropping any commands.
    raw = json.loads(TK_V1_PROJECT.read_text(encoding="utf-8"))
    expected_commands = {key for key, value in raw.items() if isinstance(value, dict)}

    assert set(loaded.tasks) == expected_commands
    # Concrete spot-check: real Tk-saved option values must survive.
    extract = loaded.tasks["extract"]
    assert extract["Detector"] == "scrfd"
    assert extract["Aligner"] == "spiga"
    assert extract["Size"] == 512
    assert extract["Re Align"] is True


def test_legacy_v1_alt_project_active_tab_is_train(store: ProjectStore) -> None:
    """The alternate fixture's active tab should be `train` post-migration."""
    loaded = store.load(str(TK_V1_PROJECT_ALT))

    assert loaded.tab_name == "train"
    train = loaded.tasks["train"]
    assert train["Trainer"] == "phaze-a"
    assert train["Batch Size"] == 4


@pytest.mark.parametrize(
    "fixture",
    [TK_V1_PROJECT, TK_V1_PROJECT_ALT, TK_V1_TASK, QT_V2_TASK],
)
def test_round_trip_load_save_load_is_stable(
    store: ProjectStore, tmp_path: Path, fixture: Path
) -> None:
    """Load -> save -> load should yield an equivalent in-memory model."""
    original = store.load(str(fixture))
    target = tmp_path / fixture.name
    store.save(str(target), original)
    reloaded = store.load(str(target))

    assert reloaded.version == original.version
    assert reloaded.tab_name == original.tab_name
    assert reloaded.tasks == original.tasks


def test_saved_file_uses_current_versioned_shape(store: ProjectStore, tmp_path: Path) -> None:
    """`ProjectStore.save` should always write the current versioned shape."""
    original = store.load(str(TK_V1_PROJECT))
    target = tmp_path / "rewritten.fsw"
    store.save(str(target), original)

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["version"] == ProjectFile.CURRENT_VERSION
    assert payload["tab_name"] == "extract"
    assert isinstance(payload["tasks"], dict)
    assert "extract" in payload["tasks"]


def test_task_fixture_selected_command_matches_active_tab(
    store: ProjectStore,
) -> None:
    """`selected_task` should resolve to the fixture's active tab values."""
    loaded = store.load(str(TK_V1_TASK))

    command, values = ProjectSessionService.selected_task(loaded)

    assert command == "extract"
    assert values["Detector"] == "scrfd"
    assert values["Re Align"] is True


def test_recent_files_store_parses_tk_legacy_shape(tmp_path: Path) -> None:
    """`RecentFilesStore` should parse Tk-written recent-files JSON."""
    target = tmp_path / "recent.json"
    target.write_bytes(TK_RECENT_FILES.read_bytes())
    store = RecentFilesStore(_serializer(), str(target))

    entries = store.load()

    assert entries == [
        RecentFile("/tmp/some_project.fsw", "project"),
        RecentFile("/tmp/some_task.fst", "task"),
        RecentFile("/tmp/legacy.fsw", "extract"),
    ]


def test_recent_files_legacy_kinds_normalize_to_task() -> None:
    """Legacy command-kind strings like `extract` should normalize to `task`."""
    raw = json.loads(TK_RECENT_FILES.read_text(encoding="utf-8"))
    entries = RecentFilesStore.decode_many(raw)
    legacy = next(item for item in entries if item.kind == "extract")

    normalized = ProjectSessionService.normalize_kind(legacy.kind, legacy.filename)

    assert normalized == TASK_KIND


def test_last_session_store_round_trip(tmp_path: Path, store: ProjectStore) -> None:
    """`LastSessionStore` should persist and restore a fixture-pointing entry."""
    target_project = tmp_path / "session.fsw"
    store.save(str(target_project), store.load(str(TK_V1_PROJECT)))
    cache = tmp_path / "last_session.json"
    last_store = LastSessionStore(_serializer(), str(cache))

    last_store.save(str(target_project), PROJECT_KIND, ui_state={"window_size": [1280, 760]})
    restored = last_store.load()

    assert isinstance(restored, LastSession)
    assert restored.filename == str(target_project)
    assert restored.kind == PROJECT_KIND
    assert restored.ui_state == {"window_size": [1280, 760]}


def test_main_window_reload_applies_fixture_state(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Loading a real Tk fixture through the Qt shell should update the panel."""
    monkeypatch.setenv("HOME", str(tmp_path))

    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec
    from lib.gui.qt_shell.main_window import MainWindow

    schema = CommandSchema(
        (
            CommandSpec(
                "faceswap",
                "extract",
                (
                    OptionSpec("Input Dir", "-i"),
                    OptionSpec("Detector", "-D"),
                ),
            ),
            CommandSpec("faceswap", "train", (OptionSpec("Model Dir", "-m"),)),
        )
    )
    window = MainWindow(schema)
    qtbot.addWidget(window)

    # Place the fixture under tmp_path so the dirty prompt and recent-files
    # path stay isolated from the developer's real GUI cache.
    target = tmp_path / "fixture_project.fsw"
    target.write_bytes(TK_V1_PROJECT.read_bytes())

    assert window._open_session_file(str(target), PROJECT_KIND) is True  # pylint:disable=protected-access

    _, command, _values = window._command_panel.command_spec()  # pylint:disable=protected-access
    assert command == "extract"
    assert window._project_filename == str(target)  # pylint:disable=protected-access
    assert window._file_kind == PROJECT_KIND  # pylint:disable=protected-access
    assert window._modified is False  # pylint:disable=protected-access
    # `extract` task survives migration and is now in the in-memory project.
    assert "extract" in window._project.tasks  # pylint:disable=protected-access
    assert window._project.tasks["extract"]["Detector"] == "scrfd"  # pylint:disable=protected-access
