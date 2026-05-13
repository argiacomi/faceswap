#!/usr/bin/env python3
"""Tests for project/task session lifecycle helpers."""

from __future__ import annotations

from pathlib import Path

from lib.gui.models.project import ProjectFile
from lib.gui.services.project_session_service import (
    PROJECT_KIND,
    TASK_KIND,
    LastSessionStore,
    ProjectSessionService,
)
from lib.serializer import get_serializer


def test_kind_from_filename_routes_project_and_task_extensions() -> None:
    """File extensions should route project vs task behavior."""
    service = ProjectSessionService()

    assert service.kind_from_filename("my-project.fsw") == PROJECT_KIND
    assert service.kind_from_filename("my-task.fst") == TASK_KIND
    assert service.kind_from_filename("unknown.json") == PROJECT_KIND
    assert service.kind_from_filename("unknown.json", default=TASK_KIND) == TASK_KIND


def test_title_renders_basename_and_dirty_marker() -> None:
    """Window title should reflect current file and dirty state."""
    service = ProjectSessionService()

    assert service.title(None, modified=False) == "Faceswap Qt Shell Prototype - Untitled"
    assert service.title("/tmp/example.fsw", modified=False) == (
        "Faceswap Qt Shell Prototype - example.fsw"
    )
    assert service.title("/tmp/example.fsw", modified=True) == (
        "Faceswap Qt Shell Prototype - example.fsw*"
    )


def test_snapshot_project_merges_current_command() -> None:
    """Project snapshots should preserve existing tasks and replace current task values."""
    service = ProjectSessionService()
    project = ProjectFile(tab_name="extract", tasks={"extract": {"-i": "old"}})

    snapshot = service.snapshot_project(project, "train", {"-m": "/models"})

    assert snapshot.tab_name == "train"
    assert snapshot.tasks == {"extract": {"-i": "old"}, "train": {"-m": "/models"}}


def test_snapshot_task_contains_only_current_command() -> None:
    """Task snapshots should only contain the selected command."""
    service = ProjectSessionService()

    snapshot = service.snapshot_task("convert", {"-i": "/input"})

    assert snapshot.tab_name == "convert"
    assert snapshot.tasks == {"convert": {"-i": "/input"}}


def test_selected_task_prefers_tab_name_then_first_task() -> None:
    """Loaded project selection should prefer tab_name when available."""
    service = ProjectSessionService()
    project = ProjectFile(tab_name="train", tasks={"extract": {"-i": "in"}, "train": {"-m": "m"}})
    fallback = ProjectFile(tab_name="missing", tasks={"extract": {"-i": "in"}})

    assert service.selected_task(project) == ("train", {"-m": "m"})
    assert service.selected_task(fallback) == ("extract", {"-i": "in"})


def test_last_session_store_round_trips_existing_file(tmp_path: Path) -> None:
    """LastSessionStore should persist valid project/task entries."""
    session_file = tmp_path / "last.json"
    project_file = tmp_path / "example.fsw"
    project_file.write_text("{}", encoding="utf-8")
    store = LastSessionStore(get_serializer("json"), str(session_file))

    store.save(str(project_file), PROJECT_KIND)
    loaded = store.load()

    assert loaded is not None
    assert loaded.filename == str(project_file)
    assert loaded.kind == PROJECT_KIND


def test_last_session_store_ignores_missing_file(tmp_path: Path) -> None:
    """LastSessionStore should not restore entries whose target file disappeared."""
    store = LastSessionStore(get_serializer("json"), str(tmp_path / "last.json"))
    missing = tmp_path / "missing.fst"

    store.save(str(missing), TASK_KIND)

    assert store.load() is None


def test_last_session_store_clear_removes_cache(tmp_path: Path) -> None:
    """LastSessionStore.clear should delete the cache file."""
    session_file = tmp_path / "last.json"
    target = tmp_path / "target.fsw"
    target.write_text("{}", encoding="utf-8")
    store = LastSessionStore(get_serializer("json"), str(session_file))
    store.save(str(target), PROJECT_KIND)

    store.clear()

    assert session_file.exists() is False
