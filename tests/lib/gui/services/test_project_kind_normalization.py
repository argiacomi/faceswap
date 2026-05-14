#!/usr/bin/env python3
"""Focused tests for Qt project/task kind normalization."""

from __future__ import annotations

from pathlib import Path

from lib.gui.services.project_session_service import PROJECT_KIND, TASK_KIND, LastSessionStore, ProjectSessionService
from lib.serializer import get_serializer


def test_normalize_kind_preserves_current_kinds() -> None:
    """Current project/task kind values should be preserved."""
    service = ProjectSessionService()

    assert service.normalize_kind(PROJECT_KIND, "task.fst") == PROJECT_KIND
    assert service.normalize_kind(TASK_KIND, "project.fsw") == TASK_KIND


def test_normalize_kind_maps_legacy_command_kinds_to_task() -> None:
    """Legacy command-kind entries should restore as task entries."""
    service = ProjectSessionService()

    assert service.normalize_kind("extract", "project.fsw") == TASK_KIND
    assert service.normalize_kind("convert", "task.fst") == TASK_KIND
    assert service.normalize_kind("train", "unknown.json") == TASK_KIND


def test_normalize_kind_falls_back_to_extension_for_invalid_values() -> None:
    """Unexpected kind values should fall back to the file extension, not task by default."""
    service = ProjectSessionService()

    assert service.normalize_kind("bad", "project.fsw") == PROJECT_KIND
    assert service.normalize_kind("bad", "task.fst") == TASK_KIND
    assert service.normalize_kind("bad", "unknown.json") == PROJECT_KIND
    assert service.normalize_kind(None, "project.fsw") == PROJECT_KIND


def test_last_session_load_uses_extension_for_invalid_cached_kind(tmp_path: Path) -> None:
    """Invalid cached last-session kind should restore according to file extension."""
    session_file = tmp_path / "last.json"
    project_file = tmp_path / "example.fsw"
    project_file.write_text("{}", encoding="utf-8")
    serializer = get_serializer("json")
    serializer.save(str(session_file), {"filename": str(project_file), "kind": "bad"})
    store = LastSessionStore(serializer, str(session_file))

    loaded = store.load()

    assert loaded is not None
    assert loaded.kind == PROJECT_KIND
