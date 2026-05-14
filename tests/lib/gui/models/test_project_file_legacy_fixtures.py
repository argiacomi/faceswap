#!/usr/bin/env python3
"""Fixture-style tests for Qt project/task legacy compatibility."""

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

LEGACY_FLAT_PROJECT = {
    "tab_name": "train",
    "extract": {"-i": "/data/input", "-o": "/data/faces"},
    "train": {"-m": "/models", "-t": "original"},
}

LEGACY_WRAPPED_PROJECT = {
    "command": "convert",
    "options": {
        "extract": {"-i": "/data/input"},
        "convert": {"-i": "/data/faces", "-o": "/data/output"},
    },
}

CURRENT_PROJECT = {
    "version": 2,
    "tab_name": "train",
    "tasks": {
        "extract": {"-i": "/data/input"},
        "train": {"-m": "/models", "-t": "model"},
    },
}

LEGACY_TASK = {
    "task": {
        "convert": {"-i": "/data/faces", "-o": "/data/output"},
        "tab_name": "convert",
    }
}


def test_project_file_loads_flat_legacy_fsw_shape() -> None:
    """Legacy flat .fsw-like payloads should migrate to versioned project files."""
    project = ProjectFile.from_mapping(LEGACY_FLAT_PROJECT)

    assert project.version == ProjectFile.CURRENT_VERSION
    assert project.tab_name == "train"
    assert project.tasks == {
        "extract": {"-i": "/data/input", "-o": "/data/faces"},
        "train": {"-m": "/models", "-t": "original"},
    }


def test_project_file_loads_wrapped_legacy_project_shape() -> None:
    """Legacy wrapped project payloads should resolve command/tab and task mappings."""
    project = ProjectFile.from_mapping(LEGACY_WRAPPED_PROJECT)

    assert project.tab_name == "convert"
    assert project.tasks["convert"] == {"-i": "/data/faces", "-o": "/data/output"}


def test_project_file_loads_current_versioned_project_shape() -> None:
    """Current .fsw payloads should round-trip through the versioned model."""
    project = ProjectFile.from_mapping(CURRENT_PROJECT)

    assert project.model_dump() == CURRENT_PROJECT


def test_project_file_loads_legacy_task_wrapper_shape() -> None:
    """Legacy .fst-like task wrappers should migrate to selected task models."""
    project = ProjectFile.from_mapping(LEGACY_TASK)

    assert project.tab_name == "convert"
    assert project.tasks == {"convert": {"-i": "/data/faces", "-o": "/data/output"}}


def test_project_session_selected_task_uses_active_tab_or_first_task() -> None:
    """Session selection should prefer active tab and fall back to first available task."""
    service = ProjectSessionService()

    assert service.selected_task(ProjectFile.from_mapping(CURRENT_PROJECT)) == (
        "train",
        {"-m": "/models", "-t": "model"},
    )
    assert service.selected_task(
        ProjectFile(tab_name="missing", tasks={"extract": {"-i": "in"}})
    ) == (
        "extract",
        {"-i": "in"},
    )


def test_last_session_store_preserves_display_ui_state(tmp_path: Path) -> None:
    """Last-session restore cache should preserve display/runtime UI state dictionaries."""
    project_file = tmp_path / "legacy.fsw"
    project_file.write_text("{}", encoding="utf-8")
    store = LastSessionStore(get_serializer("json"), str(tmp_path / "last-session.json"))
    ui_state = {
        "display_tab": "Preview",
        "analysis_source": "/models/model_state.json",
        "preview_source": "/tmp/preview",
        "graph_source": "/models/model_state.json",
        "window_size": [1280, 760],
        "splitter_sizes": [400, 880],
    }

    store.save(str(project_file), PROJECT_KIND, ui_state)
    loaded = store.load()

    assert loaded is not None
    assert loaded.kind == PROJECT_KIND
    assert loaded.ui_state == ui_state


def test_last_session_store_normalizes_legacy_task_kind(tmp_path: Path) -> None:
    """Legacy command kind values in last-session cache should restore as task files."""
    task_file = tmp_path / "legacy.fst"
    task_file.write_text("{}", encoding="utf-8")
    serializer = get_serializer("json")
    serializer.save(
        str(tmp_path / "last-session.json"),
        {"filename": str(task_file), "kind": "train"},
    )
    store = LastSessionStore(serializer, str(tmp_path / "last-session.json"))

    loaded = store.load()

    assert loaded is not None
    assert loaded.kind == TASK_KIND


def test_last_session_store_uses_extension_for_invalid_kind(tmp_path: Path) -> None:
    """Invalid cached kind values should fall back to .fsw/.fst extension semantics."""
    project_file = tmp_path / "legacy.fsw"
    project_file.write_text("{}", encoding="utf-8")
    serializer = get_serializer("json")
    serializer.save(
        str(tmp_path / "last-session.json"),
        {"filename": str(project_file), "kind": "unexpected"},
    )
    store = LastSessionStore(serializer, str(tmp_path / "last-session.json"))

    loaded = store.load()

    assert loaded is not None
    assert loaded.kind == PROJECT_KIND
