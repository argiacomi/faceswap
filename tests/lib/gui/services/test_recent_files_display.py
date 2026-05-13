#!/usr/bin/env python3
"""Tests for recent-file display label helpers."""

from __future__ import annotations

from pathlib import Path

from lib.gui.services.recent_files_store import RecentFile, RecentFilesStore
from lib.serializer import get_serializer


def _store(tmp_path: Path) -> RecentFilesStore:
    """Return a recent-files store in a temp cache."""
    return RecentFilesStore(get_serializer("json"), str(tmp_path / "recent.json"))


def test_recent_file_display_uses_basename_for_unique_files(tmp_path: Path) -> None:
    """Unique recent filenames should use compact basename labels."""
    store = _store(tmp_path)
    recent_files = [RecentFile(str(tmp_path / "project.fsw"), "project")]

    display_items = store.display_items(recent_files)

    assert len(display_items) == 1
    assert display_items[0].file == recent_files[0]
    assert display_items[0].label == "Project: project.fsw"
    assert display_items[0].tooltip == str(tmp_path / "project.fsw")


def test_recent_file_display_disambiguates_duplicate_basenames(tmp_path: Path) -> None:
    """Duplicate basenames should include their parent folder in the label."""
    first = tmp_path / "first" / "project.fsw"
    second = tmp_path / "second" / "project.fsw"
    recent_files = [RecentFile(str(first), "project"), RecentFile(str(second), "project")]
    store = _store(tmp_path)

    display_items = store.display_items(recent_files)

    assert display_items[0].label == f"Project: project.fsw ({first.parent})"
    assert display_items[1].label == f"Project: project.fsw ({second.parent})"


def test_recent_file_display_keeps_kind_in_label(tmp_path: Path) -> None:
    """Display labels should include the project/task kind prefix."""
    task = tmp_path / "task.fst"
    project = tmp_path / "project.fsw"
    recent_files = [RecentFile(str(task), "task"), RecentFile(str(project), "project")]
    store = _store(tmp_path)

    display_items = store.display_items(recent_files)

    assert [item.label for item in display_items] == [
        "Task: task.fst",
        "Project: project.fsw",
    ]


def test_recent_file_display_loads_from_store_when_not_provided(tmp_path: Path) -> None:
    """display_items should load saved recents when no explicit list is provided."""
    store = _store(tmp_path)
    filename = str(tmp_path / "project.fsw")
    store.add(filename, "project")

    display_items = store.display_items()

    assert len(display_items) == 1
    assert display_items[0].label == "Project: project.fsw"
    assert display_items[0].tooltip == filename
