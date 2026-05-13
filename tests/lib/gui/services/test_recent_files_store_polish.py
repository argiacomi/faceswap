#!/usr/bin/env python3
"""Tests for recent-file cleanup helpers."""

from __future__ import annotations

from pathlib import Path

from lib.gui.services.recent_files_store import RecentFilesStore
from lib.serializer import get_serializer


def _store(tmp_path: Path) -> RecentFilesStore:
    """Return a recent-files store in a temp cache."""
    return RecentFilesStore(get_serializer("json"), str(tmp_path / "recent.json"))


def test_recent_files_store_prunes_missing_files(tmp_path: Path) -> None:
    """prune_missing should keep only entries whose target files exist."""
    store = _store(tmp_path)
    existing = tmp_path / "existing.fsw"
    missing = tmp_path / "missing.fst"
    existing.write_text("{}", encoding="utf-8")
    store.add(str(missing), "task")
    store.add(str(existing), "project")

    recent = store.prune_missing()

    assert [(item.filename, item.kind) for item in recent] == [(str(existing), "project")]
    assert [(item.filename, item.kind) for item in store.load()] == [(str(existing), "project")]


def test_recent_files_store_clear_removes_all_entries(tmp_path: Path) -> None:
    """clear should leave an empty saved recent-file list."""
    store = _store(tmp_path)
    existing = tmp_path / "existing.fsw"
    existing.write_text("{}", encoding="utf-8")
    store.add(str(existing), "project")

    store.clear()

    assert store.load() == []
