#!/usr/bin/env python3
"""Tests for GUI recent files persistence."""

from __future__ import annotations

from pathlib import Path

from lib.gui.services.recent_files_store import RecentFile, RecentFilesStore


class _Serializer:
    """In-memory serializer test double."""

    def __init__(self, payload: object | None = None) -> None:
        self.payload = payload
        self.saved_filename: str | None = None
        self.saved_payload: object | None = None

    def load(self, filename: str) -> object:
        """Return stored payload."""
        return self.payload

    def save(self, filename: str, payload: object) -> None:
        """Store save call arguments."""
        self.saved_filename = filename
        self.saved_payload = payload
        self.payload = payload


def _store(
    tmp_path: Path, payload: object | None = None
) -> tuple[RecentFilesStore, _Serializer]:
    """Build a recent files store test subject."""
    filename = tmp_path / ".recent.json"
    serializer = _Serializer(payload)
    if payload is not None:
        filename.write_text("{}", encoding="utf-8")
    return RecentFilesStore(serializer, str(filename)), serializer


def test_load_returns_empty_list_when_missing(tmp_path: Path) -> None:
    """Missing recent files should load as empty."""
    store, _ = _store(tmp_path)

    assert store.load() == []


def test_load_decodes_valid_rows_and_skips_invalid_rows(tmp_path: Path) -> None:
    """Recent files payloads are decoded defensively."""
    store, _ = _store(
        tmp_path,
        payload=[
            ["/project.fsw", "project"],
            ["/task.fst", "extract"],
            ["missing-kind"],
            [123, "project"],
            "not-a-row",
        ],
    )

    assert store.load() == [
        RecentFile(filename="/project.fsw", kind="project"),
        RecentFile(filename="/task.fst", kind="extract"),
    ]


def test_add_deduplicates_moves_to_front_and_saves(tmp_path: Path) -> None:
    """Adding an existing recent file should move it to the front."""
    store, serializer = _store(
        tmp_path,
        payload=[
            ["/old.fsw", "project"],
            ["/task.fst", "extract"],
        ],
    )

    recent_files = store.add("/task.fst", "train")

    assert recent_files == [
        RecentFile(filename="/task.fst", kind="train"),
        RecentFile(filename="/old.fsw", kind="project"),
    ]
    assert serializer.saved_payload == [
        ("/task.fst", "train"),
        ("/old.fsw", "project"),
    ]


def test_add_limits_saved_items(tmp_path: Path) -> None:
    """Recent files should be capped to the configured limit."""
    payload = [[f"/{idx}.fsw", "project"] for idx in range(3)]
    store, serializer = _store(tmp_path, payload=payload)
    store = RecentFilesStore(serializer, store.filename, limit=2)

    recent_files = store.add("/new.fsw", "project")

    assert recent_files == [
        RecentFile(filename="/new.fsw", kind="project"),
        RecentFile(filename="/0.fsw", kind="project"),
    ]
    assert serializer.saved_payload == [
        ("/new.fsw", "project"),
        ("/0.fsw", "project"),
    ]


def test_remove_handles_missing_files(tmp_path: Path) -> None:
    """Removing a missing file should leave the list unchanged."""
    store, serializer = _store(tmp_path, payload=[["/old.fsw", "project"]])

    recent_files = store.remove("/missing.fsw")

    assert recent_files == [RecentFile(filename="/old.fsw", kind="project")]
    assert serializer.saved_payload == [("/old.fsw", "project")]
