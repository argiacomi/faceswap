#!/usr/bin/env python3
"""Tests for project persistence service."""

from __future__ import annotations

import pytest

from lib.gui.models.project import ProjectFile
from lib.gui.services.project_store import ProjectStore


class _Serializer:
    """In-memory serializer test double."""

    def __init__(self, payload: object | None = None) -> None:
        self.payload = payload or {}
        self.saved_filename: str | None = None
        self.saved_payload: dict[str, object] | None = None

    def load(self, filename: str) -> object:
        """Return stored payload."""
        return self.payload

    def save(self, filename: str, payload: dict[str, object]) -> None:
        """Store save call arguments."""
        self.saved_filename = filename
        self.saved_payload = payload


def test_project_store_load_migrates_payload() -> None:
    """Loading returns a ProjectFile model regardless of source shape."""
    serializer = _Serializer({"tab_name": "extract", "extract": {"A": 1}})

    project = ProjectStore(serializer).load("project.fsw")

    assert project == ProjectFile(tab_name="extract", tasks={"extract": {"A": 1}})


def test_project_store_save_writes_versioned_payload() -> None:
    """Saving writes the versioned model dump."""
    serializer = _Serializer()
    project = ProjectFile(tab_name="train", tasks={"train": {"Model": "x"}})

    ProjectStore(serializer).save("project.fsw", project)

    assert serializer.saved_filename == "project.fsw"
    assert serializer.saved_payload == {
        "version": 2,
        "tab_name": "train",
        "tasks": {"train": {"Model": "x"}},
    }


def test_project_store_rejects_non_mapping_payload() -> None:
    """Corrupt project payloads should fail with a clear error."""
    serializer = _Serializer(["not", "a", "mapping"])

    with pytest.raises(ValueError, match="Project file payload must be a mapping"):
        ProjectStore(serializer).load("project.fsw")
