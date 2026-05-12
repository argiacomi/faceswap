#!/usr/bin/env python3
"""Tests for GUI project session state."""

from __future__ import annotations

from pathlib import Path

from lib.gui.models.project import ProjectFile
from lib.gui.services.project_session_state import ProjectSessionState


def test_empty_state() -> None:
    """Empty session state should expose empty values."""
    state = ProjectSessionState()

    assert state.filename is None
    assert state.project is None
    assert state.has_file is False
    assert state.dirname is None
    assert state.basename is None
    assert state.tab_name is None
    assert state.legacy_options == {}


def test_load_exposes_file_and_project_helpers(tmp_path: Path) -> None:
    """Loaded state should expose filename and project-derived helpers."""
    filename = tmp_path / "project.fsw"
    filename.write_text("{}", encoding="utf-8")
    project = ProjectFile(tab_name="train", tasks={"train": {"Model Dir": "/model"}})
    state = ProjectSessionState()

    state.load(str(filename), project)

    assert state.filename == str(filename)
    assert state.project == project
    assert state.has_file is True
    assert state.dirname == str(tmp_path)
    assert state.basename == "project.fsw"
    assert state.tab_name == "train"
    assert state.legacy_options == {
        "tab_name": "train",
        "train": {"Model Dir": "/model"},
    }


def test_has_file_false_for_missing_loaded_filename(tmp_path: Path) -> None:
    """Loaded state should not report a missing filename as existing."""
    filename = tmp_path / "missing.fsw"
    state = ProjectSessionState()

    state.load(str(filename), ProjectFile())

    assert state.has_file is False
    assert state.dirname == str(tmp_path)
    assert state.basename == "missing.fsw"


def test_clear() -> None:
    """Clearing state should remove filename and project references."""
    state = ProjectSessionState(
        filename="/project.fsw",
        project=ProjectFile(tab_name="extract", tasks={"extract": {}}),
    )

    state.clear()

    assert state.filename is None
    assert state.project is None
    assert state.has_file is False
    assert state.legacy_options == {}
