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
    assert state.has_file is False
    assert state.has_options is False
    assert state.dirname is None
    assert state.basename is None
    assert state.tab_name is None
    assert state.options is None
    assert state.cli_options == {}
    assert state.legacy_options == {}


def test_load_exposes_file_and_option_helpers(tmp_path: Path) -> None:
    """Loaded state should expose filename and project-derived options."""
    filename = tmp_path / "project.fsw"
    filename.write_text("{}", encoding="utf-8")
    project = ProjectFile(tab_name="train", tasks={"train": {"Model Dir": "/model"}})
    state = ProjectSessionState()

    state.load(str(filename), project)

    assert state.filename == str(filename)
    assert state.has_file is True
    assert state.has_options is True
    assert state.dirname == str(tmp_path)
    assert state.basename == "project.fsw"
    assert state.tab_name == "train"
    assert state.options == {
        "tab_name": "train",
        "train": {"Model Dir": "/model"},
    }
    assert state.cli_options == {"train": {"Model Dir": "/model"}}
    assert state.legacy_options == {
        "tab_name": "train",
        "train": {"Model Dir": "/model"},
    }


def test_loaded_options_are_mutable_source_of_truth(tmp_path: Path) -> None:
    """Loaded ProjectFile should be converted to mutable session options only."""
    filename = tmp_path / "project.fsw"
    filename.write_text("{}", encoding="utf-8")
    state = ProjectSessionState()

    state.load(
        str(filename),
        ProjectFile(tab_name="extract", tasks={"extract": {"Mode": "bad-choice"}}),
    )

    assert state.options is not None
    extract_options = state.options["extract"]
    assert isinstance(extract_options, dict)

    extract_options["Mode"] = "default-choice"

    assert state.legacy_options == {
        "tab_name": "extract",
        "extract": {"Mode": "default-choice"},
    }
    assert state.cli_options == {"extract": {"Mode": "default-choice"}}


def test_set_project_replaces_options_with_project_legacy_shape() -> None:
    """Setting a project should replace session options with its legacy shape."""
    state = ProjectSessionState()
    state.set_options(
        {
            "tab_name": "extract",
            "extract": {"Input Dir": "/old"},
        }
    )

    state.set_project(
        ProjectFile(tab_name="train", tasks={"train": {"Model Dir": "/model"}})
    )

    assert state.tab_name == "train"
    assert state.options == {
        "tab_name": "train",
        "train": {"Model Dir": "/model"},
    }
    assert state.cli_options == {"train": {"Model Dir": "/model"}}


def test_set_options_exposes_tab_name_and_cli_options() -> None:
    """Raw legacy options should be exposed as the current mutable state."""
    state = ProjectSessionState()
    options = {
        "tab_name": "convert",
        "convert": {"Input Dir": "/input"},
        "project": "/project.fsw",
    }

    state.set_options(options)

    assert state.has_options is True
    assert state.tab_name == "convert"
    assert state.options is options
    assert state.legacy_options is options
    assert state.cli_options == {"convert": {"Input Dir": "/input"}}


def test_set_legacy_sets_filename_and_options(tmp_path: Path) -> None:
    """Legacy task state should set filename and options together."""
    filename = tmp_path / "task.fst"
    filename.write_text("{}", encoding="utf-8")
    options = {
        "tab_name": "extract",
        "extract": {"Input Dir": "/input"},
    }
    state = ProjectSessionState()

    state.set_legacy(str(filename), options)

    assert state.filename == str(filename)
    assert state.has_file is True
    assert state.tab_name == "extract"
    assert state.options is options


def test_clear_filename_keeps_options() -> None:
    """Clearing filename should not clear loaded options."""
    state = ProjectSessionState(
        filename="/project.fsw",
        _options={
            "tab_name": "extract",
            "extract": {"Input Dir": "/input"},
        },
    )

    state.clear_filename()

    assert state.filename is None
    assert state.options == {
        "tab_name": "extract",
        "extract": {"Input Dir": "/input"},
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
    """Clearing state should remove filename and options."""
    state = ProjectSessionState(
        filename="/project.fsw",
        _options={
            "tab_name": "extract",
            "extract": {},
        },
    )

    state.clear()

    assert state.filename is None
    assert state.has_file is False
    assert state.has_options is False
    assert state.options is None
    assert state.cli_options == {}
    assert state.legacy_options == {}
