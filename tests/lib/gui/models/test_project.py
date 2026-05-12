#!/usr/bin/env python3
"""Tests for GUI project file models."""

from __future__ import annotations

import pytest

from lib.gui.models.project import ProjectFile


def test_project_file_from_v2_mapping() -> None:
    """Versioned project files are parsed directly."""
    project = ProjectFile.from_mapping(
        {
            "version": 2,
            "tab_name": "train",
            "tasks": {"train": {"Model Dir": "/model"}},
        }
    )

    assert project.version == 2
    assert project.tab_name == "train"
    assert project.tasks == {"train": {"Model Dir": "/model"}}


def test_project_file_from_legacy_flat_mapping() -> None:
    """Legacy flat GUI option files are migrated to task payloads."""
    project = ProjectFile.from_mapping(
        {
            "tab_name": "extract",
            "extract": {"Input Dir": "/input"},
            "train": {"Model Dir": "/model"},
        }
    )

    assert project.version == 2
    assert project.tab_name == "extract"
    assert project.tasks == {
        "extract": {"Input Dir": "/input"},
        "train": {"Model Dir": "/model"},
    }


def test_project_file_rejects_unsupported_version() -> None:
    """Unsupported versioned project files should fail clearly."""
    with pytest.raises(ValueError, match="Unsupported project file version: 3"):
        ProjectFile.from_mapping({"version": 3, "tasks": {}})


def test_project_file_rejects_missing_version_for_versioned_shape() -> None:
    """Files that look versioned must include an explicit version."""
    with pytest.raises(ValueError, match="Project file version must be an integer"):
        ProjectFile.from_mapping({"tasks": {"extract": {}}})


def test_project_file_rejects_non_mapping_tasks() -> None:
    """Versioned project files must store tasks as a mapping."""
    with pytest.raises(ValueError, match="Project file tasks must be a mapping"):
        ProjectFile.from_mapping({"version": 2, "tasks": []})


def test_project_file_rejects_non_mapping_task_options() -> None:
    """Each versioned task payload must store options as a mapping."""
    with pytest.raises(
        ValueError, match="Project file task 'extract' options must be a mapping"
    ):
        ProjectFile.from_mapping({"version": 2, "tasks": {"extract": []}})


def test_project_file_to_legacy_options() -> None:
    """Project files can be converted back to the GUI's existing flat option shape."""
    project = ProjectFile(
        tab_name="convert",
        tasks={"convert": {"Input Dir": "/input"}},
    )

    assert project.to_legacy_options() == {
        "tab_name": "convert",
        "convert": {"Input Dir": "/input"},
    }


def test_project_file_model_dump() -> None:
    """Project files dump to the versioned on-disk format."""
    project = ProjectFile(tab_name="train", tasks={"train": {"A": 1}})

    assert project.model_dump() == {
        "version": 2,
        "tab_name": "train",
        "tasks": {"train": {"A": 1}},
    }
