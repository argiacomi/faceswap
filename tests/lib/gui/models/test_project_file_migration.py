#!/usr/bin/env python3
"""Tests for project file migration shapes."""

from __future__ import annotations

import pytest

from lib.gui.models.project import ProjectFile


def test_project_file_loads_current_versioned_shape() -> None:
    """Current versioned payloads should load unchanged."""
    project = ProjectFile.from_mapping(
        {
            "version": ProjectFile.CURRENT_VERSION,
            "tab_name": "train",
            "tasks": {"train": {"-m": "/models"}},
        }
    )

    assert project.version == ProjectFile.CURRENT_VERSION
    assert project.tab_name == "train"
    assert project.tasks == {"train": {"-m": "/models"}}


def test_project_file_loads_flat_legacy_options_shape() -> None:
    """Flat legacy option mappings should migrate to versioned tasks."""
    project = ProjectFile.from_mapping(
        {
            "tab_name": "extract",
            "extract": {"-i": "/input"},
            "train": {"-m": "/models"},
        }
    )

    assert project.tab_name == "extract"
    assert project.tasks == {
        "extract": {"-i": "/input"},
        "train": {"-m": "/models"},
    }


def test_project_file_loads_wrapped_legacy_options_shape() -> None:
    """Wrapped legacy option mappings should migrate from known option keys."""
    project = ProjectFile.from_mapping(
        {
            "command": "train",
            "options": {
                "extract": {"-i": "/input"},
                "train": {"-m": "/models"},
            },
        }
    )

    assert project.tab_name == "train"
    assert project.tasks == {
        "extract": {"-i": "/input"},
        "train": {"-m": "/models"},
    }


def test_project_file_loads_legacy_project_wrapper_shape() -> None:
    """Project/task wrappers should be accepted for migration."""
    project = ProjectFile.from_mapping(
        {
            "tab": "convert",
            "project": {
                "convert": {"-i": "/input", "-o": "/output"},
            },
        }
    )

    assert project.tab_name == "convert"
    assert project.tasks == {"convert": {"-i": "/input", "-o": "/output"}}


def test_project_file_rejects_unsupported_version() -> None:
    """Unsupported explicit versions should still fail."""
    with pytest.raises(ValueError, match="Unsupported project file version"):
        ProjectFile.from_mapping({"version": 999, "tasks": {"extract": {}}})
