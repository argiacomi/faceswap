#!/usr/bin/env python3
"""Tests for GUI session file picking."""

from __future__ import annotations

from pathlib import Path

from lib.gui.services.session_file_picker import PickedFile, SessionFilePicker


class _File:
    """File dialog return test double."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False

    def close(self) -> None:
        """Mark the file as closed."""
        self.closed = True


class _DialogResult:
    """File handler result test double."""

    def __init__(self, return_file: _File | None) -> None:
        self.return_file = return_file


class _FileHandler:
    """File handler test double."""

    def __init__(self, return_file: _File | None) -> None:
        self.return_file = return_file
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def __call__(self, action: str, handler: str, **kwargs) -> _DialogResult:
        """Capture handler calls and return the configured file."""
        self.calls.append((action, handler, kwargs))
        return _DialogResult(self.return_file)


def test_picked_file_path_helpers(tmp_path: Path) -> None:
    """PickedFile should expose dirname and basename helpers."""
    filename = tmp_path / "project.fsw"
    picked = PickedFile(str(filename))

    assert picked.dirname == str(tmp_path)
    assert picked.basename == "project.fsw"


def test_open_uses_provided_existing_project_filename(tmp_path: Path) -> None:
    """Provided project filenames should be validated and returned."""
    filename = tmp_path / "project.fsw"
    filename.write_text("{}", encoding="utf-8")
    file_handler = _FileHandler(None)
    picker = SessionFilePicker(file_handler)

    picked = picker.open("project", str(filename))

    assert picked == PickedFile(str(filename))
    assert file_handler.calls == []


def test_open_rejects_missing_file(tmp_path: Path) -> None:
    """Missing existing files should be rejected."""
    picker = SessionFilePicker(_FileHandler(None))

    assert picker.open("project", str(tmp_path / "missing.fsw")) is None


def test_open_rejects_wrong_project_extension(tmp_path: Path) -> None:
    """Project files must use the .fsw extension."""
    filename = tmp_path / "task.fst"
    filename.write_text("{}", encoding="utf-8")
    picker = SessionFilePicker(_FileHandler(None))

    assert picker.open("project", str(filename)) is None


def test_open_rejects_wrong_task_extension(tmp_path: Path) -> None:
    """Task files must use the .fst extension."""
    filename = tmp_path / "project.fsw"
    filename.write_text("{}", encoding="utf-8")
    picker = SessionFilePicker(_FileHandler(None))

    assert picker.open("task", str(filename)) is None


def test_open_all_accepts_any_existing_extension(tmp_path: Path) -> None:
    """The all session type should accept any existing file extension."""
    filename = tmp_path / "session.json"
    filename.write_text("{}", encoding="utf-8")
    picker = SessionFilePicker(_FileHandler(None))

    assert picker.open("all", str(filename)) == PickedFile(str(filename))


def test_open_uses_file_handler_when_filename_missing(tmp_path: Path) -> None:
    """Open should call the matching file handler when no filename is provided."""
    filename = tmp_path / "project.fsw"
    filename.write_text("{}", encoding="utf-8")
    return_file = _File(str(filename))
    file_handler = _FileHandler(return_file)
    picker = SessionFilePicker(file_handler)

    picked = picker.open("project")

    assert picked == PickedFile(str(filename))
    assert return_file.closed is True
    assert file_handler.calls == [("open", "config_project", {})]


def test_open_rejects_dialog_file_with_wrong_extension(tmp_path: Path) -> None:
    """Dialog-selected project files with the wrong extension should be rejected."""
    filename = tmp_path / "task.fst"
    filename.write_text("{}", encoding="utf-8")
    return_file = _File(str(filename))
    picker = SessionFilePicker(_FileHandler(return_file))

    assert picker.open("project") is None
    assert return_file.closed is True


def test_open_returns_none_when_dialog_cancelled() -> None:
    """Open should return None when the file dialog is cancelled."""
    file_handler = _FileHandler(None)
    picker = SessionFilePicker(file_handler)

    assert picker.open("task") is None
    assert file_handler.calls == [("open", "config_task", {})]


def test_save_as_returns_selected_file(tmp_path: Path) -> None:
    """Save-as should return the selected file and close the file object."""
    filename = tmp_path / "task.fst"
    return_file = _File(str(filename))
    file_handler = _FileHandler(return_file)
    picker = SessionFilePicker(file_handler)

    picked = picker.save_as(
        "task", title="Save Task As...", initial_folder=str(tmp_path)
    )

    assert picked == PickedFile(str(filename))
    assert return_file.closed is True
    assert file_handler.calls == [
        (
            "save",
            "config_task",
            {"title": "Save Task As...", "initial_folder": str(tmp_path)},
        )
    ]


def test_save_as_returns_none_when_dialog_cancelled() -> None:
    """Save-as should return None when the file dialog is cancelled."""
    file_handler = _FileHandler(None)
    picker = SessionFilePicker(file_handler)

    assert picker.save_as("project", title="Save Project As...") is None
    assert file_handler.calls == [
        (
            "save",
            "config_project",
            {"title": "Save Project As...", "initial_folder": None},
        )
    ]
