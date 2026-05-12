#!/usr/bin/env python3
"""Integration tests for GUI session service wiring."""

from __future__ import annotations

import typing as T
from pathlib import Path

from lib.gui.project import _GuiSession  # pylint:disable=protected-access
from lib.gui.services.project_session_state import ProjectSessionState
from lib.gui.services.project_store import ProjectStore


class _ProjectStore:
    """Project store test double."""

    def __init__(self) -> None:
        self.saved_filename: str | None = None
        self.saved_project = None

    def load(self, filename: str):
        """Return a migrated project model."""
        return ProjectStore(
            _Serializer(
                {
                    "extract": {"Input Dir": "/input"},
                    "tab_name": "extract",
                }
            )
        ).load(filename)

    def save(self, filename: str, project) -> None:
        """Capture save arguments."""
        self.saved_filename = filename
        self.saved_project = project


class _Serializer:
    """Serializer test double."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def load(self, filename: str) -> dict[str, object]:
        """Return the configured payload."""
        return self.payload

    def save(self, filename: str, payload: dict[str, object]) -> None:
        """Unused save stub."""


class _CliOptions:
    """CLI option state test double."""

    def get_option_values(self, command: str | None = None):
        """Return current GUI option values."""
        if command is None:
            return {"extract": {"Input Dir": "/input"}}
        return {command: {"Input Dir": "/input"}}


class _BoolVar:
    """Boolean variable test double."""

    def __init__(self, value: bool = False) -> None:
        self.value = value

    def get(self) -> bool:
        """Return the stored value."""
        return self.value

    def set(self, value: bool) -> None:
        """Set the stored value."""
        self.value = value


class _TkVars:
    """Minimal tk vars test double."""

    def __init__(self) -> None:
        self.console_clear = _BoolVar(False)


class _Config:
    """Minimal config test double."""

    def __init__(self) -> None:
        self.cli_opts = _CliOptions()
        self.modified_vars = {}
        self.tk_vars = _TkVars()


class _ModifiedTracker:
    """Modified-state tracker test double."""

    def __init__(self) -> None:
        self.reset_command: str | None = "unset"

    def reset(self, command: str | None = None) -> None:
        """Capture reset command."""
        self.reset_command = command


class _RecentFiles:
    """Recent-files store test double."""

    def __init__(self) -> None:
        self.added: tuple[str, str] | None = None

    def add(self, filename: str, kind: str) -> None:
        """Capture recent-file add arguments."""
        self.added = (filename, kind)


class _OptionApplier:
    """GUI option applier test double."""

    def __init__(self, applied: bool) -> None:
        self.applied = applied
        self.calls: list[tuple[dict[str, str | dict[str, T.Any]], str | None]] = []

    def apply_project(
        self,
        options: dict[str, str | dict[str, T.Any]],
        *,
        command: str | None = None,
    ) -> bool:
        """Capture apply arguments and return configured result."""
        self.calls.append((options, command))
        return self.applied


def _session(tmp_path: Path) -> tuple[_GuiSession, _ProjectStore]:
    """Build a minimally initialized GUI session."""
    store = _ProjectStore()
    session = _GuiSession.__new__(_GuiSession)
    filename = str(tmp_path / "project.fsw")
    Path(filename).touch()

    session._project_store = store  # pylint:disable=protected-access
    session._config = _Config()
    session._file_picker = None  # pylint:disable=protected-access
    session._saved_tasks = None  # pylint:disable=protected-access
    session._modified = False  # pylint:disable=protected-access
    session._modified_tracker = _ModifiedTracker()  # pylint:disable=protected-access
    session._recent_files = _RecentFiles()  # pylint:disable=protected-access
    session._option_applier = _OptionApplier(True)  # pylint:disable=protected-access
    session._state = ProjectSessionState(filename=filename)  # pylint:disable=protected-access
    return session, store


def test_gui_session_load_uses_project_store(tmp_path: Path) -> None:
    """Loading migrates project files into session state."""
    session, _ = _session(tmp_path)
    filename = session._state.filename  # pylint:disable=protected-access

    session._load_state()  # pylint:disable=protected-access

    assert session._state.filename == filename  # pylint:disable=protected-access
    assert session._state.options == {  # pylint:disable=protected-access
        "extract": {"Input Dir": "/input"},
        "tab_name": "extract",
    }
    assert session._state.project is not None  # pylint:disable=protected-access


def test_gui_session_save_uses_project_store(tmp_path: Path, monkeypatch) -> None:
    """Saving writes a versioned ProjectFile through ProjectStore."""
    session, store = _session(tmp_path)
    monkeypatch.setattr(type(session), "_active_tab", property(lambda self: "extract"))

    session._save()  # pylint:disable=protected-access

    assert store.saved_filename == session._state.filename  # pylint:disable=protected-access
    assert store.saved_project.model_dump() == {
        "version": 2,
        "tab_name": "extract",
        "tasks": {"extract": {"Input Dir": "/input"}},
    }
    assert session._state.options == {  # pylint:disable=protected-access
        "extract": {"Input Dir": "/input"},
        "tab_name": "extract",
    }
    assert session._modified_tracker.reset_command is None  # pylint:disable=protected-access
    assert session._recent_files.added == (  # pylint:disable=protected-access
        session._state.filename,  # pylint:disable=protected-access
        "project",
    )


def test_gui_session_set_options_delegates_to_option_applier(
    tmp_path: Path,
) -> None:
    """Loaded options should be delegated to the GUI option applier."""
    session, _ = _session(tmp_path)
    options: dict[str, str | dict[str, T.Any]] = {
        "tab_name": "extract",
        "extract": {"Input Dir": "/input"},
    }
    session._state.set_options(options)  # pylint:disable=protected-access

    session._set_options()  # pylint:disable=protected-access

    assert session._option_applier.calls == [  # pylint:disable=protected-access
        (options, None)
    ]
    assert session._config.tk_vars.console_clear.get() is False


def test_gui_session_set_options_delegates_to_option_applier_missing_command(
    tmp_path: Path,
) -> None:
    """Missing command option application should preserve console clear behavior."""
    session, _ = _session(tmp_path)
    options: dict[str, str | dict[str, T.Any]] = {
        "tab_name": "extract",
        "extract": {"Input Dir": "/input"},
    }
    session._state.set_options(options)  # pylint:disable=protected-access
    session._option_applier = _OptionApplier(False)  # pylint:disable=protected-access

    session._set_options("train")  # pylint:disable=protected-access

    assert session._option_applier.calls == [  # pylint:disable=protected-access
        (options, "train")
    ]
    assert session._config.tk_vars.console_clear.get() is True


def test_gui_session_uses_state_as_source_of_truth(tmp_path: Path) -> None:
    """Session helpers should use ProjectSessionState rather than legacy attributes."""
    session, _ = _session(tmp_path)
    options: dict[str, str | dict[str, T.Any]] = {
        "tab_name": "extract",
        "extract": {"Input Dir": "/input"},
    }
    session._state.set_options(options)  # pylint:disable=protected-access

    assert not hasattr(session, "_filename")
    assert not hasattr(session, "_options")
    assert session._cli_options == {"extract": {"Input Dir": "/input"}}  # pylint:disable=protected-access
