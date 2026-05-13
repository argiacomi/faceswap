#!/usr/bin/env python3
"""Project, task and last-session helpers shared by GUI shells."""

from __future__ import annotations

import typing as T
from dataclasses import dataclass, field
from pathlib import Path

from lib.utils import get_module_objects

from ..models.project import ProjectFile

PROJECT_KIND = "project"
TASK_KIND = "task"
PROJECT_EXTENSION = ".fsw"
TASK_EXTENSION = ".fst"


@dataclass(frozen=True)
class LastSession:
    """Last opened project/task entry for startup restore."""

    filename: str
    kind: str
    ui_state: dict[str, T.Any] = field(default_factory=dict)


class ProjectSessionService:
    """Pure helpers for Qt project/task/session lifecycle behavior."""

    @staticmethod
    def kind_from_filename(filename: str | Path, *, default: str = PROJECT_KIND) -> str:
        """Return file kind for a project/task filename."""
        suffix = Path(filename).suffix.lower()
        if suffix == TASK_EXTENSION:
            return TASK_KIND
        if suffix == PROJECT_EXTENSION:
            return PROJECT_KIND
        return default

    @classmethod
    def normalize_kind(
        cls,
        kind: str | None,
        filename: str | Path | None = None,
    ) -> str:
        """Normalize current and legacy project/task kind values."""
        if kind == PROJECT_KIND:
            return PROJECT_KIND
        if kind == TASK_KIND:
            return TASK_KIND
        if filename is not None and Path(filename).suffix.lower() == PROJECT_EXTENSION:
            return PROJECT_KIND if kind is None else TASK_KIND
        if filename is not None and Path(filename).suffix.lower() == TASK_EXTENSION:
            return TASK_KIND
        return TASK_KIND if kind else PROJECT_KIND

    @staticmethod
    def title(filename: str | None, *, modified: bool) -> str:
        """Return a window title for current filename and dirty state."""
        basename = Path(filename).name if filename else "Untitled"
        dirty = "*" if modified else ""
        return f"Faceswap Qt Shell Prototype - {basename}{dirty}"

    @staticmethod
    def snapshot_project(
        project: ProjectFile,
        command: str,
        values: T.Mapping[str, object],
    ) -> ProjectFile:
        """Return a full project snapshot with the current command state merged in."""
        tasks = {name: dict(options) for name, options in project.tasks.items()}
        if command:
            tasks[command] = dict(values)
        return ProjectFile(tab_name=command or project.tab_name, tasks=tasks)

    @staticmethod
    def snapshot_task(command: str, values: T.Mapping[str, object]) -> ProjectFile:
        """Return a single-task project model suitable for `.fst` files."""
        if not command:
            raise ValueError("Cannot save a task without a selected command")
        return ProjectFile(tab_name=command, tasks={command: dict(values)})

    @staticmethod
    def selected_task(project: ProjectFile) -> tuple[str, dict[str, object]]:
        """Return selected command and values from a loaded project/task file."""
        command = project.tab_name if project.tab_name in project.tasks else None
        if command is None and project.tasks:
            command = next(iter(project.tasks))
        if command is None:
            raise ValueError("Project does not contain any tasks")
        return command, dict(project.tasks.get(command, {}))


class LastSessionStore:
    """Persistence wrapper for restoring the last opened project/task."""

    def __init__(self, serializer: T.Any, filename: str) -> None:
        self._serializer = serializer
        self._filename = Path(filename)

    @property
    def filename(self) -> str:
        """Return the last-session cache filename."""
        return str(self._filename)

    def load(self) -> LastSession | None:
        """Load the last session entry, if it is valid and still exists."""
        if not self._filename.exists() or self._filename.stat().st_size == 0:
            return None
        payload = self._serializer.load(str(self._filename))
        if not isinstance(payload, dict):
            return None
        filename = payload.get("filename")
        kind = payload.get("kind")
        ui_state = payload.get("ui_state", {})
        if not isinstance(filename, str) or not isinstance(kind, str):
            return None
        kind = ProjectSessionService.normalize_kind(kind, filename)
        if not Path(filename).exists():
            return None
        if not isinstance(ui_state, dict):
            ui_state = {}
        return LastSession(filename, kind, dict(ui_state))

    def save(
        self,
        filename: str,
        kind: str,
        ui_state: T.Mapping[str, T.Any] | None = None,
    ) -> None:
        """Save a last-session entry."""
        self._filename.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, T.Any] = {
            "filename": filename,
            "kind": ProjectSessionService.normalize_kind(kind, filename),
        }
        if ui_state is not None:
            payload["ui_state"] = dict(ui_state)
        self._serializer.save(str(self._filename), payload)

    def clear(self) -> None:
        """Clear the last-session entry."""
        if self._filename.exists():
            self._filename.unlink()


__all__ = get_module_objects(__name__)
