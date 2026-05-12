#!/usr/bin/env python3
"""Mutable project/task session state for the Faceswap GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import typing as T

from lib.utils import get_module_objects

from ..models.project import ProjectFile


@dataclass
class ProjectSessionState:
    """Mutable state for the currently loaded project or task file."""

    filename: str | None = None
    project: ProjectFile | None = None

    @property
    def has_file(self) -> bool:
        """Return ``True`` when the state has an existing filename."""
        return self.filename is not None and Path(self.filename).is_file()

    @property
    def dirname(self) -> str | None:
        """Return the current file's directory name."""
        return None if self.filename is None else str(Path(self.filename).parent)

    @property
    def basename(self) -> str | None:
        """Return the current file's base name."""
        return None if self.filename is None else Path(self.filename).name

    @property
    def tab_name(self) -> str | None:
        """Return the loaded project's stored tab name."""
        return None if self.project is None else self.project.tab_name

    @property
    def legacy_options(self) -> dict[str, str | dict[str, T.Any]]:
        """Return loaded project options in the GUI's legacy flat shape."""
        if self.project is None:
            return {}
        return self.project.to_legacy_options()

    def load(self, filename: str, project: ProjectFile) -> None:
        """Load state for a project or task file."""
        self.filename = filename
        self.project = project

    def clear(self) -> None:
        """Clear the current project or task file state."""
        self.filename = None
        self.project = None


__all__ = get_module_objects(__name__)
