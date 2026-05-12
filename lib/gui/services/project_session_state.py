#!/usr/bin/env python3
"""Mutable project/task session state for the Faceswap GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lib.utils import get_module_objects

from ..models.project import ProjectFile

LegacyOptions = dict[str, str | dict[str, bool | int | float | str]]


@dataclass
class ProjectSessionState:
    """Mutable state for the currently loaded project or task file.

    ``ProjectFile`` is intentionally used only at load/save boundaries. The GUI mutates
    option dictionaries during normalization, validation and task reloads, so storing both
    a project model and a mutable options dict can make one stale relative to the other.

    This state object keeps the legacy GUI options as the single mutable source of truth.
    """

    filename: str | None = None
    _options: LegacyOptions | None = None

    @property
    def has_file(self) -> bool:
        """Return ``True`` when the state has an existing filename."""
        return self.filename is not None and Path(self.filename).is_file()

    @property
    def has_options(self) -> bool:
        """Return ``True`` when the state contains GUI options."""
        return self._options is not None

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
        """Return the stored tab name."""
        if self._options is None:
            return None
        retval = self._options.get("tab_name", None)
        assert retval is None or isinstance(retval, str)
        return retval

    @property
    def options(self) -> LegacyOptions | None:
        """Return loaded project options in the GUI's legacy flat shape."""
        return self._options

    @property
    def cli_options(self) -> dict[str, dict[str, bool | int | float | str]]:
        """Return GUI CLI options with non-task session fields removed."""
        if self._options is None:
            return {}
        return {key: val for key, val in self._options.items() if isinstance(val, dict)}

    @property
    def legacy_options(self) -> LegacyOptions:
        """Return loaded project options in the GUI's legacy flat shape."""
        if self._options is None:
            return {}
        return self._options

    def set_filename(self, filename: str | None) -> None:
        """Set the current project or task filename."""
        self.filename = filename

    def clear_filename(self) -> None:
        """Clear the current project or task filename without clearing options."""
        self.filename = None

    def set_project(self, project: ProjectFile) -> None:
        """Set options from a versioned project model.

        The model is not retained after conversion. ``options`` remains the mutable
        source of truth for the active GUI session.
        """
        self.set_options(project.to_legacy_options())

    def set_options(self, options: LegacyOptions | None) -> None:
        """Set raw legacy GUI options as the session source of truth."""
        self._options = options

    def set_legacy(self, filename: str | None, options: LegacyOptions | None) -> None:
        """Set filename and raw legacy options together."""
        self.filename = filename
        self.set_options(options)

    def load(self, filename: str, project: ProjectFile) -> None:
        """Load state for a project or task file."""
        self.filename = filename
        self.set_project(project)

    def clear(self) -> None:
        """Clear the current project or task file state."""
        self.filename = None
        self._options = None


__all__ = get_module_objects(__name__)
