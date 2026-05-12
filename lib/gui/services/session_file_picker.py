#!/usr/bin/env python3
"""File selection adapter for Faceswap GUI project and task sessions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import typing as T

from lib.utils import get_module_objects

SessionType = T.Literal["all", "project", "task"]


@dataclass(frozen=True)
class PickedFile:
    """A file selected for a GUI session operation."""

    filename: str

    @property
    def dirname(self) -> str:
        """Return the selected file's directory name."""
        return str(Path(self.filename).parent)

    @property
    def basename(self) -> str:
        """Return the selected file's base name."""
        return Path(self.filename).name


class SessionFilePicker:
    """Wrap FileHandler calls and session-file extension checks."""

    _HANDLER_BY_TYPE: T.ClassVar[dict[SessionType, str]] = {
        "all": "config_all",
        "project": "config_project",
        "task": "config_task",
    }
    _EXTENSION_BY_TYPE: T.ClassVar[dict[SessionType, str | None]] = {
        "all": None,
        "project": ".fsw",
        "task": ".fst",
    }

    def __init__(self, file_handler: T.Any) -> None:
        self._file_handler = file_handler

    def open(
        self, session_type: SessionType, filename: str | None = None
    ) -> PickedFile | None:
        """Return an existing selected file for the given session type."""
        if filename is None:
            file_obj = self._file_handler(
                "open", self._HANDLER_BY_TYPE[session_type]
            ).return_file
            if not file_obj:
                return None
            filename = file_obj.name
            file_obj.close()

        return self._validate_existing(filename, session_type)

    def save_as(
        self,
        session_type: SessionType,
        *,
        title: str,
        initial_folder: str | None = None,
    ) -> PickedFile | None:
        """Return a save-as selected file for the given session type."""
        file_obj = self._file_handler(
            "save",
            self._HANDLER_BY_TYPE[session_type],
            title=title,
            initial_folder=initial_folder,
        ).return_file
        if not file_obj:
            return None

        filename = file_obj.name
        file_obj.close()
        return PickedFile(filename)

    def _validate_existing(
        self, filename: str, session_type: SessionType
    ) -> PickedFile | None:
        """Validate an existing filename for a session type."""
        path = Path(filename)
        if not path.is_file():
            return None

        expected_ext = self._EXTENSION_BY_TYPE[session_type]
        if expected_ext is not None and path.suffix != expected_ext:
            return None

        return PickedFile(filename)


__all__ = get_module_objects(__name__)
