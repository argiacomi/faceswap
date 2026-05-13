#!/usr/bin/env python3
"""Versioned project and task models for the Faceswap GUI."""

from __future__ import annotations

import typing as T
from dataclasses import dataclass, field

from lib.utils import get_module_objects


@dataclass
class ProjectFile:
    """Versioned project/task file model."""

    CURRENT_VERSION: T.ClassVar[int] = 2

    tab_name: str = "extract"
    tasks: dict[str, dict[str, T.Any]] = field(default_factory=dict)
    version: int = CURRENT_VERSION

    @classmethod
    def from_mapping(cls, payload: T.Mapping[str, T.Any]) -> ProjectFile:
        """Build a versioned model from either current or legacy on-disk data."""
        if "tasks" in payload:
            return cls._from_versioned_mapping(payload)
        legacy_options = cls._legacy_options_payload(payload)
        if legacy_options is not None:
            return cls.from_legacy_options(
                legacy_options,
                tab_name=cls._legacy_tab_name(payload, legacy_options),
            )
        if "version" in payload:
            return cls._from_versioned_mapping(payload)
        return cls.from_legacy_options(payload)

    @classmethod
    def _from_versioned_mapping(cls, payload: T.Mapping[str, T.Any]) -> ProjectFile:
        """Build a project file from the current versioned on-disk shape."""
        version = cls._version(payload.get("version"))
        tasks = payload.get("tasks")
        if not isinstance(tasks, dict):
            raise ValueError("Project file tasks must be a mapping")
        return cls(
            version=version,
            tab_name=cls._tab_name(payload.get("tab_name")),
            tasks=cls._versioned_tasks_from_mapping(tasks),
        )

    @classmethod
    def from_legacy_options(
        cls, options: T.Mapping[str, T.Any], *, tab_name: str | None = None
    ) -> ProjectFile:
        """Build a versioned model from the legacy flat GUI option shape."""
        tab_name = cls._tab_name(tab_name or options.get("tab_name"))
        tasks = {
            command: dict(values)
            for command, values in options.items()
            if isinstance(values, dict)
        }
        return cls(tab_name=tab_name, tasks=tasks)

    def to_legacy_options(self) -> dict[str, str | dict[str, bool | int | float | str]]:
        """Return the flat option dictionary expected by existing GUI widgets."""
        options: dict[str, str | dict[str, bool | int | float | str]] = {
            command: T.cast(dict[str, bool | int | float | str], dict(values))
            for command, values in self.tasks.items()
        }
        options["tab_name"] = self.tab_name
        return options

    def model_dump(self) -> dict[str, int | str | dict[str, dict[str, T.Any]]]:
        """Return a serializer-friendly dictionary."""
        return {
            "version": self.version,
            "tab_name": self.tab_name,
            "tasks": {command: dict(values) for command, values in self.tasks.items()},
        }

    @classmethod
    def _legacy_options_payload(
        cls, payload: T.Mapping[str, T.Any]
    ) -> T.Mapping[str, T.Any] | None:
        """Return embedded legacy options from known wrapper shapes."""
        for key in ("options", "project", "task"):
            options = payload.get(key)
            if isinstance(options, dict):
                return options
        return None

    @classmethod
    def _legacy_tab_name(
        cls,
        payload: T.Mapping[str, T.Any],
        options: T.Mapping[str, T.Any],
    ) -> str:
        """Resolve active tab from known legacy wrapper and option keys."""
        for value in (
            payload.get("tab_name"),
            payload.get("tab"),
            payload.get("command"),
            options.get("tab_name"),
            options.get("tab"),
            options.get("command"),
        ):
            if isinstance(value, str) and value:
                return value
        return "extract"

    @classmethod
    def _version(cls, value: T.Any) -> int:
        """Validate and normalize the stored project file version."""
        if isinstance(value, bool):
            raise ValueError("Project file version must be an integer")
        try:
            version = int(value)
        except (TypeError, ValueError) as err:
            raise ValueError("Project file version must be an integer") from err
        if version != cls.CURRENT_VERSION:
            raise ValueError(f"Unsupported project file version: {version}")
        return version

    @classmethod
    def _tab_name(cls, value: T.Any) -> str:
        """Normalize the stored active tab name."""
        return value if isinstance(value, str) and value else "extract"

    @classmethod
    def _versioned_tasks_from_mapping(
        cls, tasks: T.Mapping[str, T.Any]
    ) -> dict[str, dict[str, T.Any]]:
        """Validate and normalize a versioned tasks payload."""
        normalized: dict[str, dict[str, T.Any]] = {}
        for command, values in tasks.items():
            if not isinstance(values, dict):
                raise ValueError(f"Project file task '{command}' options must be a mapping")
            normalized[str(command)] = dict(values)
        return normalized


__all__ = get_module_objects(__name__)
