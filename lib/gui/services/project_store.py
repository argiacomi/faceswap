#!/usr/bin/env python3
"""Persistence service for Faceswap GUI project and task files."""

from __future__ import annotations

from collections.abc import Mapping
import typing as T

from lib.utils import get_module_objects

from ..models.project import ProjectFile


class ProjectStore:
    """Load, migrate and save versioned project/task files."""

    def __init__(self, serializer: T.Any) -> None:
        self._serializer = serializer

    def load(self, filename: str) -> ProjectFile:
        """Load a project/task file from disk and migrate it to the current model."""
        payload = self._serializer.load(filename)
        return self.migrate(payload)

    def save(self, filename: str, project: ProjectFile) -> None:
        """Save a project/task file using the current versioned shape."""
        self._serializer.save(filename, project.model_dump())

    def migrate(self, payload: object) -> ProjectFile:
        """Migrate a raw project/task payload to the current model."""
        if not isinstance(payload, Mapping):
            raise ValueError("Project file payload must be a mapping")
        return ProjectFile.from_mapping(payload)


__all__ = get_module_objects(__name__)
