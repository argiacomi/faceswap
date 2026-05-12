#!/usr/bin/env python3
"""Persistence service for Faceswap GUI recent files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import typing as T

from lib.utils import get_module_objects


@dataclass(frozen=True)
class RecentFile:
    """A recent GUI project or task file entry."""

    filename: str
    kind: str


class RecentFilesStore:
    """Read, update and save the GUI recent-files list."""

    def __init__(self, serializer: T.Any, filename: str, *, limit: int = 20) -> None:
        self._serializer = serializer
        self._filename = Path(filename)
        self._limit = limit

    @property
    def filename(self) -> str:
        """The recent files cache filename."""
        return str(self._filename)

    def load(self) -> list[RecentFile]:
        """Load recent files from disk."""
        if not self._filename.exists() or self._filename.stat().st_size == 0:
            return []

        payload = self._serializer.load(str(self._filename))
        return self.decode_many(payload)

    def save(self, recent_files: list[RecentFile]) -> None:
        """Save recent files to disk."""
        payload = [(item.filename, item.kind) for item in recent_files[: self._limit]]
        self._serializer.save(str(self._filename), payload)

    def add(self, filename: str, kind: str) -> list[RecentFile]:
        """Add a file to the front of the recent files list."""
        recent_files = self.remove(filename, save=False)
        recent_files.insert(0, RecentFile(filename=filename, kind=kind))
        recent_files = recent_files[: self._limit]
        self.save(recent_files)
        return recent_files

    def remove(self, filename: str, *, save: bool = True) -> list[RecentFile]:
        """Remove a file from the recent files list."""
        recent_files = [item for item in self.load() if item.filename != filename]

        if save:
            self.save(recent_files)

        return recent_files

    @classmethod
    def decode_many(cls, payload: object) -> list[RecentFile]:
        """Decode a serializer payload into recent-file entries."""
        if not isinstance(payload, list):
            return []
        return [
            item
            for item in (cls._decode(row) for row in payload)
            if item is not None
        ]

    @staticmethod
    def _decode(row: object) -> RecentFile | None:
        """Decode a serializer row into a recent-file entry."""
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            return None

        filename, kind = row
        if not isinstance(filename, str) or not isinstance(kind, str):
            return None

        return RecentFile(filename=filename, kind=kind)


__all__ = get_module_objects(__name__)
