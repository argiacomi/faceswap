#!/usr/bin/env python3
"""GUI-neutral Manual Tool session state.

This module intentionally does not import tkinter or Qt.  It is a small, safe
service layer that the legacy Tk editor and the native Qt shell can both grow
around while the Manual Tool migration is in progress.
"""

from __future__ import annotations

import os
import typing as T
from dataclasses import dataclass
from pathlib import Path

from lib.utils import get_module_objects

_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})
_VIDEO_SUFFIXES = frozenset({".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"})


@dataclass(frozen=True)
class ManualFrame:
    """One source frame candidate for the Manual Tool."""

    index: int
    name: str
    path: str


@dataclass(frozen=True)
class ManualSession:
    """Serializable Manual Tool session metadata shared by GUI implementations."""

    frames: str
    alignments_path: str | None = None
    thumb_regenerate: bool = False
    single_process: bool = False
    frame_list: tuple[ManualFrame, ...] = ()
    is_video_input: bool = False

    @classmethod
    def from_cli_values(cls, values: T.Mapping[str, object]) -> "ManualSession":
        """Build a session from switch-keyed Qt command-panel values."""
        frames = cls._string_value(
            values.get("-f")
            or values.get("--frames")
            or values.get("frames")
        )
        alignments = cls._string_value(
            values.get("-a")
            or values.get("--alignments")
            or values.get("alignments_path")
        )
        return cls.create(
            frames=frames,
            alignments_path=alignments or None,
            thumb_regenerate=bool(values.get("-t") or values.get("--thumb-regen")),
            single_process=bool(values.get("-s") or values.get("--single-process")),
        )

    @classmethod
    def create(
        cls,
        *,
        frames: str,
        alignments_path: str | None = None,
        thumb_regenerate: bool = False,
        single_process: bool = False,
    ) -> "ManualSession":
        """Validate input and return a GUI-neutral Manual Tool session."""
        if not frames:
            raise ValueError("Frames input is required")
        input_path = Path(os.path.expanduser(frames)).resolve()
        if not input_path.exists():
            raise ValueError(f"Frames input does not exist: {input_path}")
        is_video = input_path.is_file() and input_path.suffix.lower() in _VIDEO_SUFFIXES
        frame_list = cls._discover_frames(input_path) if input_path.is_dir() else ()
        if input_path.is_dir() and not frame_list:
            raise ValueError(f"No supported image frames found in: {input_path}")
        alignments = None
        if alignments_path:
            alignments_candidate = Path(os.path.expanduser(alignments_path)).resolve()
            alignments = str(alignments_candidate)
        return cls(
            frames=str(input_path),
            alignments_path=alignments,
            thumb_regenerate=thumb_regenerate,
            single_process=single_process,
            frame_list=frame_list,
            is_video_input=is_video,
        )

    @property
    def has_images(self) -> bool:
        """Return whether this session can expose image frames directly."""
        return bool(self.frame_list)

    @property
    def frame_count(self) -> int:
        """Return the number of directly discoverable image frames."""
        return len(self.frame_list)

    @staticmethod
    def _discover_frames(input_path: Path) -> tuple[ManualFrame, ...]:
        """Return sorted image frames from a folder."""
        paths = sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        )
        return tuple(
            ManualFrame(index=index, name=path.name, path=str(path))
            for index, path in enumerate(paths)
        )

    @staticmethod
    def _string_value(value: object) -> str:
        """Normalize command values to strings."""
        if value is None or value is False:
            return ""
        if isinstance(value, (list, tuple)):
            return " ".join(str(item) for item in value)
        return str(value)


__all__ = get_module_objects(__name__)
