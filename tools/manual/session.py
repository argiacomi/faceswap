#!/usr/bin/env python3
"""GUI-neutral Manual Tool session state.

This module intentionally does not import tkinter or Qt.  It is a small, safe
service layer that the legacy Tk editor and the native Qt shell can both grow
around while the Manual Tool migration is in progress.
"""

from __future__ import annotations

import logging
import os
import typing as T
from dataclasses import dataclass, field
from pathlib import Path

from lib.utils import get_module_objects
from lib.video import VIDEO_EXTENSIONS

if T.TYPE_CHECKING:
    from argparse import Namespace

    from lib.align import Alignments

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})
_VIDEO_SUFFIXES = frozenset(ext.lower() for ext in VIDEO_EXTENSIONS)
_ALIGNMENTS_FILENAME = "alignments.fsa"


@dataclass(frozen=True)
class ManualFrame:
    """One source frame candidate for the Manual Tool."""

    index: int
    name: str
    path: str


@dataclass(frozen=True)
class ManualVideoMetadata:
    """GUI-neutral view of alignments-file video metadata."""

    pts_time: tuple[int, ...]
    keyframes: tuple[int, ...]

    @property
    def frame_count(self) -> int:
        """Return the number of frames described by the metadata."""
        return len(self.pts_time)

    @property
    def is_valid(self) -> bool:
        """Return whether metadata is non-empty and aligned."""
        return self.frame_count > 0


@dataclass
class ManualEditorState:
    """GUI-neutral editor state with simple observer callbacks.

    The legacy Tk implementation stores frame/face navigation, filter mode and
    dirty flags in :mod:`tkinter` variables.  The Qt shell cannot import those
    safely, so this object mirrors the same values through plain Python and a
    light observer API.  Both UIs subscribe to the same state changes.
    """

    frame_index: int = 0
    face_index: int = 0
    filter_mode: str = ""
    filter_distance: int = 10
    faces_size: str = ""
    is_zoomed: bool = False
    unsaved: bool = False
    edited: bool = False
    face_count_changed: bool = False
    _listeners: dict[str, list[T.Callable[[T.Any], None]]] = field(
        default_factory=dict, repr=False
    )

    _FIELDS = (
        "frame_index",
        "face_index",
        "filter_mode",
        "filter_distance",
        "faces_size",
        "is_zoomed",
        "unsaved",
        "edited",
        "face_count_changed",
    )

    def subscribe(self, name: str, callback: T.Callable[[T.Any], None]) -> T.Callable[[], None]:
        """Register a callback for the given field. Returns an unsubscribe handle."""
        if name not in self._FIELDS:
            raise ValueError(f"Unknown editor state field: {name}")
        listeners = self._listeners.setdefault(name, [])
        listeners.append(callback)

        def _unsubscribe() -> None:
            if callback in listeners:
                listeners.remove(callback)

        return _unsubscribe

    def set(self, name: str, value: T.Any) -> None:
        """Set a field and notify listeners only when the value changes."""
        if name not in self._FIELDS:
            raise ValueError(f"Unknown editor state field: {name}")
        if getattr(self, name) == value:
            return
        setattr(self, name, value)
        for callback in list(self._listeners.get(name, ())):
            try:
                callback(value)
            except Exception:  # pragma: no cover - defensive
                logger.exception("Editor state listener for %s failed", name)


class ManualAlignmentsHandle:
    """GUI-neutral handle for the Manual Tool alignments file.

    Resolves the alignments folder and filename using the same rules as the
    legacy Tk implementation and lazily opens a :class:`lib.align.Alignments`
    instance only when the consumer asks for it.  Importing this handle does
    not pull in tkinter or Qt.
    """

    def __init__(self, folder: str, filename: str, *, is_video: bool) -> None:
        self._folder = folder
        self._filename = filename
        self._is_video = is_video
        self._alignments: Alignments | None = None

    @property
    def folder(self) -> str:
        """Return the folder containing the alignments file."""
        return self._folder

    @property
    def filename(self) -> str:
        """Return the alignments filename."""
        return self._filename

    @property
    def path(self) -> str:
        """Return the full resolved alignments file path."""
        return os.path.join(self._folder, self._filename)

    @property
    def exists(self) -> bool:
        """Return whether the alignments file currently exists on disk."""
        return os.path.isfile(self.path)

    def open(self) -> Alignments:
        """Open and cache a :class:`lib.align.Alignments` instance."""
        if self._alignments is None:
            from lib.align import Alignments

            self._alignments = Alignments(self._folder, self._filename)
        return self._alignments

    def video_metadata(self) -> ManualVideoMetadata | None:
        """Return GUI-neutral video metadata or ``None`` when unavailable."""
        if not self._is_video or not self.exists:
            return None
        meta = self.open().video_meta_data
        if not meta:
            return None
        pts_time = meta.get("pts_time") or []
        keyframes = meta.get("keyframes") or []
        if not pts_time:
            return None
        return ManualVideoMetadata(
            pts_time=tuple(pts_time),
            keyframes=tuple(keyframes),
        )

    def has_thumbnails(self) -> bool:
        """Return whether the alignments file already stores thumbnails."""
        if not self.exists:
            return False
        return bool(self.open().thumbnails.has_thumbnails)


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
    def from_cli_values(cls, values: T.Mapping[str, object]) -> ManualSession:
        """Build a session from switch-keyed Qt command-panel values."""
        frames = cls._string_value(
            values.get("-f") or values.get("--frames") or values.get("frames")
        )
        alignments = cls._string_value(
            values.get("-a") or values.get("--alignments") or values.get("alignments_path")
        )
        return cls.create(
            frames=frames,
            alignments_path=alignments or None,
            thumb_regenerate=bool(values.get("-t") or values.get("--thumb-regen")),
            single_process=bool(values.get("-s") or values.get("--single-process")),
        )

    @classmethod
    def from_namespace(cls, namespace: Namespace) -> ManualSession:
        """Build a session from a parsed argparse namespace."""
        return cls.create(
            frames=getattr(namespace, "frames", "") or "",
            alignments_path=getattr(namespace, "alignments_path", "") or None,
            thumb_regenerate=bool(getattr(namespace, "thumb_regenerate", False)),
            single_process=bool(getattr(namespace, "single_process", False)),
        )

    @classmethod
    def create(
        cls,
        *,
        frames: str,
        alignments_path: str | None = None,
        thumb_regenerate: bool = False,
        single_process: bool = False,
    ) -> ManualSession:
        """Validate input and return a GUI-neutral Manual Tool session."""
        if not frames:
            raise ValueError("Frames input is required")
        input_path = Path(os.path.expanduser(frames)).resolve()
        if not input_path.exists():
            raise ValueError(f"Frames input does not exist: {input_path}")
        is_video = input_path.is_file() and input_path.suffix.lower() in _VIDEO_SUFFIXES
        if input_path.is_file() and not is_video:
            raise ValueError(f"Frames input is not a supported video file: {input_path}")
        frame_list = cls._discover_frames(input_path) if input_path.is_dir() else ()
        if input_path.is_dir() and not frame_list:
            raise ValueError(f"No supported image frames found in: {input_path}")
        if input_path.is_dir():
            cls._reject_extracted_faces(input_path)
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

    def alignments_handle(self) -> ManualAlignmentsHandle:
        """Resolve and return a GUI-neutral handle to the alignments file."""
        folder, filename = self._resolve_alignments_location()
        return ManualAlignmentsHandle(folder, filename, is_video=self.is_video_input)

    def video_metadata(self) -> ManualVideoMetadata | None:
        """Convenience accessor for video metadata via the alignments handle."""
        return self.alignments_handle().video_metadata()

    def has_thumbnails(self) -> bool:
        """Return whether thumbnails already exist in the alignments file."""
        return self.alignments_handle().has_thumbnails()

    def needs_thumbnail_regeneration(self) -> bool:
        """Return whether thumbnails must be generated before the editor loads."""
        return self.thumb_regenerate or not self.has_thumbnails()

    def create_editor_state(self) -> ManualEditorState:
        """Return a fresh GUI-neutral editor state for this session."""
        return ManualEditorState()

    def _resolve_alignments_location(self) -> tuple[str, str]:
        """Return (folder, filename) for the alignments file using legacy rules."""
        if self.alignments_path:
            folder, filename = os.path.split(self.alignments_path)
            return folder, filename
        if self.is_video_input:
            folder, vid = os.path.split(os.path.splitext(self.frames)[0])
            return folder, f"{vid}_{_ALIGNMENTS_FILENAME}"
        return self.frames, _ALIGNMENTS_FILENAME

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
    def _reject_extracted_faces(input_path: Path) -> None:
        """Refuse a folder of extracted Faceswap faces as Manual Tool input."""
        png_candidate = next(
            (path for path in input_path.iterdir() if path.suffix.lower() == ".png"),
            None,
        )
        if png_candidate is None:
            return
        from lib.image import read_image_meta

        try:
            meta = read_image_meta(str(png_candidate))
        except Exception:  # pragma: no cover - defensive
            logger.debug("Could not read PNG header for %s", png_candidate, exc_info=True)
            return
        if isinstance(meta, dict) and "itxt" in meta and "alignments" in meta["itxt"]:
            raise ValueError(
                f"Input folder contains extracted faces, not source frames: {input_path}"
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
