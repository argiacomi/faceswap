#!/usr/bin/env python3
"""Service helpers for discovering preview output image files."""

from __future__ import annotations

import typing as T
from dataclasses import dataclass
from pathlib import Path

from lib.gui.utils.config import PATH_CACHE
from lib.utils import get_module_objects


class PreviewOutputError(ValueError):
    """Raised when preview output cannot be resolved."""


@dataclass(frozen=True)
class PreviewOutputImage:
    """One preview output image file."""

    path: Path

    @property
    def name(self) -> str:
        """Return the image filename."""
        return self.path.name


class PreviewOutputService:
    """Discover preview output images from a file, folder, batch folder or train cache."""

    IMAGE_SUFFIXES = (".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp")
    TRAINING_PREVIEW = ".gui_training_preview.png"

    def __init__(self) -> None:
        self._source: Path | None = None
        self._resolved_source: Path | None = None
        self._images: tuple[PreviewOutputImage, ...] = ()
        self._mode: T.Literal["output", "batch", "train"] = "output"

    @property
    def source(self) -> Path | None:
        """Return the currently configured preview source."""
        return self._source

    @property
    def resolved_source(self) -> Path | None:
        """Return the actual folder/file used during the last refresh."""
        return self._resolved_source

    @property
    def images(self) -> tuple[PreviewOutputImage, ...]:
        """Return discovered preview image files."""
        return self._images

    @property
    def mode(self) -> str:
        """Return the source discovery mode."""
        return self._mode

    def configure(
        self,
        source: str | Path | None,
        *,
        batch_mode: bool = False,
    ) -> None:
        """Set an extract/convert preview source without requiring it to exist yet."""
        self._source = None if source is None else Path(source)
        self._resolved_source = None
        self._images = ()
        self._mode = "batch" if batch_mode else "output"

    def configure_training(self, source: str | Path | None = None) -> None:
        """Set the training preview cache folder used by the Tk PreviewTrain flow."""
        self._source = Path(source) if source is not None else Path(PATH_CACHE) / "preview"
        self._resolved_source = None
        self._images = ()
        self._mode = "train"

    def load(self, source: str | Path) -> tuple[PreviewOutputImage, ...]:
        """Load preview images from an existing source file or folder."""
        self.configure(source)
        return self.refresh(validate=True)

    def refresh(self, *, validate: bool = False) -> tuple[PreviewOutputImage, ...]:
        """Refresh images from the current source."""
        if self._source is None:
            self._resolved_source = None
            self._images = ()
            return ()
        if validate:
            self.resolve_source(self._source)
        elif not self._source.exists():
            self._resolved_source = self._source
            self._images = ()
            return ()
        source = self._refresh_source(self._source)
        self._resolved_source = source
        self._images = (
            self._find_training_images(source)
            if self._mode == "train"
            else self._find_images(source)
        )
        return self._images

    def clear(self) -> None:
        """Clear current source and image list."""
        self._source = None
        self._resolved_source = None
        self._images = ()
        self._mode = "output"

    def resolve_source(self, source: str | Path) -> Path:
        """Resolve and validate a preview source path."""
        path = Path(source)
        if not path.exists():
            raise PreviewOutputError(f"Preview source does not exist: {path}")
        if path.is_file() and not self.is_image(path):
            raise PreviewOutputError(f"Preview source is not an image file: {path}")
        if not path.is_file() and not path.is_dir():
            raise PreviewOutputError(f"Preview source is not a file or folder: {path}")
        return path

    def _refresh_source(self, source: Path) -> Path:
        """Return the concrete source used for the current discovery mode."""
        if self._mode != "batch" or not source.is_dir():
            return source
        folders = [path for path in source.iterdir() if path.is_dir()]
        return max(folders, key=self._batch_folder_sort_key, default=source)

    def _batch_folder_sort_key(self, folder: Path) -> tuple[int, str]:
        """Return deterministic freshness key for a batch child folder."""
        image_mtimes = (
            path.stat().st_mtime_ns for path in folder.iterdir() if self.is_image(path)
        )
        return (max(image_mtimes, default=folder.stat().st_mtime_ns), folder.name)

    def _find_training_images(self, source: Path) -> tuple[PreviewOutputImage, ...]:
        """Return the current training preview image from the cache folder."""
        if source.is_file():
            return (PreviewOutputImage(source),) if source.name == self.TRAINING_PREVIEW else ()
        preview = source / self.TRAINING_PREVIEW
        return (PreviewOutputImage(preview),) if self.is_image(preview) else ()

    def _find_images(self, source: Path) -> tuple[PreviewOutputImage, ...]:
        """Find supported image files below a source file or folder."""
        if source.is_file():
            return (PreviewOutputImage(source),)
        images = [
            PreviewOutputImage(path) for path in sorted(source.iterdir()) if self.is_image(path)
        ]
        return tuple(images)

    @classmethod
    def is_image(cls, path: Path) -> bool:
        """Return whether a path is a supported image file."""
        return path.is_file() and path.suffix.lower() in cls.IMAGE_SUFFIXES


__all__ = get_module_objects(__name__)
