#!/usr/bin/env python3
"""Service helpers for discovering preview output image files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    """Discover preview output images from a file or folder path."""

    IMAGE_SUFFIXES = (".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp")

    def __init__(self) -> None:
        self._source: Path | None = None
        self._images: tuple[PreviewOutputImage, ...] = ()

    @property
    def source(self) -> Path | None:
        """Return the currently loaded preview source."""
        return self._source

    @property
    def images(self) -> tuple[PreviewOutputImage, ...]:
        """Return discovered preview image files."""
        return self._images

    def configure(self, source: str | Path | None) -> None:
        """Set a preview source without requiring it to exist yet."""
        self._source = None if source is None else Path(source)
        self._images = ()

    def load(self, source: str | Path) -> tuple[PreviewOutputImage, ...]:
        """Load preview images from an existing source file or folder."""
        self.configure(source)
        return self.refresh(validate=True)

    def refresh(self, *, validate: bool = False) -> tuple[PreviewOutputImage, ...]:
        """Refresh images from the current source."""
        if self._source is None:
            self._images = ()
            return ()
        if validate:
            self.resolve_source(self._source)
        elif not self._source.exists():
            self._images = ()
            return ()
        self._images = self._find_images(self._source)
        return self._images

    def clear(self) -> None:
        """Clear current source and image list."""
        self._source = None
        self._images = ()

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
