#!/usr/bin python3
"""Utilities for the Faceswap GUI"""

from .config import PATH_CACHE, get_config, initialize_config
from .misc import LongRunningTask


def __getattr__(name: str):
    """Lazily expose Tk-backed utilities without importing Tk for Qt-only paths."""
    if name == "FileHandler":
        from .file_handler import FileHandler

        return FileHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_images():
    """Return the lazily initialized GUI image manager."""
    from .image import get_images as _get_images

    return _get_images()


def initialize_images() -> None:
    """Initialize the GUI image manager without importing image dependencies early."""
    from .image import initialize_images as _initialize_images

    _initialize_images()


def preview_trigger():
    """Return the lazily initialized preview trigger."""
    from .image import preview_trigger as _preview_trigger

    return _preview_trigger()
