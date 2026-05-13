#!/usr/bin/env python3
"""Tests for preview output image discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from lib.gui.services.preview_output_service import PreviewOutputError, PreviewOutputService


def _touch(path: Path) -> Path:
    """Create an empty file and return it."""
    path.write_bytes(b"")
    return path


def test_preview_output_service_loads_supported_images_from_folder(tmp_path: Path) -> None:
    """PreviewOutputService should discover supported image files in sorted order."""
    image_b = _touch(tmp_path / "b.jpg")
    image_a = _touch(tmp_path / "a.png")
    _touch(tmp_path / "notes.txt")
    service = PreviewOutputService()

    images = service.load(tmp_path)

    assert [image.path for image in images] == [image_a, image_b]
    assert [image.name for image in images] == ["a.png", "b.jpg"]
    assert service.source == tmp_path
    assert service.images == images


def test_preview_output_service_loads_single_image_file(tmp_path: Path) -> None:
    """PreviewOutputService should accept a single image source."""
    image = _touch(tmp_path / "preview.webp")
    service = PreviewOutputService()

    images = service.load(image)

    assert len(images) == 1
    assert images[0].path == image
    assert service.source == image


def test_preview_output_service_rejects_missing_source(tmp_path: Path) -> None:
    """Explicit load validation should reject missing sources."""
    service = PreviewOutputService()

    with pytest.raises(PreviewOutputError, match="does not exist"):
        service.load(tmp_path / "missing")


def test_preview_output_service_rejects_non_image_file(tmp_path: Path) -> None:
    """Explicit load validation should reject unsupported files."""
    service = PreviewOutputService()
    source = _touch(tmp_path / "not_image.txt")

    with pytest.raises(PreviewOutputError, match="not an image file"):
        service.load(source)


def test_preview_output_service_configures_pending_source(tmp_path: Path) -> None:
    """Pending output paths should be stored before the process creates output."""
    service = PreviewOutputService()
    source = tmp_path / "future-output"

    service.configure(source)
    images = service.refresh()

    assert images == ()
    assert service.source == source
    assert service.images == ()


def test_preview_output_service_refresh_picks_up_later_images(tmp_path: Path) -> None:
    """Refresh should pick up files created after configuration."""
    service = PreviewOutputService()
    service.configure(tmp_path)

    assert service.refresh() == ()
    image = _touch(tmp_path / "preview.png")

    images = service.refresh()

    assert [item.path for item in images] == [image]


def test_preview_output_service_clear_resets_state(tmp_path: Path) -> None:
    """Clear should remove source and cached image records."""
    service = PreviewOutputService()
    service.load(_touch(tmp_path / "preview.png"))

    service.clear()

    assert service.source is None
    assert service.images == ()
