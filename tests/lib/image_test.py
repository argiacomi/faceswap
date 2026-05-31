#!/usr/bin/env python3
"""Regression tests for image loading helpers."""

from __future__ import annotations

import numpy as np

from lib.image import FacesLoader, encode_image, read_image_meta


def _write_malformed_faceswap_png(filename: str) -> None:
    """Create a PNG with malformed Faceswap iTXt metadata."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)  # type: ignore[var-annotated]
    encoded = encode_image(image, ".png", metadata=b"not_a_python_literal(")
    with open(filename, "wb") as out_file:
        out_file.write(encoded)


def test_read_image_meta_ignores_malformed_faceswap_itxt(tmp_path) -> None:
    """Malformed Faceswap iTXt should not crash metadata reads."""
    filename = tmp_path / "bad_face.png"
    _write_malformed_faceswap_png(str(filename))

    metadata = read_image_meta(str(filename))

    assert metadata["width"] == 8
    assert metadata["height"] == 8
    assert "itxt" not in metadata


def test_faces_loader_skips_malformed_faceswap_png_without_crashing(tmp_path) -> None:
    """FacesLoader should skip malformed Faceswap PNG metadata instead of crashing its thread."""
    filename = tmp_path / "bad_face.png"
    _write_malformed_faceswap_png(str(filename))

    loader = FacesLoader(str(tmp_path))

    assert list(loader.load()) == []
