#!/usr/bin/env python3
"""Small Google Drive download helper used by dataset source resolution."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


class GoogleDriveDownloadError(OSError):
    """Raised when a Google Drive file cannot be downloaded or validated."""


def download_google_drive_file(
    file_id: str,
    destination: str | Path,
    *,
    expected_sha256: str | None = None,
    quiet: bool = False,
) -> Path:
    """Download a public Google Drive file into ``destination`` with atomic replace.

    The implementation intentionally imports ``gdown`` lazily so normal code paths
    that use direct URLs do not need the optional dependency.
    """
    try:
        import gdown  # type: ignore[import-not-found]
    except ImportError as err:
        raise GoogleDriveDownloadError(
            "Google Drive dataset downloads require the optional 'gdown' dependency. "
            "Install gdown or provide --download-url/explicit source paths."
        ) from err

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        _validate_file(output, expected_sha256=expected_sha256)
        return output

    fd, tmp_name = tempfile.mkstemp(prefix=f"{output.name}.", suffix=".part", dir=output.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        result = gdown.download(id=file_id, output=str(tmp_path), quiet=quiet)
        if result is None or not tmp_path.is_file():
            raise GoogleDriveDownloadError(
                f"Google Drive download did not produce an output file for id {file_id!r}"
            )
        _validate_file(tmp_path, expected_sha256=expected_sha256)
        os.replace(tmp_path, output)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return output


def _validate_file(path: Path, *, expected_sha256: str | None) -> None:
    """Validate a downloaded file is non-empty and matches the optional hash."""
    if not path.is_file() or path.stat().st_size == 0:
        raise GoogleDriveDownloadError(f"downloaded file is empty: {path}")
    if expected_sha256 is not None:
        actual = _sha256_file(path)
        if actual.lower() != expected_sha256.lower():
            raise GoogleDriveDownloadError(
                f"downloaded file checksum mismatch for {path.name}: "
                f"expected {expected_sha256}, got {actual}"
            )


def _sha256_file(path: Path) -> str:
    """Return the SHA256 hex digest for ``path``."""
    sha = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()
