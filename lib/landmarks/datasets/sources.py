#!/usr/bin/env python3
"""Source resolution helpers for landmark quality datasets."""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import tarfile
import tempfile
import typing as T
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)
DEFAULT_CACHE_DIR = Path(".fs_cache/landmark_quality")
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")


@dataclass(frozen=True)
class DatasetSourceSpec:
    """Known source information for one landmark dataset."""

    dataset: str
    cache_subdir: str
    canonical_archive: str | None = None
    cache_aliases: tuple[str, ...] = ()
    extracted_aliases: tuple[str, ...] = ()
    url: str | None = None
    google_drive_file_id: str | None = None
    manual_hint: str = ""

    @property
    def cache_root_name(self) -> str:
        """Return the cache subdirectory for this dataset."""
        return self.cache_subdir.strip("/") or self.dataset.lower()


def download(
    url: str | None,
    destination: Path,
    *,
    force: bool = False,
    google_drive_file_id: str | None = None,
    label: str = "archive",
) -> Path:
    """Download a direct URL or Google Drive file into ``destination``."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not force:
        logger.info("Using cached %s: %s", label, destination)
        return destination
    if force and destination.exists():
        logger.debug("Removing cached %s due to force download: %s", label, destination)
        destination.unlink()

    if google_drive_file_id is not None:
        from lib.google_drive import download_google_drive_file

        logger.info("Downloading %s from Google Drive into cache: %s", label, destination)
        return download_google_drive_file(google_drive_file_id, destination)

    if url is None:
        raise ValueError("Either url or google_drive_file_id must be supplied")

    logger.info("Downloading %s into cache: %s", label, destination)
    with urllib.request.urlopen(url) as response, destination.open("wb") as outfile:
        shutil.copyfileobj(response, outfile)
    logger.info("Downloaded %s (%d bytes)", destination, destination.stat().st_size)
    return destination


def _is_relative_to(path: Path, base: Path) -> bool:
    """Return whether ``path`` is inside ``base`` for Python versions without Path.is_relative_to."""
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def safe_zip_extractall(zf: zipfile.ZipFile, destination: str | os.PathLike[str]) -> None:
    """Extract a zip file while blocking path traversal."""
    dest = Path(destination).resolve()
    for member in zf.infolist():
        target = (dest / member.filename).resolve()
        if not _is_relative_to(target, dest):
            raise ValueError(f"Blocked zip path traversal member: {member.filename}")
    zf.extractall(dest)


def safe_tar_extractall(tf: tarfile.TarFile, destination: str | os.PathLike[str]) -> None:
    """Extract a tar file while blocking path traversal and links."""
    dest = Path(destination).resolve()
    for member in tf.getmembers():
        if member.issym():
            raise ValueError(f"Blocked tar symlink member: {member.name}")
        if member.islnk():
            raise ValueError(f"Blocked tar hardlink member: {member.name}")
        target = (dest / member.name).resolve()
        if not _is_relative_to(target, dest):
            raise ValueError(f"Blocked tar path traversal member: {member.name}")
    tf.extractall(dest)


@contextlib.contextmanager
def extract_archive_to_temp(archive: str | os.PathLike[str]) -> T.Iterator[Path]:
    """Extract ``archive`` into a temporary directory and yield that directory."""
    archive_path = Path(archive)
    if not archive_path.is_file():
        raise FileNotFoundError(f"dataset archive not found: {archive_path}")
    with tempfile.TemporaryDirectory() as tmp:
        destination = Path(tmp)
        suffixes = "".join(archive_path.suffixes[-2:]).lower()
        if archive_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                safe_zip_extractall(zf, destination)
        elif suffixes in {".tar.gz", ".tgz"} or archive_path.suffix.lower() == ".tar":
            with tarfile.open(archive_path, "r:*") as tf:
                safe_tar_extractall(tf, destination)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path}")
        yield destination


def is_archive(path: Path) -> bool:
    """Return whether ``path`` has a supported archive suffix."""
    return path.is_file() and any(str(path).lower().endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def _existing_source(path: Path) -> Path | None:
    """Return ``path`` if it is an existing file or directory."""
    if path.is_dir() or path.is_file():
        return path
    return None


def _generic_cache_candidate(path: Path) -> Path | None:
    """Return generic cache candidates without accepting arbitrary files."""
    if path.is_dir() or is_archive(path):
        return path
    return None


def _cached_source(spec: DatasetSourceSpec, cache_dir: Path) -> Path | None:
    """Return a cached archive, explicit alias file, or extracted source directory for ``spec``."""
    cache_root = cache_dir / spec.cache_root_name
    if spec.canonical_archive:
        candidate = _existing_source(cache_root / spec.canonical_archive)
        if candidate is not None:
            logger.info("Using cached %s source: %s", spec.dataset, candidate)
            return candidate
    for name in spec.cache_aliases:
        candidate = _existing_source(cache_root / name)
        if candidate is not None:
            logger.info("Using cached %s source alias: %s", spec.dataset, candidate)
            return candidate
    for name in spec.extracted_aliases:
        candidate = cache_root / name
        if candidate.is_dir():
            logger.info("Using cached %s extracted source: %s", spec.dataset, candidate)
            return candidate
    if cache_root.is_dir():
        for child in sorted(cache_root.iterdir()):
            candidate = _generic_cache_candidate(child)
            if candidate is not None:
                logger.info("Using cached %s source candidate: %s", spec.dataset, candidate)
                return candidate
    return None


def resolve_dataset_source(
    spec: DatasetSourceSpec,
    *,
    cache_dir: str | os.PathLike[str] = DEFAULT_CACHE_DIR,
    source_dir: str | os.PathLike[str] | None = None,
    source_zip: str | os.PathLike[str] | None = None,
    download_url: str | None = None,
    force_download: bool = False,
    no_download: bool = False,
) -> Path:
    """Resolve a dataset source using explicit args, cache, then download."""
    if source_zip is not None:
        archive = Path(source_zip)
        if not archive.is_file():
            raise FileNotFoundError(f"{spec.dataset} source archive not found: {archive}")
        logger.info("Using explicit %s source archive: %s", spec.dataset, archive)
        return archive
    if source_dir is not None:
        directory = Path(source_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"{spec.dataset} source directory not found: {directory}")
        logger.info("Using explicit %s source directory: %s", spec.dataset, directory)
        return directory

    cache_root = Path(cache_dir)
    if not force_download:
        cached = _cached_source(spec, cache_root)
        if cached is not None:
            return cached

    url = download_url or spec.url
    if no_download:
        hint = f" {spec.manual_hint}" if spec.manual_hint else ""
        raise FileNotFoundError(
            f"{spec.dataset} source not found in {cache_root / spec.cache_root_name}. "
            f"Download disabled by --no-download.{hint}"
        )
    if url is None and spec.google_drive_file_id is None:
        hint = f" {spec.manual_hint}" if spec.manual_hint else ""
        raise FileNotFoundError(
            f"{spec.dataset} source not found in {cache_root / spec.cache_root_name} and no download source is configured.{hint}"
        )
    archive_name = spec.canonical_archive or f"{spec.dataset.lower()}.zip"
    try:
        return download(
            url,
            cache_root / spec.cache_root_name / archive_name,
            force=force_download,
            google_drive_file_id=spec.google_drive_file_id,
            label=f"{spec.dataset} archive",
        )
    except Exception as err:  # pragma: no cover - exercised by adapter-level tests with monkeypatches
        hint = f" {spec.manual_hint}" if spec.manual_hint else ""
        raise OSError(f"{spec.dataset} download failed: {err}.{hint}") from err
