#!/usr/bin/env python3
"""Source resolution helpers for landmark quality datasets."""

from __future__ import annotations

import contextlib
import hashlib
import json
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
EXTRACTED_DIR_NAME = "extracted"
EXTRACTION_MARKER = ".source.json"

WFLW_ANNOTATIONS_URL = "https://wywu.github.io/projects/LAB/support/WFLW_annotations.tar.gz"
WFLW_IMAGES_GOOGLE_DRIVE_FILE_ID = "1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC"
COFW_COLOR_URL = "http://www.vision.caltech.edu/xpburgos/ICCV13/Data/COFW_color.zip"
MERL_RAV_LABELS_URL = (
    "https://github.com/abhi1kumar/MERL-RAV_dataset/archive/refs/heads/master.zip"
)
AFLW2000_3D_URL = (
    "http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip"
)
AFLW2000_3D_SHA256 = "252bc35274d65ff27b6e573aa96c2f4c116ad88452cc984fb882258c0ed6e2d8"

# Defaults are intentionally conservative. Only archives that can be used by the
# current manifest builders without separately licensed image requests are enabled
# as automatic downloads. WFLW is multi-part and MERL-RAV requires separately
# requested AFLW images, so both remain explicit/cache-driven unless callers pass
# --download-url to a complete, locally approved bundle.
DEFAULT_DOWNLOAD_SOURCES: dict[str, dict[str, str | None]] = {
    "AFLW2000-3D": {
        "url": AFLW2000_3D_URL,
        "archive_name": "AFLW2000-3D.zip",
        "sha256": AFLW2000_3D_SHA256,
    },
}

OFFICIAL_SOURCE_NOTES: dict[str, str] = {
    "WFLW": (
        "Official WFLW is distributed as separate image and annotation downloads: "
        f"images via Google Drive file id {WFLW_IMAGES_GOOGLE_DRIVE_FILE_ID}, "
        f"annotations at {WFLW_ANNOTATIONS_URL}. Pass --download-url for a complete "
        "approved bundle, or place both extracted parts in the cache."
    ),
    "COFW": (
        "Official COFW color images are at "
        f"{COFW_COLOR_URL}, but the current builder expects a COFW JSON export. "
        "Pass --cofw-json, --source-dir, --source-zip, or --download-url for a complete "
        "approved JSON bundle."
    ),
    "MERL-RAV": (
        "MERL-RAV labels are available at "
        f"{MERL_RAV_LABELS_URL}, but AFLW images must be requested separately. "
        "Place an organized image+label directory in the cache or pass "
        "--source-dir/--source-zip."
    ),
    "AFLW2000-3D": f"Official AFLW2000-3D archive: {AFLW2000_3D_URL}.",
}


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
    sha256: str | None = None
    manual_hint: str = ""

    @property
    def cache_root_name(self) -> str:
        """Return the cache subdirectory for this dataset."""
        return self.cache_subdir.strip("/") or self.dataset.lower()


@dataclass(frozen=True)
class MultiSourcePart:
    """One archive/source that contributes to a multipart dataset cache."""

    name: str
    archive_name: str
    url: str | None = None
    google_drive_file_id: str | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class MultiDatasetSourceSpec:
    """Known source information for datasets distributed as multiple archives."""

    dataset: str
    cache_subdir: str
    parts: tuple[MultiSourcePart, ...]
    manual_hint: str = ""

    @property
    def cache_root_name(self) -> str:
        """Return the cache subdirectory for this dataset."""
        return self.cache_subdir.strip("/") or self.dataset.lower()


WFLW_OFFICIAL_SOURCE = MultiDatasetSourceSpec(
    dataset="WFLW",
    cache_subdir="wflw",
    parts=(
        MultiSourcePart(
            name="annotations",
            archive_name="WFLW_annotations.tar.gz",
            url=WFLW_ANNOTATIONS_URL,
        ),
        MultiSourcePart(
            name="images",
            archive_name="WFLW_images.zip",
            google_drive_file_id=WFLW_IMAGES_GOOGLE_DRIVE_FILE_ID,
        ),
    ),
    manual_hint=(
        "Official WFLW requires both annotation and image downloads. "
        "Google Drive image downloads require the optional 'gdown' dependency."
    ),
)


def _default_source_value(spec: DatasetSourceSpec, key: str) -> str | None:
    """Return a default source value for ``spec`` when the spec leaves it unset."""
    defaults = DEFAULT_DOWNLOAD_SOURCES.get(spec.dataset, {})
    value = defaults.get(key)
    return str(value) if value else None


def _effective_url(spec: DatasetSourceSpec, download_url: str | None) -> str | None:
    """Return explicit, spec, or built-in download URL for ``spec``."""
    return download_url or spec.url or _default_source_value(spec, "url")


def _effective_sha256(spec: DatasetSourceSpec) -> str | None:
    """Return the expected SHA256 for built-in or spec-configured sources."""
    return spec.sha256 or _default_source_value(spec, "sha256")


def _archive_names(spec: DatasetSourceSpec) -> tuple[str, ...]:
    """Return candidate archive filenames for cache lookup and download."""
    names: list[str] = []
    default_archive = _default_source_value(spec, "archive_name")
    for name in (default_archive, spec.canonical_archive, *spec.cache_aliases):
        if name and name not in names:
            names.append(name)
    return tuple(names)


def _archive_name(spec: DatasetSourceSpec) -> str:
    """Return the canonical archive name used for new downloads."""
    names = _archive_names(spec)
    archive_names = [
        name for name in names if any(name.lower().endswith(suffix) for suffix in ARCHIVE_SUFFIXES)
    ]
    if archive_names:
        return archive_names[0]
    return names[0] if names else f"{spec.dataset.lower()}.zip"


def _manual_hint(spec: DatasetSourceSpec | MultiDatasetSourceSpec) -> str:
    """Return the best manual setup hint for ``spec``."""
    parts = [
        item for item in (spec.manual_hint, OFFICIAL_SOURCE_NOTES.get(spec.dataset, "")) if item
    ]
    return " " + " ".join(parts) if parts else ""


def sha256_file(path: Path) -> str:
    """Return the SHA256 hex digest for ``path``."""
    sha = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def verify_sha256(path: Path, expected_sha256: str | None, *, label: str = "archive") -> None:
    """Raise when ``path`` does not match ``expected_sha256``."""
    if expected_sha256 is None:
        return
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"{label} checksum mismatch for {path.name}: expected {expected_sha256}, got {actual}"
        )


def download(
    url: str | None,
    destination: Path,
    *,
    force: bool = False,
    google_drive_file_id: str | None = None,
    expected_sha256: str | None = None,
    label: str = "archive",
) -> Path:
    """Download a direct URL or Google Drive file into ``destination``."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not force:
        verify_sha256(destination, expected_sha256, label=label)
        logger.info("Using cached %s: %s", label, destination)
        return destination
    if force and destination.exists():
        logger.debug("Removing cached %s due to force download: %s", label, destination)
        destination.unlink()

    if google_drive_file_id is not None:
        from lib.google_drive import download_google_drive_file

        logger.info("Downloading %s from Google Drive into cache: %s", label, destination)
        return download_google_drive_file(
            google_drive_file_id,
            destination,
            expected_sha256=expected_sha256,
        )

    if url is None:
        raise ValueError("Either url or google_drive_file_id must be supplied")

    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{destination.name}.", suffix=".part", dir=destination.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        logger.info("Downloading %s into cache: %s", label, destination)
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as outfile:
            shutil.copyfileobj(response, outfile)
        if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
            raise OSError(f"download produced an empty file: {tmp_path}")
        verify_sha256(tmp_path, expected_sha256, label=label)
        os.replace(tmp_path, destination)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    logger.info("Downloaded %s (%d bytes)", destination, destination.stat().st_size)
    return destination


def _is_relative_to(path: Path, base: Path) -> bool:
    """Return whether `path` is inside `base` for Python versions without Path.is_relative_to."""
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


def _extract_archive(archive_path: Path, destination: Path) -> None:
    """Extract ``archive_path`` into ``destination``."""
    suffixes = "".join(archive_path.suffixes[-2:]).lower()
    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            safe_zip_extractall(zf, destination)
    elif suffixes in {".tar.gz", ".tgz"} or archive_path.suffix.lower() == ".tar":
        with tarfile.open(archive_path, "r:*") as tf:
            safe_tar_extractall(tf, destination)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


@contextlib.contextmanager
def extract_archive_to_temp(archive: str | os.PathLike[str]) -> T.Iterator[Path]:
    """Extract ``archive`` into a temporary directory and yield that directory."""
    archive_path = Path(archive)
    if not archive_path.is_file():
        raise FileNotFoundError(f"dataset archive not found: {archive_path}")
    with tempfile.TemporaryDirectory() as tmp:
        destination = Path(tmp)
        _extract_archive(archive_path, destination)
        yield destination


def _archive_fingerprint(archive: Path, expected_sha256: str | None) -> dict[str, T.Any]:
    """Return a stable fingerprint for extraction freshness checks."""
    stat = archive.stat()
    fingerprint: dict[str, T.Any] = {
        "archive": archive.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if expected_sha256 is not None:
        fingerprint["sha256"] = expected_sha256.lower()
    return fingerprint


def _read_extraction_marker(destination: Path) -> dict[str, T.Any] | None:
    """Return extraction marker JSON when present and valid."""
    marker = destination / EXTRACTION_MARKER
    if not marker.is_file():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _write_extraction_marker(destination: Path, payload: dict[str, T.Any]) -> None:
    """Write extraction freshness metadata."""
    marker = destination / EXTRACTION_MARKER
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _extracted_cache_has_content(destination: Path) -> bool:
    """Return whether an extracted cache directory contains source payload files."""
    return destination.is_dir() and any(
        item.name != EXTRACTION_MARKER for item in destination.iterdir()
    )


def _extraction_is_fresh(destination: Path, payload: dict[str, T.Any]) -> bool:
    """Return whether ``destination`` already contains a matching extraction."""
    if not _extracted_cache_has_content(destination):
        return False
    current = _read_extraction_marker(destination)
    return current == payload


def _extracted_matches_archive(
    extracted: Path,
    archive: Path,
    expected_sha256: str | None,
) -> bool:
    """Return whether ``extracted`` was produced from the current ``archive``."""
    if not archive.is_file() or not extracted.is_dir():
        return False
    return _extraction_is_fresh(extracted, _archive_fingerprint(archive, expected_sha256))


def _multipart_fingerprint(
    archives: T.Mapping[str, Path],
    parts: T.Sequence[MultiSourcePart],
) -> dict[str, T.Any]:
    """Return extraction freshness metadata for a multipart dataset source."""
    return {
        "multipart": True,
        "parts": {
            part.name: _archive_fingerprint(archives[part.name], part.sha256) for part in parts
        },
    }


def _multipart_archives_are_cached(
    spec: MultiDatasetSourceSpec,
    cache_root: Path,
) -> dict[str, Path] | None:
    """Return cached archive paths when all multipart source parts are present."""
    archives = {part.name: cache_root / part.archive_name for part in spec.parts}
    if all(path.is_file() for path in archives.values()):
        return archives
    return None


def _extract_multipart_archives(
    archives: T.Mapping[str, Path],
    spec: MultiDatasetSourceSpec,
    destination: Path,
    *,
    force: bool,
) -> Path:
    """Extract all multipart archives into one managed cache directory."""
    payload = _multipart_fingerprint(archives, spec.parts)
    if not force and _extraction_is_fresh(destination, payload):
        logger.info("Using cached extracted %s multipart source: %s", spec.dataset, destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(
        tempfile.mkdtemp(prefix=f"{destination.name}.", suffix=".part", dir=destination.parent)
    )
    try:
        for part in spec.parts:
            logger.info("Extracting %s %s into cache: %s", spec.dataset, part.name, destination)
            _extract_archive(archives[part.name], tmp_dir)
        _write_extraction_marker(tmp_dir, payload)
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        os.replace(tmp_dir, destination)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise
    return destination


def resolve_multipart_dataset_source(
    spec: MultiDatasetSourceSpec,
    *,
    cache_dir: str | os.PathLike[str] = DEFAULT_CACHE_DIR,
    force_download: bool = False,
    no_download: bool = False,
) -> Path:
    """Download/cache/extract a dataset distributed across multiple archives."""
    cache_root = Path(cache_dir) / spec.cache_root_name
    extracted = cache_root / EXTRACTED_DIR_NAME
    archives = None if force_download else _multipart_archives_are_cached(spec, cache_root)
    if archives is not None:
        for part in spec.parts:
            verify_sha256(archives[part.name], part.sha256, label=f"{spec.dataset} {part.name}")
        return _extract_multipart_archives(archives, spec, extracted, force=False)
    if not force_download and _extracted_cache_has_content(extracted):
        logger.info(
            "Using cached %s multipart extracted source without archives: %s",
            spec.dataset,
            extracted,
        )
        return extracted

    if no_download:
        raise FileNotFoundError(
            f"{spec.dataset} multipart source not found in {cache_root}. "
            f"Download disabled by --no-download.{_manual_hint(spec)}"
        )

    archives = {}
    for part in spec.parts:
        archives[part.name] = download(
            part.url,
            cache_root / part.archive_name,
            force=force_download,
            google_drive_file_id=part.google_drive_file_id,
            expected_sha256=part.sha256,
            label=f"{spec.dataset} {part.name}",
        )
    return _extract_multipart_archives(archives, spec, extracted, force=force_download)


def resolve_wflw_official_source(
    *,
    cache_dir: str | os.PathLike[str] = DEFAULT_CACHE_DIR,
    force_download: bool = False,
    no_download: bool = False,
) -> Path:
    """Resolve official WFLW annotations plus image archives into the dataset cache."""
    return resolve_multipart_dataset_source(
        WFLW_OFFICIAL_SOURCE,
        cache_dir=cache_dir,
        force_download=force_download,
        no_download=no_download,
    )


def extract_archive_to_cache(
    archive: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    force: bool = False,
    expected_sha256: str | None = None,
    label: str = "dataset archive",
) -> Path:
    """Extract ``archive`` into ``destination`` and reuse fresh extractions."""
    archive_path = Path(archive)
    if not archive_path.is_file():
        raise FileNotFoundError(f"dataset archive not found: {archive_path}")
    verify_sha256(archive_path, expected_sha256, label=label)
    destination_path = Path(destination)
    marker_payload = _archive_fingerprint(archive_path, expected_sha256)
    if not force and _extraction_is_fresh(destination_path, marker_payload):
        logger.info("Using cached extracted %s: %s", label, destination_path)
        return destination_path

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{destination_path.name}.", suffix=".part", dir=destination_path.parent
        )
    )
    try:
        logger.info("Extracting %s into cache: %s", label, destination_path)
        _extract_archive(archive_path, tmp_dir)
        _write_extraction_marker(tmp_dir, marker_payload)
        if destination_path.exists():
            if destination_path.is_dir():
                shutil.rmtree(destination_path)
            else:
                destination_path.unlink()
        os.replace(tmp_dir, destination_path)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise
    return destination_path


def is_archive(path: Path) -> bool:
    """Return whether ``path`` has a supported archive suffix."""
    return path.is_file() and any(
        str(path).lower().endswith(suffix) for suffix in ARCHIVE_SUFFIXES
    )


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


def _known_cached_sources(
    spec: DatasetSourceSpec, cache_root: Path
) -> tuple[Path | None, Path | None]:
    """Return the first known archive and non-archive cache candidates."""
    archive: Path | None = None
    direct: Path | None = None
    for name in _archive_names(spec):
        candidate = _existing_source(cache_root / name)
        if candidate is None:
            continue
        if is_archive(candidate):
            archive = archive or candidate
        else:
            direct = direct or candidate
    return archive, direct


def _generic_cached_archive(cache_root: Path) -> Path | None:
    """Return the first generic archive candidate in ``cache_root``."""
    if not cache_root.is_dir():
        return None
    for child in sorted(cache_root.iterdir()):
        if is_archive(child):
            return child
    return None


def _generic_cached_directory(cache_root: Path) -> Path | None:
    """Return the first generic directory candidate in ``cache_root``."""
    if not cache_root.is_dir():
        return None
    for child in sorted(cache_root.iterdir()):
        if child.name == EXTRACTED_DIR_NAME:
            continue
        candidate = _generic_cache_candidate(child)
        if candidate is not None and candidate.is_dir():
            return candidate
    return None


def _cached_extracted_source(
    spec: DatasetSourceSpec,
    cache_root: Path,
    archive: Path | None,
) -> Path | None:
    """Return a reusable extracted source only when it cannot be stale."""
    expected_sha256 = _effective_sha256(spec)
    extracted = cache_root / EXTRACTED_DIR_NAME
    if archive is not None:
        if _extracted_matches_archive(extracted, archive, expected_sha256):
            logger.info("Using fresh cached %s extracted source: %s", spec.dataset, extracted)
            return extracted
        return None
    if extracted.is_dir():
        logger.info(
            "Using cached %s extracted source without archive: %s", spec.dataset, extracted
        )
        return extracted
    for name in spec.extracted_aliases:
        candidate = cache_root / name
        if candidate.is_dir():
            logger.info("Using cached %s extracted source alias: %s", spec.dataset, candidate)
            return candidate
    return None


def _cached_source(spec: DatasetSourceSpec, cache_dir: Path) -> Path | None:
    """Return a cached archive, explicit alias file, or safe extracted source."""
    cache_root = cache_dir / spec.cache_root_name
    known_archive, known_direct = _known_cached_sources(spec, cache_root)
    generic_archive = None if known_archive is not None else _generic_cached_archive(cache_root)
    archive = known_archive or generic_archive

    extracted = _cached_extracted_source(spec, cache_root, archive)
    if extracted is not None:
        return extracted
    if archive is not None:
        logger.info("Using cached %s source archive: %s", spec.dataset, archive)
        return archive
    if known_direct is not None:
        logger.info("Using cached %s source: %s", spec.dataset, known_direct)
        return known_direct
    generic_dir = _generic_cached_directory(cache_root)
    if generic_dir is not None:
        logger.info("Using cached %s source candidate: %s", spec.dataset, generic_dir)
        return generic_dir
    return None


def _materialize_cached_source(
    spec: DatasetSourceSpec,
    source: Path,
    cache_root: Path,
    *,
    force_extract: bool = False,
) -> Path:
    """Return an extracted cache root for archives, otherwise ``source``."""
    if not is_archive(source):
        return source
    return extract_archive_to_cache(
        source,
        cache_root / spec.cache_root_name / EXTRACTED_DIR_NAME,
        force=force_extract,
        expected_sha256=_effective_sha256(spec),
        label=f"{spec.dataset} archive",
    )


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
            return _materialize_cached_source(spec, cached, cache_root)

    url = _effective_url(spec, download_url)
    google_drive_file_id = spec.google_drive_file_id
    expected_sha256 = None if download_url else _effective_sha256(spec)
    if no_download:
        raise FileNotFoundError(
            f"{spec.dataset} source not found in {cache_root / spec.cache_root_name}. "
            f"Download disabled by --no-download.{_manual_hint(spec)}"
        )
    if url is None and google_drive_file_id is None:
        raise FileNotFoundError(
            f"{spec.dataset} source not found in {cache_root / spec.cache_root_name} and no complete download source is configured.{_manual_hint(spec)}"  # noqa: E501
        )
    archive_name = _archive_name(spec)
    try:
        archive = download(
            url,
            cache_root / spec.cache_root_name / archive_name,
            force=force_download,
            google_drive_file_id=google_drive_file_id,
            expected_sha256=expected_sha256,
            label=f"{spec.dataset} archive",
        )
        return _materialize_cached_source(
            spec,
            archive,
            cache_root,
            force_extract=force_download,
        )
    except Exception as err:  # pragma: no cover - adapter-level tests monkeypatch failures
        raise OSError(
            f"{spec.dataset} download or extraction failed: {err}.{_manual_hint(spec)}"
        ) from err
