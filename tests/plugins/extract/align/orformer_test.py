#!/usr/bin python3
"""Pytest unit tests for the ORFormer aligner archive-extraction helpers."""
# pylint:disable=protected-access

import hashlib
import os
import zipfile

import pytest

from plugins.extract.align import orformer


def _build_archive(tmp_path: str, members: dict[str, bytes]) -> str:
    """Write a zip archive containing the supplied members and return its path."""
    archive_path = os.path.join(tmp_path, "weights.zip")
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, payload in members.items():
            with archive.open(name, "w") as out:
                out.write(payload)
    return archive_path


def _checkpoint(member: str, filename: str, payload: bytes) -> orformer.ORFormerCheckpoint:
    """Build a checkpoint descriptor with the SHA256 of ``payload``."""
    return orformer.ORFormerCheckpoint(
        archive_member=member,
        filename=filename,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_extract_checkpoint_writes_and_validates(
    tmp_path: pytest.TempdirFactory,
) -> None:
    """A fresh extract writes the file and matches the expected SHA256."""
    cache_dir = str(tmp_path)
    payload = b"hgnet-state-dict-bytes" * 16
    archive_path = _build_archive(cache_dir, {"weights/HGNet/300W/best_model.pt": payload})
    checkpoint = _checkpoint("weights/HGNet/300W/best_model.pt", "hgnet.pt", payload)

    output_path = orformer._extract_checkpoint(archive_path, cache_dir, checkpoint)

    assert output_path == os.path.join(cache_dir, "hgnet.pt")
    assert os.path.exists(output_path)
    with open(output_path, "rb") as infile:
        assert infile.read() == payload
    assert not os.path.exists(f"{output_path}.part")


def test_extract_checkpoint_skips_when_cached(tmp_path: pytest.TempdirFactory, mocker) -> None:
    """A correctly hashed cached file is reused without touching the archive."""
    cache_dir = str(tmp_path)
    payload = b"orformer-state" * 32
    archive_path = _build_archive(cache_dir, {"weights/ORFormer/WFLW/best_model.pt": payload})
    checkpoint = _checkpoint("weights/ORFormer/WFLW/best_model.pt", "orformer.pt", payload)

    cached_path = os.path.join(cache_dir, "orformer.pt")
    with open(cached_path, "wb") as outfile:
        outfile.write(payload)

    spy = mocker.spy(zipfile, "ZipFile")
    output_path = orformer._extract_checkpoint(archive_path, cache_dir, checkpoint)

    assert output_path == cached_path
    assert spy.call_count == 0


def test_extract_checkpoint_replaces_bad_cache(tmp_path: pytest.TempdirFactory) -> None:
    """A cached file with a wrong hash is replaced with the correct extracted payload."""
    cache_dir = str(tmp_path)
    payload = b"good-payload-bytes" * 8
    archive_path = _build_archive(cache_dir, {"weights/HGNet/WFLW/best_model.pt": payload})
    checkpoint = _checkpoint("weights/HGNet/WFLW/best_model.pt", "hgnet.pt", payload)

    cached_path = os.path.join(cache_dir, "hgnet.pt")
    with open(cached_path, "wb") as outfile:
        outfile.write(b"bad-payload")

    output_path = orformer._extract_checkpoint(archive_path, cache_dir, checkpoint)

    with open(output_path, "rb") as infile:
        assert infile.read() == payload


def test_extract_checkpoint_cleans_stale_part(tmp_path: pytest.TempdirFactory) -> None:
    """A leftover .part file from a prior interrupted extract is removed first."""
    cache_dir = str(tmp_path)
    payload = b"clean-extract" * 16
    archive_path = _build_archive(cache_dir, {"weights/HGNet/300W/best_model.pt": payload})
    checkpoint = _checkpoint("weights/HGNet/300W/best_model.pt", "hgnet.pt", payload)

    output_path = os.path.join(cache_dir, "hgnet.pt")
    partial_path = f"{output_path}.part"
    with open(partial_path, "wb") as outfile:
        outfile.write(b"leftover")

    orformer._extract_checkpoint(archive_path, cache_dir, checkpoint)

    assert not os.path.exists(partial_path)
    with open(output_path, "rb") as infile:
        assert infile.read() == payload


def test_extract_checkpoint_hash_mismatch_raises(
    tmp_path: pytest.TempdirFactory,
) -> None:
    """A SHA256 mismatch on the extracted payload raises and removes the .part file."""
    cache_dir = str(tmp_path)
    payload = b"actual-payload"
    archive_path = _build_archive(cache_dir, {"weights/HGNet/WFLW/best_model.pt": payload})
    checkpoint = orformer.ORFormerCheckpoint(
        archive_member="weights/HGNet/WFLW/best_model.pt",
        filename="hgnet.pt",
        sha256=hashlib.sha256(b"different-payload").hexdigest(),
    )

    with pytest.raises(RuntimeError, match="hash mismatch"):
        orformer._extract_checkpoint(archive_path, cache_dir, checkpoint)

    output_path = os.path.join(cache_dir, "hgnet.pt")
    assert not os.path.exists(output_path)
    assert not os.path.exists(f"{output_path}.part")


def test_hash_file_matches_hashlib(tmp_path: pytest.TempdirFactory) -> None:
    """``_hash_file`` matches ``hashlib.sha256`` for arbitrary content sizes."""
    cache_dir = str(tmp_path)
    payload = b"abc123" * 100_000  # multi-chunk
    target = os.path.join(cache_dir, "blob.bin")
    with open(target, "wb") as outfile:
        outfile.write(payload)
    assert orformer._hash_file(target) == hashlib.sha256(payload).hexdigest()
