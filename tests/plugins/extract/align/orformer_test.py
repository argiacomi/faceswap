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


def _list_part_files(cache_dir: str, filename: str) -> list[str]:
    """Return leftover ``tempfile.mkstemp``-style extract artefacts for ``filename``.

    Matches names like ``filename.<random>.part`` while ignoring a literal
    ``filename.part`` (which the new extract path does not produce and may be
    left in place from an older release).
    """
    prefix = f"{filename}."
    return [
        entry
        for entry in os.listdir(cache_dir)
        if entry.startswith(prefix) and entry.endswith(".part") and entry != f"{filename}.part"
    ]


def test_extract_checkpoint_writes_and_validates(
    tmp_path: pytest.TempdirFactory,
) -> None:
    """A fresh extract writes the file and matches the expected SHA256."""
    cache_dir = str(tmp_path)
    payload = b"hgnet-state-dict-bytes" * 16
    archive_path = _build_archive(cache_dir, {"weights/HGNet/300W/best_model.pt": payload})
    checkpoint = _checkpoint("weights/HGNet/300W/best_model.pt", "hgnet.pt", payload)

    with zipfile.ZipFile(archive_path) as archive:
        output_path = orformer._extract_checkpoint(archive, cache_dir, checkpoint)

    assert output_path == os.path.join(cache_dir, "hgnet.pt")
    assert os.path.exists(output_path)
    with open(output_path, "rb") as infile:
        assert infile.read() == payload
    assert _list_part_files(cache_dir, "hgnet.pt") == []


def test_get_checkpoint_paths_skips_when_cached(tmp_path: pytest.TempdirFactory, mocker) -> None:
    """``_get_checkpoint_paths`` reuses a correctly hashed cached file without opening the archive."""
    cache_dir = str(tmp_path)
    hgnet_payload = b"hgnet-cached" * 16
    orformer_payload = b"orformer-cached" * 16
    archive_path = _build_archive(
        cache_dir,
        {
            "weights/HGNet/WFLW/best_model.pt": hgnet_payload,
            "weights/ORFormer/WFLW/best_model.pt": orformer_payload,
        },
    )
    hgnet_checkpoint = _checkpoint("weights/HGNet/WFLW/best_model.pt", "hgnet.pt", hgnet_payload)
    orformer_checkpoint = _checkpoint(
        "weights/ORFormer/WFLW/best_model.pt", "orformer.pt", orformer_payload
    )

    with open(os.path.join(cache_dir, "hgnet.pt"), "wb") as outfile:
        outfile.write(hgnet_payload)
    with open(os.path.join(cache_dir, "orformer.pt"), "wb") as outfile:
        outfile.write(orformer_payload)

    model_config = orformer.ORFormerModelConfig(
        num_landmarks=98,
        num_edges=15,
        edge_info=lambda: [],
        hgnet=hgnet_checkpoint,
        orformer=orformer_checkpoint,
    )
    mocker.patch.object(
        orformer.GetModelFromUrl,
        "__init__",
        lambda self, *_args, **_kwargs: None,
    )
    mocker.patch.object(
        orformer.GetModelFromUrl, "model_path", property(lambda self: archive_path)
    )
    spy = mocker.spy(zipfile, "ZipFile")

    resolved = orformer._get_checkpoint_paths(model_config)

    assert resolved == {
        "hgnet": os.path.join(cache_dir, "hgnet.pt"),
        "orformer": os.path.join(cache_dir, "orformer.pt"),
    }
    assert spy.call_count == 0


def test_checkpoint_is_valid_uses_sentinel(tmp_path: pytest.TempdirFactory, mocker) -> None:
    """``_checkpoint_is_valid`` short-circuits via the sentinel on the second call."""
    cache_dir = str(tmp_path)
    payload = b"sentinel-test" * 16
    target = os.path.join(cache_dir, "blob.pt")
    with open(target, "wb") as outfile:
        outfile.write(payload)
    expected = hashlib.sha256(payload).hexdigest()

    spy = mocker.spy(orformer, "_hash_file")
    assert orformer._checkpoint_is_valid(target, expected)
    assert spy.call_count == 1

    # Second call must use the sentinel and not re-hash.
    assert orformer._checkpoint_is_valid(target, expected)
    assert spy.call_count == 1


def test_extract_checkpoint_replaces_bad_cache(tmp_path: pytest.TempdirFactory) -> None:
    """Extraction overwrites a wrong-hash cache file with the correct payload."""
    cache_dir = str(tmp_path)
    payload = b"good-payload-bytes" * 8
    archive_path = _build_archive(cache_dir, {"weights/HGNet/WFLW/best_model.pt": payload})
    checkpoint = _checkpoint("weights/HGNet/WFLW/best_model.pt", "hgnet.pt", payload)

    cached_path = os.path.join(cache_dir, "hgnet.pt")
    with open(cached_path, "wb") as outfile:
        outfile.write(b"bad-payload")

    with zipfile.ZipFile(archive_path) as archive:
        orformer._extract_checkpoint(archive, cache_dir, checkpoint)

    with open(cached_path, "rb") as infile:
        assert infile.read() == payload
    assert _list_part_files(cache_dir, "hgnet.pt") == []


def test_extract_checkpoint_uses_per_process_tempfile(
    tmp_path: pytest.TempdirFactory,
) -> None:
    """A pre-existing static ``<output>.part`` is irrelevant: extraction uses a tempfile."""
    cache_dir = str(tmp_path)
    payload = b"clean-extract" * 16
    archive_path = _build_archive(cache_dir, {"weights/HGNet/300W/best_model.pt": payload})
    checkpoint = _checkpoint("weights/HGNet/300W/best_model.pt", "hgnet.pt", payload)

    output_path = os.path.join(cache_dir, "hgnet.pt")
    static_part = f"{output_path}.part"
    with open(static_part, "wb") as outfile:
        outfile.write(b"leftover from old version")

    with zipfile.ZipFile(archive_path) as archive:
        orformer._extract_checkpoint(archive, cache_dir, checkpoint)

    with open(output_path, "rb") as infile:
        assert infile.read() == payload
    # The per-process tempfile must be cleaned up; the static ``.part`` is intentionally untouched.
    assert _list_part_files(cache_dir, "hgnet.pt") == []
    assert os.path.exists(static_part)


def test_extract_checkpoint_hash_mismatch_raises_and_cleans_up(
    tmp_path: pytest.TempdirFactory,
) -> None:
    """A SHA256 mismatch raises and removes the tempfile rather than the final path."""
    cache_dir = str(tmp_path)
    payload = b"actual-payload"
    archive_path = _build_archive(cache_dir, {"weights/HGNet/WFLW/best_model.pt": payload})
    checkpoint = orformer.ORFormerCheckpoint(
        archive_member="weights/HGNet/WFLW/best_model.pt",
        filename="hgnet.pt",
        sha256=hashlib.sha256(b"different-payload").hexdigest(),
    )

    with (
        zipfile.ZipFile(archive_path) as archive,
        pytest.raises(RuntimeError, match="hash mismatch"),
    ):
        orformer._extract_checkpoint(archive, cache_dir, checkpoint)

    output_path = os.path.join(cache_dir, "hgnet.pt")
    assert not os.path.exists(output_path)
    assert _list_part_files(cache_dir, "hgnet.pt") == []


def test_hash_file_matches_hashlib(tmp_path: pytest.TempdirFactory) -> None:
    """``_hash_file`` matches ``hashlib.sha256`` for arbitrary content sizes."""
    cache_dir = str(tmp_path)
    payload = b"abc123" * 100_000  # multi-chunk
    target = os.path.join(cache_dir, "blob.bin")
    with open(target, "wb") as outfile:
        outfile.write(payload)
    assert orformer._hash_file(target) == hashlib.sha256(payload).hexdigest()
