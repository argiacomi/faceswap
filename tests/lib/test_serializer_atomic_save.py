#!/usr/bin/env python3
"""Regression tests for crash-safe serializer writes."""

from __future__ import annotations

import os
import stat

import pytest

from lib.serializer import get_serializer
from lib.utils import FaceswapError


def test_atomic_save_preserves_original_when_replace_fails(tmp_path, monkeypatch):
    serializer = get_serializer("compressed")
    target = tmp_path / "alignments.fsa"
    serializer.save(str(target), {"version": 1, "value": "original"})
    original_bytes = target.read_bytes()

    def _raise_replace(_src, _dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", _raise_replace)

    with pytest.raises(FaceswapError):
        serializer.save(str(target), {"version": 2, "value": "new"})

    assert target.read_bytes() == original_bytes
    assert serializer.load(str(target)) == {"version": 1, "value": "original"}
    assert not list(tmp_path.glob(".alignments.fsa.*.tmp"))


def test_atomic_save_preserves_original_when_roundtrip_validation_fails(
    tmp_path,
    monkeypatch,
):
    serializer = get_serializer("json")
    target = tmp_path / "data.json"
    serializer.save(str(target), {"value": "original"})
    original_bytes = target.read_bytes()

    def _raise_unmarshal(_payload):
        raise RuntimeError("simulated validation failure")

    monkeypatch.setattr(serializer, "unmarshal", _raise_unmarshal)

    with pytest.raises(FaceswapError):
        serializer.save(str(target), {"value": "new"})

    assert target.read_bytes() == original_bytes
    assert not list(tmp_path.glob(".data.json.*.tmp"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX file mode test")
def test_atomic_save_preserves_existing_file_mode(tmp_path):
    serializer = get_serializer("json")
    target = tmp_path / "data.json"
    serializer.save(str(target), {"value": "original"})
    target.chmod(0o640)

    serializer.save(str(target), {"value": "updated"})

    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert serializer.load(str(target)) == {"value": "updated"}
