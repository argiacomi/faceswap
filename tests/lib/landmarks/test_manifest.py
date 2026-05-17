#!/usr/bin/env python3
"""Tests for :mod:`lib.landmarks.datasets.manifest_io` (Ticket 1).

Three legacy bbox normalizers (in ``eval.harness``, ``eval.geometry_metrics``,
and ``tools.landmarks.cache_predictions``) used to drift on the COFW-68 xywh
shape. The canonical coercer below has to handle every shape the legacy code
saw plus the dict variants.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.datasets.manifest_io import (
    LandmarkSample,
    bbox_for_sample,
    bbox_from_truth_fallback,
    coerce_bbox,
    coerce_visibility,
    load_manifest,
)


def test_coerce_bbox_accepts_ltrb_list() -> None:
    assert coerce_bbox([10, 20, 110, 220]) == (10.0, 20.0, 110.0, 220.0)


def test_coerce_bbox_treats_degenerate_ltrb_as_xywh_fallback() -> None:
    """Sequence inputs where 3rd/4th values would yield a degenerate ltrb fall back to xywh.

    Pure numeric sequences are ambiguous (``[10, 20, 100, 200]`` could be
    either shape and we trust upstream dataset modules to disambiguate).
    The coercer only auto-converts the unambiguous degenerate case where
    the third value is ``<=`` the first (or fourth ``<=`` second) — those
    are clearly width/height, not right/bottom.
    """
    assert coerce_bbox([10, 20, 5, 5]) == (10.0, 20.0, 15.0, 25.0)


def test_coerce_bbox_accepts_ltrb_dict() -> None:
    assert coerce_bbox({"left": 1, "top": 2, "right": 11, "bottom": 22}) == (
        1.0,
        2.0,
        11.0,
        22.0,
    )


def test_coerce_bbox_accepts_xywh_dict() -> None:
    assert coerce_bbox({"x": 1, "y": 2, "w": 10, "h": 20}) == (1.0, 2.0, 11.0, 22.0)


def test_coerce_bbox_rejects_none_and_short_sequences() -> None:
    assert coerce_bbox(None) is None
    assert coerce_bbox([1, 2, 3]) is None


def test_coerce_bbox_rejects_zero_width_xywh() -> None:
    assert coerce_bbox({"x": 1, "y": 2, "w": 0, "h": 10}) is None


def test_coerce_bbox_handles_numpy_array() -> None:
    bbox = coerce_bbox(np.array([10.0, 20.0, 110.0, 220.0]))
    assert bbox == (10.0, 20.0, 110.0, 220.0)


def test_coerce_bbox_returns_ltrb_for_already_ltrb_array() -> None:
    """Plain ltrb arrays must round-trip unchanged (no xywh re-interpretation)."""
    assert coerce_bbox(np.array([0.0, 0.0, 256.0, 256.0])) == (0.0, 0.0, 256.0, 256.0)


def test_coerce_visibility_returns_bool_tuple() -> None:
    assert coerce_visibility([1, 0, True, False]) == (True, False, True, False)


def test_coerce_visibility_handles_empty() -> None:
    assert coerce_visibility([]) is None
    assert coerce_visibility(None) is None


def test_bbox_from_truth_fallback_returns_landmark_extent() -> None:
    truth = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 100.0]], dtype="float32")
    assert bbox_from_truth_fallback(truth) == (10.0, 20.0, 50.0, 100.0)


def test_bbox_from_truth_fallback_rejects_degenerate_truth() -> None:
    assert bbox_from_truth_fallback(np.zeros((0, 2))) is None
    assert bbox_from_truth_fallback(np.array([[1.0, 2.0]])) is None


def test_load_manifest_resolves_relative_paths_and_metadata(tmp_path: Path) -> None:
    """``load_manifest`` reads bbox + visibility from either top-level or metadata."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    truth = np.array([[0.0, 0.0], [10.0, 10.0]], dtype="float32")
    np.save(str(dataset / "truth.npy"), truth)
    samples = [
        {
            "sample_id": "top-level-bbox",
            "image": "image.png",
            "landmarks": "truth.npy",
            "dataset": "fixture",
            "condition": "clean",
            "face_bbox": [0.0, 0.0, 100.0, 100.0],
        },
        {
            "sample_id": "metadata-xywh-bbox",
            "image": "image.png",
            "landmarks": "truth.npy",
            "dataset": "fixture",
            "condition": "clean",
            "metadata": {
                "face_bbox": {"x": 5, "y": 5, "w": 50, "h": 50},
                "visibility": [True, False] * 34,
            },
        },
        {
            "sample_id": "generic-bbox-fallback",
            "image": "image.png",
            "landmarks": "truth.npy",
            "dataset": "fixture",
            "condition": "clean",
            # Pure-numeric bbox: trust dataset module's pre-coercion. The
            # canonical coercer treats this as ltrb because both values
            # exceed the origin.
            "bbox": [10.0, 10.0, 60.0, 60.0],
        },
    ]
    manifest_path = dataset / "manifest.json"
    manifest_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")

    loaded = load_manifest(manifest_path)
    by_id = {sample.sample_id: sample for sample in loaded}

    assert by_id["top-level-bbox"].face_bbox == (0.0, 0.0, 100.0, 100.0)
    assert by_id["metadata-xywh-bbox"].face_bbox == (5.0, 5.0, 55.0, 55.0)
    assert by_id["metadata-xywh-bbox"].visibility is not None
    assert len(by_id["metadata-xywh-bbox"].visibility) == 68
    assert by_id["generic-bbox-fallback"].face_bbox == (10.0, 10.0, 60.0, 60.0)
    # Image / landmark paths resolve against the manifest's parent directory.
    assert by_id["top-level-bbox"].image == str((dataset / "image.png").resolve())
    assert by_id["top-level-bbox"].landmarks == str((dataset / "truth.npy").resolve())


def test_load_manifest_rejects_missing_landmarks(tmp_path: Path) -> None:
    """Manifest entries without a landmarks path fail fast (hard contract)."""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"samples": [{"sample_id": "broken"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing landmarks path"):
        load_manifest(manifest_path)


def test_bbox_for_sample_prefers_manifest_bbox(tmp_path: Path) -> None:
    truth = np.array([[0.0, 0.0], [10.0, 10.0]], dtype="float32")
    np.save(str(tmp_path / "truth.npy"), truth)
    sample = LandmarkSample(
        sample_id="s",
        image="",
        landmarks=str(tmp_path / "truth.npy"),
        face_bbox=(100.0, 100.0, 200.0, 200.0),
    )
    assert bbox_for_sample(sample) == (100.0, 100.0, 200.0, 200.0)


def test_bbox_for_sample_falls_back_to_truth_extent(tmp_path: Path) -> None:
    truth = np.array([[0.0, 0.0], [10.0, 20.0]], dtype="float32")
    np.save(str(tmp_path / "truth.npy"), truth)
    sample = LandmarkSample(
        sample_id="s",
        image="",
        landmarks=str(tmp_path / "truth.npy"),
        face_bbox=None,
    )
    assert bbox_for_sample(sample) == (0.0, 0.0, 10.0, 20.0)


def test_bbox_for_sample_can_disable_fallback(tmp_path: Path) -> None:
    """Tools that need a real detector bbox can opt out of the truth fallback."""
    truth = np.array([[0.0, 0.0], [10.0, 20.0]], dtype="float32")
    np.save(str(tmp_path / "truth.npy"), truth)
    sample = LandmarkSample(
        sample_id="s",
        image="",
        landmarks=str(tmp_path / "truth.npy"),
        face_bbox=None,
    )
    assert bbox_for_sample(sample, allow_truth_fallback=False) is None
