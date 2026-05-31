#!/usr/bin/env python3
"""Behaviour-preservation tests for the P3 ``lib/align`` simplifications (#191)."""

from __future__ import annotations

import numpy as np

from lib.align.alignments import Alignments
from lib.align.objects import AlignmentsEntry, FileAlignments


def _face_with_masks(masks: list[str]) -> FileAlignments:
    face = FileAlignments(x=0, y=0, w=10, h=10, landmarks_xy=np.zeros((68, 2), dtype="float32"))
    face.mask = {name: None for name in masks}  # type: ignore[misc]
    return face


def test_mask_summary_counter_behaviour_matches_legacy() -> None:
    """``mask_summary`` returns the same counts the manual ``dict.get(...)+1``
    loop produced before being refactored to ``collections.Counter``.
    """
    alignments = Alignments.__new__(Alignments)
    alignments._data = {
        "frame_a.png": AlignmentsEntry(
            faces=[
                _face_with_masks(["bisenet-fp", "components"]),
                _face_with_masks([]),  # no mask → counts as "none"
            ]
        ),
        "frame_b.png": AlignmentsEntry(
            faces=[
                _face_with_masks(["bisenet-fp"]),
                _face_with_masks(["components"]),
            ]
        ),
    }

    summary = alignments.mask_summary

    assert summary == {"none": 1, "bisenet-fp": 2, "components": 2}


def test_numpy_to_list_update_handles_legacy_alignment_without_thumb() -> None:
    """``NumpyToList.update`` must use ``alignment.get('thumb')`` so legacy
    alignments without a ``thumb`` field don't raise ``KeyError`` on update.
    """
    from lib.align.updater import NumpyToList

    alignments = {
        "frame_a.png": {
            "faces": [
                {"landmarks_xy": np.zeros((68, 2), dtype="float32")},
                {"landmarks_xy": [[0.0, 0.0]]},
            ]
        }
    }

    updater = NumpyToList.__new__(NumpyToList)
    updater._alignments = alignments

    # Must not raise — and only the ndarray entry should flip to list.
    count = updater.update()

    assert count == 1
    first = alignments["frame_a.png"]["faces"][0]
    second = alignments["frame_a.png"]["faces"][1]
    assert first["landmarks_xy"] == [[0.0, 0.0]] * 68  # type: ignore[index]
    assert second["landmarks_xy"] == [[0.0, 0.0]]  # type: ignore[index]
    # ``thumb`` was never set; the updater must not have created it.
    assert "thumb" not in first  # type: ignore[operator]
    assert "thumb" not in second  # type: ignore[operator]


def test_solve_pnp_writes_rotation_and_translation_into_preallocated_buffer(monkeypatch) -> None:
    """``Batch3D.solve_pnp`` must use the preallocated ``(N, 2, 3, 1)``
    float32 buffer rather than building a Python list of
    ``cv2.solvePnP(...)[1:]`` results then ``np.array``-ing it (#191).

    Spies on ``cv2.solvePnP`` to feed back deterministic rotation /
    translation vectors and asserts they land in the
    ``(rotation_first, translation_first)`` row order the preallocation
    contract guarantees, then swap-axes to the documented
    ``(2, N, 3, 1)`` shape.
    """
    import cv2

    from lib.align.pose import _CORE_LMS, Batch3D

    n = 3
    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    for idx in range(n):
        rotations.append(np.array([[float(idx) + 0.1], [0.2], [0.3]], dtype=np.float64))
        translations.append(np.array([[1.0 + idx], [2.0 + idx], [3.0 + idx]], dtype=np.float64))

    call_index = {"n": 0}

    def _fake_solve_pnp(_obj, _img, _camera, _dist, flags):
        idx = call_index["n"]
        call_index["n"] += 1
        return True, rotations[idx], translations[idx]

    monkeypatch.setattr(cv2, "solvePnP", _fake_solve_pnp)

    rng = np.random.default_rng(0)
    landmarks = rng.uniform(0.2, 0.8, (n, 68, 2)).astype(np.float32)

    result = Batch3D.solve_pnp(landmarks)

    assert result.shape == (2, n, 3, 1)
    assert result.dtype == np.float32

    for idx in range(n):
        np.testing.assert_allclose(result[0, idx], rotations[idx].astype(np.float32))
        np.testing.assert_allclose(result[1, idx], translations[idx].astype(np.float32))
    del _CORE_LMS  # quiet unused import


def test_to_alignment_uses_landmarks_xy_property_without_recast() -> None:
    """``to_alignment`` must hand the property straight through —
    ``_as_landmarks_array`` already enforced float32 on the way in
    (#191).
    """
    from lib.align.detected_face import DetectedFace

    landmarks = np.zeros((68, 2), dtype=np.float32)  # type: ignore[var-annotated]
    face = DetectedFace(left=0, top=0, width=10, height=10, landmarks_xy=landmarks)

    out: FileAlignments = face.to_alignment()

    # ``to_alignment`` must not have built a fresh copy via the redundant
    # ``np.asarray(..., dtype=np.float32)`` cast — the same float32 array
    # the property exposes should land in the alignment payload.
    assert out.landmarks_xy is face.landmarks_xy


def test_to_png_meta_uses_landmarks_xy_property_without_recast() -> None:
    """Same contract as ``to_alignment`` for the PNG-meta path."""
    from lib.align.detected_face import DetectedFace
    from lib.align.objects import PNGAlignments

    landmarks = np.zeros((68, 2), dtype=np.float32)  # type: ignore[var-annotated]
    face = DetectedFace(left=0, top=0, width=10, height=10, landmarks_xy=landmarks)

    out: PNGAlignments = face.to_png_meta()

    assert out.landmarks_xy is face.landmarks_xy
