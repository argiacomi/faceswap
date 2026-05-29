#!/usr/bin/env python3
"""Behaviour-preservation tests for the P3 ``lib/align`` simplifications (#191)."""

from __future__ import annotations

import numpy as np

from lib.align.alignments import Alignments
from lib.align.objects import AlignmentsEntry, FileAlignments


def _face_with_masks(masks: list[str]) -> FileAlignments:
    face = FileAlignments(x=0, y=0, w=10, h=10, landmarks_xy=np.zeros((68, 2), dtype="float32"))
    face.mask = {name: None for name in masks}  # type: ignore[arg-type]
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
    updater._alignments = alignments  # type: ignore[attr-defined]

    # Must not raise — and only the ndarray entry should flip to list.
    count = updater.update()

    assert count == 1
    first = alignments["frame_a.png"]["faces"][0]
    second = alignments["frame_a.png"]["faces"][1]
    assert first["landmarks_xy"] == [[0.0, 0.0]] * 68
    assert second["landmarks_xy"] == [[0.0, 0.0]]
    # ``thumb`` was never set; the updater must not have created it.
    assert "thumb" not in first
    assert "thumb" not in second
