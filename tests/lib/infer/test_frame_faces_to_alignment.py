#!/usr/bin/env python3
"""Tests for FrameFaces alignment serialization."""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

from lib.infer.objects import FrameFaces, frame_faces_to_alignment


def _media(
    identities: dict[str, npt.NDArray[np.float32]],
) -> FrameFaces:
    """Build a two-face FrameFaces object for serialization tests."""
    image: npt.NDArray[np.uint8] = np.zeros((64, 64, 3), dtype=np.uint8)
    bboxes: npt.NDArray[np.int32] = np.array(
        [[1, 2, 11, 22], [3, 4, 13, 24]],
        dtype=np.int32,
    )
    landmarks: npt.NDArray[np.float32] = np.zeros((2, 68, 2), dtype=np.float32)
    return FrameFaces(
        "frame_000001.png",
        image,
        bboxes=bboxes,
        landmarks=landmarks,
        identities=identities,
    )


def test_frame_faces_to_alignment_keeps_well_formed_identities() -> None:
    """Identity rows matching the face count are serialized per face."""
    identities: dict[str, npt.NDArray[np.float32]] = {
        "vggface2": np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=np.float32,
        )
    }

    alignments = frame_faces_to_alignment(_media(identities))

    assert alignments[0].identity == {"vggface2": [1.0, 2.0, 3.0]}
    assert alignments[1].identity == {"vggface2": [4.0, 5.0, 6.0]}


def test_frame_faces_to_alignment_drops_malformed_identities(
    caplog,
) -> None:
    """Malformed identity rows are optional metadata, not an extraction-stopping crash."""
    identities: dict[str, npt.NDArray[np.float32]] = {
        "vggface2": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
    }

    with caplog.at_level(logging.WARNING):
        alignments = frame_faces_to_alignment(_media(identities))

    assert [alignment.identity for alignment in alignments] == [{}, {}]
    assert "Dropping malformed identity metadata" in caplog.text
    assert "1 identity row(s) for 2 face(s)" in caplog.text
