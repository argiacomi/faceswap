#!/usr/bin/env python3
"""Tests for persisting per-face alignment metadata."""

from __future__ import annotations

import numpy as np

from lib.align.detected_face import DetectedFace
from lib.infer.objects import FrameFaces, frame_faces_to_alignment


def test_frame_faces_to_alignment_persists_landmark_metadata() -> None:
    """Resolver debug metadata is serialized into alignment face entries."""
    landmarks = np.zeros((1, 68, 2), dtype="float32")  # type: ignore[var-annotated]
    media = FrameFaces(
        "frame.png",
        np.zeros((16, 16, 3), dtype="uint8"),
        bboxes=np.array([[1, 2, 11, 12]], dtype="int32"),
        landmarks=landmarks,
    )
    media.aligned.metadata = [
        {"landmark_ensemble": {"selected_candidate": "spiga", "bucket": "profile_left"}}
    ]

    alignment = frame_faces_to_alignment(media)[0]

    assert alignment.metadata["landmark_ensemble"]["selected_candidate"] == "spiga"
    face = DetectedFace().from_alignment(alignment)
    assert face.metadata["landmark_ensemble"]["bucket"] == "profile_left"
