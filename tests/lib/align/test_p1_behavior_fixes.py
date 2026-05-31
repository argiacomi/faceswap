#!/usr/bin/env python3
"""Regression tests for the P1 ``lib/align`` behaviour fixes (issue #191)."""

from __future__ import annotations

import numpy as np
import pytest

from lib.align.alignments import Alignments
from lib.align.objects import AlignmentsEntry, FileAlignments
from lib.align.pose import get_camera_matrix

# ---------------------------------------------------------------------------
# P1 #1: alignments.frame_has_multiple_faces — missing frame must return False
# ---------------------------------------------------------------------------


def test_frame_has_multiple_faces_returns_false_for_unknown_frame(tmp_path) -> None:
    """A missing frame name must return ``False``, not raise ``AttributeError``.

    Previously ``self._data.get(frame_name, AlignmentsEntry)`` returned the
    CLASS object as the default, then ``.faces`` raised ``AttributeError``
    because the class has no instance attribute populated.
    """
    alignments = Alignments.__new__(Alignments)
    alignments._data = {
        "frame_known.png": AlignmentsEntry(
            faces=[
                FileAlignments(x=0, y=0, w=10, h=10, landmarks_xy=np.zeros((68, 2), "float32")),
            ]
        )
    }

    assert alignments.frame_has_multiple_faces("frame_missing.png") is False
    assert alignments.frame_has_multiple_faces("frame_known.png") is False
    alignments._data["frame_known.png"].faces.append(
        FileAlignments(x=0, y=0, w=10, h=10, landmarks_xy=np.zeros((68, 2), "float32"))
    )
    assert alignments.frame_has_multiple_faces("frame_known.png") is True


# ---------------------------------------------------------------------------
# P1 #2: pose.get_camera_matrix preserves caller-supplied focal length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("focal", [1, 2, 4, 8, 16])
def test_get_camera_matrix_preserves_focal_length(focal: int) -> None:
    """``focal_length`` argument must flow into the [0,0] / [1,1] diagonal."""
    matrix = get_camera_matrix(focal_length=focal)
    assert matrix.shape == (3, 3)
    assert matrix[0, 0] == focal
    assert matrix[1, 1] == focal
    # Principal point stays at the normalised image centre.
    assert matrix[0, 2] == 0.5
    assert matrix[1, 2] == 0.5


# ---------------------------------------------------------------------------
# P1 #3: aligned_utils.batch_sub_crop respects channel count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_batch_sub_crop_handles_non_three_channel_images(channels: int) -> None:
    """``batch_sub_crop`` must reshape with the actual channel count, not 3.

    Regression: the function previously hard-coded ``3`` in the final
    ``gathered.reshape``, so RGBA (4-channel) and grayscale (1-channel)
    crops crashed or were silently mis-shaped. The fix reuses
    ``channels = images[0].shape[-1]`` (already computed for the
    output buffer) for the reshape too.
    """
    from lib.align.aligned_utils import batch_sub_crop

    height = width = 32
    out_size = 16
    batch = 2
    images = np.full((batch, height, width, channels), 200, dtype=np.uint8)  # type: ignore[var-annotated]
    offsets = np.zeros((batch, 2), dtype=np.int32)  # type: ignore[var-annotated]

    out = batch_sub_crop(images, offsets, out_size)

    assert out.shape == (batch, out_size, out_size, channels)
    assert out.dtype == images.dtype


# ---------------------------------------------------------------------------
# P1 #4: aligned_face.relative_eye_mouth_position uses the converted lms
# ---------------------------------------------------------------------------


def test_relative_eye_mouth_position_indexes_converted_landmarks(monkeypatch) -> None:
    """For 98-point landmarks the property must index the points_to_68 result.

    Regression: the previous code overwrote ``lms`` with
    ``points_to_68(lms)`` and then indexed
    ``self.normalized_landmarks`` (the un-converted 98-point array)
    instead of ``lms``. With 98-point landmarks, indices like ``48:68``
    point at the wrong landmarks under the 98-point convention so the
    computed eye/mouth offset is wrong.
    """
    import lib.align.aligned_face as aligned_face_mod
    from lib.align.aligned_face import AlignedFace, LandmarkType

    # 98 landmarks with a deterministic y-gradient so any indexing
    # mistake produces a different numeric position.
    points = np.stack(
        [
            np.linspace(0.0, 1.0, 98, dtype="float32"),
            np.arange(98, dtype="float32") / 100.0,
        ],
        axis=1,
    )

    face = AlignedFace.__new__(AlignedFace)
    face._landmark_type = LandmarkType.LM_2D_98

    class _Cache:
        relative_eye_mouth_position = None

        def lock(self, _name):
            class _Ctx:
                def __enter__(self_inner):
                    return None

                def __exit__(self_inner, *args):
                    return False

            return _Ctx()

    face._cache = _Cache()  # type: ignore[assignment]
    # monkeypatch.setattr handles teardown for us so we never strip the
    # base class's real ``normalized_landmarks`` property after the test
    # — that previous corruption broke unrelated AlignedFace consumers
    # (FaceQA records) in the same pytest session.
    monkeypatch.setattr(AlignedFace, "normalized_landmarks", property(lambda _self: points))

    converted = aligned_face_mod.points_to_68(points)
    expected_lowest = float(np.max(converted[np.r_[17:27, 36:48], 1]))
    expected_highest = float(np.min(converted[48:68, 1]))
    expected_position = expected_highest - expected_lowest

    actual = face.relative_eye_mouth_position
    assert actual == pytest.approx(expected_position)


# ---------------------------------------------------------------------------
# P1 #5: detected_face.from_alignment treats all-black frames as real images
# ---------------------------------------------------------------------------


def test_from_alignment_invokes_image_to_face_on_all_black_frame() -> None:
    """All-black frames must still trigger ``_image_to_face``.

    Regression: the previous guard ``image is not None and image.any()``
    short-circuited on any frame whose every pixel was 0 — an
    all-black frame would silently skip ``_image_to_face``. The new
    guard checks ``image.size`` instead, which only rejects an empty
    array.
    """
    from lib.align.detected_face import DetectedFace
    from lib.align.objects import FileAlignments

    face = DetectedFace.__new__(DetectedFace)
    DetectedFace.__init__(face)
    alignment = FileAlignments(
        x=0,
        y=0,
        w=32,
        h=32,
        landmarks_xy=np.zeros((68, 2), dtype="float32"),
    )

    calls: list[np.ndarray] = []
    face._image_to_face = lambda image: calls.append(image)  # type: ignore[method-assign]

    all_black = np.zeros((64, 64, 3), dtype=np.uint8)  # type: ignore[var-annotated]
    face.from_alignment(alignment, image=all_black)

    assert len(calls) == 1
    assert calls[0] is all_black


def test_from_alignment_skips_image_to_face_on_empty_array() -> None:
    """An empty array (``size == 0``) must still skip ``_image_to_face``."""
    from lib.align.detected_face import DetectedFace
    from lib.align.objects import FileAlignments

    face = DetectedFace.__new__(DetectedFace)
    DetectedFace.__init__(face)
    alignment = FileAlignments(
        x=0,
        y=0,
        w=32,
        h=32,
        landmarks_xy=np.zeros((68, 2), dtype="float32"),
    )

    calls: list[np.ndarray] = []
    face._image_to_face = lambda image: calls.append(image)  # type: ignore[method-assign]

    face.from_alignment(alignment, image=np.empty((0, 0, 3), dtype=np.uint8))

    assert calls == []
