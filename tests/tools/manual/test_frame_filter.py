#!/usr/bin/env python3
"""Tests for :mod:`tools.manual.frame_filter` (#107)."""

from __future__ import annotations

import pytest

from tools.manual.frame_filter import (
    DEFAULT_FILTER_MODE,
    FILTER_MODES,
    MISALIGNED_THRESHOLD_DEFAULT,
    MISALIGNED_THRESHOLD_MAX,
    MISALIGNED_THRESHOLD_MIN,
    filtered_frame_indices,
    frame_misaligned,
    misaligned_predicate_for_model,
)
from tools.manual.session import EditableFace, ManualEditableAlignments

# ---------------------------------------------------------------------------
# Filter mode constants
# ---------------------------------------------------------------------------


def test_filter_modes_match_legacy_cycle_order() -> None:
    """``FILTER_MODES`` is the Tk Manual Tool's cycle order, exactly."""
    assert FILTER_MODES == (
        "All Frames",
        "Has Face(s)",
        "No Faces",
        "Single Face",
        "Multiple Faces",
        "Misaligned Faces",
    )


def test_default_filter_mode_is_all_frames() -> None:
    """The first filter at launch must be ``All Frames``."""
    assert DEFAULT_FILTER_MODE == "All Frames"


def test_misaligned_threshold_bounds_match_legacy_slider() -> None:
    """The Tk threshold slider runs from 5..20 with a default near 10."""
    assert MISALIGNED_THRESHOLD_MIN == 5
    assert MISALIGNED_THRESHOLD_MAX == 20
    assert MISALIGNED_THRESHOLD_MIN <= MISALIGNED_THRESHOLD_DEFAULT <= MISALIGNED_THRESHOLD_MAX


# ---------------------------------------------------------------------------
# Predicates — non-misaligned modes
# ---------------------------------------------------------------------------


def _counts(*counts: int) -> dict[int, int]:
    """Return a frame_index → face_count map for predictable face_count()."""
    return {index: count for index, count in enumerate(counts)}


def _count_callable(counts: dict[int, int]):
    """Return a ``face_count(frame_index)`` callable that defaults to 0."""

    def _face_count(index: int) -> int:
        return int(counts.get(index, 0))

    return _face_count


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ([0, 0, 0], (0, 1, 2)),
        ([1, 0, 2, 0], (0, 1, 2, 3)),
        ([], ()),
    ],
)
def test_all_frames_returns_every_frame(counts: list[int], expected: tuple[int, ...]) -> None:
    """``All Frames`` returns ``frame_indices`` verbatim."""
    indices = tuple(range(len(counts)))
    result = filtered_frame_indices(indices, _count_callable(_counts(*counts)), "All Frames")
    assert result == expected


def test_has_faces_includes_only_nonzero_counts() -> None:
    counts = _counts(0, 1, 0, 2, 1)
    indices = tuple(range(5))
    assert filtered_frame_indices(indices, _count_callable(counts), "Has Face(s)") == (1, 3, 4)


def test_no_faces_includes_only_zero_counts() -> None:
    counts = _counts(0, 1, 0, 2, 1)
    indices = tuple(range(5))
    assert filtered_frame_indices(indices, _count_callable(counts), "No Faces") == (0, 2)


def test_single_face_includes_only_one_count() -> None:
    counts = _counts(0, 1, 0, 2, 1)
    indices = tuple(range(5))
    assert filtered_frame_indices(indices, _count_callable(counts), "Single Face") == (1, 4)


def test_multiple_faces_excludes_zero_and_one() -> None:
    counts = _counts(0, 1, 2, 3, 1)
    indices = tuple(range(5))
    assert filtered_frame_indices(indices, _count_callable(counts), "Multiple Faces") == (2, 3)


def test_unknown_filter_mode_falls_back_to_all_frames() -> None:
    """A stale editor-state filter must not crash navigation."""
    counts = _counts(0, 1)
    assert filtered_frame_indices((0, 1), _count_callable(counts), "Bogus Mode") == (0, 1)


def test_filtered_frame_indices_preserves_input_order() -> None:
    """The output order matches the input even when frame_indices are sparse."""
    counts = {10: 1, 5: 0, 12: 2}
    assert filtered_frame_indices((10, 5, 12), _count_callable(counts), "Has Face(s)") == (10, 12)


# ---------------------------------------------------------------------------
# Misaligned predicate
# ---------------------------------------------------------------------------


def test_misaligned_mode_requires_predicate() -> None:
    """No misaligned predicate means an empty result — never a crash."""
    counts = _counts(1, 1)
    assert filtered_frame_indices((0, 1), _count_callable(counts), "Misaligned Faces") == ()


def test_misaligned_mode_uses_predicate_for_filter() -> None:
    """The predicate decides which frames count as misaligned."""
    counts = _counts(1, 1, 1)
    misaligned = {1}

    def _predicate(index: int) -> bool:
        return index in misaligned

    result = filtered_frame_indices(
        (0, 1, 2),
        _count_callable(counts),
        "Misaligned Faces",
        misaligned_predicate=_predicate,
    )
    assert result == (1,)


def test_frame_misaligned_empty_frames_is_false() -> None:
    """No faces means not misaligned."""
    assert frame_misaligned([], MISALIGNED_THRESHOLD_DEFAULT) is False


def test_frame_misaligned_faces_without_landmarks_treated_as_aligned() -> None:
    """A freshly-added face with no landmarks isn't flagged as misaligned."""
    face = EditableFace(face_index=0, bbox=(0.0, 0.0, 10.0, 10.0), landmarks=())
    assert frame_misaligned([face], MISALIGNED_THRESHOLD_DEFAULT) is False


def test_frame_misaligned_threshold_above_score_returns_false() -> None:
    """A high threshold suppresses the flag for typical-looking landmarks."""
    # 68 canonical-ish points clustered near the centre.
    landmarks = tuple((50.0 + i * 0.1, 50.0 + i * 0.1) for i in range(68))
    face = EditableFace(face_index=0, bbox=(0.0, 0.0, 100.0, 100.0), landmarks=landmarks)
    # 20 / 100 = 0.2 — large enough that a well-formed landmark cloud doesn't
    # cross the threshold.  This locks in that the Tk metric isn't trivially
    # always > threshold for realistic inputs.
    assert frame_misaligned([face], MISALIGNED_THRESHOLD_MAX) is False


def test_misaligned_predicate_for_model_uses_editable_faces() -> None:
    """``misaligned_predicate_for_model`` consults the live editable model."""
    model = ManualEditableAlignments()
    model.add_face(
        0,
        (0.0, 0.0, 50.0, 50.0),
        landmarks=tuple((float(i), float(i)) for i in range(68)),
    )
    predicate = misaligned_predicate_for_model(model, MISALIGNED_THRESHOLD_DEFAULT)
    # The result depends on the actual canonical mean face — but at minimum
    # the predicate is callable and returns a bool for the frame indices it
    # is asked about, including frames with no faces.
    assert isinstance(predicate(0), bool)
    assert predicate(99) is False  # frame with no faces
