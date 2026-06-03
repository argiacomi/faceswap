#!/usr/bin/env python3
"""Tests for :mod:`tools.manual.frame_filter` (#107)."""

from __future__ import annotations

from typing import cast

import numpy as np
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
        "Two Faces",
        "Multiple Faces",
        "Misaligned Faces",
        "Neighbor Outliers",
        "Landmarks Outside Thumbnail",
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


def test_two_faces_matches_exactly_two() -> None:
    counts = _counts(0, 1, 2, 3, 1)
    indices = tuple(range(5))
    assert filtered_frame_indices(indices, _count_callable(counts), "Two Faces") == (2,)


def test_multiple_faces_excludes_zero_one_and_two() -> None:
    counts = _counts(0, 1, 2, 3, 1)
    indices = tuple(range(5))
    assert filtered_frame_indices(indices, _count_callable(counts), "Multiple Faces") == (3,)


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


# ---------------------------------------------------------------------------
# Neighbor-outlier and thumbnail-outside predicates
# ---------------------------------------------------------------------------


def _grid_landmarks(
    offset_x: float = 0.0, offset_y: float = 0.0
) -> tuple[tuple[float, float], ...]:
    """Return 68 spread-out source-pixel landmarks, optionally translated as a whole.

    The cloud spans ~84x96 px (diagonal ~128), so a uniform translation is a realistic
    whole-face position jump rather than a degenerate point.
    """
    return tuple(
        (20.0 + (i % 8) * 12.0 + offset_x, 20.0 + (i // 8) * 12.0 + offset_y) for i in range(68)
    )


def test_neighbor_outlier_flags_current_face_vs_previous_and_next() -> None:
    """A face that jumps position relative to its adjacent frames is flagged."""
    from tools.manual import frame_filter as _frame_filter

    model = ManualEditableAlignments()
    model.add_face(0, (20.0, 20.0, 110.0, 110.0), landmarks=_grid_landmarks())
    # Frame 1's face jumps ~40px (a third of its size) off the neighbours' position.
    model.add_face(1, (60.0, 20.0, 110.0, 110.0), landmarks=_grid_landmarks(offset_x=40.0))
    model.add_face(2, (20.0, 20.0, 110.0, 110.0), landmarks=_grid_landmarks())

    assert _frame_filter.frame_neighbor_landmark_outlier_from_provider(model.faces, 1, 10) is True
    assert _frame_filter.neighbor_outlier_predicate_for_model(model, 10)(1) is True
    # Stable frames whose neighbours interpolate their position are not flagged.
    assert _frame_filter.neighbor_outlier_predicate_for_model(model, 10)(0) is False


def test_neighbor_outlier_requires_adjacent_faces() -> None:
    """Edge frames without both neighbors are not treated as outliers."""
    from tools.manual import frame_filter as _frame_filter

    model = ManualEditableAlignments()
    model.add_face(0, (60.0, 20.0, 110.0, 110.0), landmarks=_grid_landmarks(offset_x=40.0))
    model.add_face(1, (20.0, 20.0, 110.0, 110.0), landmarks=_grid_landmarks())

    assert _frame_filter.frame_neighbor_landmark_outlier_from_provider(model.faces, 0, 10) is False


def test_landmarks_outside_thumbnail_predicate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A face is flagged when transformed thumbnail landmarks leave the 96px crop."""
    import numpy as np

    from tools.manual import frame_filter as _frame_filter

    face = EditableFace(
        face_index=0,
        bbox=(0.0, 0.0, 10.0, 10.0),
        landmarks=_grid_landmarks(),
    )
    monkeypatch.setattr(
        _frame_filter,
        "_thumbnail_landmarks_for_face",
        lambda face, *, size=96, centering="head": np.asarray(
            [(-1.0, 10.0), (50.0, 50.0)],
            dtype=np.float32,
        ),
    )

    assert _frame_filter.face_landmarks_outside_thumbnail(face) is True
    assert _frame_filter.frame_landmarks_outside_thumbnail((face,)) is True


def test_new_filter_modes_use_supplied_predicates() -> None:
    """The neutral filter dispatcher routes the new modes through their predicates."""
    counts = _counts(1, 1, 1)
    indices = (0, 1, 2)

    assert filtered_frame_indices(
        indices,
        _count_callable(counts),
        "Neighbor Outliers",
        neighbor_outlier_predicate=lambda idx: idx == 1,
    ) == (1,)
    assert filtered_frame_indices(
        indices,
        _count_callable(counts),
        "Landmarks Outside Thumbnail",
        thumbnail_outlier_predicate=lambda idx: idx == 2,
    ) == (2,)


# ---------------------------------------------------------------------------
# Real DetectedFace path: reuse the cached AlignedFace (fast) and flag outliers
# ---------------------------------------------------------------------------


def _stable_landmarks() -> np.ndarray:
    """Return a realistic, roughly frontal 68-point landmark array."""

    rng = np.random.default_rng(1234)
    return cast(np.ndarray, (rng.random((68, 2)).astype("float32") * 120.0) + 40.0)


def _detected_face(landmarks: np.ndarray) -> object:
    """Build a manual-tool DetectedFace with its aligned face loaded, as the tool does."""
    from lib.align.detected_face import DetectedFace

    face = DetectedFace(landmarks_xy=landmarks)
    face.load_aligned(None)
    return face


def test_thumbnail_landmarks_built_once_and_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """The face-centered transform is built once per face, then served from the memo."""
    from tools.manual import frame_filter as _frame_filter

    face = _detected_face(_stable_landmarks())
    first = _frame_filter._thumbnail_landmarks_for_face(face)
    assert first is not None and first.shape == (68, 2)

    # Any subsequent call for the same (unedited) face must hit the memo, not rebuild.
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("repeated calls must reuse the memoized landmarks, not rebuild")

    monkeypatch.setattr(_frame_filter, "_aligned_landmarks_for_face", _boom)
    second = _frame_filter._thumbnail_landmarks_for_face(face)
    assert second is first


def test_neighbor_outlier_flags_real_outlier_frame_only() -> None:
    """A single bad frame between two stable frames is flagged; stable frames are not."""
    import numpy as np

    from tools.manual import frame_filter as _frame_filter

    stable = _stable_landmarks()
    # A whole-face position jump: the previous pose-normalized metric scored this as 0.0
    # (translation invariant) and silently missed it; the raw-space metric catches it.
    outlier = np.array(stable, dtype="float32") + 50.0

    frames = {
        0: (_detected_face(stable),),
        1: (_detected_face(outlier),),
        2: (_detected_face(stable),),
    }
    provider = lambda idx: frames.get(idx, ())  # noqa: E731

    threshold_raw = 10  # shared Misaligned/Neighbor slider value -> 0.10 fraction of face size
    assert _frame_filter.frame_neighbor_landmark_outlier_from_provider(provider, 1, threshold_raw)
    # Edge frames lack both neighbours, so they are never treated as outliers.
    assert not _frame_filter.frame_neighbor_landmark_outlier_from_provider(
        provider, 0, threshold_raw
    )
    assert not _frame_filter.frame_neighbor_landmark_outlier_from_provider(
        provider, 2, threshold_raw
    )


def test_thumbnail_landmarks_match_displayed_face_crop() -> None:
    """The thumbnail check uses the same face-centered crop the viewer displays, not head."""
    import numpy as np

    from lib.align.aligned_face import AlignedFace
    from tools.manual.frame_filter import (
        THUMBNAIL_OUTSIDE_CENTERING,
        THUMBNAIL_OUTSIDE_SIZE,
        _thumbnail_landmarks_for_face,
    )

    assert THUMBNAIL_OUTSIDE_CENTERING == "face"
    lms = _stable_landmarks()
    points = _thumbnail_landmarks_for_face(_detected_face(lms))
    assert points is not None

    expected = np.asarray(
        AlignedFace(lms, centering="face", size=THUMBNAIL_OUTSIDE_SIZE).landmarks,
        dtype="float32",
    )
    assert np.allclose(points, expected)


def _realistic_face(jaw_drop: float = 0.0) -> np.ndarray:
    """Return a realistic 68-point face (mean inner-51 + synthesized jaw), optionally dropping
    the jaw line to simulate a blown-out detection whose landmarks leave the displayed crop."""
    from lib.align.constants import MEAN_FACE, LandmarkType

    inner = np.asarray(MEAN_FACE[LandmarkType.LM_2D_51], dtype="float32")
    inner = (inner - inner.mean(0)) * 150.0 + np.array([110.0, 110.0], dtype="float32")
    cx, bot = inner[:, 0].mean(), inner[:, 1].max()
    jaw = [
        (
            cx - np.cos((i / 16) * np.pi) * 150.0 * 0.62,
            (bot + 10.0) + np.sin((i / 16) * np.pi) * 150.0 * 0.35,
        )
        for i in range(17)
    ]
    face: np.ndarray = np.vstack([np.array(jaw, dtype="float32"), inner]).astype("float32")
    if jaw_drop:
        face[0:17, 1] += jaw_drop
        face[8, 1] += jaw_drop * 0.5
    return face


def test_outside_thumbnail_flags_blown_out_jaw_not_clean_face() -> None:
    """A face whose jaw spills past the displayed crop is flagged; a clean face is not."""
    from tools.manual import frame_filter as _frame_filter

    clean = _detected_face(_realistic_face())
    blown_jaw = _detected_face(_realistic_face(jaw_drop=80.0))

    assert _frame_filter.face_landmarks_outside_thumbnail(clean) is False
    assert _frame_filter.face_landmarks_outside_thumbnail(blown_jaw) is True
    assert _frame_filter.frame_landmarks_outside_thumbnail((clean, blown_jaw)) is True
