#!/usr/bin/env python3
"""GUI-neutral frame filtering for the Manual Tool (#107).

The legacy Tk Manual Tool exposes six filter modes that restrict the
navigation slider + face viewer to a subset of frames:

* ``All Frames``         — every known source frame.
* ``Has Face(s)``        — frames with at least one face.
* ``No Faces``           — frames with zero faces.
* ``Single Face``        — frames with exactly one face.
* ``Multiple Faces``     — frames with more than one face.
* ``Misaligned Faces``   — frames containing at least one face whose mean
  normalized-landmark distance from the canonical mean face exceeds a
  user-adjustable threshold (Tk's threshold is a 5..20 slider that is
  scaled to ``threshold / 100`` on the way into the comparison).

This module is intentionally Qt-free and tkinter-free so the predicates can
be exercised by unit tests without instantiating any UI. The host
(``ManualToolWindow``) drives the filtered frame list off this module and
re-runs ``filtered_frame_indices`` whenever the editable model or filter
state changes.
"""

from __future__ import annotations

import typing as T

import numpy as np
import numpy.typing as npt

from lib.utils import get_module_objects

if T.TYPE_CHECKING:  # pragma: no cover - typing only
    from tools.manual.session import EditableFace, ManualEditableAlignments

LandmarkArray: T.TypeAlias = npt.NDArray[np.float32]
CenteringType: T.TypeAlias = T.Literal["face", "head", "legacy"]
LandmarkCache: T.TypeAlias = dict[tuple[int, int], "LandmarkArray | None"]
"""Per-pass memo of normalized landmarks keyed by ``(frame_index, face_index)``.

The neighbor-outlier scan revisits each frame's landmarks up to three times (as a
frame's own faces, then as the previous/next neighbour of adjacent frames). Sharing
one cache across a full ``filtered_frame_indices`` pass turns that ~3N ``AlignedFace``
builds back into ~N."""

FilterMode = str
"""Type alias for filter-mode names — kept as ``str`` so editor-state
storage and dropdown labels stay identical to the Tk shell."""

FILTER_MODES: tuple[str, ...] = (
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
"""User-visible filter modes in legacy Tk cycle order."""

DEFAULT_FILTER_MODE = FILTER_MODES[0]
"""``"All Frames"`` — selected on first launch and after a reset."""

MISALIGNED_THRESHOLD_MIN = 5
MISALIGNED_THRESHOLD_MAX = 20
MISALIGNED_THRESHOLD_DEFAULT = 10
"""Tk-parity bounds for the misaligned-faces slider (``5..20`` raw, scaled
to ``threshold / 100`` when compared to ``AlignedFace.average_distance``)."""

THUMBNAIL_OUTSIDE_SIZE = 96
"""Saved manual-tool face thumbnails are generated from a 96px head-centered crop."""

THUMBNAIL_OUTSIDE_CENTERING: CenteringType = "head"
"""Centering used by the legacy thumbnail refresh path."""


def filtered_frame_indices(
    frame_indices: T.Sequence[int],
    face_count: T.Callable[[int], int],
    filter_mode: FilterMode,
    *,
    misaligned_predicate: T.Callable[[int], bool] | None = None,
    neighbor_outlier_predicate: T.Callable[[int], bool] | None = None,
    thumbnail_outlier_predicate: T.Callable[[int], bool] | None = None,
) -> tuple[int, ...]:
    """Return the subset of ``frame_indices`` matching ``filter_mode``.

    ``face_count(frame_index) -> int`` is required for every count-based
    mode.  ``misaligned_predicate(frame_index) -> bool`` is required only
    when ``filter_mode == "Misaligned Faces"`` — it returns ``True`` when
    *any* face on the frame's editable list exceeds the configured
    threshold.  Unknown filter modes fall back to ``"All Frames"`` so a
    stale editor-state value never produces a hard failure.

    The returned sequence preserves the order of ``frame_indices``.
    """
    if filter_mode == "All Frames" or filter_mode not in FILTER_MODES:
        return tuple(frame_indices)
    if filter_mode == "Has Face(s)":
        return tuple(i for i in frame_indices if face_count(i) > 0)
    if filter_mode == "No Faces":
        return tuple(i for i in frame_indices if face_count(i) == 0)
    if filter_mode == "Single Face":
        return tuple(i for i in frame_indices if face_count(i) == 1)
    if filter_mode == "Two Faces":
        return tuple(i for i in frame_indices if face_count(i) == 2)
    if filter_mode == "Multiple Faces":
        return tuple(i for i in frame_indices if face_count(i) > 2)
    if filter_mode == "Misaligned Faces":
        if misaligned_predicate is None:
            return ()
        return tuple(i for i in frame_indices if misaligned_predicate(i))
    if filter_mode == "Neighbor Outliers":
        if neighbor_outlier_predicate is None:
            return ()
        return tuple(i for i in frame_indices if neighbor_outlier_predicate(i))
    if filter_mode == "Landmarks Outside Thumbnail":
        if thumbnail_outlier_predicate is None:
            return ()
        return tuple(i for i in frame_indices if thumbnail_outlier_predicate(i))
    return tuple(frame_indices)  # pragma: no cover - exhausted above


def frame_misaligned(
    faces: T.Sequence[EditableFace],
    threshold_raw: int,
) -> bool:
    """Return whether any face on the frame exceeds the misaligned threshold.

    Mirrors ``tools.manual.detected_faces.Filter._frame_meets_criteria``'s
    Misaligned-Faces branch:

    * The Tk shell scales the raw slider value ``5..20`` by ``/100`` and
      compares against ``face.aligned.average_distance``.
    * Faces without landmarks (e.g. freshly-added boxes that haven't
      been run through an aligner yet) cannot have a sensible average
      distance — Tk treats those as "not misaligned" so the slider
      doesn't immediately pin a freshly-added face.
    """
    if not faces:
        return False
    threshold = float(threshold_raw) / 100.0
    for face in faces:
        if not face.landmarks:
            continue
        distance = face_average_distance(face)
        if distance is None:
            continue
        if distance > threshold:
            return True
    return False


def face_average_distance(face: EditableFace) -> float | None:
    """Return the Tk-parity average distance for ``face`` or ``None``.

    The 68-point normalized-landmark mean distance from ``MEAN_FACE`` is the
    same metric Tk uses for the Misaligned filter.  We construct a
    lightweight ``AlignedFace`` (no image) so we don't have to re-derive the
    affine + normalization in this module — the alignment library is the
    canonical source of truth and we want to track its changes if the
    metric ever evolves.

    Returns ``None`` when:

    * The face has no landmarks (alignment can't be computed).
    * The landmark count isn't one of the supported shapes (68 / 98).
    """
    if not face.landmarks:
        return None
    landmarks = np.asarray(face.landmarks, dtype=np.float32)
    if landmarks.shape[0] < 17:  # core landmarks indices 17..67 are required
        return None
    try:
        from lib.align.aligned_face import AlignedFace
    except ImportError:  # pragma: no cover - alignment lib always present
        return None
    try:
        aligned = AlignedFace(landmarks)
    except Exception:  # noqa: BLE001 - alignment lib may reject odd shapes
        return None
    try:
        return float(aligned.average_distance)
    except Exception:  # noqa: BLE001 - tolerate downstream failure
        return None


def _landmarks_array(face: T.Any) -> LandmarkArray | None:
    """Return a ``(N, 2)`` landmark array from either neutral or legacy face objects."""
    raw = getattr(face, "landmarks", None)
    if raw is not None:
        try:
            if len(raw) == 0:
                raw = None
        except TypeError:
            pass
    if raw is None and getattr(face, "has_landmarks", True):
        # ``DetectedFace.landmarks_xy`` asserts instead of returning ``None`` when no
        # landmarks are present, so honour ``has_landmarks`` and swallow the assertion
        # to keep this helper safe for landmark-less legacy faces.
        try:
            raw = getattr(face, "landmarks_xy", None)
        except AssertionError:
            raw = None
    if raw is None:
        return None

    try:
        landmarks = T.cast(LandmarkArray, np.asarray(raw, dtype=np.float32))
    except (TypeError, ValueError):
        return None

    if landmarks.ndim != 2 or landmarks.shape[1] != 2 or landmarks.shape[0] < 17:
        return None
    if not np.all(np.isfinite(landmarks)):
        return None
    return landmarks


def _aligned_landmarks_for_face(
    face: T.Any,
    *,
    centering: CenteringType,
    size: int,
) -> LandmarkArray | None:
    """Return landmarks transformed into an ``AlignedFace`` coordinate system."""
    landmarks = _landmarks_array(face)
    if landmarks is None:
        return None

    try:
        from lib.align.aligned_face import AlignedFace
    except ImportError:  # pragma: no cover
        return None

    try:
        aligned = AlignedFace(landmarks, centering=centering, size=int(size))
        points = T.cast(LandmarkArray, np.asarray(aligned.landmarks, dtype=np.float32))
    except Exception:  # noqa: BLE001
        return None

    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        return None
    return points


def _normalized_landmarks_for_face(face: T.Any, *, size: int = 100) -> LandmarkArray | None:
    """Return face-centered normalized landmarks in roughly 0..1 coordinates."""
    points = _aligned_landmarks_for_face(face, centering="face", size=size)
    if points is None:
        return None
    return T.cast(LandmarkArray, points / float(size))


def _thumbnail_landmarks_for_face(
    face: T.Any,
    *,
    size: int = THUMBNAIL_OUTSIDE_SIZE,
    centering: CenteringType = THUMBNAIL_OUTSIDE_CENTERING,
) -> LandmarkArray | None:
    """Return landmarks in the same crop coordinate space used for saved thumbnails."""
    return _aligned_landmarks_for_face(face, centering=centering, size=size)


def _safe_faces_for_frame(
    faces_for_frame: T.Callable[[int], T.Sequence[T.Any]],
    frame_index: int,
) -> T.Sequence[T.Any]:
    try:
        return faces_for_frame(frame_index)
    except (IndexError, KeyError):
        return ()


def _face_by_index(faces: T.Sequence[T.Any], face_index: int) -> T.Any | None:
    if 0 <= face_index < len(faces):
        return faces[face_index]
    for face in faces:
        if int(getattr(face, "face_index", -1)) == face_index:
            return face
    return None


def _cached_normalized_landmarks(
    faces_for_frame: T.Callable[[int], T.Sequence[T.Any]],
    frame_index: int,
    face_index: int,
    cache: LandmarkCache,
) -> LandmarkArray | None:
    """Return memoized normalized landmarks for ``(frame_index, face_index)``."""
    key = (int(frame_index), int(face_index))
    if key in cache:
        return cache[key]
    face = _face_by_index(_safe_faces_for_frame(faces_for_frame, frame_index), int(face_index))
    points = None if face is None else _normalized_landmarks_for_face(face)
    cache[key] = points
    return points


def face_neighbor_landmark_distance_from_provider(
    faces_for_frame: T.Callable[[int], T.Sequence[T.Any]],
    frame_index: int,
    face_index: int,
    *,
    window: int = 1,
    require_both_sides: bool = True,
    cache: LandmarkCache | None = None,
) -> float | None:
    """Return current-face distance from the average of previous/next neighboring faces.

    The initial implementation matches faces by ``face_index`` because this is the same
    ordering exposed by the Manual Tool's editable model and legacy face lists.  Frames at
    the edge, missing landmarks, or missing adjacent matching faces return ``None``.

    Pass a shared ``cache`` across a full-frame scan to avoid rebuilding the same
    ``AlignedFace`` once per neighbouring frame.
    """
    if cache is None:
        cache = {}
    current = _cached_normalized_landmarks(faces_for_frame, frame_index, int(face_index), cache)
    if current is None:
        return None

    neighbors: list[np.ndarray] = []
    max_window = max(1, int(window))
    for offset in range(1, max_window + 1):
        points = _cached_normalized_landmarks(
            faces_for_frame, frame_index - offset, int(face_index), cache
        )
        if points is not None and points.shape == current.shape:
            neighbors.append(points)
            break
    for offset in range(1, max_window + 1):
        points = _cached_normalized_landmarks(
            faces_for_frame, frame_index + offset, int(face_index), cache
        )
        if points is not None and points.shape == current.shape:
            neighbors.append(points)
            break

    if require_both_sides and len(neighbors) < 2:
        return None
    if not neighbors:
        return None
    reference = np.mean(np.stack(neighbors, axis=0), axis=0)
    deltas = current - reference
    return float(np.sqrt(np.sum(deltas * deltas, axis=1)).mean())


def frame_neighbor_landmark_outlier_from_provider(
    faces_for_frame: T.Callable[[int], T.Sequence[T.Any]],
    frame_index: int,
    threshold_raw: int,
    *,
    window: int = 1,
    cache: LandmarkCache | None = None,
) -> bool:
    """Return whether any face in ``frame_index`` is an outlier vs adjacent frames.

    Pass a shared ``cache`` across a full-frame scan to reuse normalized landmarks
    computed for neighbouring frames.
    """
    threshold = float(threshold_raw) / 100.0
    for face_index, _face in enumerate(_safe_faces_for_frame(faces_for_frame, frame_index)):
        distance = face_neighbor_landmark_distance_from_provider(
            faces_for_frame,
            frame_index,
            face_index,
            window=window,
            cache=cache,
        )
        if distance is not None and distance > threshold:
            return True
    return False


def face_landmarks_outside_thumbnail(
    face: T.Any,
    *,
    size: int = THUMBNAIL_OUTSIDE_SIZE,
    margin: float = 0.0,
) -> bool:
    """Return whether any landmark lands outside the saved-thumbnail crop."""
    points = _thumbnail_landmarks_for_face(face, size=size)
    if points is None:
        return False
    low = -float(margin)
    high = float(size) + float(margin)
    return bool(
        np.any(points[:, 0] < low)
        or np.any(points[:, 1] < low)
        or np.any(points[:, 0] >= high)
        or np.any(points[:, 1] >= high)
    )


def frame_landmarks_outside_thumbnail(faces: T.Sequence[T.Any]) -> bool:
    """Return whether any face on the frame has landmarks outside its thumbnail."""
    return any(face_landmarks_outside_thumbnail(face) for face in faces)


def misaligned_predicate_for_model(
    model: ManualEditableAlignments,
    threshold_raw: int,
) -> T.Callable[[int], bool]:
    """Return a ``frame_index -> bool`` callable for the Misaligned filter.

    Convenience wrapper used by :func:`filtered_frame_indices` callers that
    have a :class:`ManualEditableAlignments` instance available; tests can
    inject their own callable instead.
    """

    def _predicate(frame_index: int) -> bool:
        return frame_misaligned(model.faces(frame_index), threshold_raw)

    return _predicate


def neighbor_outlier_predicate_for_model(
    model: ManualEditableAlignments,
    threshold_raw: int,
) -> T.Callable[[int], bool]:
    """Return a ``frame_index -> bool`` predicate for adjacent-frame outliers."""

    def _predicate(frame_index: int) -> bool:
        return frame_neighbor_landmark_outlier_from_provider(
            model.faces, frame_index, threshold_raw
        )

    return _predicate


def thumbnail_outlier_predicate_for_model(
    model: ManualEditableAlignments,
) -> T.Callable[[int], bool]:
    """Return a ``frame_index -> bool`` predicate for thumbnail-outside landmarks."""

    def _predicate(frame_index: int) -> bool:
        return frame_landmarks_outside_thumbnail(model.faces(frame_index))

    return _predicate


__all__ = get_module_objects(__name__)
