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

from lib.utils import get_module_objects

if T.TYPE_CHECKING:  # pragma: no cover - typing only
    from tools.manual.session import EditableFace, ManualEditableAlignments


FilterMode = str
"""Type alias for filter-mode names — kept as ``str`` so editor-state
storage and dropdown labels stay identical to the Tk shell."""

FILTER_MODES: tuple[str, ...] = (
    "All Frames",
    "Has Face(s)",
    "No Faces",
    "Single Face",
    "Multiple Faces",
    "Misaligned Faces",
)
"""User-visible filter modes in legacy Tk cycle order."""

DEFAULT_FILTER_MODE = FILTER_MODES[0]
"""``"All Frames"`` — selected on first launch and after a reset."""

MISALIGNED_THRESHOLD_MIN = 5
MISALIGNED_THRESHOLD_MAX = 20
MISALIGNED_THRESHOLD_DEFAULT = 10
"""Tk-parity bounds for the misaligned-faces slider (``5..20`` raw, scaled
to ``threshold / 100`` when compared to ``AlignedFace.average_distance``)."""


def filtered_frame_indices(
    frame_indices: T.Sequence[int],
    face_count: T.Callable[[int], int],
    filter_mode: FilterMode,
    *,
    misaligned_predicate: T.Callable[[int], bool] | None = None,
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
    if filter_mode == "Multiple Faces":
        return tuple(i for i in frame_indices if face_count(i) > 1)
    if filter_mode == "Misaligned Faces":
        if misaligned_predicate is None:
            return ()
        return tuple(i for i in frame_indices if misaligned_predicate(i))
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


__all__ = get_module_objects(__name__)
