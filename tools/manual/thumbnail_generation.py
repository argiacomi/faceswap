#!/usr/bin/env python3
"""GUI-neutral thumbnail regeneration for the Manual Tool.

Encapsulates the work of decoding source frames (image folder or video),
rebuilding the face thumbnails stored in the alignments file, and persisting
the results.  This module is intentionally free of any Qt or tkinter imports
so both the legacy Tk shell and the native Qt shell can drive it through the
same API.

The public entry point is :func:`regenerate_thumbnails`, called by
:meth:`tools.manual.session.ManualAlignmentsHandle.regenerate_thumbnails`.
Progress is reported through an optional ``ProgressCallback`` — callers pass
their own emitter (the Qt startup worker bridges this to a Qt signal) so
this module never needs to know what UI it is feeding.
"""

from __future__ import annotations

import logging
import typing as T

import numpy as np

if T.TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from tools.manual.session import ManualAlignmentsHandle, ManualSession

logger = logging.getLogger(__name__)

ProgressCallback = T.Callable[[int, int, str], None]
"""``(done, total, message)`` — done/total are 1-based and total is constant."""


def regenerate_thumbnails(
    handle: ManualAlignmentsHandle,
    session: ManualSession,
    *,
    progress: ProgressCallback | None = None,
) -> int:
    """Rebuild cached face thumbnails and persist the result.

    The operation mutates only ``face.thumb`` payloads.  Existing payloads are
    snapshotted before any modification and restored if a failure occurs
    before :meth:`Alignments.save`, so a partial run cannot corrupt the
    alignments file or its in-memory object.

    Returns the number of frames whose thumbnails were regenerated.  ``0`` is
    a valid result (e.g. when no faces are stored yet, or every source frame
    is missing).
    """
    if not handle.exists:
        return 0
    alignments = handle.open()
    targets = _frames_with_faces(alignments)
    if not targets:
        return 0
    snapshot = {
        (frame_name, face_index): face.thumb
        for frame_name in targets
        for face_index, face in enumerate(alignments.data[frame_name].faces)
    }
    try:
        generated = (
            _regenerate_video(handle, session, targets, progress)
            if session.is_video_input
            else _regenerate_image_folder(session, alignments, targets, progress)
        )
    except Exception:
        for (frame_name, face_index), thumb in snapshot.items():
            alignments.data[frame_name].faces[face_index].thumb = thumb
        logger.exception("Manual Tool thumbnail regeneration failed")
        raise
    if generated:
        alignments.save()
    return generated


def _frames_with_faces(alignments: T.Any) -> tuple[str, ...]:
    """Return alignments frame keys that currently contain at least one face."""
    return tuple(frame_name for frame_name, entry in alignments.data.items() if entry.faces)


def _regenerate_image_folder(
    session: ManualSession,
    alignments: T.Any,
    targets: tuple[str, ...],
    progress: ProgressCallback | None,
) -> int:
    """Regenerate thumbnails for an image-folder source."""
    from lib.image import read_image

    frame_paths = {frame.name: frame.path for frame in session.frame_list}
    generated = 0
    total = len(targets)
    for done, frame_name in enumerate(targets, start=1):
        frame_path = frame_paths.get(frame_name)
        if frame_path is None:
            logger.warning("Manual Tool thumbnail source frame not found: %s", frame_name)
            _emit(progress, done, total, f"Skipped {frame_name}")
            continue
        image = read_image(frame_path, raise_error=True)
        _attach_thumbnails(alignments, frame_name, image)
        generated += 1
        _emit(progress, done, total, f"Generated thumbnails for {frame_name}")
    return generated


def _regenerate_video(
    handle: ManualAlignmentsHandle,
    session: ManualSession,
    targets: tuple[str, ...],
    progress: ProgressCallback | None,
) -> int:
    """Regenerate thumbnails for a video source."""
    from lib.image import SingleFrameLoader

    metadata = handle.video_metadata()
    meta_dict = (
        None
        if metadata is None or not metadata.is_valid
        else {"pts_time": list(metadata.pts_time), "keyframes": list(metadata.keyframes)}
    )
    loader = SingleFrameLoader(session.frames, video_meta_data=meta_dict)
    try:
        frame_index_by_name = {
            session.frame_name_for_index(index): index for index in range(loader.count)
        }
        alignments = handle.open()
        total = len(targets)
        generated = 0
        for done, frame_name in enumerate(targets, start=1):
            frame_index = frame_index_by_name.get(frame_name)
            if frame_index is None:
                logger.warning("Manual Tool thumbnail video frame not found: %s", frame_name)
                _emit(progress, done, total, f"Skipped {frame_name}")
                continue
            _filename, image = loader.image_from_index(frame_index)
            _attach_thumbnails(alignments, frame_name, image)
            generated += 1
            _emit(progress, done, total, f"Generated thumbnails for {frame_name}")
        return generated
    finally:
        loader.close()


def _attach_thumbnails(alignments: T.Any, frame_name: str, image: np.ndarray) -> None:
    """Rebuild and store the thumbnail payload for every face on one frame."""
    for face_index, face in enumerate(alignments.data[frame_name].faces):
        alignments.thumbnails.add_thumbnail(
            frame_name,
            face_index,
            _thumbnail_from_face(face, image),
        )


def _thumbnail_from_face(face: T.Any, image: np.ndarray) -> np.ndarray:
    """Build one encoded thumbnail from a FileAlignments-like face object."""
    from lib.align import AlignedFace
    from lib.image import generate_thumbnail

    landmarks = np.asarray(getattr(face, "landmarks_xy", np.zeros((0, 2))), dtype=np.float32)
    if landmarks.size:
        aligned = AlignedFace(landmarks, image=image, centering="head", size=96)
        if aligned.face is not None:
            return generate_thumbnail(aligned.face, size=96)
    crop = _bbox_crop(face, image)
    return generate_thumbnail(crop, size=96)


def _bbox_crop(face: T.Any, image: np.ndarray) -> np.ndarray:
    """Return a safe bbox crop fallback for faces without landmarks."""
    height, width = image.shape[:2]
    x = max(0, int(round(getattr(face, "x", 0))))
    y = max(0, int(round(getattr(face, "y", 0))))
    w = max(1, int(round(getattr(face, "w", width))))
    h = max(1, int(round(getattr(face, "h", height))))
    x2 = min(width, x + w)
    y2 = min(height, y + h)
    if x >= x2 or y >= y2:
        return image
    return image[y:y2, x:x2]


def _emit(progress: ProgressCallback | None, done: int, total: int, message: str) -> None:
    """Emit optional progress without binding this module to any UI framework."""
    if progress is not None:
        progress(done, total, message)


__all__ = ("ProgressCallback", "regenerate_thumbnails")
