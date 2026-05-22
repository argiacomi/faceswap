#!/usr/bin/env python3
"""Qt Manual Tool thumbnail regeneration bridge.

This module keeps thumbnail-generation behavior GUI-neutral and deliberately
avoids importing Qt or tkinter. It installs a narrow compatibility shim over
``tools.manual.session`` so the native Qt startup worker can honor
``--thumb-regen`` and missing thumbnail caches through its existing background
thread path.
"""

from __future__ import annotations

import inspect
import logging
import os
import typing as T

import numpy as np

from . import session as _session

logger = logging.getLogger(__name__)

ProgressCallback = T.Callable[[int, int, str], None]

_INSTALLED = False
_ORIGINAL_ALIGNMENTS_HANDLE = _session.ManualSession.alignments_handle
_ORIGINAL_HAS_THUMBNAILS = _session.ManualAlignmentsHandle.has_thumbnails


def install() -> None:
    """Install the Qt thumbnail-regeneration shim once per interpreter."""
    global _INSTALLED  # noqa: PLW0603 - module-level one-shot guard
    if _INSTALLED:
        return
    _session.ManualSession.alignments_handle = _session_alignments_handle  # type:ignore[method-assign]
    _session.ManualAlignmentsHandle.has_thumbnails = _handle_has_thumbnails  # type:ignore[method-assign]
    _session.ManualAlignmentsHandle.regenerate_thumbnails = _handle_regenerate_thumbnails  # type:ignore[attr-defined,method-assign]
    _INSTALLED = True


def _session_alignments_handle(self: _session.ManualSession) -> _session.ManualAlignmentsHandle:
    """Attach session context to the handle returned by the original resolver."""
    handle = _ORIGINAL_ALIGNMENTS_HANDLE(self)
    setattr(handle, "_qt_manual_session", self)
    return handle


def _handle_has_thumbnails(self: _session.ManualAlignmentsHandle) -> bool:
    """Return thumbnail state, generating missing/forced thumbnails when possible."""
    if not self.exists:
        return False
    if getattr(self, "_qt_manual_thumbs_generated", False) and _ORIGINAL_HAS_THUMBNAILS(self):
        return True
    manual_session = _session_from_handle(self)
    force = bool(manual_session and manual_session.thumb_regenerate)
    if not force and _ORIGINAL_HAS_THUMBNAILS(self):
        return True
    if manual_session is None:
        return _ORIGINAL_HAS_THUMBNAILS(self)
    generated = _regenerate_thumbnails(
        self,
        manual_session,
        progress=_progress_callback_from_stack(),
    )
    setattr(self, "_qt_manual_thumbs_generated", True)
    if generated:
        logger.info("Manual Tool generated thumbnails for %s frame(s)", generated)
    return _ORIGINAL_HAS_THUMBNAILS(self)


def _handle_regenerate_thumbnails(
    self: _session.ManualAlignmentsHandle,
    progress: ProgressCallback | None = None,
) -> int:
    """Public shim method for tests and future UI workers."""
    manual_session = _session_from_handle(self)
    if manual_session is None:
        raise ValueError("Manual thumbnail regeneration requires ManualSession context")
    generated = _regenerate_thumbnails(self, manual_session, progress=progress)
    setattr(self, "_qt_manual_thumbs_generated", True)
    return generated


def _session_from_handle(handle: _session.ManualAlignmentsHandle) -> _session.ManualSession | None:
    """Return the ManualSession attached by ``_session_alignments_handle``."""
    candidate = getattr(handle, "_qt_manual_session", None)
    return candidate if isinstance(candidate, _session.ManualSession) else None


def _progress_callback_from_stack() -> ProgressCallback | None:
    """Find a running Qt startup task on the call stack and bridge progress to it."""
    frame = inspect.currentframe()
    frame = None if frame is None else frame.f_back
    try:
        while frame is not None:
            task = frame.f_locals.get("self")
            if task is not None and hasattr(task, "progress") and hasattr(task, "STAGE_PERCENT"):
                return _progress_emitter(task, base_percent=66, span_percent=33)
            frame = frame.f_back
    finally:
        del frame
    return None


def _progress_emitter(task: T.Any, *, base_percent: int, span_percent: int) -> ProgressCallback:
    """Return a neutral-to-Qt progress bridge for thumbnail regeneration."""

    def _emit(done: int, total: int, message: str) -> None:
        denominator = max(1, total)
        percent = min(99, base_percent + round((done / denominator) * span_percent))
        stage = f"thumbs:{done}:{total}"
        task.STAGE_PERCENT[stage] = percent
        task.progress.emit(stage, message)

    return _emit


def _regenerate_thumbnails(
    handle: _session.ManualAlignmentsHandle,
    manual_session: _session.ManualSession,
    progress: ProgressCallback | None = None,
) -> int:
    """Regenerate cached face thumbnails and persist them to the alignments file.

    The operation mutates only ``face.thumb`` payloads. Existing thumb values
    are snapshotted and restored if any error occurs before save, so a failed
    regeneration does not corrupt the in-memory alignments object or its file.
    """
    if not handle.exists:
        return 0
    alignments = handle.open()
    targets = _frames_with_faces(alignments)
    if not targets:
        return 0
    previous = {
        (frame_name, face_index): face.thumb
        for frame_name in targets
        for face_index, face in enumerate(alignments.data[frame_name].faces)
    }
    try:
        generated = (
            _regenerate_video_thumbnails(handle, manual_session, targets, progress)
            if manual_session.is_video_input
            else _regenerate_image_folder_thumbnails(manual_session, alignments, targets, progress)
        )
    except Exception:
        for (frame_name, face_index), thumb in previous.items():
            alignments.data[frame_name].faces[face_index].thumb = thumb
        logger.exception("Manual Tool thumbnail regeneration failed")
        raise
    if generated:
        alignments.save()
    return generated


def _frames_with_faces(alignments: T.Any) -> tuple[str, ...]:
    """Return alignments frame keys that currently contain at least one face."""
    return tuple(frame_name for frame_name, entry in alignments.data.items() if entry.faces)


def _regenerate_image_folder_thumbnails(
    manual_session: _session.ManualSession,
    alignments: T.Any,
    targets: tuple[str, ...],
    progress: ProgressCallback | None,
) -> int:
    """Regenerate thumbnails for an image-folder source."""
    from lib.image import read_image

    frame_paths = {frame.name: frame.path for frame in manual_session.frame_list}
    generated = 0
    total = len(targets)
    for done, frame_name in enumerate(targets, start=1):
        frame_path = frame_paths.get(frame_name)
        if frame_path is None:
            logger.warning("Manual Tool thumbnail source frame not found: %s", frame_name)
            _emit_progress(progress, done, total, f"Skipped {frame_name}")
            continue
        image = read_image(frame_path, raise_error=True)
        for face_index, face in enumerate(alignments.data[frame_name].faces):
            alignments.thumbnails.add_thumbnail(
                frame_name,
                face_index,
                _thumbnail_from_face(face, image),
            )
        generated += 1
        _emit_progress(progress, done, total, f"Generated thumbnails for {frame_name}")
    return generated


def _regenerate_video_thumbnails(
    handle: _session.ManualAlignmentsHandle,
    manual_session: _session.ManualSession,
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
    loader = SingleFrameLoader(manual_session.frames, video_meta_data=meta_dict)
    try:
        frame_index_by_name = {
            manual_session.frame_name_for_index(index): index for index in range(loader.count)
        }
        generated = 0
        total = len(targets)
        alignments = handle.open()
        for done, frame_name in enumerate(targets, start=1):
            frame_index = frame_index_by_name.get(frame_name)
            if frame_index is None:
                logger.warning("Manual Tool thumbnail video frame not found: %s", frame_name)
                _emit_progress(progress, done, total, f"Skipped {frame_name}")
                continue
            _filename, image = loader.image_from_index(frame_index)
            for face_index, face in enumerate(alignments.data[frame_name].faces):
                alignments.thumbnails.add_thumbnail(
                    frame_name,
                    face_index,
                    _thumbnail_from_face(face, image),
                )
            generated += 1
            _emit_progress(progress, done, total, f"Generated thumbnails for {frame_name}")
        return generated
    finally:
        loader.close()


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


def _emit_progress(
    progress: ProgressCallback | None,
    done: int,
    total: int,
    message: str,
) -> None:
    """Emit optional deterministic progress without binding this module to Qt."""
    if progress is not None:
        progress(done, total, message)


__all__ = ("install",)
