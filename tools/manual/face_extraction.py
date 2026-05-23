#!/usr/bin/env python3
"""GUI-neutral Extract Faces workflow for the Manual Tool.

Reads source frames (image folder or video) and writes one aligned-face PNG
per detected face into ``output_folder``, embedding the legacy Tk alignments
+ source metadata in the PNG header so downstream Faceswap tools can re-open
the extracted faces with the same provenance.

This module deliberately has no Qt or tkinter imports — the Qt worker
constructs an :class:`FaceExtractionRequest`, hands it to
:func:`extract_faces`, and bridges progress / cancellation through its own
callbacks.

Returns an :class:`ExtractFacesResult` summarizing the run.  The caller is
responsible for surfacing the totals (and the optional ``errors`` list) in
its UI.
"""

from __future__ import annotations

import logging
import os
import typing as T
from dataclasses import dataclass, field

if T.TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from tools.manual.session import ManualAlignmentsHandle, ManualSession

logger = logging.getLogger(__name__)

ProgressCallback = T.Callable[[int, int, str], None]
"""``(done, total, message)`` — ``done``/``total`` are 1-based frame counts."""

CancelPredicate = T.Callable[[], bool]
"""Optional probe the caller polls to abort extraction between frames."""

EXTRACT_FACE_SIZE = 512
"""Aligned face size (px) — matches the legacy Tk Manual Tool default."""


@dataclass(frozen=True)
class FaceExtractionRequest:
    """Inputs for one :func:`extract_faces` call."""

    handle: ManualAlignmentsHandle
    session: ManualSession
    output_folder: str


@dataclass
class ExtractFacesResult:
    """Outputs from one :func:`extract_faces` call."""

    frames_processed: int = 0
    """Number of source frames the worker read."""
    faces_written: int = 0
    """Number of aligned-face PNGs written to disk."""
    skipped_frames: int = 0
    """Source frames the worker could not read (logged + reported)."""
    cancelled: bool = False
    """``True`` when the cancel predicate returned ``True`` mid-run."""
    errors: list[str] = field(default_factory=list)
    """User-facing error strings for any per-frame failures."""


def extract_faces(
    request: FaceExtractionRequest,
    *,
    progress: ProgressCallback | None = None,
    is_cancelled: CancelPredicate | None = None,
) -> ExtractFacesResult:
    """Run Tk-parity Extract Faces against ``request``.

    Writes one PNG per face in the order frames are encountered.  Progress is
    reported per source frame (a frame with two faces still counts as one
    progress tick).  Returns even when ``output_folder`` is partially
    populated — the caller decides whether to surface this as a partial
    success or a failure depending on the cancellation / error state on the
    returned :class:`ExtractFacesResult`.
    """
    result = ExtractFacesResult()
    if not request.handle.exists:
        return result
    alignments = request.handle.open()
    targets = tuple(
        (frame_name, entry) for frame_name, entry in alignments.data.items() if entry.faces
    )
    if not targets:
        return result
    output_folder = _ensure_output_folder(request.output_folder)
    saver = _build_saver(output_folder)
    try:
        if request.session.is_video_input:
            _extract_video(request, alignments, targets, saver, progress, is_cancelled, result)
        else:
            _extract_image_folder(
                request, alignments, targets, saver, progress, is_cancelled, result
            )
    finally:
        saver.close()
    return result


def _ensure_output_folder(path: str) -> str:
    """Create ``path`` if needed and return the canonical folder string."""
    from lib.utils import get_folder

    return get_folder(path)


def _build_saver(output_folder: str):  # type:ignore[no-untyped-def]
    """Return a Tk-parity ``ImagesSaver`` for writing PNGs as raw bytes."""
    from lib.image import ImagesSaver

    return ImagesSaver(output_folder, as_bytes=True)


def _extract_image_folder(
    request: FaceExtractionRequest,
    alignments: T.Any,
    targets: tuple[tuple[str, T.Any], ...],
    saver: T.Any,
    progress: ProgressCallback | None,
    is_cancelled: CancelPredicate | None,
    result: ExtractFacesResult,
) -> None:
    """Extract faces for an image-folder session."""
    from lib.image import read_image

    frame_paths = {frame.name: frame.path for frame in request.session.frame_list}
    total = len(targets)
    for done, (frame_name, entry) in enumerate(targets, start=1):
        if is_cancelled is not None and is_cancelled():
            result.cancelled = True
            _emit(progress, done, total, "Cancelled")
            return
        path = frame_paths.get(frame_name)
        if path is None:
            result.skipped_frames += 1
            result.errors.append(f"Source frame missing: {frame_name}")
            _emit(progress, done, total, f"Skipped {frame_name}")
            continue
        image = read_image(path, raise_error=True)
        _emit_faces_for_frame(alignments, frame_name, entry, image, saver, request, result)
        result.frames_processed += 1
        _emit(progress, done, total, f"Extracted faces from {frame_name}")


def _extract_video(
    request: FaceExtractionRequest,
    alignments: T.Any,
    targets: tuple[tuple[str, T.Any], ...],
    saver: T.Any,
    progress: ProgressCallback | None,
    is_cancelled: CancelPredicate | None,
    result: ExtractFacesResult,
) -> None:
    """Extract faces for a video session."""
    from lib.image import SingleFrameLoader

    metadata = request.handle.video_metadata()
    meta_dict = (
        None
        if metadata is None or not metadata.is_valid
        else {"pts_time": list(metadata.pts_time), "keyframes": list(metadata.keyframes)}
    )
    loader = SingleFrameLoader(request.session.frames, video_meta_data=meta_dict)
    try:
        name_to_index = {
            request.session.frame_name_for_index(index): index for index in range(loader.count)
        }
        total = len(targets)
        for done, (frame_name, entry) in enumerate(targets, start=1):
            if is_cancelled is not None and is_cancelled():
                result.cancelled = True
                _emit(progress, done, total, "Cancelled")
                return
            frame_index = name_to_index.get(frame_name)
            if frame_index is None:
                result.skipped_frames += 1
                result.errors.append(f"Video frame missing: {frame_name}")
                _emit(progress, done, total, f"Skipped {frame_name}")
                continue
            _filename, image = loader.image_from_index(frame_index)
            _emit_faces_for_frame(alignments, frame_name, entry, image, saver, request, result)
            result.frames_processed += 1
            _emit(progress, done, total, f"Extracted faces from {frame_name}")
    finally:
        loader.close()


def _emit_faces_for_frame(
    alignments: T.Any,
    frame_name: str,
    entry: T.Any,
    image: T.Any,
    saver: T.Any,
    request: FaceExtractionRequest,
    result: ExtractFacesResult,
) -> None:
    """Encode + save one aligned face per ``entry.faces`` entry."""
    from lib.align import AlignedFace
    from lib.align.objects import PNGAlignments, PNGHeader, PNGSource
    from lib.image import encode_image

    source_stem = os.path.splitext(os.path.basename(frame_name))[0]
    for face_index, face in enumerate(entry.faces):
        output_name = f"{source_stem}_{face_index}.png"
        aligned = AlignedFace(
            face.landmarks_xy, image=image, centering="head", size=EXTRACT_FACE_SIZE
        )
        if aligned.face is None:
            result.skipped_frames += 1
            result.errors.append(f"Face {face_index} in {frame_name} produced no aligned crop")
            continue
        # ``FileAlignments`` extends ``PNGAlignments`` with a ``thumb`` field we
        # don't want in the PNG header — re-project the shared fields onto a
        # fresh ``PNGAlignments`` dataclass so the saved payload matches the
        # legacy Tk header schema exactly.
        png_alignments = PNGAlignments(
            x=face.x,
            y=face.y,
            w=face.w,
            h=face.h,
            landmarks_xy=face.landmarks_xy,
            mask=face.mask,
            identity=face.identity,
            metadata=face.metadata,
        )
        meta = PNGHeader(
            alignments=png_alignments,
            source=PNGSource(
                alignments_version=alignments.version,
                original_filename=output_name,
                face_index=face_index,
                source_filename=frame_name,
                source_is_video=bool(request.session.is_video_input),
                source_frame_dims=tuple(image.shape[:2]),
            ),
        )
        payload = encode_image(aligned.face, ".png", metadata=meta)
        saver.save(output_name, payload)
        result.faces_written += 1


def _emit(progress: ProgressCallback | None, done: int, total: int, message: str) -> None:
    """Emit optional progress without binding this module to any UI framework."""
    if progress is not None:
        progress(done, total, message)


__all__ = (
    "CancelPredicate",
    "ExtractFacesResult",
    "EXTRACT_FACE_SIZE",
    "FaceExtractionRequest",
    "ProgressCallback",
    "extract_faces",
)
