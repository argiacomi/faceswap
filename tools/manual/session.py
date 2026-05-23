#!/usr/bin/env python3
"""GUI-neutral Manual Tool session state.

This module intentionally does not import tkinter or Qt.  It is a small, safe
service layer that the legacy Tk editor and the native Qt shell can both grow
around while the Manual Tool migration is in progress.
"""

from __future__ import annotations

import logging
import os
import typing as T
from dataclasses import dataclass, field
from pathlib import Path

from lib.utils import get_module_objects
from lib.video import VIDEO_EXTENSIONS

if T.TYPE_CHECKING:
    from argparse import Namespace

    from lib.align import Alignments

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})
_VIDEO_SUFFIXES = frozenset(ext.lower() for ext in VIDEO_EXTENSIONS)
_ALIGNMENTS_FILENAME = "alignments.fsa"


def _thumb_bytes(thumb: object) -> bytes:
    """Normalize a FileAlignments.thumb value to opaque JPEG bytes."""
    if thumb is None:
        return b""
    if isinstance(thumb, (bytes, bytearray, memoryview)):
        return bytes(thumb)
    try:  # numpy uint8 array
        return bytes(thumb.tobytes())  # type: ignore[attr-defined]
    except AttributeError:
        return b""


@dataclass(frozen=True)
class ManualFrame:
    """One source frame candidate for the Manual Tool."""

    index: int
    name: str
    path: str


@dataclass(frozen=True)
class FaceThumbnail:
    """One detected face entry for a Manual Tool frame.

    The thumbnail payload is the raw JPEG byte string stored inside the
    alignments file.  Callers decode it with whichever image library their UI
    layer prefers (cv2.imdecode, Pillow, QImage.loadFromData, …).
    """

    frame_index: int
    """Sorted-frame index this face belongs to."""
    frame_name: str
    """The frame's filename inside the alignments dict."""
    face_index: int
    """Position of this face inside the frame's face list."""
    thumbnail_jpeg: bytes
    """JPEG-encoded thumbnail bytes (may be empty when no thumb cached)."""

    @property
    def has_image(self) -> bool:
        """Return whether the entry has a cached thumbnail payload."""
        return bool(self.thumbnail_jpeg)


@dataclass(frozen=True)
class EditableFace:
    """One face's editable bounding box and landmarks for the Manual Tool overlay.

    Coordinates use source-image pixels. ``landmarks`` may be empty when the
    legacy alignment did not store per-point data.
    """

    face_index: int
    bbox: tuple[float, float, float, float]
    """Source-image bounding box as ``(x, y, width, height)``."""
    landmarks: tuple[tuple[float, float], ...] = ()

    def center(self) -> tuple[float, float]:
        """Return the bounding-box center in source-image pixels."""
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)

    def contains(self, px: float, py: float) -> bool:
        """Return whether ``(px, py)`` falls inside the bounding box."""
        x, y, w, h = self.bbox
        return x <= px <= x + w and y <= py <= y + h


@dataclass(frozen=True)
class ManualVideoMetadata:
    """GUI-neutral view of alignments-file video metadata."""

    pts_time: tuple[int, ...]
    keyframes: tuple[int, ...]

    @property
    def frame_count(self) -> int:
        """Return the number of frames described by the metadata."""
        return len(self.pts_time)

    @property
    def is_valid(self) -> bool:
        """Return whether metadata is non-empty and aligned."""
        return self.frame_count > 0


@dataclass
class ManualEditorState:
    """GUI-neutral editor state with simple observer callbacks.

    The legacy Tk implementation stores frame/face navigation, filter mode and
    dirty flags in :mod:`tkinter` variables.  The Qt shell cannot import those
    safely, so this object mirrors the same values through plain Python and a
    light observer API.  Both UIs subscribe to the same state changes.
    """

    frame_index: int = 0
    face_index: int = 0
    filter_mode: str = ""
    filter_distance: int = 10
    faces_size: str = ""
    is_zoomed: bool = False
    unsaved: bool = False
    edited: bool = False
    face_count_changed: bool = False
    editor_mode: str = "View"
    annotation_mode: str = ""
    is_playing: bool = False
    # Mask editor (#101) — brush + display state.  Defaults match the Tk
    # Manual Tool's first-launch values so muscle memory carries over.
    brush_size: int = 12
    brush_mode: str = "draw"
    mask_opacity: int = 50
    mask_type: str = ""
    # Bounding Box editor (#104) — aligner choices live on the editor
    # state so dropdowns + post-edit rerun share a single source of
    # truth.  Empty string means "use service default".
    aligner_name: str = ""
    aligner_normalization: str = "hist"
    aligner_auto_run: bool = True
    _listeners: dict[str, list[T.Callable[[T.Any], None]]] = field(
        default_factory=dict, repr=False
    )

    _FIELDS = (
        "frame_index",
        "face_index",
        "filter_mode",
        "filter_distance",
        "faces_size",
        "is_zoomed",
        "unsaved",
        "edited",
        "face_count_changed",
        "editor_mode",
        "annotation_mode",
        "is_playing",
        "brush_size",
        "brush_mode",
        "mask_opacity",
        "mask_type",
        "aligner_name",
        "aligner_normalization",
        "aligner_auto_run",
    )

    def subscribe(self, name: str, callback: T.Callable[[T.Any], None]) -> T.Callable[[], None]:
        """Register a callback for the given field. Returns an unsubscribe handle."""
        if name not in self._FIELDS:
            raise ValueError(f"Unknown editor state field: {name}")
        listeners = self._listeners.setdefault(name, [])
        listeners.append(callback)

        def _unsubscribe() -> None:
            if callback in listeners:
                listeners.remove(callback)

        return _unsubscribe

    def set(self, name: str, value: T.Any) -> None:
        """Set a field and notify listeners only when the value changes."""
        if name not in self._FIELDS:
            raise ValueError(f"Unknown editor state field: {name}")
        if getattr(self, name) == value:
            return
        setattr(self, name, value)
        for callback in list(self._listeners.get(name, ())):
            try:
                callback(value)
            except Exception:  # pragma: no cover - defensive
                logger.exception("Editor state listener for %s failed", name)


class ManualAlignmentsHandle:
    """GUI-neutral handle for the Manual Tool alignments file.

    Resolves the alignments folder and filename using the same rules as the
    legacy Tk implementation and lazily opens a :class:`lib.align.Alignments`
    instance only when the consumer asks for it.  Importing this handle does
    not pull in tkinter or Qt.
    """

    def __init__(self, folder: str, filename: str, *, is_video: bool) -> None:
        self._folder = folder
        self._filename = filename
        self._is_video = is_video
        self._alignments: Alignments | None = None

    @property
    def folder(self) -> str:
        """Return the folder containing the alignments file."""
        return self._folder

    @property
    def filename(self) -> str:
        """Return the alignments filename."""
        return self._filename

    @property
    def path(self) -> str:
        """Return the full resolved alignments file path."""
        return os.path.join(self._folder, self._filename)

    @property
    def exists(self) -> bool:
        """Return whether the alignments file currently exists on disk."""
        return os.path.isfile(self.path)

    def open(self) -> Alignments:
        """Open and cache a :class:`lib.align.Alignments` instance."""
        if self._alignments is None:
            from lib.align import Alignments

            self._alignments = Alignments(self._folder, self._filename)
        return self._alignments

    def video_metadata(self) -> ManualVideoMetadata | None:
        """Return GUI-neutral video metadata or ``None`` when unavailable."""
        if not self._is_video or not self.exists:
            return None
        meta = self.open().video_meta_data
        if not meta:
            return None
        pts_time = meta.get("pts_time") or []
        keyframes = meta.get("keyframes") or []
        if not pts_time:
            return None
        return ManualVideoMetadata(
            pts_time=tuple(pts_time),
            keyframes=tuple(keyframes),
        )

    def has_thumbnails(self) -> bool:
        """Return whether the alignments file already stores thumbnails.

        Pure read — does *not* regenerate.  Callers that want missing
        thumbnails materialized should invoke :meth:`regenerate_thumbnails`
        explicitly with a session.
        """
        if not self.exists:
            return False
        return bool(self.open().thumbnails.has_thumbnails)

    def regenerate_thumbnails(
        self,
        session: ManualSession,
        *,
        progress: T.Callable[[int, int, str], None] | None = None,
    ) -> int:
        """Rebuild cached face thumbnails for ``session`` and persist them.

        Delegates to :func:`tools.manual.thumbnail_generation.regenerate_thumbnails`
        so the heavy work stays in a Qt/Tk-neutral module while callers reach
        for it through the handle they already hold.  Returns the number of
        frames regenerated.
        """
        from tools.manual.thumbnail_generation import regenerate_thumbnails

        return regenerate_thumbnails(self, session, progress=progress)

    def persist(
        self,
        editable: ManualEditableAlignments,
        *,
        frame_names: T.Sequence[str] | T.Callable[[int], str | None],
    ) -> int:
        """Write ``editable``'s in-memory edits back to the alignments file.

        Opens the alignments file (bootstrapping a minimal one on disk first
        when none exists), applies the editable model's frame data via
        :meth:`ManualEditableAlignments.apply_to_alignments`, then writes the
        file out atomically through ``Alignments.save``.

        Returns the number of frames modified.  Re-raises any persistence
        error from :class:`lib.align.Alignments` so the caller (typically
        :class:`ManualToolWindow`) can surface the failure to the user
        without silently clearing dirty state.
        """
        self._bootstrap_alignments_file()
        alignments = self.open()
        try:
            modified = editable.apply_to_alignments(alignments, frame_names=frame_names)
        except Exception:  # pragma: no cover - re-raised below with context
            logger.exception("Manual Tool persist: apply_to_alignments failed")
            raise
        alignments.save()
        return modified

    def _bootstrap_alignments_file(self) -> None:
        """Create a minimal alignments file on disk if one does not yet exist.

        The legacy ``Alignments`` constructor refuses to open a missing file,
        but Manual Tool edits can be the first thing that touches the
        alignments path (especially when the user runs the Qt shell against a
        fresh folder of frames).  Write out the smallest valid payload so the
        existing ``Alignments`` load + ``save`` pipeline can take over.
        """
        if self.exists or self._alignments is not None:
            return
        os.makedirs(self._folder, exist_ok=True)
        # Lazy imports keep tkinter and heavyweight dependencies out of the
        # module's import-time graph.
        from lib.serializer import get_serializer

        serializer = get_serializer("compressed")
        serializer.save(self.path, {"__meta__": {"version": 2.4}, "__data__": {}})

    def sorted_frame_names(self) -> tuple[str, ...]:
        """Return frame names from the alignments file in sorted order."""
        if not self.exists:
            return ()
        return tuple(sorted(self.open().data))

    def faces_for_frame(self, frame_index: int) -> tuple[FaceThumbnail, ...]:
        """Return GUI-neutral face thumbnail entries for the given frame index.

        Resolves ``frame_index`` against the sorted alignments-file keys. Callers
        that drive their model from a separate frame list (eg. image-folder
        sessions whose source list is sparse relative to alignments) should use
        :meth:`faces_for_frame_name` so the thumbnail is fetched by the same
        frame name the model uses to persist.

        Returns an empty tuple when the alignments file does not exist yet, the
        index is out of range, or the frame has no detected faces.
        """
        if frame_index < 0 or not self.exists:
            return ()
        alignments = self.open()
        names = sorted(alignments.data)
        if frame_index >= len(names):
            return ()
        return self._thumbnails_for(frame_index, names[frame_index], alignments)

    def faces_for_frame_name(
        self, frame_name: str, frame_index: int | None = None
    ) -> tuple[FaceThumbnail, ...]:
        """Return GUI-neutral face thumbnail entries keyed by frame *name*.

        Use this in the Manual Tool's panel refresh: the editable model's
        ``frame_index`` is anchored to the source frame list, but the alignments
        file may store sparse entries with arbitrary names. Looking up by name
        guarantees the persisted thumbnail came from the same frame the editable
        face was seeded from.

        Returns an empty tuple when the alignments file does not exist yet or
        ``frame_name`` is not present in it.
        """
        if not frame_name or not self.exists:
            return ()
        alignments = self.open()
        if frame_name not in alignments.data:
            return ()
        resolved_index = -1 if frame_index is None else frame_index
        return self._thumbnails_for(resolved_index, frame_name, alignments)

    @staticmethod
    def _thumbnails_for(
        frame_index: int, frame_name: str, alignments: T.Any
    ) -> tuple[FaceThumbnail, ...]:
        """Build ``FaceThumbnail`` entries for one resolved frame name."""
        faces = alignments.data[frame_name].faces
        return tuple(
            FaceThumbnail(
                frame_index=frame_index,
                frame_name=frame_name,
                face_index=face_index,
                thumbnail_jpeg=_thumb_bytes(face.thumb),
            )
            for face_index, face in enumerate(faces)
        )

    def face_count_for_frame(self, frame_index: int) -> int:
        """Return the number of faces stored for a given sorted-frame index."""
        return len(self.faces_for_frame(frame_index))


class ManualEditableAlignments:
    """GUI-neutral editable alignment model for the Manual Tool overlay layer.

    Stages face edits per frame in memory so the Qt frame viewer can drive
    select / move / add / delete / update operations without touching the
    on-disk alignments file before the user saves.  Each mutation pushes a
    matching inverse onto an undo stack so the overlay can offer undo/redo.

    Seeding from an existing alignments file is opt-in: callers either pass an
    initial ``faces`` mapping at construction time, or call
    :meth:`seed_from_handle` to lazily import the read-only handle data.
    """

    _MIN_BBOX = 1.0
    MASK_STORED_SIZE: T.ClassVar[int] = 128
    """Side length (in pixels) of stored-space mask grids used by the
    Mask editor (#101).  Matches the legacy Tk Manual Tool's default."""

    def __init__(
        self,
        faces: T.Mapping[int, T.Sequence[EditableFace]] | None = None,
    ) -> None:
        self._faces: dict[int, list[EditableFace]] = {}
        if faces is not None:
            for frame_index, frame_faces in faces.items():
                self._faces[int(frame_index)] = [
                    self._normalize(face_index, face)
                    for face_index, face in enumerate(frame_faces)
                ]
        # Each entry is (do, undo, frame_index).
        self._undo: list[tuple[T.Callable[[], None], T.Callable[[], None], int]] = []
        self._redo: list[tuple[T.Callable[[], None], T.Callable[[], None], int]] = []
        self._listeners: list[T.Callable[[int], None]] = []
        # Frame indices that have been mutated since the last persist/seed.
        # Tracked separately from ``_faces`` so fully emptied frames (every
        # face deleted) are still written back to the alignments file on save.
        self._dirty_frames: set[int] = set()
        # Mask Editor (#101) — in-memory mask grids keyed by ``(frame_index,
        # face_index, mask_type)``.  Stored as uint8 numpy arrays sized to
        # :attr:`MASK_STORED_SIZE`.  Persistence to the alignments-file blob
        # format is a follow-up; today the editor operates in-memory so the
        # user can paint, undo/redo, and see live overlay feedback.
        self._mask_state: dict[tuple[int, int, str], T.Any] = {}

    # ---- read API ----
    def faces(self, frame_index: int) -> tuple[EditableFace, ...]:
        """Return the editable faces for a sorted-frame index."""
        return tuple(self._faces.get(frame_index, ()))

    def face_count(self, frame_index: int) -> int:
        """Return the number of editable faces for a frame."""
        return len(self._faces.get(frame_index, ()))

    def known_frame_indices(self) -> tuple[int, ...]:
        """Return every frame index this model currently tracks.

        Includes empty-but-known frames (fully deleted entries still appear
        until explicitly cleared) so callers can decide whether to keep,
        skip, or surface them — :meth:`apply_to_alignments` already iterates
        ``_dirty_frames`` to write empty entries back to disk.
        """
        return tuple(sorted(self._faces.keys()))

    def hit_test(self, frame_index: int, px: float, py: float) -> int | None:
        """Return the face_index whose bbox contains ``(px, py)`` or ``None``.

        Selects the topmost face when multiple bboxes overlap.
        """
        for face in reversed(self._faces.get(frame_index, ())):
            if face.contains(px, py):
                return face.face_index
        return None

    @property
    def can_undo(self) -> bool:
        """Return whether the undo stack has any reversible operations."""
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        """Return whether the redo stack has any replayable operations."""
        return bool(self._redo)

    # ---- mutation API ----
    def seed_from_handle(
        self,
        handle: ManualAlignmentsHandle,
        *,
        frame_names: T.Sequence[str] | None = None,
    ) -> None:
        """Lazily import face metadata from a :class:`ManualAlignmentsHandle`.

        Reads bbox + landmark data from the underlying alignments file.  Clears
        the existing in-memory edits and the undo/redo stacks.

        ``frame_names`` is an optional ordered list of filenames that defines
        the canonical mapping from positional ``frame_index`` to filename.
        Image-folder sessions pass their source ``frame_list`` so that sparse
        alignments (e.g. one entry for ``frame_010.png``) seed onto the
        matching frame index rather than the lexicographic position of the
        alignment key.  When ``None``, the legacy behavior of
        ``sorted(alignments.data)`` is used (the right choice for video
        inputs whose frame order *is* the sorted alignment-key order).
        """
        if not handle.exists:
            return
        alignments = handle.open()
        self._faces.clear()
        self._undo.clear()
        self._redo.clear()
        self._dirty_frames.clear()
        if frame_names is None:
            index_map = {
                frame_name: frame_index
                for frame_index, frame_name in enumerate(sorted(alignments.data))
            }
        else:
            index_map = {
                frame_name: frame_index for frame_index, frame_name in enumerate(frame_names)
            }
        for frame_name, entry in alignments.data.items():
            frame_index = index_map.get(frame_name)
            if frame_index is None:
                logger.debug(
                    "Manual Tool seed: alignments key %s has no matching source frame; skipping",
                    frame_name,
                )
                continue
            entries: list[EditableFace] = []
            for face_index, face in enumerate(entry.faces):
                landmarks: tuple[tuple[float, float], ...] = ()
                raw_landmarks = getattr(face, "landmarks_xy", None)
                if raw_landmarks is not None:
                    try:
                        landmarks = tuple((float(x), float(y)) for x, y in raw_landmarks)
                    except (TypeError, ValueError):
                        landmarks = ()
                entries.append(
                    EditableFace(
                        face_index=face_index,
                        bbox=(
                            float(getattr(face, "x", 0)),
                            float(getattr(face, "y", 0)),
                            float(getattr(face, "w", 0)),
                            float(getattr(face, "h", 0)),
                        ),
                        landmarks=landmarks,
                    )
                )
            if entries:
                self._faces[frame_index] = entries
        for callback in list(self._listeners):
            for frame in self._faces:
                callback(frame)

    def add_face(
        self,
        frame_index: int,
        bbox: tuple[float, float, float, float],
        *,
        landmarks: T.Sequence[tuple[float, float]] = (),
    ) -> int:
        """Add a new face at ``bbox`` and return its new ``face_index``."""
        validated = self._validate_bbox(bbox)
        face_index = self.face_count(frame_index)
        face = EditableFace(
            face_index=face_index,
            bbox=validated,
            landmarks=tuple((float(x), float(y)) for x, y in landmarks),
        )

        def do() -> None:
            self._faces.setdefault(frame_index, []).append(face)
            self._reindex(frame_index)

        def undo() -> None:
            faces = self._faces.get(frame_index, [])
            if faces:
                faces.pop()

        self._apply(do, undo, frame_index)
        return face_index

    def delete_face(self, frame_index: int, face_index: int) -> bool:
        """Delete the face at ``face_index`` in ``frame_index``."""
        faces = self._faces.get(frame_index, [])
        if face_index < 0 or face_index >= len(faces):
            return False
        removed = faces[face_index]

        def do() -> None:
            self._faces[frame_index].pop(face_index)
            self._reindex(frame_index)
            # Keep the (now-empty) entry in _faces so apply_to_alignments()
            # writes ``entry.faces = []`` to disk; tracking dirty frames
            # ensures the persist still hits this frame.

        def undo() -> None:
            target = self._faces.setdefault(frame_index, [])
            target.insert(face_index, removed)
            self._reindex(frame_index)

        self._apply(do, undo, frame_index)
        return True

    def resize_face(
        self,
        frame_index: int,
        face_index: int,
        bbox: tuple[float, float, float, float],
    ) -> bool:
        """Replace the face's bbox with ``bbox`` (source pixels).

        Landmarks are rescaled proportionally to the new bbox dimensions so a
        resize-drag keeps the landmark cloud anchored relative to the box; this
        mirrors the Tk Manual Tool's resize behavior.  Returns ``False`` when
        the index is invalid or the new bbox would be degenerate.
        """
        faces = self._faces.get(frame_index, [])
        if face_index < 0 or face_index >= len(faces):
            return False
        new_x, new_y, new_w, new_h = self._validate_bbox(bbox)
        previous = faces[face_index]
        prev_x, prev_y, prev_w, prev_h = previous.bbox
        if new_x == prev_x and new_y == prev_y and new_w == prev_w and new_h == prev_h:
            return True
        if prev_w > 0 and prev_h > 0 and previous.landmarks:
            scale_x = new_w / prev_w
            scale_y = new_h / prev_h
            landmarks = tuple(
                (
                    new_x + (lx - prev_x) * scale_x,
                    new_y + (ly - prev_y) * scale_y,
                )
                for lx, ly in previous.landmarks
            )
        else:
            landmarks = previous.landmarks
        resized = EditableFace(
            face_index=face_index,
            bbox=(new_x, new_y, new_w, new_h),
            landmarks=landmarks,
        )

        def do() -> None:
            self._faces[frame_index][face_index] = resized

        def undo() -> None:
            self._faces[frame_index][face_index] = previous

        self._apply(do, undo, frame_index)
        return True

    def move_face(
        self,
        frame_index: int,
        face_index: int,
        dx: float,
        dy: float,
    ) -> bool:
        """Translate the face's bbox + landmarks by ``(dx, dy)`` source pixels."""
        faces = self._faces.get(frame_index, [])
        if face_index < 0 or face_index >= len(faces):
            return False
        if dx == 0.0 and dy == 0.0:
            return True
        previous = faces[face_index]
        x, y, w, h = previous.bbox
        moved = EditableFace(
            face_index=face_index,
            bbox=(x + dx, y + dy, w, h),
            landmarks=tuple((lx + dx, ly + dy) for lx, ly in previous.landmarks),
        )

        def do() -> None:
            self._faces[frame_index][face_index] = moved

        def undo() -> None:
            self._faces[frame_index][face_index] = previous

        self._apply(do, undo, frame_index)
        return True

    def scale_face(
        self,
        frame_index: int,
        face_index: int,
        scale: float,
    ) -> bool:
        """Scale the face's landmarks + bbox uniformly around the bbox centre.

        Used by the Extract Box editor (#102) where corner-drag scales the
        landmark cloud as a unit.  ``scale`` is the multiplicative factor
        applied to every landmark coordinate relative to the centre; the
        bbox grows/shrinks by the same factor centred on the same point.

        Returns ``False`` for an invalid face_index, a non-positive scale,
        or when the result would produce a degenerate sub-pixel bbox.
        ``True`` no-op for a 1.0 scale.
        """
        faces = self._faces.get(frame_index, [])
        if face_index < 0 or face_index >= len(faces):
            return False
        if not (scale > 0.0):
            return False
        if scale == 1.0:
            return True
        face = faces[face_index]
        x, y, w, h = face.bbox
        new_w = w * scale
        new_h = h * scale
        if new_w < 1.0 or new_h < 1.0:
            return False
        cx = x + w / 2.0
        cy = y + h / 2.0
        new_bbox = (cx - new_w / 2.0, cy - new_h / 2.0, new_w, new_h)
        new_landmarks: tuple[tuple[float, float], ...]
        if face.landmarks:
            new_landmarks = tuple(
                (cx + (lx - cx) * scale, cy + (ly - cy) * scale) for lx, ly in face.landmarks
            )
        else:
            new_landmarks = face.landmarks
        previous = face
        scaled = EditableFace(
            face_index=face.face_index,
            bbox=new_bbox,
            landmarks=new_landmarks,
        )

        def do() -> None:
            self._faces[frame_index][face_index] = scaled

        def undo() -> None:
            self._faces[frame_index][face_index] = previous

        self._apply(do, undo, frame_index)
        return True

    def rotate_face(
        self,
        frame_index: int,
        face_index: int,
        angle_radians: float,
    ) -> bool:
        """Rotate the face's landmarks around the bbox centre.

        Used by the Extract Box editor (#102) where the outside-corner
        rotation zone rotates the entire landmark cloud.  The bbox is
        recomputed as the axis-aligned bounding rectangle of the rotated
        landmarks so the visible box continues to enclose the points; if
        the face has no landmarks the bbox stays put and the call no-ops.

        ``angle_radians`` is clockwise in source-image coordinates (Qt's
        y-down convention).  Returns ``True`` for a no-op zero angle and
        ``False`` for an invalid face_index.
        """
        import math

        faces = self._faces.get(frame_index, [])
        if face_index < 0 or face_index >= len(faces):
            return False
        if angle_radians == 0.0:
            return True
        face = faces[face_index]
        if not face.landmarks:
            return True
        x, y, w, h = face.bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        cos_a = math.cos(angle_radians)
        sin_a = math.sin(angle_radians)
        new_points: list[tuple[float, float]] = []
        for lx, ly in face.landmarks:
            dx = lx - cx
            dy = ly - cy
            new_points.append((cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a))
        # Axis-aligned bbox of the rotated landmark cloud — preserves the
        # visual relationship between the box and the points without
        # tracking a separate rotation state on the face object.
        xs = [px for px, _ in new_points]
        ys = [py for _, py in new_points]
        nx = min(xs)
        ny = min(ys)
        nw = max(1.0, max(xs) - nx)
        nh = max(1.0, max(ys) - ny)
        previous = face
        rotated = EditableFace(
            face_index=face.face_index,
            bbox=(nx, ny, nw, nh),
            landmarks=tuple(new_points),
        )

        def do() -> None:
            self._faces[frame_index][face_index] = rotated

        def undo() -> None:
            self._faces[frame_index][face_index] = previous

        self._apply(do, undo, frame_index)
        return True

    # ---- Mask editor (#101) ----
    def get_mask(
        self,
        frame_index: int,
        face_index: int,
        mask_type: str,
    ) -> T.Any:
        """Return the editable mask grid for ``(frame, face, type)``.

        Returns the in-memory ``numpy.ndarray`` of dtype uint8 sized
        :attr:`MASK_STORED_SIZE` x :attr:`MASK_STORED_SIZE`, or ``None`` if
        no mask has been created/painted for this combination yet.  The
        Mask editor consults this lazily — masks are *only* allocated on
        first paint stroke.
        """
        key = (int(frame_index), int(face_index), str(mask_type))
        return self._mask_state.get(key)

    def has_mask(self, frame_index: int, face_index: int, mask_type: str) -> bool:
        """Return whether an editable mask has been populated for this slot."""
        return (int(frame_index), int(face_index), str(mask_type)) in self._mask_state

    def known_mask_types(self, frame_index: int, face_index: int) -> tuple[str, ...]:
        """Return every mask type currently materialised for the given face."""
        return tuple(
            sorted(
                mask_type
                for (fi, faci, mask_type) in self._mask_state
                if fi == int(frame_index) and faci == int(face_index)
            )
        )

    def paint_mask_stroke(
        self,
        frame_index: int,
        face_index: int,
        mask_type: str,
        source_x: float,
        source_y: float,
        radius: float,
        value: int,
    ) -> bool:
        """Stamp a circular brush of ``radius`` at ``(source_x, source_y)``.

        Source coordinates are mapped into stored-space (the mask grid's
        coordinate frame) using a simple axis-aligned projection from the
        face's bounding box: that's lossier than the legacy affine-matrix
        approach but sufficient for an editor-side preview and keeps the
        in-memory model free of additional matrix state.

        ``value`` is ``255`` to draw or ``0`` to erase.  Pixels are
        full-on/full-off — feathered brushes are a follow-up.  Returns
        ``False`` for an invalid face_index, out-of-bbox coordinates, or
        a non-positive radius.  Mutates the mask array in place and
        records undo/redo via a snapshot of the affected stamp region.
        """
        import numpy as np

        faces = self._faces.get(frame_index, [])
        if face_index < 0 or face_index >= len(faces):
            return False
        if not (radius > 0.0):
            return False
        face = faces[face_index]
        x, y, w, h = face.bbox
        if w <= 0.0 or h <= 0.0:
            return False
        # Source → stored-space mapping (axis-aligned bbox projection).
        rel_x = (source_x - x) / w
        rel_y = (source_y - y) / h
        if not (0.0 <= rel_x <= 1.0 and 0.0 <= rel_y <= 1.0):
            return False
        size = self.MASK_STORED_SIZE
        cx = rel_x * size
        cy = rel_y * size
        # Stored-space brush radius: scale by the smaller bbox dimension so
        # the on-screen brush size feels uniform regardless of bbox aspect.
        scale = size / max(1.0, min(w, h))
        stored_radius = max(1.0, radius * scale)
        key = (int(frame_index), int(face_index), str(mask_type))
        existing = self._mask_state.get(key)
        if existing is None:
            existing = np.zeros((size, size), dtype=np.uint8)
            self._mask_state[key] = existing
        # Carve out the stamp region for undo/redo bookkeeping.
        r_ceil = int(np.ceil(stored_radius)) + 1
        x0 = max(0, int(cx) - r_ceil)
        y0 = max(0, int(cy) - r_ceil)
        x1 = min(size, int(cx) + r_ceil + 1)
        y1 = min(size, int(cy) + r_ceil + 1)
        if x0 >= x1 or y0 >= y1:
            return False
        prior_region = existing[y0:y1, x0:x1].copy()
        # Apply the circular stamp.
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= stored_radius * stored_radius
        if int(value) > 0:
            existing[y0:y1, x0:x1][mask] = int(value)
        else:
            existing[y0:y1, x0:x1][mask] = 0
        # Snapshot AFTER paint so redo replays the same write.
        new_region = existing[y0:y1, x0:x1].copy()

        def do() -> None:
            existing[y0:y1, x0:x1] = new_region

        def undo() -> None:
            existing[y0:y1, x0:x1] = prior_region

        self._apply(do, undo, frame_index)
        return True

    def clear_mask(self, frame_index: int, face_index: int, mask_type: str) -> bool:
        """Drop the editable mask for ``(frame, face, type)``.

        Used as the explicit clear control (and by full-frame revert).
        Records undo/redo so a clear can be reversed.  Returns ``False``
        when no mask was materialised in the first place.
        """
        key = (int(frame_index), int(face_index), str(mask_type))
        existing = self._mask_state.get(key)
        if existing is None:
            return False
        prior = existing.copy()

        def do() -> None:
            self._mask_state.pop(key, None)

        def undo() -> None:
            self._mask_state[key] = prior

        self._apply(do, undo, frame_index)
        return True

    def update_landmark(
        self,
        frame_index: int,
        face_index: int,
        landmark_index: int,
        px: float,
        py: float,
    ) -> bool:
        """Move a single landmark to ``(px, py)`` in source coordinates."""
        faces = self._faces.get(frame_index, [])
        if face_index < 0 or face_index >= len(faces):
            return False
        face = faces[face_index]
        if landmark_index < 0 or landmark_index >= len(face.landmarks):
            return False
        previous = face
        new_landmarks = list(face.landmarks)
        new_landmarks[landmark_index] = (float(px), float(py))
        updated = EditableFace(
            face_index=face.face_index,
            bbox=face.bbox,
            landmarks=tuple(new_landmarks),
        )

        def do() -> None:
            self._faces[frame_index][face_index] = updated

        def undo() -> None:
            self._faces[frame_index][face_index] = previous

        self._apply(do, undo, frame_index)
        return True

    def set_landmarks(
        self,
        frame_index: int,
        face_index: int,
        landmarks: T.Sequence[tuple[float, float]],
    ) -> bool:
        """Replace the full landmark set for a face (Aligner integration, #104).

        Used by the host after a Bounding Box edit reruns the selected
        aligner: the editable bbox stays put while landmarks are
        refreshed in one undoable step.  Returns ``False`` for an
        invalid face_index; an empty ``landmarks`` sequence is treated
        as "clear" and is allowed so the host can roll back to landmark-
        less state if the aligner refuses to produce anything.
        """
        faces = self._faces.get(frame_index, [])
        if face_index < 0 or face_index >= len(faces):
            return False
        previous = faces[face_index]
        new_landmarks = tuple((float(x), float(y)) for x, y in landmarks)
        updated = EditableFace(
            face_index=previous.face_index,
            bbox=previous.bbox,
            landmarks=new_landmarks,
        )

        def do() -> None:
            self._faces[frame_index][face_index] = updated

        def undo() -> None:
            self._faces[frame_index][face_index] = previous

        self._apply(do, undo, frame_index)
        return True

    def move_landmarks(
        self,
        frame_index: int,
        face_index: int,
        landmark_indices: T.Iterable[int],
        dx: float,
        dy: float,
    ) -> bool:
        """Translate the named landmarks by ``(dx, dy)`` source pixels.

        Used by the Landmark editor's group-move gesture: after a marquee
        selection the user can drag the highlighted points as a unit.  Each
        ``landmark_index`` is validated; unknown indices are silently dropped
        rather than raising, so a stale selection from a previous frame
        cannot crash the editor.

        Returns ``False`` when the frame/face index is invalid or every
        supplied index is out of range; ``True`` (no-op) when the delta is
        zero or when nothing remains after validation.
        """
        faces = self._faces.get(frame_index, [])
        if face_index < 0 or face_index >= len(faces):
            return False
        if dx == 0.0 and dy == 0.0:
            return True
        face = faces[face_index]
        if not face.landmarks:
            return False
        indices = {int(idx) for idx in landmark_indices if 0 <= int(idx) < len(face.landmarks)}
        if not indices:
            return True
        previous = face
        new_landmarks = list(face.landmarks)
        for idx in indices:
            lx, ly = new_landmarks[idx]
            new_landmarks[idx] = (lx + dx, ly + dy)
        updated = EditableFace(
            face_index=face.face_index,
            bbox=face.bbox,
            landmarks=tuple(new_landmarks),
        )

        def do() -> None:
            self._faces[frame_index][face_index] = updated

        def undo() -> None:
            self._faces[frame_index][face_index] = previous

        self._apply(do, undo, frame_index)
        return True

    def undo(self) -> bool:
        """Replay the last inverse operation; returns ``False`` when empty."""
        if not self._undo:
            return False
        do, undo, frame_index = self._undo.pop()
        undo()
        self._redo.append((do, undo, frame_index))
        self._dirty_frames.add(frame_index)
        self._notify(frame_index)
        return True

    def redo(self) -> bool:
        """Replay the last undone operation; returns ``False`` when empty."""
        if not self._redo:
            return False
        do, undo, frame_index = self._redo.pop()
        do()
        self._undo.append((do, undo, frame_index))
        self._dirty_frames.add(frame_index)
        self._notify(frame_index)
        return True

    # ---- observer hooks ----
    def subscribe(self, callback: T.Callable[[int], None]) -> T.Callable[[], None]:
        """Register a callback fired with the modified ``frame_index``."""
        self._listeners.append(callback)

        def _unsubscribe() -> None:
            if callback in self._listeners:
                self._listeners.remove(callback)

        return _unsubscribe

    # ---- internals ----
    def _apply(
        self,
        do: T.Callable[[], None],
        undo: T.Callable[[], None],
        frame_index: int,
    ) -> None:
        """Run ``do``, push the inverse onto undo, and notify listeners.

        Also marks ``frame_index`` dirty so :meth:`apply_to_alignments`
        knows the frame needs writing — even if the operation ends up
        leaving the editable face list empty (e.g. deleting the last
        face), the touched-frames set guarantees the alignments file is
        updated to match on the next save.
        """
        do()
        self._undo.append((do, undo, frame_index))
        self._redo.clear()
        self._dirty_frames.add(frame_index)
        self._notify(frame_index)

    def clear_history(self) -> None:
        """Drop the undo and redo stacks.

        Called after a successful persist so the user cannot undo back past
        the save point in the editor and so the next save only writes frames
        that have actually been mutated since this point.
        """
        self._undo.clear()
        self._redo.clear()
        self._dirty_frames.clear()

    def revert_frame(self, frame_index: int) -> int:
        """Undo only the operations recorded against ``frame_index``.

        The undo stack is rewritten so that records tagged with other frame
        indices are preserved in their original order.  Returns the number
        of operations reverted.  The redo stack is cleared because the
        partial replay produces a state that cannot be re-derived from the
        existing redo records.
        """
        kept: list[tuple[T.Callable[[], None], T.Callable[[], None], int]] = []
        reverted = 0
        for record in reversed(self._undo):
            do, undo, op_frame = record
            if op_frame == frame_index:
                undo()
                reverted += 1
            else:
                kept.append(record)
        # ``kept`` is built reverse-of-reverse so it preserves stack order.
        self._undo = list(reversed(kept))
        if reverted:
            self._redo.clear()
            if not any(op_frame == frame_index for _do, _undo, op_frame in self._undo):
                self._dirty_frames.discard(frame_index)
            self._notify(frame_index)
        return reverted

    def apply_to_alignments(
        self,
        alignments: T.Any,
        *,
        frame_names: T.Sequence[str] | T.Callable[[int], str | None],
    ) -> int:
        """Write in-memory edits back to a :class:`lib.align.Alignments` object.

        Mutates ``alignments.data`` in place: matching face indices keep their
        ``mask`` / ``identity`` / ``metadata`` payloads while bbox + landmarks
        come from the editable model.  Thumbnails are invalidated whenever the
        bbox changes so a stale crop is never written.  Added faces produce a
        fresh :class:`lib.align.objects.FileAlignments` entry; removed faces
        are dropped.

        ``frame_names`` may be a :class:`Sequence` (image-folder mode, where the
        source frame list is bounded and pre-known) or a callable
        ``Callable[[int], str | None]`` (video mode, where the frame name is
        synthesized on demand from the frame index — see
        :meth:`ManualSession.frame_name_for_index`). Returning ``None`` from
        the callable indicates the index cannot be mapped and raises
        ``ValueError`` just like falling past the end of a sequence does.

        Returns the number of frames that were modified.

        Raises ``ValueError`` if any frame_index referenced by the editable
        model cannot be mapped to a frame name.
        """
        import numpy as np

        from lib.align.objects import AlignmentsEntry, FileAlignments

        if callable(frame_names):
            resolver: T.Callable[[int], str | None] = frame_names
        else:
            sequence = frame_names

            def resolver(frame_index: int) -> str | None:
                if frame_index < 0 or frame_index >= len(sequence):
                    return None
                return sequence[frame_index]

        modified = 0
        targets = sorted(self._dirty_frames or self._faces.keys())
        for frame_index in targets:
            frame_name = resolver(frame_index)
            if not frame_name:
                raise ValueError(f"Frame index {frame_index} has no matching frame name")
            entry = alignments.data.get(frame_name)
            if entry is None:
                entry = AlignmentsEntry(faces=[], video_meta={})
                alignments.data[frame_name] = entry
            original_faces = list(entry.faces)
            new_faces: list[FileAlignments] = []
            for face in self._faces.get(frame_index, []):
                landmarks_array = (
                    np.asarray(face.landmarks, dtype=np.float32)
                    if face.landmarks
                    else np.zeros((0, 2), dtype=np.float32)
                )
                prev = (
                    original_faces[face.face_index]
                    if face.face_index < len(original_faces)
                    else None
                )
                x, y, w, h = (int(round(v)) for v in face.bbox)
                if prev is not None and (
                    int(prev.x),
                    int(prev.y),
                    int(prev.w),
                    int(prev.h),
                ) == (x, y, w, h):
                    # Bbox unchanged — keep the cached thumb + landmark data
                    # but accept any landmark replacement from the editable
                    # model so update_landmark survives a round-trip.
                    prev.landmarks_xy = landmarks_array
                    new_faces.append(prev)
                else:
                    if prev is not None:
                        new_faces.append(
                            FileAlignments(
                                x=x,
                                y=y,
                                w=w,
                                h=h,
                                landmarks_xy=landmarks_array,
                                mask=prev.mask,
                                identity=prev.identity,
                                metadata=prev.metadata,
                                thumb=None,
                            )
                        )
                    else:
                        new_faces.append(
                            FileAlignments(
                                x=x,
                                y=y,
                                w=w,
                                h=h,
                                landmarks_xy=landmarks_array,
                            )
                        )
            entry.faces = new_faces
            modified += 1
        return modified

    def _notify(self, frame_index: int) -> None:
        for callback in list(self._listeners):
            try:
                callback(frame_index)
            except Exception:  # pragma: no cover - defensive
                logger.exception("Editable alignment listener raised")

    def _reindex(self, frame_index: int) -> None:
        """Reassign monotonically increasing ``face_index`` after add/delete."""
        faces = self._faces.get(frame_index, [])
        for new_index, face in enumerate(faces):
            if face.face_index != new_index:
                faces[new_index] = EditableFace(
                    face_index=new_index,
                    bbox=face.bbox,
                    landmarks=face.landmarks,
                )

    def _validate_bbox(
        self, bbox: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Reject degenerate bounding boxes before recording the edit."""
        x, y, w, h = (float(v) for v in bbox)
        if w < self._MIN_BBOX or h < self._MIN_BBOX:
            raise ValueError(f"Bounding box must have width and height >= {self._MIN_BBOX}")
        return (x, y, w, h)

    @staticmethod
    def _normalize(face_index: int, face: EditableFace) -> EditableFace:
        """Ensure a seeded face uses the expected positional index."""
        if face.face_index == face_index:
            return face
        return EditableFace(
            face_index=face_index,
            bbox=face.bbox,
            landmarks=face.landmarks,
        )


@dataclass(frozen=True)
class ManualSession:
    """Serializable Manual Tool session metadata shared by GUI implementations."""

    frames: str
    alignments_path: str | None = None
    thumb_regenerate: bool = False
    single_process: bool = False
    frame_list: tuple[ManualFrame, ...] = ()
    is_video_input: bool = False

    @classmethod
    def from_cli_values(cls, values: T.Mapping[str, object]) -> ManualSession:
        """Build a session from switch-keyed Qt command-panel values."""
        frames = cls._string_value(
            values.get("-f") or values.get("--frames") or values.get("frames")
        )
        alignments = cls._string_value(
            values.get("-a") or values.get("--alignments") or values.get("alignments_path")
        )
        return cls.create(
            frames=frames,
            alignments_path=alignments or None,
            thumb_regenerate=bool(values.get("-t") or values.get("--thumb-regen")),
            single_process=bool(values.get("-s") or values.get("--single-process")),
        )

    @classmethod
    def from_namespace(cls, namespace: Namespace) -> ManualSession:
        """Build a session from a parsed argparse namespace."""
        return cls.create(
            frames=getattr(namespace, "frames", "") or "",
            alignments_path=getattr(namespace, "alignments_path", "") or None,
            thumb_regenerate=bool(getattr(namespace, "thumb_regenerate", False)),
            single_process=bool(getattr(namespace, "single_process", False)),
        )

    @classmethod
    def create(
        cls,
        *,
        frames: str,
        alignments_path: str | None = None,
        thumb_regenerate: bool = False,
        single_process: bool = False,
    ) -> ManualSession:
        """Validate input and return a GUI-neutral Manual Tool session."""
        if not frames:
            raise ValueError("Frames input is required")
        input_path = Path(os.path.expanduser(frames)).resolve()
        if not input_path.exists():
            raise ValueError(f"Frames input does not exist: {input_path}")
        is_video = input_path.is_file() and input_path.suffix.lower() in _VIDEO_SUFFIXES
        if input_path.is_file() and not is_video:
            raise ValueError(f"Frames input is not a supported video file: {input_path}")
        frame_list = cls._discover_frames(input_path) if input_path.is_dir() else ()
        if input_path.is_dir() and not frame_list:
            raise ValueError(f"No supported image frames found in: {input_path}")
        if input_path.is_dir():
            cls._reject_extracted_faces(input_path)
        alignments = None
        if alignments_path:
            alignments_candidate = Path(os.path.expanduser(alignments_path)).resolve()
            alignments = str(alignments_candidate)
        return cls(
            frames=str(input_path),
            alignments_path=alignments,
            thumb_regenerate=thumb_regenerate,
            single_process=single_process,
            frame_list=frame_list,
            is_video_input=is_video,
        )

    @property
    def has_images(self) -> bool:
        """Return whether this session can expose image frames directly."""
        return bool(self.frame_list)

    @property
    def frame_count(self) -> int:
        """Return the number of directly discoverable image frames."""
        return len(self.frame_list)

    def frame_name_for_index(self, frame_index: int) -> str | None:
        """Return the on-disk frame name that maps to ``frame_index``.

        Used by Manual Tool persistence: image-folder sessions resolve through
        the source frame list (so sparse alignments still align to the source
        ordering), while video sessions synthesize the Faceswap-standard dummy
        frame name (``<video_basename>_<NNNNNN><ext>``, matching
        ``lib.image.ImagesLoader._dummy_video_frame_name``).

        Returns ``None`` when no valid name can be derived (e.g. negative index,
        image-folder index past the discovered frame list).
        """
        if frame_index < 0:
            return None
        if self.frame_list:
            if frame_index >= len(self.frame_list):
                return None
            return self.frame_list[frame_index].name
        if self.is_video_input:
            video_path = Path(self.frames)
            return f"{video_path.stem}_{frame_index + 1:06d}{video_path.suffix}"
        return None

    def alignments_handle(self) -> ManualAlignmentsHandle:
        """Resolve and return a GUI-neutral handle to the alignments file."""
        folder, filename = self._resolve_alignments_location()
        return ManualAlignmentsHandle(folder, filename, is_video=self.is_video_input)

    def video_metadata(self) -> ManualVideoMetadata | None:
        """Convenience accessor for video metadata via the alignments handle."""
        return self.alignments_handle().video_metadata()

    def faces_for_frame(self, frame_index: int) -> tuple[FaceThumbnail, ...]:
        """Convenience accessor for face thumbnails via the alignments handle."""
        return self.alignments_handle().faces_for_frame(frame_index)

    def has_thumbnails(self) -> bool:
        """Return whether thumbnails already exist in the alignments file."""
        return self.alignments_handle().has_thumbnails()

    def needs_thumbnail_regeneration(self) -> bool:
        """Return whether thumbnails must be generated before the editor loads."""
        return self.thumb_regenerate or not self.has_thumbnails()

    def create_editor_state(self) -> ManualEditorState:
        """Return a fresh GUI-neutral editor state for this session."""
        return ManualEditorState()

    def _resolve_alignments_location(self) -> tuple[str, str]:
        """Return (folder, filename) for the alignments file using legacy rules."""
        if self.alignments_path:
            folder, filename = os.path.split(self.alignments_path)
            return folder, filename
        if self.is_video_input:
            folder, vid = os.path.split(os.path.splitext(self.frames)[0])
            return folder, f"{vid}_{_ALIGNMENTS_FILENAME}"
        return self.frames, _ALIGNMENTS_FILENAME

    @staticmethod
    def _discover_frames(input_path: Path) -> tuple[ManualFrame, ...]:
        """Return sorted image frames from a folder."""
        paths = sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        )
        return tuple(
            ManualFrame(index=index, name=path.name, path=str(path))
            for index, path in enumerate(paths)
        )

    @staticmethod
    def _reject_extracted_faces(input_path: Path) -> None:
        """Refuse a folder of extracted Faceswap faces as Manual Tool input."""
        png_candidate = next(
            (path for path in input_path.iterdir() if path.suffix.lower() == ".png"),
            None,
        )
        if png_candidate is None:
            return
        from lib.image import read_image_meta

        try:
            meta = read_image_meta(str(png_candidate))
        except Exception:  # pragma: no cover - defensive
            logger.debug("Could not read PNG header for %s", png_candidate, exc_info=True)
            return
        if isinstance(meta, dict) and "itxt" in meta and "alignments" in meta["itxt"]:
            raise ValueError(
                f"Input folder contains extracted faces, not source frames: {input_path}"
            )

    @staticmethod
    def _string_value(value: object) -> str:
        """Normalize command values to strings."""
        if value is None or value is False:
            return ""
        if isinstance(value, (list, tuple)):
            return " ".join(str(item) for item in value)
        return str(value)


__all__ = get_module_objects(__name__)
