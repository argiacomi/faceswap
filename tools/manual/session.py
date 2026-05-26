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
    face_grid_mesh_visible: bool = False
    face_grid_mask_visible: bool = False
    is_playing: bool = False
    # Mask editor (#101) — brush + display state.  Defaults match the Tk
    # Manual Tool's first-launch values so muscle memory carries over.
    brush_size: int = 12
    brush_mode: str = "draw"
    brush_shape: str = "Circle"
    brush_color: str = "#ffffff"
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
        "face_grid_mesh_visible",
        "face_grid_mask_visible",
        "is_playing",
        "brush_size",
        "brush_mode",
        "brush_shape",
        "brush_color",
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
        # :attr:`MASK_STORED_SIZE`.
        self._mask_state: dict[tuple[int, int, str], T.Any] = {}
        # Keys the user *touched* (paint or clear) since the last persist.
        # Persistence re-encodes only these so an untouched mask keeps the
        # legacy landmark-aligned affine matrix it was originally saved
        # with — our axis-aligned bbox projection is fine for new edits
        # but would degrade an untouched mask if we encoded it blindly.
        self._painted_masks: set[tuple[int, int, str]] = set()
        # Pre-existing mask blobs from the on-disk alignments, surfaced
        # lazily via :meth:`get_mask` so the editor can preview / edit
        # masks that were persisted by a previous session.
        self._persisted_mask_blobs: dict[tuple[int, int, str], T.Any] = {}

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
        # Mask editor (#101): re-seed clears both the editor cache and the
        # tracking sets so a second seed doesn't smuggle in stale state
        # from a previous session.
        self._mask_state.clear()
        self._painted_masks.clear()
        self._persisted_mask_blobs.clear()
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
                # Mask editor (#101): record persisted blobs so the editor can
                # display the existing mask without forcing a re-encode on
                # save.  Decoding happens lazily through ``get_mask``.
                face_mask = getattr(face, "mask", None) or {}
                for mask_type, blob in face_mask.items():
                    self._persisted_mask_blobs[(frame_index, face_index, str(mask_type))] = blob
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

    def delete_faces(self, selections: T.Iterable[tuple[int, int]]) -> int:
        """Delete multiple ``(frame_index, face_index)`` pairs as one undoable edit.

        Deletions are applied in descending face-index order per frame so
        reindexing cannot shift a later removal target.  The undo closure
        restores the exact pre-delete face lists for every touched frame.
        """
        by_frame: dict[int, set[int]] = {}
        for frame_index, face_index in selections:
            by_frame.setdefault(int(frame_index), set()).add(int(face_index))
        valid: dict[int, tuple[int, ...]] = {}
        before: dict[int, list[EditableFace]] = {}
        for frame_index, face_indices in by_frame.items():
            faces = self._faces.get(frame_index, [])
            indices = tuple(
                sorted(
                    (index for index in face_indices if 0 <= index < len(faces)),
                    reverse=True,
                )
            )
            if not indices:
                continue
            valid[frame_index] = indices
            before[frame_index] = list(faces)
        if not valid:
            return 0

        def do() -> None:
            for frame_index, indices in valid.items():
                faces = self._faces.setdefault(frame_index, [])
                for face_index in indices:
                    if 0 <= face_index < len(faces):
                        faces.pop(face_index)
                self._reindex(frame_index)

        def undo() -> None:
            for frame_index, faces in before.items():
                self._faces[frame_index] = list(faces)
                self._reindex(frame_index)

        first_frame = min(valid)
        do()
        undo_frames = tuple(valid)
        self._undo.append((lambda: do(), lambda: undo(), first_frame))
        self._redo.clear()
        self._dirty_frames.update(undo_frames)
        for frame_index in undo_frames:
            self._notify(frame_index)
        return sum(len(indices) for indices in valid.values())

    def replace_frame_faces(
        self,
        frame_index: int,
        faces: T.Sequence[EditableFace],
        *,
        mask_state: T.Mapping[tuple[int, int, str], T.Any] | None = None,
        persisted_mask_blobs: T.Mapping[tuple[int, int, str], T.Any] | None = None,
    ) -> int:
        """Replace one frame's faces and mask metadata as one undoable edit."""
        frame_index = int(frame_index)
        previous_faces = list(self._faces.get(frame_index, ()))
        previous_masks = {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in self._mask_state.items()
            if key[0] == frame_index
        }
        previous_blobs = {
            key: value
            for key, value in self._persisted_mask_blobs.items()
            if key[0] == frame_index
        }
        new_faces = [self._normalize(index, face) for index, face in enumerate(faces)]
        new_masks = dict(mask_state or {})
        new_blobs = dict(persisted_mask_blobs or {})

        def clear_frame_masks() -> None:
            for key in tuple(self._mask_state):
                if key[0] == frame_index:
                    self._mask_state.pop(key, None)
            for key in tuple(self._persisted_mask_blobs):
                if key[0] == frame_index:
                    self._persisted_mask_blobs.pop(key, None)
            for key in tuple(self._painted_masks):
                if key[0] == frame_index:
                    self._painted_masks.discard(key)

        def do() -> None:
            self._faces[frame_index] = list(new_faces)
            self._reindex(frame_index)
            clear_frame_masks()
            self._mask_state.update(
                {
                    key: value.copy() if hasattr(value, "copy") else value
                    for key, value in new_masks.items()
                }
            )
            self._persisted_mask_blobs.update(new_blobs)

        def undo() -> None:
            self._faces[frame_index] = list(previous_faces)
            self._reindex(frame_index)
            clear_frame_masks()
            self._mask_state.update(
                {
                    key: value.copy() if hasattr(value, "copy") else value
                    for key, value in previous_masks.items()
                }
            )
            self._persisted_mask_blobs.update(previous_blobs)

        self._apply(do, undo, frame_index)
        return len(new_faces)

    def copied_frame_payload(
        self,
        source_frame_index: int,
        target_frame_index: int,
    ) -> tuple[
        tuple[EditableFace, ...],
        dict[tuple[int, int, str], T.Any],
        dict[tuple[int, int, str], T.Any],
    ]:
        """Return deep-ish face and mask payload copied from source to target."""
        faces = tuple(
            EditableFace(index, face.bbox, face.landmarks)
            for index, face in enumerate(self._faces.get(int(source_frame_index), ()))
        )
        mask_state: dict[tuple[int, int, str], T.Any] = {}
        for (_frame, face_index, mask_type), value in self._mask_state.items():
            if _frame == int(source_frame_index):
                mask_state[(int(target_frame_index), int(face_index), mask_type)] = (
                    value.copy() if hasattr(value, "copy") else value
                )
        persisted: dict[tuple[int, int, str], T.Any] = {}
        for (_frame, face_index, mask_type), value in self._persisted_mask_blobs.items():
            if _frame == int(source_frame_index):
                persisted[(int(target_frame_index), int(face_index), mask_type)] = value
        return (faces, mask_state, persisted)

    def restore_frame_from_handle(
        self,
        handle: ManualAlignmentsHandle,
        *,
        frame_index: int,
        frame_name: str | None,
    ) -> bool:
        """Restore one frame from saved alignments, dropping local edits."""
        frame_index = int(frame_index)
        saved_faces: list[EditableFace] = []
        saved_blobs: dict[tuple[int, int, str], T.Any] = {}
        if frame_name and handle.exists:
            alignments = handle.open()
            entry = alignments.data.get(frame_name)
            if entry is not None:
                for face_index, face in enumerate(entry.faces):
                    raw_landmarks = getattr(face, "landmarks_xy", None)
                    landmarks: tuple[tuple[float, float], ...] = ()
                    if raw_landmarks is not None:
                        try:
                            landmarks = tuple((float(x), float(y)) for x, y in raw_landmarks)
                        except (TypeError, ValueError):
                            landmarks = ()
                    saved_faces.append(
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
                    for mask_type, blob in (getattr(face, "mask", None) or {}).items():
                        saved_blobs[(frame_index, face_index, str(mask_type))] = blob
        previous_faces = list(self._faces.get(frame_index, ()))
        previous_blobs = {
            key: value
            for key, value in self._persisted_mask_blobs.items()
            if key[0] == frame_index
        }
        previous_masks = {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in self._mask_state.items()
            if key[0] == frame_index
        }
        if previous_faces == saved_faces and previous_blobs == saved_blobs and not previous_masks:
            return False
        self.replace_frame_faces(
            frame_index,
            saved_faces,
            persisted_mask_blobs=saved_blobs,
        )
        self._undo = [record for record in self._undo if record[2] != frame_index]
        self._redo = [record for record in self._redo if record[2] != frame_index]
        self._dirty_frames.discard(frame_index)
        self._notify(frame_index)
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

    def update_face_bbox_live(
        self,
        frame_index: int,
        face_index: int,
        bbox: tuple[float, float, float, float],
    ) -> bool:
        """Replace a face bbox during an in-progress drag without recording undo."""
        faces = self._faces.get(frame_index, [])
        if face_index < 0 or face_index >= len(faces):
            return False
        new_x, new_y, new_w, new_h = self._validate_bbox(bbox)
        previous = faces[face_index]
        prev_x, prev_y, prev_w, prev_h = previous.bbox
        if new_x == prev_x and new_y == prev_y and new_w == prev_w and new_h == prev_h:
            return True
        if previous.landmarks and new_w == prev_w and new_h == prev_h:
            dx = new_x - prev_x
            dy = new_y - prev_y
            landmarks = tuple((lx + dx, ly + dy) for lx, ly in previous.landmarks)
        elif prev_w > 0 and prev_h > 0 and previous.landmarks:
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
        faces[face_index] = EditableFace(
            face_index=face_index,
            bbox=(new_x, new_y, new_w, new_h),
            landmarks=landmarks,
        )
        self._dirty_frames.add(frame_index)
        self._notify(frame_index)
        return True

    def set_landmarks_live(
        self,
        frame_index: int,
        face_index: int,
        landmarks: T.Sequence[tuple[float, float]],
    ) -> bool:
        """Replace landmarks during a live drag without recording undo."""
        faces = self._faces.get(frame_index, [])
        if face_index < 0 or face_index >= len(faces):
            return False
        previous = faces[face_index]
        faces[face_index] = EditableFace(
            face_index=previous.face_index,
            bbox=previous.bbox,
            landmarks=tuple((float(x), float(y)) for x, y in landmarks),
        )
        self._dirty_frames.add(frame_index)
        self._notify(frame_index)
        return True

    def record_face_update(
        self,
        frame_index: int,
        face_index: int,
        previous: EditableFace,
    ) -> bool:
        """Record one undo entry for a live-updated face."""
        faces = self._faces.get(frame_index, [])
        if face_index < 0 or face_index >= len(faces):
            return False
        final = faces[face_index]
        if final == previous:
            return True

        def do() -> None:
            self._faces[frame_index][face_index] = final

        def undo() -> None:
            self._faces[frame_index][face_index] = previous

        self._undo.append((do, undo, frame_index))
        self._redo.clear()
        self._dirty_frames.add(frame_index)
        self._notify(frame_index)
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
        *,
        source_size: tuple[int, int] | None = None,
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
        if new_w < 20.0 or new_h < 20.0:
            return False
        cx = x + w / 2.0
        cy = y + h / 2.0
        new_bbox = (cx - new_w / 2.0, cy - new_h / 2.0, new_w, new_h)
        if source_size is not None:
            source_width, source_height = (float(source_size[0]), float(source_size[1]))
            if (
                new_bbox[0] < 0.0
                or new_bbox[1] < 0.0
                or new_bbox[0] + new_bbox[2] > source_width
                or new_bbox[1] + new_bbox[3] > source_height
            ):
                return False
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
        no mask has been created/painted *and* no persisted blob exists
        for this combination.  If a persisted blob exists from
        :meth:`seed_from_handle` it is decoded on first access and
        cached in :attr:`_mask_state` so subsequent calls are O(1).
        """
        key = (int(frame_index), int(face_index), str(mask_type))
        grid = self._mask_state.get(key)
        if grid is not None:
            return grid
        blob = self._persisted_mask_blobs.get(key)
        if blob is None:
            return None
        decoded = self.decode_mask_blob(blob)
        if decoded is None:
            return None
        # Cache the decoded grid so further reads hit memory; do NOT mark
        # this as a touched mask (so a save that never paints it leaves
        # the legacy blob intact).
        self._mask_state[key] = decoded
        return decoded

    def _decoded_or_blank_grid(
        self,
        key: tuple[int, int, str],
        size: int,
    ) -> T.Any:
        """Return the persisted decoded grid for ``key`` or a fresh zero grid.

        Centralises the "edit an existing persisted mask before any render
        path has triggered ``get_mask``" path used by ``paint_mask_stroke``
        and ``clear_mask``.  Falls back to a zero grid when no persisted
        blob exists or the blob can't be decoded.
        """
        import numpy as np

        blob = self._persisted_mask_blobs.get(key)
        if blob is not None:
            decoded = self.decode_mask_blob(blob)
            if decoded is not None:
                return decoded
        return np.zeros((size, size), dtype=np.uint8)

    def has_mask(self, frame_index: int, face_index: int, mask_type: str) -> bool:
        """Return whether a mask exists for ``(frame, face, type)``.

        ``True`` when an editable grid has been allocated *or* a persisted
        blob is available from the last seed.  Counts the legacy file's
        masks so the editor's dropdown can populate from disk even before
        the user paints anything.
        """
        key = (int(frame_index), int(face_index), str(mask_type))
        return key in self._mask_state or key in self._persisted_mask_blobs

    def known_mask_types(self, frame_index: int, face_index: int) -> tuple[str, ...]:
        """Return every mask type with editable OR persisted state for the face.

        Combines in-memory edits and the seed-time persisted-blob set so
        the Mask editor's dropdown surfaces masks that exist on disk
        even before the user has painted anything new.
        """
        types: set[str] = set()
        for fi, faci, mask_type in self._mask_state:
            if fi == int(frame_index) and faci == int(face_index):
                types.add(mask_type)
        for fi, faci, mask_type in self._persisted_mask_blobs:
            if fi == int(frame_index) and faci == int(face_index):
                types.add(mask_type)
        return tuple(sorted(types))

    def mask_affine_matrix(
        self,
        frame_index: int,
        face_index: int,
        mask_type: str,
    ) -> T.Any:
        """Return the source-frame to stored-mask affine matrix for a mask."""
        import cv2
        import numpy as np

        key = (int(frame_index), int(face_index), str(mask_type))
        blob = self._persisted_mask_blobs.get(key)
        if blob is not None:
            # Persisted MaskAlignmentsFile blobs store the opposite direction:
            # stored-mask coordinates -> source-frame coordinates. Painting
            # receives source-frame points, so invert the saved matrix before
            # transforming brush locations into stored-mask space.
            matrix = getattr(blob, "affine_matrix", None)
            if matrix is not None:
                matrix_array = np.asarray(matrix, dtype=np.float32)[:2]
                if matrix_array.shape == (2, 3):
                    return cv2.invertAffineTransform(matrix_array)
        faces = self._faces.get(frame_index, [])
        if face_index < 0 or face_index >= len(faces):
            return None
        return self._bbox_to_mask_affine(faces[face_index].bbox)

    def mask_original_roi(
        self,
        frame_index: int,
        face_index: int,
        mask_type: str,
    ) -> tuple[tuple[float, float], ...]:
        """Return the source-frame ROI polygon for ``mask_type``."""
        import cv2
        import numpy as np

        matrix = self.mask_affine_matrix(frame_index, face_index, mask_type)
        if matrix is None:
            return ()
        size = self.MASK_STORED_SIZE
        points = np.array(
            [[0, 0], [0, size - 1], [size - 1, size - 1], [size - 1, 0]],
            dtype=np.float32,
        ).reshape((-1, 1, 2))
        inverse = cv2.invertAffineTransform(np.asarray(matrix, dtype=np.float32)[:2])
        roi = cv2.transform(points, inverse).reshape((4, 2))
        return tuple((float(x), float(y)) for x, y in roi.tolist())

    def paint_mask_stroke(
        self,
        frame_index: int,
        face_index: int,
        mask_type: str,
        source_x: float,
        source_y: float,
        radius: float,
        value: int,
        *,
        previous_source_x: float | None = None,
        previous_source_y: float | None = None,
        shape: str = "Circle",
    ) -> bool:
        """Paint a brush stroke in stored-mask space using the mask affine.

        ``value`` is ``255`` to draw or ``0`` to erase.  Pixels are
        full-on/full-off — feathered brushes are a follow-up.  Returns
        ``False`` for an invalid face_index, out-of-ROI coordinates, or
        a non-positive radius.  Mutates the mask array in place and
        records undo/redo via a snapshot of the mask grid.
        """
        import cv2
        import numpy as np

        faces = self._faces.get(frame_index, [])
        if face_index < 0 or face_index >= len(faces):
            return False
        if not (radius > 0.0):
            return False
        matrix = self.mask_affine_matrix(frame_index, face_index, mask_type)
        if matrix is None:
            return False
        matrix_array = np.asarray(matrix, dtype=np.float32)[:2]
        points = [(float(source_x), float(source_y))]
        if previous_source_x is not None and previous_source_y is not None:
            points.insert(0, (float(previous_source_x), float(previous_source_y)))
        transformed = cv2.transform(
            np.asarray(points, dtype=np.float32).reshape(1, -1, 2), matrix_array
        )
        transformed = transformed.reshape((-1, 2))
        if not self._stored_point_in_bounds(transformed[-1]):
            return False
        size = self.MASK_STORED_SIZE
        stored_radius = max(1.0, float(radius) * self._mask_radius_scale(matrix_array))
        key = (int(frame_index), int(face_index), str(mask_type))
        existing = self._mask_state.get(key)
        if existing is None:
            # Editing a mask type that already exists on disk must start
            # from the decoded persisted grid, not a fresh zero grid — the
            # next save (which only re-encodes touched keys) would
            # otherwise replace the well-aligned legacy blob with the
            # single new brush stamp.  Fall back to zeros only when no
            # persisted blob exists for this combination.
            existing = self._decoded_or_blank_grid(key, size)
            self._mask_state[key] = existing
        prior = existing.copy()
        was_touched = key in self._painted_masks
        self._paint_mask_shape(
            existing,
            transformed,
            int(round(stored_radius)),
            255 if int(value) > 0 else 0,
            str(shape),
        )
        # Snapshot AFTER paint so redo replays the same write.
        new = existing.copy()

        def do() -> None:
            existing[:, :] = new
            self._painted_masks.add(key)

        def undo() -> None:
            existing[:, :] = prior
            self._set_mask_touched(key, was_touched)

        self._apply(do, undo, frame_index)
        return True

    def clear_mask(self, frame_index: int, face_index: int, mask_type: str) -> bool:
        """Drop the editable mask for ``(frame, face, type)``.

        Used as the explicit clear control (and by full-frame revert).
        Records undo/redo so a clear can be reversed.  Returns ``False``
        when no mask exists — either in memory *or* on disk for the
        given key.

        A persisted-but-not-yet-decoded blob is materialised first so the
        clear actually deletes the disk content the user can see.  The
        recorded undo entry restores both the decoded grid *and* the
        persisted blob entry so :meth:`get_mask` doesn't lazily re-
        materialise the original after an undo.
        """
        key = (int(frame_index), int(face_index), str(mask_type))
        existing = self._mask_state.get(key)
        persisted = self._persisted_mask_blobs.get(key)
        if existing is None:
            if persisted is None:
                return False
            decoded = self.decode_mask_blob(persisted)
            if decoded is None:
                return False
            self._mask_state[key] = decoded
            existing = decoded
        prior = existing.copy()
        had_persisted = persisted is not None
        was_touched = key in self._painted_masks

        def do() -> None:
            self._mask_state.pop(key, None)
            # Drop the persisted blob too so ``get_mask`` doesn't lazily
            # re-decode it.  Saving in this state leaves the corresponding
            # ``mask_dict`` key removed; the undo path below restores it.
            if had_persisted:
                self._persisted_mask_blobs.pop(key, None)
            self._painted_masks.add(key)

        def undo() -> None:
            self._mask_state[key] = prior
            if had_persisted:
                self._persisted_mask_blobs[key] = persisted
            self._set_mask_touched(key, was_touched)

        self._apply(do, undo, frame_index)
        return True

    def adopt_mask(
        self,
        frame_index: int,
        face_index: int,
        mask_type: str,
        grid: T.Any,
    ) -> None:
        """Install a pre-built mask grid without recording undo history.

        Used by the lazy decoder on first access so seeding a persisted
        mask doesn't pollute the user-facing undo stack.  ``grid`` must
        be a uint8 ``numpy.ndarray`` sized
        :attr:`MASK_STORED_SIZE` x :attr:`MASK_STORED_SIZE`.
        """
        import numpy as np

        if grid is None:
            return
        array = np.asarray(grid, dtype=np.uint8)
        if array.shape != (self.MASK_STORED_SIZE, self.MASK_STORED_SIZE):
            return
        key = (int(frame_index), int(face_index), str(mask_type))
        self._mask_state[key] = array

    @staticmethod
    def _stored_point_in_bounds(point: T.Any) -> bool:
        """Return whether a transformed stored-space point is paintable."""
        x, y = float(point[0]), float(point[1])
        return 0.0 <= x <= ManualEditableAlignments.MASK_STORED_SIZE - 1 and 0.0 <= y <= (
            ManualEditableAlignments.MASK_STORED_SIZE - 1
        )

    @staticmethod
    def _mask_radius_scale(matrix: T.Any) -> float:
        """Return source-pixel to stored-pixel scale from an affine matrix."""
        import numpy as np

        matrix_array = np.asarray(matrix, dtype=np.float32)
        scale_x = float((matrix_array[0, 0] ** 2 + matrix_array[1, 0] ** 2) ** 0.5)
        scale_y = float((matrix_array[0, 1] ** 2 + matrix_array[1, 1] ** 2) ** 0.5)
        return max(1.0, (scale_x + scale_y) / 2.0)

    @staticmethod
    def _paint_mask_shape(
        image: T.Any,
        points: T.Any,
        radius: int,
        value: int,
        shape: str,
    ) -> None:
        """Paint a circle or rectangle stamp/line into ``image`` in-place."""
        import cv2
        import numpy as np

        radius = max(1, int(radius))
        int_points = np.rint(points).astype("int32")
        if str(shape).lower() == "rectangle":
            if len(int_points) == 1:
                point = int_points[0]
                cv2.rectangle(
                    image,
                    (int(point[0] - radius), int(point[1] - radius)),
                    (int(point[0] + radius), int(point[1] + radius)),
                    int(value),
                    thickness=-1,
                )
                return
            start, end = int_points[0], int_points[-1]
            distance = max(1, int(np.linalg.norm(end - start)))
            for step in range(distance + 1):
                t = step / distance
                point = np.rint(start + (end - start) * t).astype("int32")
                cv2.rectangle(
                    image,
                    (int(point[0] - radius), int(point[1] - radius)),
                    (int(point[0] + radius), int(point[1] + radius)),
                    int(value),
                    thickness=-1,
                )
            return
        if len(int_points) == 1:
            point = int_points[0]
            cv2.circle(image, (int(point[0]), int(point[1])), radius, int(value), thickness=-1)
            return
        start, end = int_points[0], int_points[-1]
        cv2.line(
            image,
            (int(start[0]), int(start[1])),
            (int(end[0]), int(end[1])),
            int(value),
            max(1, radius * 2),
        )

    # ---- Mask blob persistence (#101) ----

    MASK_STORED_CENTERING: T.ClassVar[str] = "head"
    """Centering of the stored-space mask grid.  Faceswap masks are stored at
    head centering by convention; the Qt editor follows the same default so a
    round-trip with the production alignment pipeline stays compatible."""

    @classmethod
    def encode_mask_blob(
        cls,
        grid: T.Any,
        bbox: tuple[float, float, float, float],
    ) -> T.Any:
        """Encode a stored-space mask grid into a :class:`MaskAlignmentsFile`.

        Compresses the uint8 grid with zlib and computes the 2x3 affine
        that maps stored-space coordinates back to source-frame pixels via
        the axis-aligned bbox projection ``paint_mask_stroke`` uses on the
        way in.  Returns a :class:`lib.align.objects.MaskAlignmentsFile`
        that can be slotted directly into ``face.mask[mask_type]`` so
        :meth:`lib.align.Alignments.save` writes it out in the legacy
        format.
        """
        import zlib

        import numpy as np

        from lib.align.objects import MaskAlignmentsFile

        size = cls.MASK_STORED_SIZE
        array = np.asarray(grid, dtype=np.uint8)
        if array.shape != (size, size):
            # Defensive: reshape or pad so the writer never receives a
            # mismatched grid.  An invalid grid is replaced with zeros so
            # save still produces a valid blob.
            array = np.zeros((size, size), dtype=np.uint8)
        x, y, w, h = bbox
        affine = cls._bbox_to_frame_affine((x, y, w, h))
        # cv2.INTER_LINEAR is 1; we don't import cv2 here to keep this module
        # tkinter/Qt-neutral and import-light.  The legacy Tk path stores the
        # cv2 interpolator constant so we mirror the numeric value directly.
        interpolator = 1
        return MaskAlignmentsFile(
            mask=zlib.compress(array.tobytes()),
            affine_matrix=affine,
            interpolator=interpolator,
            stored_size=size,
            stored_centering=cls.MASK_STORED_CENTERING,
        )

    @classmethod
    def _bbox_to_mask_affine(
        cls,
        bbox: tuple[float, float, float, float],
    ) -> T.Any:
        """Return source-frame to stored-mask affine for an axis-aligned bbox."""
        import numpy as np

        x, y, w, h = bbox
        size = cls.MASK_STORED_SIZE
        scale_x = float(size) / max(1.0, float(w))
        scale_y = float(size) / max(1.0, float(h))
        return np.array(
            [
                [scale_x, 0.0, -float(x) * scale_x],
                [0.0, scale_y, -float(y) * scale_y],
            ],
            dtype=np.float32,
        )

    @classmethod
    def _bbox_to_frame_affine(
        cls,
        bbox: tuple[float, float, float, float],
    ) -> T.Any:
        """Return stored-mask to source-frame affine for an axis-aligned bbox."""
        import numpy as np

        x, y, w, h = bbox
        size = cls.MASK_STORED_SIZE
        scale_x = float(w) / float(size)
        scale_y = float(h) / float(size)
        return np.array(
            [
                [scale_x, 0.0, float(x)],
                [0.0, scale_y, float(y)],
            ],
            dtype=np.float32,
        )

    @classmethod
    def decode_mask_blob(cls, blob: T.Any) -> T.Any:
        """Decode a :class:`MaskAlignmentsFile` back to a uint8 grid.

        Returns ``None`` when the blob is missing or malformed so the
        caller can fall back to "no mask painted yet".  Decoded grids
        are sized :attr:`MASK_STORED_SIZE` x :attr:`MASK_STORED_SIZE`;
        blobs with a different ``stored_size`` are rejected (the editor
        only paints at the canonical size today; a follow-up can resize
        smaller / larger blobs on the way in).
        """
        import zlib

        import numpy as np

        if blob is None:
            return None
        size = cls.MASK_STORED_SIZE
        stored_size = getattr(blob, "stored_size", size)
        if int(stored_size) != size:
            return None
        raw = getattr(blob, "mask", None)
        if raw is None:
            return None
        try:
            decompressed = zlib.decompress(bytes(raw))
        except (zlib.error, TypeError):
            return None
        expected = size * size
        if len(decompressed) != expected:
            return None
        return np.frombuffer(decompressed, dtype=np.uint8).reshape(size, size).copy()

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
        self._painted_masks.clear()

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
                    self._apply_editable_masks(prev.mask, frame_index, face.face_index, face.bbox)
                    new_faces.append(prev)
                else:
                    if prev is not None:
                        merged_mask = dict(prev.mask) if prev.mask else {}
                        self._apply_editable_masks(
                            merged_mask, frame_index, face.face_index, face.bbox
                        )
                        new_faces.append(
                            FileAlignments(
                                x=x,
                                y=y,
                                w=w,
                                h=h,
                                landmarks_xy=landmarks_array,
                                mask=merged_mask,
                                identity=prev.identity,
                                metadata=prev.metadata,
                                thumb=None,
                            )
                        )
                    else:
                        fresh_mask: dict[str, T.Any] = {}
                        self._apply_editable_masks(
                            fresh_mask, frame_index, face.face_index, face.bbox
                        )
                        new_faces.append(
                            FileAlignments(
                                x=x,
                                y=y,
                                w=w,
                                h=h,
                                landmarks_xy=landmarks_array,
                                mask=fresh_mask,
                            )
                        )
            entry.faces = new_faces
            modified += 1
        return modified

    def _apply_editable_masks(
        self,
        mask_dict: dict[str, T.Any],
        frame_index: int,
        face_index: int,
        bbox: tuple[float, float, float, float],
    ) -> None:
        """Encode every *user-touched* mask grid into ``mask_dict``.

        Mutates ``mask_dict`` in place.  Only entries the user explicitly
        painted or cleared (tracked in :attr:`_painted_masks`) are
        rewritten; untouched masks keep their on-disk blob so we don't
        downgrade a legacy landmark-aligned affine matrix with our
        axis-aligned bbox approximation.  A touched-and-now-empty grid
        deletes the corresponding key from ``mask_dict`` so the user can
        fully erase a mask through the editor.
        """
        import numpy as np

        for key, blob in self._persisted_mask_blobs.items():
            if key[0] == int(frame_index) and key[1] == int(face_index):
                mask_dict.setdefault(key[2], blob)
        touched_for_face = [
            key
            for key in self._painted_masks
            if key[0] == int(frame_index) and key[1] == int(face_index)
        ]
        for key in touched_for_face:
            mask_type = key[2]
            grid = self._mask_state.get(key)
            if grid is None or not np.any(grid):
                mask_dict.pop(mask_type, None)
            else:
                mask_dict[mask_type] = self.encode_mask_blob(grid, bbox)
            self._painted_masks.discard(key)

    def _set_mask_touched(self, key: tuple[int, int, str], touched: bool) -> None:
        """Restore ``_painted_masks`` membership for undo/redo bookkeeping."""
        if touched:
            self._painted_masks.add(key)
        else:
            self._painted_masks.discard(key)

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
