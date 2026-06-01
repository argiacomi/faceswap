#!/usr/bin/env python3
"""Tools for manipulating the alignments serialized file"""

from __future__ import annotations

import logging
import os
import re
import sys
import typing as T
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.optimize import linear_sum_assignment
from sklearn import decomposition
from tqdm import tqdm

from lib.logger import parse_class_init
from lib.serializer import get_serializer
from lib.utils import FaceswapError, get_module_objects

from .jobs_faces import FaceToFile
from .media import Faces, Frames

if T.TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Generator

    from lib.align.objects import FileAlignments, PNGHeader

    from .media import AlignmentData

logger = logging.getLogger(__name__)


class Check:
    """Frames and faces checking tasks.

    Parameters
    ---------
    alignments
        The loaded alignments corresponding to the frames to be annotated
    arguments
        The command line arguments that have called this job
    """

    def __init__(self, alignments: AlignmentData, arguments: Namespace) -> None:
        logger.debug(parse_class_init(locals()))
        self._alignments = alignments
        self._job = arguments.job
        self._type: T.Literal["faces", "frames"] | None = None
        self._is_video = False  # Set when getting items
        self._output = arguments.output
        self._source_dir = self._get_source_dir(arguments)
        self._validate()
        self._items = self._get_items()

        self.output_message = ""
        logger.debug("Initialized %s", self.__class__.__name__)

    def _get_source_dir(self, arguments: Namespace) -> str:
        """Set the correct source folder

        Parameters
        ----------
        arguments
            The command line arguments for the Alignments tool

        Returns
        -------
        Full path to the source folder
        """
        if (
            hasattr(arguments, "faces_dir")
            and arguments.faces_dir
            and hasattr(arguments, "frames_dir")
            and arguments.frames_dir
        ):
            logger.error("Only select a source frames (-fr) or source faces (-fc) folder")
            sys.exit(1)
        elif hasattr(arguments, "faces_dir") and arguments.faces_dir:
            self._type = "faces"
            source_dir = arguments.faces_dir
        elif hasattr(arguments, "frames_dir") and arguments.frames_dir:
            self._type = "frames"
            source_dir = arguments.frames_dir
        else:
            logger.error("No source folder (-fr or -fc) was provided")
            sys.exit(1)
        logger.debug("type: '%s', source_dir: '%s'", self._type, source_dir)
        return source_dir  # type: ignore[no-any-return]

    def _get_items(self) -> list[dict[str, str]] | list[tuple[str, PNGHeader]]:
        """Set the correct items to process

        Returns
        -------
        Sorted list of dictionaries for either faces or frames. If faces the dictionaries
        have the current filename as key, with the header source data as value. If frames
        the dictionaries will contain the keys 'frame_fullname', 'frame_name', 'extension'.
        """
        assert self._type is not None
        items: Frames | Faces = globals()[self._type.title()](self._source_dir)
        self._is_video = items.is_video
        return T.cast(list[dict[str, str]] | list[tuple[str, "PNGHeader"]], items.file_list_sorted)  # type: ignore[redundant-cast]

    def process(self) -> None:
        """Process the frames check against the alignments file"""
        assert self._type is not None
        logger.info("[CHECK %s]", self._type.upper())
        items_output = self._compile_output()

        if self._type == "faces":
            filelist = T.cast(list[tuple[str, "PNGHeader"]], self._items)
            check_update = FaceToFile(self._alignments, [val[1] for val in filelist])
            if check_update():
                self._alignments.save()

        self._output_results(items_output)

    def _validate(self) -> None:
        """Check that the selected type is valid for selected task and job"""
        if self._job == "missing-frames" and self._output == "move":
            logger.warning(
                "Missing_frames was selected with move output, but there will "
                "be nothing to move. Defaulting to output: console"
            )
            self._output = "console"
        if self._type == "faces" and self._job != "multi-faces":
            logger.error(
                "The selected folder is not valid. Faces folder (-fc) is only "
                "supported for 'multi-faces'"
            )
            sys.exit(1)

    def _compile_output(self) -> list[str] | list[tuple[str, int]]:
        """Compile list of frames that meet criteria

        Returns
        -------
        List of filenames or filenames and face indices for the selected criteria
        """
        action = self._job.replace("-", "_")
        processor = getattr(self, f"_get_{action}")
        logger.debug("Processor: %s", processor)
        return [item for item in processor()]  # pylint:disable=unnecessary-comprehension

    def _get_no_faces(self) -> Generator[str, None, None]:
        """yield each frame that has no face match in alignments file

        Yields
        ------
        The frame name of any frames which have no faces
        """
        self.output_message = "Frames with no faces"
        for frame in tqdm(
            T.cast(list[dict[str, str]], self._items),
            desc=self.output_message,
            leave=False,
        ):
            logger.trace(frame)  # type:ignore
            frame_name = frame["frame_fullname"]
            if not self._alignments.frame_has_faces(frame_name):
                logger.debug("Returning: '%s'", frame_name)
                yield frame_name

    def _get_multi_faces(
        self,
    ) -> Generator[str, None, None] | Generator[tuple[str, int], None, None]:
        """yield each frame or face that has multiple faces matched in alignments file

        Yields
        ------
        The frame name of any frames which have multiple faces and potentially the face id
        """
        process_type = getattr(self, f"_get_multi_faces_{self._type}")
        yield from process_type()

    def _get_multi_faces_frames(self) -> Generator[str, None, None]:
        """Return Frames that contain multiple faces

        Yields
        ------
        The frame name of any frames which have multiple faces
        """
        self.output_message = "Frames with multiple faces"
        for item in tqdm(
            T.cast(list[dict[str, str]], self._items),
            desc=self.output_message,
            leave=False,
        ):
            filename = item["frame_fullname"]
            if not self._alignments.frame_has_multiple_faces(filename):
                continue
            logger.trace("Returning: '%s'", filename)  # type:ignore
            yield filename

    def _get_multi_faces_faces(self) -> Generator[tuple[str, int], None, None]:
        """Return Faces when there are multiple faces in a frame

        Yields
        ------
        The frame name and the face id of any frames which have multiple faces
        """
        self.output_message = "Multiple faces in frame"
        for item in tqdm(
            T.cast(list[tuple[str, "PNGHeader"]], self._items),
            desc=self.output_message,
            leave=False,
        ):
            src = item[1].source
            if not self._alignments.frame_has_multiple_faces(src.source_filename):
                continue
            retval = (item[0], src.face_index)
            logger.trace("Returning: '%s'", retval)  # type:ignore
            yield retval

    def _get_missing_alignments(self) -> Generator[str, None, None]:
        """yield each frame that does not exist in alignments file

        Yields
        ------
        The frame name of any frames missing alignments
        """
        self.output_message = "Frames missing from alignments file"
        exclude_filetypes = set(["yaml", "yml", "p", "json", "txt"])
        for frame in tqdm(
            T.cast(list[dict[str, str]], self._items),
            desc=self.output_message,
            leave=False,
        ):
            frame_name = frame["frame_fullname"]
            if frame[
                "frame_extension"
            ] not in exclude_filetypes and not self._alignments.frame_exists(frame_name):
                logger.debug("Returning: '%s'", frame_name)
                yield frame_name

    def _get_missing_frames(self) -> Generator[str, None, None]:
        """yield each frame in alignments that does not have a matching file

        Yields
        ------
        The frame name of any frames in alignments with no matching file
        """
        self.output_message = "Missing frames that are in alignments file"
        frames = set(item["frame_fullname"] for item in T.cast(list[dict[str, str]], self._items))
        for frame in tqdm(self._alignments.data.keys(), desc=self.output_message, leave=False):
            if frame not in frames:
                logger.debug("Returning: '%s'", frame)
                yield frame

    def _output_results(self, items_output: list[str] | list[tuple[str, int]]) -> None:
        """Output the results in the requested format

        Parameters
        ----------
        items_output
            The list of frame names, and potentially face ids, of any items which met the
            selection criteria
        """
        logger.trace("items_output: %s", items_output)  # type:ignore
        if self._output == "move" and self._is_video and self._type == "frames":
            logger.warning(
                "Move was selected with an input video. This is not possible so "
                "falling back to console output"
            )
            self._output = "console"
        if not items_output:
            logger.info("No %s were found meeting the criteria", self._type)
            return
        if self._output == "move":
            self._move_file(items_output)
            return
        if self._job == "multi-faces" and self._type == "faces":
            # Strip the index for printed/file output
            final_output = [item[0] for item in items_output]
        else:
            final_output = T.cast(list[str], items_output)
        output_message = "-----------------------------------------------\r\n"
        output_message += f" {self.output_message} ({len(final_output)})\r\n"
        output_message += "-----------------------------------------------\r\n"
        output_message += "\r\n".join(final_output)
        if self._output == "console":
            for line in output_message.splitlines():
                logger.info(line)
        if self._output == "file":
            self.output_file(output_message, len(final_output))

    def _get_output_folder(self) -> str:
        """Return output folder. Needs to be in the root if input is a video and processing
        frames

        Returns
        -------
        Full path to the output folder
        """
        if self._is_video and self._type == "frames":
            return os.path.dirname(self._source_dir)
        return self._source_dir

    def _get_filename_prefix(self) -> str:
        """Video name needs to be prefixed to filename if input is a video and processing frames

        Returns
        -------
        The common filename prefix to use
        """
        if self._is_video and self._type == "frames":
            return f"{os.path.basename(self._source_dir)}_"
        return ""

    def output_file(self, output_message: str, items_discovered: int) -> None:
        """Save the output to a text file in the frames directory

        Parameters
        ----------
        output_message
            The message to write out to file
        items_discovered
            The number of items which matched the criteria
        """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst_dir = self._get_output_folder()
        filename = (
            f"{self._get_filename_prefix()}{self.output_message.replace(' ', '_').lower()}"
            f"_{now}.txt"
        )
        output_file = os.path.join(dst_dir, filename)
        logger.info("Saving %s result(s) to '%s'", items_discovered, output_file)
        with open(output_file, "w", encoding="utf8") as f_output:
            f_output.write(output_message)

    def _move_file(self, items_output: list[str] | list[tuple[str, int]]) -> None:
        """Move the identified frames to a new sub folder

        Parameters
        ----------
        items_output
            List of items to move
        """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = (
            f"{self._get_filename_prefix()}{self.output_message.replace(' ', '_').lower()}_{now}"
        )
        dst_dir = self._get_output_folder()
        output_folder = os.path.join(dst_dir, folder_name)
        logger.debug("Creating folder: '%s'", output_folder)
        os.makedirs(output_folder)
        move = getattr(self, f"_move_{self._type}")
        logger.debug("Move function: %s", move)
        move(output_folder, items_output)

    def _move_frames(self, output_folder: str, items_output: list[str]) -> None:
        """Move frames into single sub folder

        Parameters
        ----------
        output_folder
            The folder to move the output to
        items_output
            List of items to move
        """
        logger.info("Moving %s frame(s) to '%s'", len(items_output), output_folder)
        for frame in items_output:
            src = os.path.join(self._source_dir, frame)
            dst = os.path.join(output_folder, frame)
            logger.debug("Moving: '%s' to '%s'", src, dst)
            os.rename(src, dst)

    def _move_faces(self, output_folder: str, items_output: list[tuple[str, int]]) -> None:
        """Make additional sub folders for each face that appears Enables easier manual sorting

        Parameters
        ----------
        output_folder
            The folder to move the output to
        items_output
            List of items and face indices to move
        """
        logger.info("Moving %s faces(s) to '%s'", len(items_output), output_folder)
        for frame, idx in items_output:
            src = os.path.join(self._source_dir, frame)
            dst_folder = os.path.join(output_folder, str(idx)) if idx != -1 else output_folder
            if not os.path.isdir(dst_folder):
                logger.debug("Creating folder: '%s'", dst_folder)
                os.makedirs(dst_folder)
            dst = os.path.join(dst_folder, frame)
            logger.debug("Moving: '%s' to '%s'", src, dst)
            os.rename(src, dst)


class Export:
    """Export alignments from a Faceswap .fsa file to a json formatted file.

    Parameters
    ----------
    alignments
        The alignments data loaded from an alignments file for this rename job
    arguments
        The :mod:`argparse` arguments as passed in from :mod:`tools.py`. Unused
    """

    def __init__(self, alignments: AlignmentData, arguments: Namespace) -> None:  # pylint:disable=unused-argument
        logger.debug(parse_class_init(locals()))
        self._alignments = alignments
        self._serializer = get_serializer("json")
        self._output_file = self._get_output_file()
        logger.debug("Initialized %s", self.__class__.__name__)

    def _get_output_file(self) -> str:
        """Obtain the name of an output file. If a file of the request name exists, then append a
        digit to the end until a unique filename is found

        Returns
        -------
        Full path to an output json file
        """
        in_file = self._alignments.file
        base_filename = f"{os.path.splitext(in_file)[0]}"
        out_file = f"{base_filename}.json"
        idx = 1
        while True:
            if not os.path.exists(out_file):
                break
            logger.debug("Output file exists: '%s'", out_file)
            out_file = f"{base_filename}_{idx}.json"
            idx += 1
        logger.debug("Setting output file to '%s'", out_file)
        return out_file

    @classmethod
    def _format_face(cls, face: FileAlignments) -> dict[str, list[int] | list[list[float]]]:
        """Format the relevant keys from an alignment file's face into the correct format for
        export/import

        Parameters
        ----------
        face
            The alignment dictionary for a face to process

        Returns
        -------
        The face formatted for exporting to a json file
        """
        lms = face.landmarks_xy
        assert isinstance(lms, list)
        box = [
            int(round(face.x, 0)),
            int(round(face.y, 0)),
            int(round(face.x + face.w, 0)),
            int(round(face.y + face.h, 0)),
        ]
        retval = T.cast(
            dict[str, list[int] | list[list[float]]],
            {"detected": box, "landmarks_2d": lms},
        )
        return retval

    def process(self) -> None:
        """Parse the imported alignments file and output relevant information to a json file"""
        logger.info("[EXPORTING ALIGNMENTS]")  # Tidy up cli output
        formatted = {
            key: [self._format_face(face) for face in val.faces]
            for key, val in self._alignments.data.items()
        }
        logger.info("Saving export alignments to '%s'...", self._output_file)
        self._serializer.save(self._output_file, formatted)


class Sort:
    """Sort alignments' index by the order they appear in an image in left to right order.

    Parameters
    ----------
    alignments
        The alignments data loaded from an alignments file for this rename job
    arguments
        The :mod:`argparse` arguments as passed in from :mod:`tools.py`. Unused
    """

    def __init__(self, alignments: AlignmentData, arguments: Namespace) -> None:  # pylint:disable=unused-argument
        logger.debug(parse_class_init(locals()))
        self._alignments = alignments
        logger.debug("Initialized %s", self.__class__.__name__)

    def process(self) -> None:
        """Execute the sort process"""
        logger.info("[SORT INDEXES]")  # Tidy up cli output
        reindexed = self.reindex_faces()
        if reindexed:
            self._alignments.save()
            logger.warning(
                "If you have a face-set corresponding to the alignment file you "
                "processed then you should run the 'Extract' job to regenerate it."
            )

    def reindex_faces(self) -> int:
        """Re-Index the faces

        Returns
        -------
        The count of re-indexed faces
        """
        reindexed = 0
        for alignment in tqdm(
            self._alignments.yield_faces(),
            desc="Sort alignment indexes",
            total=self._alignments.frames_count,
            leave=False,
        ):
            frame, alignments, count, key = alignment
            if count <= 1:
                logger.trace("0 or 1 face in frame. Not sorting: '%s'", frame)  # type:ignore
                continue
            sorted_alignments = sorted(alignments, key=lambda a: a.x)
            if sorted_alignments == alignments:
                logger.trace(  # type: ignore[attr-defined]
                    "Alignments already in correct order. Not sorting: '%s'",
                    frame,
                )
                continue
            logger.trace("Sorting alignments for frame: '%s'", frame)  # type:ignore
            self._alignments.data[key].faces = sorted_alignments
            reindexed += 1
        logger.info("%s Frames had their faces reindexed", reindexed)
        return reindexed


_SPATIAL_TRACK_REJECT_COST = 1_000_000.0
_SPATIAL_MAX_TRACK_GAP = 10
_SPATIAL_MAX_IDENTITY_DISTANCE = 0.45
_SPATIAL_STRONG_IDENTITY_DISTANCE = 0.10
_SPATIAL_MAX_WEAK_IDENTITY_JUMP = 3.0


@dataclass
class _SpatialFaceObservation:
    """One detected face instance in one frame for identity-aware smoothing."""

    frame_key: str
    frame_number: int
    face_index: int
    landmarks: np.ndarray
    embedding: np.ndarray
    center_x: float
    center_y: float
    width: float
    height: float
    frame_half: str | None = None
    same_identity_duplicate: bool = False


@dataclass
class _SpatialIdentityTrack:
    """A persistent face instance track across frames."""

    track_id: int
    observations: list[_SpatialFaceObservation]

    @property
    def last(self) -> _SpatialFaceObservation:
        """Return most recent observation in this track."""
        return self.observations[-1]

    @property
    def embedding(self) -> np.ndarray:
        """Return normalized running mean identity embedding for this track."""
        values = np.stack([obs.embedding for obs in self.observations], axis=0)
        mean = values.mean(axis=0).astype(np.float32, copy=False)
        norm = np.linalg.norm(mean)
        if norm == 0:
            return T.cast(np.ndarray, mean)
        return T.cast(np.ndarray, mean / norm)


class Spatial:
    """Apply spatial temporal filtering to landmarks

    Parameters
    ----------
    alignments
        The alignments data loaded from an alignments file for this rename job
    arguments
        The :mod:`argparse` arguments as passed in from :mod:`tools.py`

    Reference
    ---------
    https://www.kaggle.com/selfishgene/animating-and-smoothing-3d-facial-keypoints/notebook
    """

    def __init__(self, alignments: AlignmentData, arguments: Namespace) -> None:
        logger.debug(parse_class_init(locals()))
        self.arguments = arguments
        self._alignments = alignments
        self._mappings: dict[int, str | tuple[str, int]] = {}
        self._normalized: dict[str, np.ndarray] = {}
        self._shapes_model: decomposition.PCA | None = None
        logger.debug("Initialized %s", self.__class__.__name__)

    def process(self) -> None:
        """Perform spatial filtering."""
        logger.info("[SPATIAL-TEMPORAL FILTERING]")  # Tidy up cli output

        if self._has_identity_embeddings():
            logger.info("Identity embeddings found. Smoothing by persistent face instance tracks.")
            if not self._process_by_identity_tracks():
                logger.info(
                    "Identity embeddings were present, but no contiguous identity segments "
                    "were long enough to smooth. Skipping face-index fallback."
                )
            return

        self._process_by_face_index()

    def _process_by_face_index(self) -> None:
        """Run the legacy slot-based smoothing fallback."""
        logger.info(
            "NB: Falling back to face-index smoothing. For best results only run this "
            "when face ordering is consistent across frames and all false positives have been "
            "removed"
        )

        smoothed = 0
        for face_index in range(self._max_face_count()):
            self._mappings = {}
            self._normalized = {}
            self._shapes_model = None

            if not self._normalize(face_index):
                continue

            self._shape_model()
            landmarks = self._spatially_filter()
            landmarks = self._temporally_smooth(landmarks)
            self._update_alignments(landmarks, face_index)
            smoothed += 1

        if not smoothed:
            logger.info("No faces were found to spatially smooth")
            return

        self._save_smoothed_alignments()

    def _process_by_identity_tracks(self) -> bool:
        """Run identity-assisted instance-track smoothing.

        Returns
        -------
        bool
            ``True`` if any identity track was smoothed, otherwise ``False``.
        """
        tracks = self._build_identity_instance_tracks()
        if not tracks:
            logger.info("No identity tracks were built")
            return False

        smoothed = 0
        for track in tracks:
            for segment in self._split_track_segments(track):
                self._mappings = {}
                self._normalized = {}
                self._shapes_model = None

                if not self._normalize_track(segment):
                    continue

                self._shape_model()
                landmarks = self._spatially_filter()
                landmarks = self._temporally_smooth(landmarks)
                self._update_track_alignments(landmarks)
                smoothed += 1

        if not smoothed:
            logger.info(
                "No contiguous identity track segments were long enough to spatially smooth"
            )
            return False

        self._save_smoothed_alignments()
        return True

    def _save_smoothed_alignments(self) -> None:
        """Persist smoothed alignments and warn about regenerated facesets."""
        self._alignments.save()
        logger.warning(
            "If you have a face-set corresponding to the alignment file you "
            "processed then you should run the 'Extract' job to regenerate it."
        )

    def _max_face_count(self) -> int:
        """Return the maximum number of faces found in any frame.

        Returns
        -------
        Maximum number of faces in a single frame
        """
        retval = max((len(val.faces) for val in self._alignments.data.values()), default=0)
        logger.debug("Max face count: %s", retval)
        return retval

    def _has_identity_embeddings(self) -> bool:
        """Return ``True`` if any face has a usable identity embedding."""
        return any(
            self._embedding_for_face(face) is not None
            for frame in self._alignments.data.values()
            for face in frame.faces
        )

    @staticmethod
    def _frame_number_for_key(frame_key: str, fallback: int) -> int:
        """Return source frame number from a frame key when available.

        Alignment keys are commonly image filenames such as ``video_000123.png`` or
        ``frame_001.png``. For gap-aware identity smoothing, use that numeric suffix so missing
        detections do not get treated as adjacent samples. If no trailing number exists, fall back
        to dense sorted-key order.
        """
        stem = Path(frame_key).stem
        match = re.search(r"(\d+)$", stem)
        if match is None:
            return fallback
        return int(match.group(1))

    def _build_identity_instance_tracks(self) -> list[_SpatialIdentityTrack]:
        """Build persistent face-instance tracks from identity and motion continuity.

        This is intentionally instance-based, not pure identity clustering. Multiple tracks may
        share the same identity embedding, which is required for same-person duplicate cases such
        as side-by-side stereo frames.
        """
        tracks: list[_SpatialIdentityTrack] = []
        next_track_id = 0

        for fallback_frame_number, frame_key in enumerate(sorted(self._alignments.data.keys())):
            frame_number = self._frame_number_for_key(frame_key, fallback_frame_number)
            observations = self._observations_for_frame(frame_key, frame_number)
            if not observations:
                continue

            if not tracks:
                for observation in observations:
                    tracks.append(_SpatialIdentityTrack(next_track_id, [observation]))
                    next_track_id += 1
                continue

            active_tracks = [
                track
                for track in tracks
                if frame_number - track.last.frame_number <= _SPATIAL_MAX_TRACK_GAP
            ]

            matched_observations: set[int] = set()

            if active_tracks:
                cost_matrix: np.ndarray = np.full(
                    (len(active_tracks), len(observations)),
                    _SPATIAL_TRACK_REJECT_COST,
                    dtype=np.float32,
                )

                for row, track in enumerate(active_tracks):
                    for col, observation in enumerate(observations):
                        cost_matrix[row, col] = self._track_match_cost(track, observation)

                rows, cols = linear_sum_assignment(cost_matrix)
                for row, col in zip(rows, cols, strict=False):
                    cost = float(cost_matrix[row, col])
                    if cost >= _SPATIAL_TRACK_REJECT_COST:
                        continue
                    active_tracks[row].observations.append(observations[col])
                    matched_observations.add(col)

            for col, observation in enumerate(observations):
                if col in matched_observations:
                    continue
                tracks.append(_SpatialIdentityTrack(next_track_id, [observation]))
                next_track_id += 1

        logger.info("Built %s identity-aware face instance track(s)", len(tracks))
        return tracks

    @staticmethod
    def _split_track_segments(
        track: _SpatialIdentityTrack,
        *,
        max_gap: int = 1,
    ) -> list[_SpatialIdentityTrack]:
        """Split an identity track into gap-contiguous smoothing segments.

        The PCA + temporal moving-average smoother expects a dense sequence. If a face disappears
        for one or more frames, the observed landmarks on either side of that gap must not be
        smoothed as adjacent samples.
        """
        segments: list[_SpatialIdentityTrack] = []
        current: list[_SpatialFaceObservation] = []

        for observation in track.observations:
            if not current:
                current = [observation]
                continue

            gap = observation.frame_number - current[-1].frame_number
            if gap > max_gap:
                segments.append(_SpatialIdentityTrack(track.track_id, current))
                current = [observation]
                continue

            current.append(observation)

        if current:
            segments.append(_SpatialIdentityTrack(track.track_id, current))

        return segments

    def _observations_for_frame(
        self,
        frame_key: str,
        frame_number: int,
    ) -> list[_SpatialFaceObservation]:
        """Return usable identity observations for one frame."""
        observations: list[_SpatialFaceObservation] = []
        faces = self._alignments.data[frame_key].faces

        for face_index, face in enumerate(faces):
            embedding = self._embedding_for_face(face)
            if embedding is None:
                continue

            landmarks_raw = getattr(face, "landmarks_xy", None)
            if landmarks_raw is None:
                continue
            landmarks = np.asarray(landmarks_raw, dtype=np.float32)
            if landmarks.ndim != 2 or landmarks.shape[1] != 2:
                continue

            x_pos = float(getattr(face, "x", 0.0) or 0.0)
            y_pos = float(getattr(face, "y", 0.0) or 0.0)
            width = float(getattr(face, "w", 0.0) or 0.0)
            height = float(getattr(face, "h", 0.0) or 0.0)

            observations.append(
                _SpatialFaceObservation(
                    frame_key=frame_key,
                    frame_number=frame_number,
                    face_index=face_index,
                    landmarks=landmarks,
                    embedding=embedding,
                    center_x=x_pos + (width / 2.0),
                    center_y=y_pos + (height / 2.0),
                    width=width,
                    height=height,
                )
            )

        self._assign_frame_halves(observations)
        self._mark_same_identity_duplicates(observations)
        return observations

    @staticmethod
    def _embedding_for_face(face: T.Any) -> np.ndarray | None:
        """Return a normalized deterministic identity embedding for a face."""
        identity = getattr(face, "identity", None)
        if not identity:
            return None

        for key in sorted(identity):
            embedding = np.asarray(identity[key], dtype=np.float32)
            if embedding.ndim != 1 or embedding.size == 0:
                continue
            if not np.isfinite(embedding).all():
                continue
            norm = np.linalg.norm(embedding)
            if norm == 0:
                continue
            return T.cast(np.ndarray, embedding / norm)
        return None

    @staticmethod
    def _identity_distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
        """Return cosine distance for normalized identity embeddings."""
        return 1.0 - float(np.clip(np.dot(lhs, rhs), -1.0, 1.0))

    @staticmethod
    def _assign_frame_halves(observations: list[_SpatialFaceObservation]) -> None:
        """Assign coarse left/right positions within a frame.

        This is only used as a guardrail when the same identity appears more than once in a
        single frame, as in side-by-side stereo.
        """
        if len(observations) < 2:
            return

        centers = [obs.center_x for obs in observations]
        left = min(centers)
        right = max(centers)
        if right <= left:
            return

        midpoint = (left + right) / 2.0
        for observation in observations:
            observation.frame_half = "left" if observation.center_x < midpoint else "right"

    def _mark_same_identity_duplicates(
        self,
        observations: list[_SpatialFaceObservation],
    ) -> None:
        """Mark observations that share a near-identical identity within the same frame."""
        for idx, lhs in enumerate(observations):
            for rhs in observations[idx + 1 :]:
                if (
                    self._identity_distance(lhs.embedding, rhs.embedding)
                    <= _SPATIAL_STRONG_IDENTITY_DISTANCE
                ):
                    lhs.same_identity_duplicate = True
                    rhs.same_identity_duplicate = True

    def _track_match_cost(
        self,
        track: _SpatialIdentityTrack,
        observation: _SpatialFaceObservation,
    ) -> float:
        """Return assignment cost for matching a track to an observation.

        Lower is better. ``_SPATIAL_TRACK_REJECT_COST`` means do not match.
        """
        last = track.last
        gap = max(1, observation.frame_number - last.frame_number)
        if gap > _SPATIAL_MAX_TRACK_GAP:
            return _SPATIAL_TRACK_REJECT_COST

        identity_distance = self._identity_distance(track.embedding, observation.embedding)
        if identity_distance > _SPATIAL_MAX_IDENTITY_DISTANCE:
            return _SPATIAL_TRACK_REJECT_COST

        spatial_distance = np.hypot(
            observation.center_x - last.center_x,
            observation.center_y - last.center_y,
        )
        spatial_scale = max(
            last.width,
            last.height,
            observation.width,
            observation.height,
            1.0,
        )
        spatial_distance = float(spatial_distance / spatial_scale)

        if (
            spatial_distance > _SPATIAL_MAX_WEAK_IDENTITY_JUMP
            and identity_distance > _SPATIAL_STRONG_IDENTITY_DISTANCE
        ):
            return _SPATIAL_TRACK_REJECT_COST

        if (
            last.frame_half is not None
            and observation.frame_half is not None
            and last.frame_half != observation.frame_half
            and (last.same_identity_duplicate or observation.same_identity_duplicate)
            and identity_distance <= _SPATIAL_STRONG_IDENTITY_DISTANCE
        ):
            # Same-person duplicates in a single frame, e.g. side-by-side stereo, should remain
            # separate left/right face-instance tracks instead of collapsing into one identity.
            return _SPATIAL_TRACK_REJECT_COST

        spatial_component = min(spatial_distance, 1.0)
        gap_component = min(1.0, 0.1 * (gap - 1))

        return 0.80 * identity_distance + 0.15 * spatial_component + 0.05 * gap_component

    # Define shape normalization utility functions
    @staticmethod
    def _normalize_shapes(
        shapes_im_coords: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize a 2D or 3D shape

        Parameters
        ----------
        shaped_im_coords
            The facial landmarks

        Returns
        -------
        shapes_normalized
            The normalized shapes
        scale_factors
            The scale factors
        mean_coords
            The mean coordinates
        """
        logger.debug("Normalize shapes")
        (num_pts, num_dims, _) = shapes_im_coords.shape

        # Calculate mean coordinates and subtract from shapes
        mean_coords = shapes_im_coords.mean(axis=0)
        shapes_centered = np.zeros(shapes_im_coords.shape)
        shapes_centered = shapes_im_coords - np.tile(mean_coords, [num_pts, 1, 1])

        # Calculate scale factors and divide shapes
        scale_factors = np.sqrt((shapes_centered**2).sum(axis=1)).mean(axis=0)
        shapes_normalized = np.zeros(shapes_centered.shape)
        shapes_normalized = shapes_centered / np.tile(scale_factors, [num_pts, num_dims, 1])

        logger.debug(
            "Normalized shapes: (shapes_normalized: %s, scale_factors: %s, mean_coords: %s",
            shapes_normalized,
            scale_factors,
            mean_coords,
        )
        return shapes_normalized, scale_factors, mean_coords

    @staticmethod
    def _normalized_to_original(
        shapes_normalized: np.ndarray,
        scale_factors: np.ndarray,
        mean_coords: np.ndarray,
    ) -> np.ndarray:
        """Transform a normalized shape back to original image coordinates

        Parameters
        ----------
        shapes_normalized
            The normalized shapes
        scale_factors
            The scale factors
        mean_coords
            The mean coordinates

        Returns
        -------
        The normalized shape transformed back to original coordinates
        """
        logger.debug("Normalize to original")
        (num_pts, num_dims, _) = shapes_normalized.shape

        # move back to the correct scale
        shapes_centered = shapes_normalized * np.tile(scale_factors, [num_pts, num_dims, 1])
        # move back to the correct location
        shapes_im_coords = shapes_centered + np.tile(mean_coords, [num_pts, 1, 1])

        logger.debug("Normalized to original: %s", shapes_im_coords)
        return shapes_im_coords  # type: ignore[no-any-return]

    def _normalize(self, face_index: int) -> bool:
        """Compile all original and normalized alignments for a face index

        Parameters
        ----------
        face_index
            The face index to compile landmarks for

        Returns
        -------
        ``True`` if the face index has enough landmarks to smooth otherwise ``False``
        """
        logger.debug("Normalize face index: %s", face_index)
        count = sum(1 for val in self._alignments.data.values() if len(val.faces) > face_index)
        if count < 2:
            logger.info(
                "Skipping face index %s. Not enough frames with this face index.",
                face_index,
            )
            return False

        sample_lm = next(
            (
                val.faces[face_index].landmarks_xy
                for val in self._alignments.data.values()
                if len(val.faces) > face_index
            ),
            None,
        )
        if sample_lm is None:
            return False

        assert isinstance(sample_lm, np.ndarray)
        lm_count = sample_lm.shape[0]
        if lm_count != 68:
            raise FaceswapError("Spatial smoothing only supports 68 point facial landmarks")

        landmarks_all = np.zeros((lm_count, 2, int(count)))

        end = 0
        for key in tqdm(
            sorted(self._alignments.data.keys()),
            desc=f"Compiling face {face_index}",
            leave=False,
        ):
            val = self._alignments.data[key].faces
            if len(val) <= face_index:
                continue
            landmarks = np.array(val[face_index].landmarks_xy).reshape((lm_count, 2, 1))
            start = end
            end = start + landmarks.shape[2]
            # Store in one big array
            landmarks_all[:, :, start:end] = landmarks
            # Make sure we keep track of the mapping to the original frame
            self._mappings[start] = key

        # Normalize shapes
        normalized_shape = self._normalize_shapes(landmarks_all)
        self._normalized["landmarks"] = normalized_shape[0]
        self._normalized["scale_factors"] = normalized_shape[1]
        self._normalized["mean_coords"] = normalized_shape[2]
        logger.debug("Normalized: %s", self._normalized)
        return True

    def _normalize_track(self, track: _SpatialIdentityTrack) -> bool:
        """Compile original and normalized alignments for one identity instance track."""
        logger.debug("Normalize identity track: %s", track.track_id)
        observations = track.observations
        if len(observations) < 2:
            logger.info(
                "Skipping identity track %s. Not enough observations.",
                track.track_id,
            )
            return False

        sample_lm = observations[0].landmarks
        lm_count = sample_lm.shape[0]
        if lm_count != 68:
            raise FaceswapError("Spatial smoothing only supports 68 point facial landmarks")

        landmarks_all = np.zeros((lm_count, 2, len(observations)))
        self._mappings = {}

        for idx, observation in enumerate(observations):
            landmarks = np.asarray(observation.landmarks, dtype=np.float32).reshape((lm_count, 2))
            landmarks_all[:, :, idx] = landmarks
            self._mappings[idx] = (observation.frame_key, observation.face_index)

        normalized_shape = self._normalize_shapes(landmarks_all)
        self._normalized["landmarks"] = normalized_shape[0]
        self._normalized["scale_factors"] = normalized_shape[1]
        self._normalized["mean_coords"] = normalized_shape[2]
        logger.debug("Normalized identity track %s: %s", track.track_id, self._normalized)
        return True

    def _shape_model(self) -> None:
        """build 2D shape model"""
        logger.debug("Shape model")
        landmarks_norm = self._normalized["landmarks"]
        normalized_shapes_tbl = np.reshape(landmarks_norm, [68 * 2, landmarks_norm.shape[2]]).T
        num_components = min(20, normalized_shapes_tbl.shape[0], normalized_shapes_tbl.shape[1])
        self._shapes_model = decomposition.PCA(
            n_components=num_components, whiten=True, random_state=1
        ).fit(normalized_shapes_tbl)
        explained = self._shapes_model.explained_variance_ratio_.sum()
        logger.info(
            "Total explained percent by PCA model with %s components is %s%%",
            num_components,
            round(100 * explained, 1),
        )
        logger.debug("Shaped model")

    def _spatially_filter(self) -> np.ndarray:
        """interpret the shapes using our shape model (project and reconstruct)

        Returns
        -------
        The filtered landmarks in original coordinate space
        """
        logger.debug("Spatially Filter")
        assert self._shapes_model is not None
        landmarks_norm = self._normalized["landmarks"]
        # Convert to matrix form
        landmarks_norm_table = np.reshape(landmarks_norm, [68 * 2, landmarks_norm.shape[2]]).T
        # Project onto shapes model and reconstruct
        landmarks_norm_table_rec = self._shapes_model.inverse_transform(
            self._shapes_model.transform(landmarks_norm_table)
        )
        # Convert back to shapes (num key points, num_dims, numFrames)
        landmarks_norm_rec = np.reshape(
            landmarks_norm_table_rec.T, [68, 2, landmarks_norm.shape[2]]
        )
        # Transform back to image co-ordinates
        retval = self._normalized_to_original(
            landmarks_norm_rec,
            self._normalized["scale_factors"],
            self._normalized["mean_coords"],
        )

        logger.debug("Spatially Filtered: %s", retval)
        return retval

    @staticmethod
    def _temporally_smooth(landmarks: np.ndarray) -> np.ndarray:
        """apply temporal filtering on the 2D points

        Parameters
        ----------
        landmarks
            68 point landmarks to be temporally smoothed

        Returns
        -------
        The temporally smoothed landmarks
        """
        logger.debug("Temporally Smooth")
        filter_half_length = 2
        temporal_filter = np.ones((1, 1, 2 * filter_half_length + 1))
        temporal_filter = temporal_filter / temporal_filter.sum()

        start_tile_block = np.tile(
            landmarks[:, :, 0][:, :, np.newaxis], [1, 1, filter_half_length]
        )
        end_tile_block = np.tile(landmarks[:, :, -1][:, :, np.newaxis], [1, 1, filter_half_length])
        landmarks_padded = np.dstack((start_tile_block, landmarks, end_tile_block))

        retval = signal.convolve(landmarks_padded, temporal_filter, mode="valid", method="fft")
        logger.debug("Temporally Smoothed: %s", retval)
        return retval  # type: ignore[no-any-return]

    def _update_alignments(self, landmarks: np.ndarray, face_index: int) -> None:
        """Update smoothed landmarks back to alignments

        Parameters
        ----------
        landmarks
            The smoothed landmarks
        face_index
            The face index to update
        """
        logger.debug("Update alignments for face index: %s", face_index)
        for idx, frame in tqdm(
            self._mappings.items(), desc=f"Updating face {face_index}", leave=False
        ):
            logger.trace("Updating: (frame: %s, face_index: %s)", frame, face_index)  # type:ignore
            landmarks_update = landmarks[:, :, idx]
            landmarks_xy = landmarks_update.reshape(68, 2)
            self._alignments.data[frame].faces[face_index].landmarks_xy = landmarks_xy
            logger.trace(  # type: ignore[attr-defined]
                "Updated: (frame: '%s', face_index: %s, landmarks: %s)",
                frame,
                face_index,
                landmarks_xy,
            )
        logger.debug("Updated alignments")

    def _update_track_alignments(self, landmarks: np.ndarray) -> None:
        """Update smoothed landmarks back to their original frame and face index."""
        logger.debug("Update alignments for identity track")
        for idx, mapping in tqdm(
            self._mappings.items(),
            desc="Updating identity track",
            leave=False,
        ):
            assert isinstance(mapping, tuple)
            frame, face_index = mapping
            logger.trace("Updating: (frame: %s, face_index: %s)", frame, face_index)  # type:ignore
            landmarks_update = landmarks[:, :, idx]
            landmarks_xy = landmarks_update.reshape(68, 2)
            self._alignments.data[frame].faces[face_index].landmarks_xy = landmarks_xy
            logger.trace(  # type: ignore[attr-defined]
                "Updated: (frame: '%s', face_index: %s, landmarks: %s)",
                frame,
                face_index,
                landmarks_xy,
            )
        logger.debug("Updated identity track alignments")


__all__ = get_module_objects(__name__)
