#!/usr/bin/env python3
"""Thumbnail generator for the manual tool"""

from __future__ import annotations

import logging
import os
import typing as T
from dataclasses import dataclass
from threading import Lock
from time import sleep

import numpy as np
from tqdm import tqdm

from lib.align import AlignedFace
from lib.image import SingleFrameLoader, generate_thumbnail
from lib.logger import parse_class_init
from lib.multithreading import MultiThread
from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    from .detected_faces import DetectedFaces

logger = logging.getLogger(__name__)


@dataclass
class ProgressBar:
    """Thread-safe progress bar for tracking thumbnail generation progress"""

    p_bar: tqdm | None = None
    lock = Lock()


@dataclass
class VideoMeta:
    """Holds meta information about a video file

    Parameters
    ----------
    key_frames
        List of key frame indices for the video
    pts_times
        List of presentation timestamps for the video
    """

    key_frames: list[int] | None = None
    pts_times: list[int] | None = None


class ThumbsCreator:
    """Background loader to generate thumbnails for the alignments file. Generates low resolution
    thumbnails in parallel threads for faster processing.

    Parameters
    ----------
    detected_faces
        The :class:`~lib.align.DetectedFace` objects for this video
    input_location
        The location of the input folder of frames or video file
    single_process
        ``True`` to generated thumbs in a single process otherwise ``False``
    """

    def __init__(
        self, detected_faces: DetectedFaces, input_location: str, single_process: bool
    ) -> None:
        logger.debug(parse_class_init(locals()))
        self._p_bar = ProgressBar()
        self._meta = detected_faces.video_meta_data
        self._location = input_location
        self._alignments = detected_faces._alignments
        self._frame_faces = detected_faces._frame_faces
        self._folder_reader: SingleFrameLoader | None = None

        self._is_video = self._meta is not None

        cpu_count = os.cpu_count()
        max_threads = 1 if cpu_count is None or cpu_count <= 2 else cpu_count - 2

        if self._is_video and single_process:
            self._num_threads = 1
        else:
            self._num_threads = min(max_threads, 32)
        self._threads: list[MultiThread] = []
        logger.debug("[THUMBS] Initialized %s", self.__class__.__name__)

    @property
    def has_thumbs(self) -> bool:
        """``True`` if the alignments file holds thumbnail images otherwise ``False``."""
        return bool(self._alignments.thumbnails.has_thumbnails)

    def generate_cache(self) -> None:
        """Extract the face thumbnails from a video or folder of images into the
        alignments file"""
        self._p_bar.p_bar = tqdm(
            desc="Caching Thumbnails", leave=False, total=len(self._frame_faces)
        )
        if self._is_video:
            self._launch_video()
        else:
            self._launch_folder()
        while True:
            self._check_and_raise_error()
            if all(not thread.is_alive() for thread in self._threads):
                break
            sleep(1)
        self._join_threads()
        folder_reader = getattr(self, "_folder_reader", None)
        if folder_reader is not None:
            folder_reader.close()
            self._folder_reader = None
        self._p_bar.p_bar.close()
        self._alignments.save()

    # << PRIVATE METHODS >> #
    def _check_and_raise_error(self) -> None:
        """Monitor the loading threads for errors and raise if any occur."""
        for thread in self._threads:
            thread.check_and_raise_error()

    def _join_threads(self) -> None:
        """Join the loading threads"""
        logger.debug("[THUMBS] Joining face viewer loading threads")
        for thread in self._threads:
            thread.join()

    def _launch_video(self) -> None:
        """Launch thumbnail workers for a video file."""
        assert self._meta is not None
        if self._meta["keyframes"] and self._meta["keyframes"][0] != 0:
            logger.warning("Your video does not start on a Key Frame. This can lead to issues.")

        frame_face_indices = [i for i, v in enumerate(self._alignments.data.values()) if v.faces]
        num_frames = len(frame_face_indices)
        if num_frames == 0:
            logger.debug("[THUMBS] No video frames with faces to cache")
            return

        num_threads = min(num_frames, self._num_threads)
        for idx, chunk in enumerate(np.array_split(frame_face_indices, num_threads)):
            indices = [int(index) for index in chunk.tolist()]
            if not indices:
                continue
            logger.debug(
                "[THUMBS] thread index: %s, frame_start: %s, frame_end: %s, segment_count: %s",
                idx,
                indices[0],
                indices[-1],
                len(indices),
            )
            thread = MultiThread(self._load_from_video, indices)
            thread.start()
            self._threads.append(thread)

    def _launch_folder(self) -> None:
        """Launch thumbnail workers for a folder of images."""
        reader = SingleFrameLoader(self._location)
        self._folder_reader = reader
        skip_list = [
            idx
            for idx, f in enumerate(reader.file_list)
            if os.path.basename(f) not in self._alignments.data
        ]
        if skip_list:
            reader.add_skip_list(skip_list)
        if reader.process_count == 0:
            logger.debug("[THUMBS] No image frames with faces to cache")
            return

        num_threads = min(reader.process_count, self._num_threads)
        logger.debug(
            "[THUMBS] total images: %s, num_threads: %s",
            reader.process_count,
            num_threads,
        )
        for chunk in np.array_split(np.arange(reader.process_count), num_threads):
            if len(chunk) == 0:
                continue
            start_idx = int(chunk[0])
            end_idx = int(chunk[-1]) + 1
            thread = MultiThread(self._load_from_folder, reader, start_idx, end_idx)
            thread.start()
            self._threads.append(thread)

    def _load_from_video(self, indices: list[int]) -> None:
        """Loads faces from video for the given segment of the source video."""
        if not indices:
            return
        logger.debug(
            "[THUMBS] Segment start: frame_start: %s, frame_end: %s, segment_count: %s",
            indices[0],
            indices[-1],
            len(indices),
        )
        assert self._meta is not None
        reader = SingleFrameLoader(self._location, video_meta_data=self._meta)
        proc_count = 0
        try:
            for frame_index in indices:
                filename, image = reader.image_from_index(frame_index)
                self._set_thumbnail(filename, image, frame_index)
                proc_count += 1
        finally:
            reader.close()
        logger.debug(
            "[THUMBS] Segment complete: (starting_frame_index: %s, processed_count: %s)",
            indices[0],
            proc_count,
        )

    def _load_from_folder(
        self, reader: SingleFrameLoader, start_index: int, end_index: int
    ) -> None:
        """Loads faces from the given range of frame indices from a folder of images.

        Each frame range is extracted in a different background thread.

        Parameters
        ----------
        reader
            The reader that is used to retrieve the requested frame
        start_index
            The starting frame index for the images to extract faces from
        end_index
            The end frame index for the images to extract faces from
        """
        logger.debug(
            "[THUMBS] reader: %s, start_index: %s, end_index: %s",
            reader,
            start_index,
            end_index,
        )
        for frame_index in range(start_index, end_index):
            filename, frame = reader.image_from_index(frame_index)
            self._set_thumbnail(filename, frame, frame_index)
        logger.debug(
            "[THUMBS] Segment complete: (start_index: %s, processed_count: %s)",
            start_index,
            end_index - start_index,
        )

    def _set_thumbnail(self, filename: str, frame: np.ndarray, frame_index: int) -> None:
        """Extracts the faces from the frame and adds to alignments file."""
        for face_idx, face in enumerate(self._frame_faces[frame_index]):
            thumbnail_source = None
            try:
                aligned = AlignedFace(face.landmarks_xy, image=frame, centering="head", size=96)
                thumbnail_source = aligned.face
            except Exception:  # pylint:disable=broad-except
                logger.debug(
                    "[THUMBS] Could not build aligned thumbnail for %s face %s",
                    filename,
                    face_idx,
                    exc_info=True,
                )

            if thumbnail_source is None:
                height, width = frame.shape[:2]
                x = max(0, int(round(getattr(face, "x", getattr(face, "left", 0)))))
                y = max(0, int(round(getattr(face, "y", getattr(face, "top", 0)))))
                w = max(1, int(round(getattr(face, "w", getattr(face, "width", width)))))
                h = max(1, int(round(getattr(face, "h", getattr(face, "height", height)))))
                x2 = min(width, x + w)
                y2 = min(height, y + h)
                thumbnail_source = frame if x >= x2 or y >= y2 else frame[y:y2, x:x2]

            face.thumbnail = generate_thumbnail(thumbnail_source, size=96)
            assert face.thumbnail is not None
            self._alignments.thumbnails.add_thumbnail(filename, face_idx, face.thumbnail)
        with self._p_bar.lock:
            assert self._p_bar.p_bar is not None
            self._p_bar.p_bar.update(1)


__all__ = get_module_objects(__name__)
