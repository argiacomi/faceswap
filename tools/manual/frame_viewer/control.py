#!/usr/bin/env python3
# mypy: disable-error-code="arg-type, attr-defined, call-overload"
"""Handles Navigation and Background Image for the Frame Viewer section of the manual
tool GUI."""

import logging
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

from lib.align import AlignedFace

logger = logging.getLogger(__name__)


class Navigation:
    """Handles playback and frame navigation for the Frame Viewer Window.

    Parameters
    ----------
    display_frame: :class:`DisplayFrame`
        The parent frame viewer window
    """

    def __init__(self, display_frame):
        logger.debug("Initializing %s", self.__class__.__name__)
        self._display_frame = display_frame
        self._globals = display_frame._globals
        self._det_faces = display_frame._det_faces
        self._nav = display_frame._nav
        self._tk_is_playing = tk.BooleanVar()
        self._tk_is_playing.set(False)
        self._det_faces.tk_face_count_changed.trace("w", self._update_total_frame_count)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def _current_nav_frame_count(self) -> int:
        """The current frame count for the transport slider."""
        try:
            return int(float(self._nav["scale"].cget("to"))) + 1
        except (TypeError, ValueError, tk.TclError):
            logger.debug(
                "Could not read navigation scale maximum. Falling back to 0.", exc_info=True
            )
            return 0

    def nav_scale_callback(self, *args, reset_progress=True):  # pylint:disable=unused-argument
        """Adjust transport slider scale for different filters.

        Dynamic landmark-based filters may need a grid refresh even when the
        resulting count is unchanged, so this method does not early-return on
        unchanged counts.
        """
        self._display_frame.pack_threshold_slider()
        if reset_progress:
            self.stop_playback()

        frame_count = self._det_faces.filter.count
        if self._current_nav_frame_count == frame_count:
            logger.trace("Filtered count has not changed.")

        if self._globals.var_filter_mode.get() in (
            "Misaligned Faces",
            "Neighbor Outliers",
            "Landmarks Outside Thumbnail",
        ):
            self._det_faces.tk_face_count_changed.set(True)

        self._update_total_frame_count()
        if reset_progress:
            self._globals.var_transport_index.set(0)

    def _update_total_frame_count(self, *args):  # pylint:disable=unused-argument
        """Update the displayed number of total frames that meet the current filter criteria.

        Parameters
        ----------
        args: tuple
            Required for tkinter trace callback but unused
        """
        frame_count = self._det_faces.filter.count
        if self._current_nav_frame_count == frame_count:
            logger.trace("Filtered count has not changed. Returning")
            return
        max_frame = max(0, frame_count - 1)
        logger.debug(
            "Filtered frame count has changed. Updating from %s to %s",
            self._current_nav_frame_count,
            frame_count,
        )
        self._nav["scale"].config(to=max_frame)
        self._nav["label"].config(text=f"/{max_frame}")
        state = "disabled" if max_frame == 0 else "normal"
        self._nav["entry"].config(state=state)

    @property
    def tk_is_playing(self):
        """:class:`tkinter.BooleanVar`: Whether the stream is currently playing."""
        return self._tk_is_playing

    def handle_play_button(self):
        """Handle the play button.

        Switches the :attr:`tk_is_playing` variable.
        """
        is_playing = self.tk_is_playing.get()
        self.tk_is_playing.set(not is_playing)

    def stop_playback(self):
        """Stop play back if playing"""
        if self.tk_is_playing.get():
            logger.trace("Stopping playback")
            self.tk_is_playing.set(False)

    def increment_face(self):
        """Update navigation to the next face in filtered frame order."""
        self._cycle_face(1)

    def decrement_face(self):
        """Update navigation to the previous face in filtered frame order."""
        self._cycle_face(-1)

    def _cycle_face(self, direction):
        """Cycle to the next/previous face, crossing filtered frames as needed."""
        self.stop_playback()

        frames = list(self._det_faces.filter.frames_list)
        if not frames:
            logger.debug("No filtered frames. Not cycling face.")
            return

        face_counts = self._det_faces.face_count_per_index
        position = self._current_filtered_position(frames)
        if position < 0 or position >= len(frames):
            logger.debug("No valid filtered position. Not cycling face.")
            return

        current_frame = frames[position]
        if current_frame < 0 or current_frame >= len(face_counts):
            logger.debug("Current filtered frame is stale: %s", current_frame)
            return

        current_face_count = face_counts[current_frame]
        current_face_index = self._globals.face_index

        if current_face_count:
            current_face_index = max(0, min(current_face_index, current_face_count - 1))
            if direction > 0 and current_face_index < current_face_count - 1:
                self._set_face_position(frames, position, current_face_index + 1)
                return
            if direction < 0 and current_face_index > 0:
                self._set_face_position(frames, position, current_face_index - 1)
                return

        search = range(position + 1, len(frames)) if direction > 0 else range(position - 1, -1, -1)
        for next_position in search:
            frame_index = frames[next_position]
            if frame_index < 0 or frame_index >= len(face_counts):
                continue
            face_count = face_counts[frame_index]
            if face_count == 0:
                continue
            face_index = 0 if direction > 0 else face_count - 1
            self._set_face_position(frames, next_position, face_index)
            return

        logger.debug("End of Stream. Not cycling face.")

    def _current_filtered_position(self, frames):
        """Return the current frame's position in the filtered frame list."""
        if not frames:
            return 0

        current_frame = self._globals.frame_index
        try:
            return frames.index(current_frame)
        except ValueError:
            return self._get_safe_frame_index(max_index=len(frames) - 1)

    def _set_face_position(self, frames, transport_index, face_index):
        """Select a face and move transport to its frame if needed."""
        if not 0 <= transport_index < len(frames):
            return

        frame_index = frames[transport_index]
        face_counts = self._det_faces.face_count_per_index
        if frame_index < 0 or frame_index >= len(face_counts):
            logger.debug("Ignoring stale frame in filtered position: %s", frame_index)
            return

        face_count = face_counts[frame_index]
        if face_count <= 0:
            logger.debug("Ignoring face selection for frame with no faces: %s", frame_index)
            return

        face_index = max(0, min(int(face_index), face_count - 1))
        logger.debug(
            "Setting face position. transport_index: %s, frame_index: %s, face_index: %s",
            transport_index,
            frame_index,
            face_index,
        )
        self._globals.set_selected_faces(((frame_index, face_index),))
        self._globals.set_face_index(face_index)

        if self._get_safe_frame_index(max_index=len(frames) - 1) != transport_index:
            self._globals.var_transport_index.set(transport_index)

        self._globals.var_update_active_viewport.set(True)
        self._globals.var_full_update.set(True)

    def increment_frame(self, frame_count=None, is_playing=False):
        """Update frame navigation to the next filtered frame."""
        if not is_playing:
            self.stop_playback()

        frames = list(self._det_faces.filter.frames_list)
        frame_count = len(frames) if frame_count is None else int(frame_count)
        if frame_count <= 0:
            logger.debug("End of Stream. Not incrementing")
            self.stop_playback()
            return

        position = self._get_safe_frame_index(max_index=frame_count - 1)
        face_count_change = not self._det_faces.filter.frame_meets_criteria
        if face_count_change:
            position -= 1

        if not face_count_change and position >= frame_count - 1:
            logger.debug("End of Stream. Not incrementing")
            self.stop_playback()
            return

        self._globals.var_transport_index.set(min(position + 1, frame_count - 1))

    def decrement_frame(self):
        """Update frame navigation to the previous filtered frame."""
        self.stop_playback()

        frame_count = self._det_faces.filter.count
        if frame_count <= 0:
            logger.debug("End of Stream. Not decrementing")
            return

        position = self._get_safe_frame_index(max_index=frame_count - 1)
        face_count_change = not self._det_faces.filter.frame_meets_criteria
        if not face_count_change and position == 0:
            logger.debug("End of Stream. Not decrementing")
            return

        self._globals.var_transport_index.set(max(0, position - 1))

    def _get_safe_frame_index(self, max_index=None):
        """Obtain a safe transport slider index.

        The historical method name is retained for compatibility, but this
        returns the filtered transport position, not an absolute frame index.
        """
        try:
            retval = int(float(self._globals.var_transport_index.get()))
        except (tk.TclError, TypeError, ValueError):
            retval = 0

        retval = max(0, min(retval, int(max_index))) if max_index is not None else max(0, retval)

        try:
            if self._globals.var_transport_index.get() != retval:
                self._globals.var_transport_index.set(retval)
        except tk.TclError:
            self._globals.var_transport_index.set(retval)

        return retval

    def goto_first_frame(self):
        """Go to the first frame that meets the filter criteria."""
        self.stop_playback()
        if self._det_faces.filter.count <= 0:
            return
        position = self._get_safe_frame_index(max_index=self._det_faces.filter.count - 1)
        if position == 0:
            return
        self._globals.var_transport_index.set(0)

    def goto_last_frame(self):
        """Go to the last frame that meets the filter criteria."""
        self.stop_playback()
        frame_count = self._det_faces.filter.count
        if frame_count <= 0:
            return

        position = self._get_safe_frame_index(max_index=frame_count - 1)
        if position == frame_count - 1:
            return
        self._globals.var_transport_index.set(frame_count - 1)


class BackgroundImage:
    """The background image of the canvas"""

    def __init__(self, canvas):
        self._canvas = canvas
        self._globals = canvas._globals
        self._det_faces = canvas._det_faces
        placeholder = np.zeros((*reversed(self._globals.frame_display_dims), 3), dtype="uint8")
        self._tk_frame = ImageTk.PhotoImage(Image.fromarray(placeholder))
        self._tk_face = ImageTk.PhotoImage(Image.fromarray(placeholder))
        self._image = self._canvas.create_image(
            self._globals.frame_display_dims[0] / 2,
            self._globals.frame_display_dims[1] / 2,
            image=self._tk_frame,
            anchor=tk.CENTER,
            tags="main_image",
        )
        self._zoomed_centering = "face"

    @property
    def _current_view_mode(self):
        """str: `frame` if global zoom mode variable is set to ``False`` other wise `face`."""
        retval = "face" if self._globals.is_zoomed else "frame"
        logger.trace(retval)
        return retval

    def _blank_display_face(self) -> np.ndarray:
        """Return a blank RGB square for invalid zoom targets."""
        size = max(1, min(self._globals.frame_display_dims))
        return np.zeros((size, size, 3), dtype="uint8")

    def _valid_frame_index(self, frame_idx: int) -> bool:
        """Return whether ``frame_idx`` can be used against detected face data."""
        return 0 <= frame_idx < len(self._det_faces.face_count_per_index)

    def _clamp_face_index(self, frame_idx: int) -> int:
        """Clamp the global face index to the available faces for ``frame_idx``."""
        if not self._valid_frame_index(frame_idx):
            self._globals.set_face_index(0)
            return 0

        faces_in_frame = self._det_faces.face_count_per_index[frame_idx]
        if faces_in_frame <= 0:
            self._globals.set_face_index(0)
            return 0

        current_face_index = int(self._globals.face_index)
        face_idx = int(max(0, min(current_face_index, faces_in_frame - 1)))
        if face_idx != current_face_index:
            self._globals.set_face_index(face_idx)
        return face_idx

    def refresh(self, view_mode):
        """Update the displayed frame."""
        self._switch_image(view_mode)
        logger.trace("Updating background frame")
        if self._current_view_mode == "face":
            self._update_tk_face()
        else:
            self._update_tk_frame()

    def _switch_image(self, view_mode):
        """Switch the image between full-frame and zoomed-face display."""
        new_centering = self._canvas.active_editor.zoomed_centering
        current_face_index = self._globals.face_index
        target_face_index = self._clamp_face_index(self._globals.frame_index)

        if view_mode == self._current_view_mode and (
            new_centering == self._zoomed_centering and target_face_index == current_face_index
        ):
            return

        if view_mode == "frame" or new_centering != self._zoomed_centering:
            self._canvas._bbox_zoom_state = None  # pylint:disable=protected-access

        self._zoomed_centering = new_centering
        logger.trace(
            "Switching background image from '%s' to '%s'",
            self._current_view_mode,
            view_mode,
        )
        img = self._tk_face if view_mode == "face" else self._tk_frame
        self._canvas.itemconfig(self._image, image=img)
        self._globals.set_zoomed(view_mode == "face")
        self._globals.set_face_index(target_face_index)

    def _update_tk_face(self):
        """Update the currently zoomed face."""
        face = self._get_zoomed_face()
        padding = self._get_padding(face.shape[:2])
        if any(padding):
            face = cv2.copyMakeBorder(face, *padding, cv2.BORDER_CONSTANT)

        if self._tk_face.height() != face.shape[0] or self._tk_face.width() != face.shape[1]:
            self._resize_frame()

        logger.trace("final shape: %s", face.shape)
        self._tk_face.paste(Image.fromarray(face))

    def _get_zoomed_face(self):
        """Get the zoomed face or a blank image if no valid face is available."""
        frame_idx = self._globals.frame_index
        if not self._valid_frame_index(frame_idx):
            logger.debug("Invalid frame index for zoomed face: %s", frame_idx)
            return self._blank_display_face()

        face_idx = self._clamp_face_index(frame_idx)
        faces_in_frame = self._det_faces.face_count_per_index[frame_idx]
        if faces_in_frame <= 0:
            return self._blank_display_face()

        faces = self._det_faces.current_faces
        if frame_idx >= len(faces) or face_idx >= len(faces[frame_idx]):
            logger.debug(
                "Stale zoomed face target. frame_idx=%s face_idx=%s",
                frame_idx,
                face_idx,
            )
            return self._blank_display_face()

        det_face = faces[frame_idx][face_idx]
        if self._zoomed_centering == "bbox":
            face = self._get_zoomed_bbox_face(det_face)
        else:
            face = AlignedFace(
                det_face.landmarks_xy,
                image=self._globals.current_frame.image,
                centering=self._zoomed_centering,
                size=max(1, min(self._globals.frame_display_dims)),
            ).face

        if face is None:
            logger.debug("AlignedFace returned no face for zoomed target.")
            return self._blank_display_face()

        logger.trace("face shape: %s", face.shape)
        return face[..., 2::-1]

    def _bbox_zoom_state(self, det_face, frame_idx: int, face_idx: int):
        """Return persistent bbox zoom state for the current frame/face/display."""
        display_dims = tuple(self._globals.frame_display_dims)
        state = getattr(self._canvas, "_bbox_zoom_state", None)
        if (
            state is not None
            and state.get("frame_index") == frame_idx
            and state.get("face_index") == face_idx
            and state.get("display_dims") == display_dims
        ):
            return state

        state = self._create_bbox_zoom_state(det_face, frame_idx, face_idx, display_dims)
        self._canvas._bbox_zoom_state = state  # pylint:disable=protected-access
        return state

    def _create_bbox_zoom_state(self, det_face, frame_idx: int, face_idx: int, display_dims):
        """Create bbox zoom geometry once when entering bbox zoom."""
        frame = self._globals.current_frame.image
        frame_h, frame_w = frame.shape[:2]
        display_w, display_h = display_dims

        if (
            det_face.left is None
            or det_face.top is None
            or det_face.right is None
            or det_face.bottom is None
        ):
            logger.debug("Face has no bbox. Falling back to full-frame bbox zoom state.")
            roi_left, roi_top, roi_right, roi_bottom = 0, 0, frame_w, frame_h
        else:
            width = max(1, int(det_face.right - det_face.left))
            height = max(1, int(det_face.bottom - det_face.top))
            roi_left = int(np.floor(det_face.left - (width / 2.0)))
            roi_top = int(np.floor(det_face.top - (height / 2.0)))
            roi_right = int(np.ceil(det_face.right + (width / 2.0)))
            roi_bottom = int(np.ceil(det_face.bottom + (height / 2.0)))

        roi_w = max(1, roi_right - roi_left)
        roi_h = max(1, roi_bottom - roi_top)
        scale = min(display_w / roi_w, display_h / roi_h)

        resized_w = max(1, int(round(roi_w * scale)))
        resized_h = max(1, int(round(roi_h * scale)))
        offset = (
            (display_w - resized_w) / 2.0,
            (display_h - resized_h) / 2.0,
        )

        return {
            "frame_index": frame_idx,
            "face_index": face_idx,
            "display_dims": display_dims,
            "roi": (roi_left, roi_top, roi_right, roi_bottom),
            "origin": (float(roi_left), float(roi_top)),
            "scale": float(scale),
            "offset": (float(offset[0]), float(offset[1])),
            "resized_dims": (resized_w, resized_h),
        }

    def _get_zoomed_bbox_face(self, det_face):
        """Return persistent bbox-centered zoom crop for bounding-box editing."""
        frame_idx = self._globals.frame_index
        face_idx = self._globals.face_index
        state = self._bbox_zoom_state(det_face, frame_idx, face_idx)

        frame = self._globals.current_frame.image
        roi_left, roi_top, roi_right, roi_bottom = state["roi"]
        resized_w, resized_h = state["resized_dims"]

        frame_h, frame_w = frame.shape[:2]
        crop_left = max(0, roi_left)
        crop_top = max(0, roi_top)
        crop_right = min(frame_w, roi_right)
        crop_bottom = min(frame_h, roi_bottom)

        crop = frame[crop_top:crop_bottom, crop_left:crop_right]
        roi_w = max(1, roi_right - roi_left)
        roi_h = max(1, roi_bottom - roi_top)

        if crop.size == 0:
            crop = np.zeros((roi_h, roi_w, 3), dtype="uint8")
        else:
            crop = cv2.copyMakeBorder(
                crop,
                max(0, -roi_top),
                max(0, roi_bottom - frame_h),
                max(0, -roi_left),
                max(0, roi_right - frame_w),
                cv2.BORDER_CONSTANT,
            )

        interpolation = (
            cv2.INTER_CUBIC
            if resized_w > crop.shape[1] or resized_h > crop.shape[0]
            else cv2.INTER_AREA
        )
        return cv2.resize(crop, (resized_w, resized_h), interpolation=interpolation)

    def _update_tk_frame(self):
        """Place the currently held frame into :attr:`_tk_frame`."""
        img = cv2.resize(
            self._globals.current_frame.image,
            self._globals.current_frame.display_dims,
            interpolation=self._globals.current_frame.interpolation,
        )[..., 2::-1]
        padding = self._get_padding(img.shape[:2])
        if any(padding):
            img = cv2.copyMakeBorder(img, *padding, cv2.BORDER_CONSTANT)
        logger.trace("final shape: %s", img.shape)

        if self._tk_frame.height() != img.shape[0] or self._tk_frame.width() != img.shape[1]:
            self._resize_frame()

        self._tk_frame.paste(Image.fromarray(img))

    def _get_padding(self, size):
        """Obtain top/bottom/left/right padding for the display image."""
        pad_lt = (
            (self._globals.frame_display_dims[1] - size[0]) // 2,
            (self._globals.frame_display_dims[0] - size[1]) // 2,
        )
        raw_padding = (
            pad_lt[0],
            self._globals.frame_display_dims[1] - size[0] - pad_lt[0],
            pad_lt[1],
            self._globals.frame_display_dims[0] - size[1] - pad_lt[1],
        )
        padding = (
            max(0, int(raw_padding[0])),
            max(0, int(raw_padding[1])),
            max(0, int(raw_padding[2])),
            max(0, int(raw_padding[3])),
        )
        logger.debug(
            "Frame dimensions: %s, size: %s, padding: %s",
            self._globals.frame_display_dims,
            size,
            padding,
        )
        return padding

    def _resize_frame(self):
        """Resize cached PhotoImages and update canvas positioning."""
        logger.trace("Resizing video frame on resize event: %s", self._globals.frame_display_dims)
        placeholder = np.zeros((*reversed(self._globals.frame_display_dims), 3), dtype="uint8")
        self._tk_frame = ImageTk.PhotoImage(Image.fromarray(placeholder))
        self._tk_face = ImageTk.PhotoImage(Image.fromarray(placeholder))
        self._canvas.coords(
            self._image,
            self._globals.frame_display_dims[0] / 2,
            self._globals.frame_display_dims[1] / 2,
        )
        img = self._tk_face if self._current_view_mode == "face" else self._tk_frame
        self._canvas.itemconfig(self._image, image=img)
