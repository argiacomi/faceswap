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
from lib.utils import get_module_objects

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
    def _current_nav_frame_count(self):
        """int: The current frame count for the transport slider"""
        return self._nav["scale"].cget("to") + 1

    def nav_scale_callback(self, *args, reset_progress=True):  # pylint:disable=unused-argument
        """Adjust transport slider scale for different filters. Hide or display optional filter
        controls.
        """
        self._display_frame.pack_threshold_slider()
        if reset_progress:
            self.stop_playback()
        frame_count = self._det_faces.filter.count
        if self._current_nav_frame_count == frame_count:
            logger.trace("Filtered count has not changed. Returning")
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

    def increment_frame(self, frame_count=None, is_playing=False):
        """Update The frame navigation position to the next frame based on filter."""
        if not is_playing:
            self.stop_playback()
        position = self._get_safe_frame_index()
        face_count_change = not self._det_faces.filter.frame_meets_criteria
        if face_count_change:
            position -= 1
        frame_count = self._det_faces.filter.count if frame_count is None else frame_count
        if not face_count_change and (frame_count == 0 or position == frame_count - 1):
            logger.debug("End of Stream. Not incrementing")
            self.stop_playback()
            return
        self._globals.var_transport_index.set(min(position + 1, max(0, frame_count - 1)))

    def decrement_frame(self):
        """Update The frame navigation position to the previous frame based on filter."""
        self.stop_playback()
        position = self._get_safe_frame_index()
        face_count_change = not self._det_faces.filter.frame_meets_criteria
        if not face_count_change and (self._det_faces.filter.count == 0 or position == 0):
            logger.debug("End of Stream. Not decrementing")
            return
        self._globals.var_transport_index.set(
            min(max(0, self._det_faces.filter.count - 1), max(0, position - 1))
        )

    def _get_safe_frame_index(self):
        """Obtain the current frame position from the var_transport_index variable in
        a safe manner (i.e. handle for non-numeric)

        Returns
        -------
        int
            The current transport frame index
        """
        try:
            retval = self._globals.var_transport_index.get()
        except tk.TclError as err:
            if "expected floating-point" not in str(err):
                raise
            val = str(err).rsplit(" ", maxsplit=1)[-1].replace('"', "")
            retval = "".join(ch for ch in val if ch.isdigit())
            retval = 0 if not retval else int(retval)
            self._globals.var_transport_index.set(retval)
        return retval

    def goto_first_frame(self):
        """Go to the first frame that meets the filter criteria."""
        self.stop_playback()
        position = self._globals.var_transport_index.get()
        if position == 0:
            return
        self._globals.var_transport_index.set(0)

    def goto_last_frame(self):
        """Go to the last frame that meets the filter criteria."""
        self.stop_playback()
        position = self._globals.var_transport_index.get()
        frame_count = self._det_faces.filter.count
        if position == frame_count - 1:
            return
        self._globals.var_transport_index.set(frame_count - 1)


class BackgroundImage:
    """The background image of the canvas"""

    def __init__(self, canvas):
        self._canvas = canvas
        self._globals = canvas._globals
        self._det_faces = canvas._det_faces
        placeholder = np.ones((*reversed(self._globals.frame_display_dims), 3), dtype="uint8")
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

    def refresh(self, view_mode):
        """Update the displayed frame.

        Parameters
        ----------
        view_mode: ["frame", "face"]
            The currently active editor's selected view mode.
        """
        self._switch_image(view_mode)
        logger.trace("Updating background frame")
        getattr(self, f"_update_tk_{self._current_view_mode}")()

    def _switch_image(self, view_mode):
        """Switch the image between the full frame image and the zoomed face image.

        Parameters
        ----------
        view_mode: ["frame", "face"]
            The currently active editor's selected view mode.
        """
        if view_mode == self._current_view_mode and (
            self._canvas.active_editor.zoomed_centering == self._zoomed_centering
        ):
            return
        new_centering = self._canvas.active_editor.zoomed_centering
        if view_mode == "frame" or new_centering != self._zoomed_centering:
            self._canvas._bbox_zoom_state = None  # pylint:disable=protected-access
        self._zoomed_centering = new_centering
        logger.trace(
            "Switching background image from '%s' to '%s'",
            self._current_view_mode,
            view_mode,
        )
        img = getattr(self, f"_tk_{view_mode}")
        self._canvas.itemconfig(self._image, image=img)
        self._globals.set_zoomed(view_mode == "face")
        self._globals.set_face_index(0)

    def _update_tk_face(self):
        """Update the currently zoomed face."""
        face = self._get_zoomed_face()
        padding = self._get_padding(face.shape[:2])
        face = cv2.copyMakeBorder(face, *padding, cv2.BORDER_CONSTANT)
        if self._tk_frame.height() != face.shape[0]:
            self._resize_frame()

        logger.trace("final shape: %s", face.shape)
        self._tk_face.paste(Image.fromarray(face))

    def _get_zoomed_face(self):
        """Get the zoomed face or a blank image if no faces are available.

        For most editors this remains an AlignedFace crop. For the bounding-box
        editor, use a bbox-centered crop with 2x bbox width and 2x bbox height,
        so the displayed bbox can remain an axis-aligned rectangle in its own
        zoom coordinate system.
        """
        frame_idx = self._globals.frame_index
        face_idx = self._globals.face_index
        faces_in_frame = self._det_faces.face_count_per_index[frame_idx]
        size = min(self._globals.frame_display_dims)

        if face_idx + 1 > faces_in_frame:
            logger.debug(
                "Resetting face index to 0 for more faces in frame than current index: ("
                "faces_in_frame: %s, zoomed_face_index: %s",
                faces_in_frame,
                face_idx,
            )
            self._globals.set_face_index(0)
            face_idx = 0

        if faces_in_frame == 0:
            face = np.ones((size, size, 3), dtype="uint8")
        else:
            det_face = self._det_faces.current_faces[frame_idx][face_idx]
            if self._zoomed_centering == "bbox":
                face = self._get_zoomed_bbox_face(det_face)
            else:
                face = AlignedFace(
                    det_face.landmarks_xy,
                    image=self._globals.current_frame.image,
                    centering=self._zoomed_centering,
                    size=size,
                ).face
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
        """Create bbox zoom geometry once when entering bbox zoom.

        The ROI is intentionally based on the bbox at zoom-entry time. It must
        not follow subsequent bbox edits, otherwise the image/overlay coordinate
        system moves under the user's cursor.
        """
        assert det_face.left is not None and det_face.top is not None
        assert det_face.right is not None and det_face.bottom is not None

        display_w, display_h = display_dims
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

        return cv2.resize(
            crop,
            (resized_w, resized_h),
            interpolation=self._globals.current_frame.interpolation,
        )

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

        if self._tk_frame.height() != img.shape[0]:
            self._resize_frame()

        self._tk_frame.paste(Image.fromarray(img))

    def _get_padding(self, size):
        """Obtain the Left, Top, Right, Bottom padding required to place the square face or frame
        in to the Photo Image

        Returns
        -------
        tuple
            The (Left, Top, Right, Bottom) padding to apply to the face image in pixels
        """
        pad_lt = (
            (self._globals.frame_display_dims[1] - size[0]) // 2,
            (self._globals.frame_display_dims[0] - size[1]) // 2,
        )
        padding = (
            pad_lt[0],
            self._globals.frame_display_dims[1] - size[0] - pad_lt[0],
            pad_lt[1],
            self._globals.frame_display_dims[0] - size[1] - pad_lt[1],
        )
        logger.debug(
            "Frame dimensions: %s, size: %s, padding: %s",
            self._globals.frame_display_dims,
            size,
            padding,
        )
        return padding

    def _resize_frame(self):
        """Resize the :attr:`_tk_frame`, attr:`_tk_face` photo images, update the canvas to
        offset the image correctly.
        """
        logger.trace("Resizing video frame on resize event: %s", self._globals.frame_display_dims)
        placeholder = np.ones((*reversed(self._globals.frame_display_dims), 3), dtype="uint8")
        self._tk_frame = ImageTk.PhotoImage(Image.fromarray(placeholder))
        self._tk_face = ImageTk.PhotoImage(Image.fromarray(placeholder))
        self._canvas.coords(
            self._image,
            self._globals.frame_display_dims[0] / 2,
            self._globals.frame_display_dims[1] / 2,
        )
        img = self._tk_face if self._current_view_mode == "face" else self._tk_frame
        self._canvas.itemconfig(self._image, image=img)


__all__ = get_module_objects(__name__)
