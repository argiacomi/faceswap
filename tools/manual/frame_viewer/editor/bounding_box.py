#!/usr/bin/env python3
# mypy: disable-error-code="attr-defined, index, union-attr"
"""Bounding Box Editor for the manual adjustments tool"""

import gettext
import platform
from functools import partial

import numpy as np

from lib.gui.custom_widgets import RightClickMenu
from tools.manual.aligners import available_aligners, default_aligner

from ._base import ControlPanelOption, Editor, logger

# LOCALES
_LANG = gettext.translation("tools.manual", localedir="locales", fallback=True)
_ = _LANG.gettext


class BoundingBox(Editor):
    """The Bounding Box Editor.

    Adjusting the bounding box feeds the aligner to generate new 68 point landmarks.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The canvas that holds the image and annotations
    detected_faces: :class:`~tools.manual.detected_faces.DetectedFaces`
        The _detected_faces data for this manual session
    """

    def __init__(self, canvas, detected_faces):
        self._tk_aligner = None
        self._right_click_menu = RightClickMenu(
            [_("Delete Face(s)")], [self._delete_context_face], ["Del"]
        )
        self._context_face_index = None
        control_text = _(
            "Bounding Box Editor\nEdit the bounding box being fed into the aligner "
            "to recalculate the landmarks.\n\n"
            " - Grab the corner anchors to resize the bounding box.\n"
            " - Click and drag the bounding box to relocate.\n"
            " - Click in empty space to create a new bounding box.\n"
            " - Right click a bounding box to delete a face.\n"
            " - Use Magnify to edit the selected face's bounding box in zoomed view."
        )
        key_bindings = {"<Delete>": self._delete_current_face}
        key_bindings["<BackSpace>"] = self._delete_current_face
        key_bindings["<Command-BackSpace>"] = self._delete_current_face
        key_bindings["<Command-Delete>"] = self._delete_current_face
        super().__init__(
            canvas, detected_faces, control_text=control_text, key_bindings=key_bindings
        )
        self.zoomed_centering = "bbox"

    @property
    def _corner_order(self):
        """dict: The position index of bounding box corners"""
        return {
            0: ("top", "left"),
            1: ("top", "right"),
            2: ("bottom", "right"),
            3: ("bottom", "left"),
        }

    @property
    def _bounding_boxes(self):
        """list: The :func:`tkinter.Canvas.coords` for all displayed bounding boxes."""
        item_ids = self._canvas.find_withtag("bb_box")
        return [
            self._canvas.coords(item_id)
            for item_id in item_ids
            if self._canvas.itemcget(item_id, "state") != "hidden"
        ]

    def _current_faces(self):
        """Return faces for the current frame, or an empty tuple for stale frame state."""
        frame_index = self._globals.frame_index
        current_faces = self._det_faces.current_faces
        if frame_index < 0 or frame_index >= len(current_faces):
            return ()
        return current_faces[frame_index]

    def _valid_face_index(self, face_index: int) -> bool:
        """Return whether ``face_index`` exists on the current frame."""
        return 0 <= int(face_index) < len(self._current_faces())

    def _face_from_index(self, face_index: int):
        """Return the current frame face for ``face_index``, or ``None`` if stale."""
        if not self._valid_face_index(face_index):
            logger.debug(
                "Ignoring stale bounding-box face index. frame_index: %s face_index: %s faces: %s",
                self._globals.frame_index,
                face_index,
                len(self._current_faces()),
            )
            return None
        return self._current_faces()[face_index]

    def _aligner_name(self) -> str:
        """Return the selected aligner, falling back to the default before controls are ready."""
        return str(self._tk_aligner.get()) if self._tk_aligner is not None else default_aligner()

    def _display_shift_to_source(self, shift: tuple[int, int], face_index: int) -> tuple[int, int]:
        """Convert a display-space drag delta to source-frame pixels."""
        if self._globals.is_zoomed:
            face = self._face_from_index(face_index)
            if face is None:
                return (0, 0)
            _, scale, _ = self._bbox_zoom_geometry(face)
        else:
            scale = self._globals.current_frame.scale

        if not scale:
            return (0, 0)
        return (int(round(shift[0] / scale)), int(round(shift[1] / scale)))

    def _add_controls(self):
        """Controls for feeding the Aligner. Exposes Normalization Method as a parameter."""
        align_ctl = ControlPanelOption(
            "Aligner",
            str,
            group="Aligner",
            choices=available_aligners(),
            default=default_aligner(),
            is_radio=True,
            helptext=_(
                "Aligner to use. Model-based aligners will usually obtain better alignments, "
                "but cv2-dnn can be useful if these cannot get decent alignments and you want "
                "to set a base to edit from."
            ),
        )
        self._tk_aligner = align_ctl.tk_var
        self._add_control(align_ctl)

        norm_ctl = ControlPanelOption(
            "Normalization method",
            str,
            group="Aligner",
            choices=["none", "clahe", "hist", "mean"],
            default="hist",
            is_radio=True,
            helptext=_(
                "Normalization method to use for feeding faces to the aligner. This can "
                "help the aligner better align faces with difficult lighting conditions. "
                "Different methods will yield different results on different sets. NB: "
                "This does not impact the output face, just the input to the aligner."
                "\n\tnone: Don't perform normalization on the face."
                "\n\tclahe: Perform Contrast Limited Adaptive Histogram Equalization on "
                "the face."
                "\n\thist: Equalize the histograms on the RGB channels."
                "\n\tmean: Normalize the face colors to the mean."
            ),
        )
        var = norm_ctl.tk_var
        self._add_trace(
            var,
            "write",
            lambda *e, v=var: self._det_faces.extractor.set_normalization_method(v.get()),
        )
        self._add_control(norm_ctl)

    def _add_actions(self):
        """Add the zoom action for detailed bounding-box editing."""
        self._add_action(
            "magnify", "zoom", _("Magnify/Demagnify the View"), group=None, hotkey="M"
        )
        self._add_trace(self._actions["magnify"]["tk_var"], "write", self._toggle_zoom)

    def _toggle_zoom(self, *args):  # pylint:disable=unused-argument
        """Trigger a redraw when zoom mode changes."""
        self._globals.var_full_update.set(True)

    def update_annotation(self):
        """Get the latest bounding box data from alignments and update."""
        key = "bb_box"
        color = self._control_color
        if self._globals.is_zoomed:
            self.hide_annotation("bb_box")
            self.hide_annotation("bb_anc_dsp")
            self.hide_annotation("bb_anc_grb")

        for idx, face in enumerate(self._face_iterator):
            face_index = self._globals.face_index if self._globals.is_zoomed else idx
            if self._globals.is_zoomed:
                box = self._zoomed_bounding_box(face)
            else:
                box = np.array([(face.left, face.top), (face.right, face.bottom)])
                box = self._scale_to_display(box).astype("int32").flatten()
            kwargs = {"outline": color, "width": 1}
            logger.trace(
                "frame_index: %s, face_index: %s, box: %s, kwargs: %s",
                self._globals.frame_index,
                face_index,
                box,
                kwargs,
            )
            self._object_tracker(key, "rectangle", face_index, box, kwargs)
            self._update_anchor_annotation(face_index, box, color)
        logger.trace("Updated bounding box annotations")

    def _zoomed_bounding_box(self, face):
        """Return bbox display coords in bbox-centered zoom space.

        Because the zoom crop is exactly 2x bbox width and 2x bbox height, the
        bbox remains axis-aligned and centered in the zoom view. No face-pose
        affine transform is applied.
        """
        assert face.left is not None and face.top is not None
        assert face.right is not None and face.bottom is not None

        origin, scale, offset = self._bbox_zoom_geometry(face)
        corners = np.array(
            [[face.left, face.top], [face.right, face.bottom]],
            dtype="float32",
        )
        display = ((corners - origin) * scale) + offset
        return np.rint(display).astype("int32").flatten()

    def _display_corners_to_source(self, coords, face_idx=None):
        """Convert display/zoomed box coords into original-frame corner coordinates."""
        coords = np.asarray(coords, dtype="float32").flatten()
        left, top, right, bottom = coords
        left, right = sorted((left, right))
        top, bottom = sorted((top, bottom))
        corners = np.array(
            [[left, top], [right, top], [right, bottom], [left, bottom]],
            dtype="float32",
        )

        if self._globals.is_zoomed and face_idx is not None:
            face = self._face_from_index(int(face_idx))
            if face is None:
                raise ValueError(f"Stale face index for bbox zoom conversion: {face_idx}")
            corners = self._scale_from_bbox_zoom(corners, face)
        else:
            corners = self.scale_from_display(corners)

        return np.rint(corners).astype("int32")

    def _update_anchor_annotation(self, face_index, bounding_box, color):
        """Update the anchor annotations for each corner of the bounding box.

        The anchors only display when the bounding box editor is active.

        Parameters
        ----------
        face_index: int
            The index of the face being annotated
        bounding_box: :class:`numpy.ndarray`
            The scaled bounding box to get the corner anchors for
        color: str
            The hex color of the bounding box line
        """
        if not self._is_active:
            self.hide_annotation("bb_anc_dsp")
            self.hide_annotation("bb_anc_grb")
            return
        fill_color = "gray"
        activefill_color = "white" if self._is_active else ""
        anchor_points = self._get_anchor_points(
            (
                (bounding_box[0], bounding_box[1]),
                (bounding_box[2], bounding_box[1]),
                (bounding_box[2], bounding_box[3]),
                (bounding_box[0], bounding_box[3]),
            )
        )
        for idx, (anc_dsp, anc_grb) in enumerate(zip(*anchor_points, strict=False)):
            dsp_kwargs = {"outline": color, "fill": fill_color, "width": 1}
            grb_kwargs = {
                "outline": "",
                "fill": "",
                "width": 1,
                "activefill": activefill_color,
            }
            dsp_key = f"bb_anc_dsp_{idx}"
            grb_key = f"bb_anc_grb_{idx}"
            self._object_tracker(dsp_key, "oval", face_index, anc_dsp, dsp_kwargs)
            self._object_tracker(grb_key, "oval", face_index, anc_grb, grb_kwargs)
        logger.trace("Updated bounding box anchor annotations")

    # << MOUSE HANDLING >>
    # Mouse cursor display
    def _update_cursor(self, event):
        """Set the cursor action.

        Update :attr:`_mouse_location` with the current cursor position and display appropriate
        icon.

        If the cursor is over a corner anchor, then pop resize icon.
        If the cursor is over a bounding box, then pop move icon.
        If the cursor is over the image, then pop add icon.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The current tkinter mouse event
        """
        if self._check_cursor_anchors():
            return
        if self._check_cursor_bounding_box(event):
            return
        if self._check_cursor_image(event):
            return

        self._canvas.config(cursor="")
        self._mouse_location = None

    def _check_cursor_anchors(self):
        """Check whether the cursor is over a corner anchor."""
        anchors = set(self._canvas.find_withtag("bb_anc_grb"))
        item_ids = set(self._canvas.find_withtag("current")).intersection(anchors)
        if not item_ids:
            return False

        item_id = list(item_ids)[0]
        tags = self._canvas.gettags(item_id)
        face_idx = int(next(tag for tag in tags if tag.startswith("face_")).split("_")[-1])
        if not self._valid_face_index(face_idx):
            self._mouse_location = None
            return False

        corner_idx = int(
            next(
                tag for tag in tags if tag.startswith("bb_anc_grb_") and "face_" not in tag
            ).split("_")[-1]
        )
        pos_x, pos_y = self._corner_order[corner_idx]
        self._canvas.config(cursor=f"{pos_x}_{pos_y}_corner")
        self._mouse_location = ("anchor", f"{face_idx}_{corner_idx}")
        return True

    def _check_cursor_bounding_box(self, event):
        """Check whether the cursor is over a bounding box."""
        item_ids = self._canvas.find_withtag("bb_box")
        for item_id in item_ids:
            if self._canvas.itemcget(item_id, "state") == "hidden":
                continue
            bbox = self._canvas.coords(item_id)
            if bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]:
                tags = self._canvas.gettags(item_id)
                face_idx = int(next(tag for tag in tags if tag.startswith("face_")).split("_")[-1])
                if not self._valid_face_index(face_idx):
                    continue
                self._canvas.config(cursor="fleur")
                self._mouse_location = ("box", str(face_idx))
                return True
        return False

    def _check_cursor_image(self, event):
        """Check whether the cursor is over the image.

        If it is, set the appropriate cursor type and set :attr:`_mouse_location` to:
        ("image", )

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event

        Returns
        -------
        bool
            ``True`` if cursor is over a bounding box otherwise ``False``
        """
        if self._globals.frame_index == -1 or self._globals.is_zoomed:
            return False
        display_dims = self._globals.current_frame.display_dims
        if (
            self._canvas.offset[0] <= event.x <= display_dims[0] + self._canvas.offset[0]
            and self._canvas.offset[1] <= event.y <= display_dims[1] + self._canvas.offset[1]
        ):
            self._canvas.config(cursor="plus")
            self._mouse_location = ("image",)
            return True
        return False

    # Mouse Actions
    def set_mouse_click_actions(self):
        """Add context menu to OS specific right click action."""
        super().set_mouse_click_actions()
        self._canvas.bind(
            "<Button-2>" if platform.system() == "Darwin" else "<Button-3>",
            self._context_menu,
        )

    def _drag_start(self, event):
        """The action to perform when the user starts clicking and dragging the mouse.

        If :attr:`_mouse_location` indicates a corner anchor, then the bounding box is resized
        based on the adjusted corner, and the alignments re-generated.

        If :attr:`_mouse_location` indicates a bounding box, then the bounding box is moved, and
        the alignments re-generated.

        If :attr:`_mouse_location` indicates being over the main image, then a new bounding box is
        created, and alignments generated.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.
        """
        if self._mouse_location is None:
            self._drag_data = {}
            self._drag_callback = None
            return
        if self._mouse_location[0] == "anchor":
            corner_idx = int(self._mouse_location[1].split("_")[-1])
            self._select_face(self._mouse_location[1].split("_")[0])
            self._drag_data["corner"] = self._corner_order[corner_idx]
            self._drag_callback = self._resize
        elif self._mouse_location[0] == "box":
            self._select_face(self._mouse_location[1])
            self._drag_data["current_location"] = (event.x, event.y)
            self._drag_callback = self._move
        elif self._mouse_location[0] == "image":
            self._create_new_bounding_box(event)
            # Refresh cursor and _mouse_location for new bounding box and reset _drag_start
            self._update_cursor(event)
            self._drag_start(event)

    def _drag_stop(self, event):  # pylint:disable=unused-argument
        """Realign moved bounding boxes once, then trigger thumbnail/grid updates."""
        try:
            if self._mouse_location is None:
                return
            if self._mouse_location[0] not in ("anchor", "box"):
                return

            face_idx = int(self._mouse_location[1].split("_")[0])
            if not self._valid_face_index(face_idx):
                self._mouse_location = None
                return

            frame_index = self._globals.frame_index
            updated = True
            if self._mouse_location[0] == "box" and self._drag_data.get("needs_realign"):
                face_tag = f"bb_box_face_{face_idx}"
                box = self._canvas.coords(face_tag)
                if not box:
                    logger.debug("Move realign ignored. No canvas coords for tag: %s", face_tag)
                    return
                try:
                    bbox = self._coords_to_bounding_box(box, face_idx)
                except ValueError:
                    logger.debug("Move realign ignored due to stale zoom state.", exc_info=True)
                    return
                updated = self._det_faces.update.bounding_box(
                    frame_index,
                    face_idx,
                    *bbox,
                    aligner=self._aligner_name(),
                )

            if updated:
                self._det_faces.update.post_edit_trigger(frame_index, face_idx)
        finally:
            super()._drag_stop(event)

    def _create_new_bounding_box(self, event):
        """Create a new bounding box when user clicks on image, outside of existing boxes.

        The bounding box is created as a square located around the click location, with dimensions
        1 quarter the size of the frame's shortest side

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event
        """
        size = min(self._globals.current_frame.display_dims) // 8
        box = (event.x - size, event.y - size, event.x + size, event.y + size)
        logger.debug("Creating new bounding box: %s ", box)
        self._det_faces.update.add(self._globals.frame_index, *self._coords_to_bounding_box(box))

    def _resize(self, event):
        """Resize a bounding box on a corner anchor drag event."""
        face_idx = int(self._mouse_location[1].split("_")[0])
        if not self._valid_face_index(face_idx):
            self._mouse_location = None
            return

        face_tag = f"bb_box_face_{face_idx}"
        box = self._canvas.coords(face_tag)
        if not box:
            logger.debug("Resize ignored. No canvas coords for tag: %s", face_tag)
            return

        logger.trace(
            "Face Index: %s, Corner Index: %s. Original ROI: %s",
            face_idx,
            self._drag_data["corner"],
            box,
        )
        limits = (
            partial(min, box[2] - 20),
            partial(min, box[3] - 20),
            partial(max, box[0] + 20),
            partial(max, box[1] + 20),
        )
        rect_xy_indices = [
            ("left", "top", "right", "bottom").index(pnt) for pnt in self._drag_data["corner"]
        ]
        box[rect_xy_indices[1]] = limits[rect_xy_indices[1]](event.x)
        box[rect_xy_indices[0]] = limits[rect_xy_indices[0]](event.y)

        try:
            bbox = self._coords_to_bounding_box(box, face_idx)
        except ValueError:
            logger.debug("Resize ignored due to stale zoom state.", exc_info=True)
            return

        self._det_faces.update.bounding_box(
            self._globals.frame_index,
            face_idx,
            *bbox,
            aligner=self._aligner_name(),
        )

    def _move(self, event):
        """Move a bounding box during drag and regenerate landmarks."""
        logger.trace("event: %s, mouse_location: %s", event, self._mouse_location)
        face_idx = int(self._mouse_location[1])
        if not self._valid_face_index(face_idx):
            self._mouse_location = None
            return

        shift = (
            event.x - self._drag_data["current_location"][0],
            event.y - self._drag_data["current_location"][1],
        )
        if shift == (0, 0):
            return

        face_tag = f"bb_box_face_{face_idx}"
        box = self._canvas.coords(face_tag)
        if not box:
            logger.debug("Move ignored. No canvas coords for tag: %s", face_tag)
            return

        moved_box = np.array(box) + (*shift, *shift)
        try:
            bbox = self._coords_to_bounding_box(moved_box, face_idx)
        except ValueError:
            logger.debug("Move ignored due to stale zoom state.", exc_info=True)
            return

        updated = self._det_faces.update.bounding_box(
            self._globals.frame_index,
            face_idx,
            *bbox,
            aligner=self._aligner_name(),
        )
        if updated:
            # Keep the item responsive even if the redraw callback is delayed.
            self._canvas.coords(face_tag, *moved_box)
            self._drag_data["current_location"] = (event.x, event.y)

    def _nudge_face_index(self):
        """Return the face index to nudge from hover/selection state."""
        frame_index = self._globals.frame_index
        current_faces = self._det_faces.current_faces
        if frame_index < 0 or frame_index >= len(current_faces):
            return None

        faces = current_faces[frame_index]
        if not faces:
            return None

        if self._mouse_location is not None and self._mouse_location[0] == "box":
            face_index = int(self._mouse_location[1].split("_")[0])
            if 0 <= face_index < len(faces):
                return face_index
            logger.debug(
                "Ignoring stale bounding box hover index. frame_index: %s face_index: %s faces: %s",
                frame_index,
                face_index,
                len(faces),
            )
            self._mouse_location = None

        face_index = self._globals.face_index
        if 0 <= face_index < len(faces):
            return face_index

        return 0

    def nudge_bounding_box(self, direction, pixels=5, *args):  # pylint:disable=unused-argument
        """Move the active bounding box by a small number of display pixels."""
        shifts = {
            "left": (-pixels, 0),
            "right": (pixels, 0),
            "up": (0, -pixels),
            "down": (0, pixels),
        }
        if direction not in shifts:
            return

        face_idx = self._nudge_face_index()
        if face_idx is None:
            logger.debug("Bounding box nudge ignored. No active face.")
            return

        frame_index = self._globals.frame_index
        face_tag = f"bb_box_face_{face_idx}"
        coords = self._canvas.coords(face_tag)
        if not coords:
            logger.debug("Bounding box nudge ignored. No canvas coords for tag: %s", face_tag)
            return

        shift_x, shift_y = shifts[direction]
        nudged = np.array(coords) + (shift_x, shift_y, shift_x, shift_y)
        try:
            bbox = self._coords_to_bounding_box(nudged, face_idx)
        except ValueError:
            logger.debug("Bounding box nudge ignored due to stale zoom state.", exc_info=True)
            return

        updated = self._det_faces.update.bounding_box(
            frame_index,
            face_idx,
            *bbox,
            aligner=self._aligner_name(),
        )
        if updated:
            self._det_faces.update.post_edit_trigger(frame_index, face_idx)

    def _coords_to_bounding_box(self, coords, face_idx=None):
        """Convert display coordinates to DetectedFace bounding-box format.

        Frame view uses normal display scaling. Bbox zoom view uses the
        bbox-centered ROI scale/offset, leaving the edited rectangle in bbox
        zoom space and mapping it back to original-frame coordinates.
        """
        logger.trace("in: %s", coords)
        corners = self._display_corners_to_source(coords, face_idx)
        mins = np.min(corners, axis=0)
        maxes = np.max(corners, axis=0)
        logger.trace("out mins: %s maxes: %s", mins, maxes)
        return (
            int(mins[0]),
            int(maxes[0] - mins[0]),
            int(mins[1]),
            int(maxes[1] - mins[1]),
        )

    def _face_index_from_bbox_event(self, event):
        """Return the face index for the bounding box under a mouse event."""
        for item_id in self._canvas.find_withtag("bb_box"):
            if self._canvas.itemcget(item_id, "state") == "hidden":
                continue
            bbox = self._canvas.coords(item_id)
            if len(bbox) < 4:
                continue
            left, top, right, bottom = bbox[:4]
            if not (left <= event.x <= right and top <= event.y <= bottom):
                continue
            tags = self._canvas.gettags(item_id)
            face_tag = next((tag for tag in tags if tag.startswith("face_")), None)
            if face_tag is None:
                continue
            face_idx = int(face_tag.split("_")[-1])
            if self._valid_face_index(face_idx):
                return face_idx
        return None

    def _context_menu(self, event):
        """Create a right-click context menu for the bounding box under the click."""
        face_idx = self._face_index_from_bbox_event(event)
        if face_idx is None:
            self._context_face_index = None
            return
        self._context_face_index = face_idx
        self._mouse_location = ("box", str(face_idx))
        self._right_click_menu.popup(event)

    def _delete_context_face(self, *args):  # pylint:disable=unused-argument
        """Delete the face that was under the right-click event."""
        face_index = self._context_face_index
        if face_index is None:
            logger.debug("Context delete ignored. No clicked face target.")
            return
        if not self._valid_face_index(face_index):
            logger.debug("Context delete ignored for stale face index: %s", face_index)
            self._context_face_index = None
            return

        logger.debug(
            "Deleting right-clicked face. frame_index: %s face_index: %s",
            self._globals.frame_index,
            face_index,
        )
        self._det_faces.update.delete_many(self._globals.frame_index, (int(face_index),))
        self._globals.clear_selected_faces()
        self._context_face_index = None
        self._mouse_location = None

    def _delete_current_face(self, *args):  # pylint:disable=unused-argument
        """Delete the selected or hovered face(s)."""
        selected_faces = self._globals.selected_faces
        if selected_faces:
            logger.debug("Deleting selected faces: %s", selected_faces)
            self._det_faces.update.delete_many_frames(selected_faces)
            self._mouse_location = None
            return

        if self._mouse_location is None or self._mouse_location[0] != "box":
            logger.debug(
                "Delete called without valid location. _mouse_location: %s",
                self._mouse_location,
            )
            return

        face_index = int(self._mouse_location[1])
        if not self._valid_face_index(face_index):
            logger.debug("Delete ignored for stale face index: %s", face_index)
            self._mouse_location = None
            return

        face_indices = (face_index,)
        logger.debug(
            "Deleting hovered face. frame_index: %s face_indices: %s",
            self._globals.frame_index,
            face_indices,
        )
        self._det_faces.update.delete_many(self._globals.frame_index, face_indices)
        self._globals.clear_selected_faces()
        self._mouse_location = None
