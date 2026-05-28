#!/usr/bin/env python3
"""Mask Editor for the manual adjustments tool"""

from __future__ import annotations

import gettext
import tkinter as tk
import typing as T

import cv2
import numpy as np
from PIL import Image, ImageTk

from lib.utils import get_module_objects

from ._base import ControlPanelOption, Editor, logger

if T.TYPE_CHECKING:
    import numpy.typing as npt

    from lib import align
    from tools.manual import detected_faces


# LOCALES
_LANG = gettext.translation("tools.manual", localedir="locales", fallback=True)
_ = _LANG.gettext


class Mask(Editor):
    """The mask Editor.

    Edit a mask in the alignments file.

    Parameters
    ----------
    canvas
        The canvas that holds the image and annotations
    detected_faces
        The _detected_faces data for this manual session
    """

    _LASSO_PREVIEW_TAG: T.ClassVar[str] = "mask_lasso_preview"
    _LASSO_LINE_WIDTH: T.ClassVar[int] = 2

    def __init__(self, canvas: tk.Canvas, detected_faces: detected_faces.DetectedFaces) -> None:
        self._meta: dict[str, T.Any] = {}
        self._tk_faces: list[ImageTk.PhotoImage] = []
        self._internal_size = 512
        control_text = _(
            "Mask Editor\nEdit the mask."
            "\n - NB: For Landmark based masks (e.g. components/extended) it is "
            "better to make sure the landmarks are correct rather than editing the "
            "mask directly. Any change to the landmarks after editing the mask will "
            "override your manual edits."
        )
        key_bindings = {
            "[": lambda *e, i=False: self._adjust_brush_radius(increase=i),
            "]": lambda *e, i=True: self._adjust_brush_radius(increase=i),
            "<Control-Delete>": self._delete_mask,
        }
        super().__init__(
            canvas, detected_faces, control_text=control_text, key_bindings=key_bindings
        )
        # Bind control click for reverse painting
        self._canvas.bind("<Control-ButtonPress-1>", self._control_click)
        self._mask_type = self._set_tk_mask_change_callback()
        self._cursor_shape = self._set_tk_cursor_shape_change_callback()
        self._mouse_location = [self._get_cursor_shape(), False]

    @property
    def _opacity(self) -> float:
        """The mask opacity setting from the control panel from 0.0 - 1.0."""
        annotation = self.__class__.__name__
        return self._annotation_formats[annotation]["mask_opacity"].get() / 100.0

    @property
    def _brush_radius(self) -> int:
        """The radius of the brush to use as set in control panel options"""
        return self._control_vars["brush"]["BrushSize"].get()

    @property
    def _edit_mode(self) -> str:
        """The currently selected edit mode based on optional action button.
        One of "draw" or "erase" """
        action = [
            name
            for name, option in self._actions.items()
            if option["group"] == "paint" and option["tk_var"].get()
        ]
        return "draw" if not action else action[0]

    @property
    def _is_lasso(self) -> bool:
        """``True`` if the lasso fill tool is active."""
        return self._actions["lasso"]["tk_var"].get()

    @property
    def _cursor_color(self) -> str:
        """The hex code for the selected cursor color"""
        return self._control_vars["brush"]["CursorColor"].get()

    @property
    def _cursor_shape_name(self) -> str:
        """The selected cursor shape"""
        return self._control_vars["display"]["CursorShape"].get()

    def _add_actions(self) -> None:
        """Add the optional action buttons to the viewer. Current actions are Draw, Erase,
        Lasso and Zoom."""
        self._add_action(
            "magnify", "zoom", _("Magnify/De-magnify the View"), group=None, hotkey="M"
        )
        self._add_action("draw", "draw", _("Draw Tool"), group="paint", hotkey="D")
        self._add_action("erase", "erase", _("Erase Tool"), group="paint", hotkey="E")
        self._add_action("lasso", "lasso", _("Lasso Fill Tool"), group=None, hotkey="L")
        self._actions["magnify"]["tk_var"].trace(
            "w", lambda *e: self._globals.var_full_update.set(True)
        )

    def _add_controls(self) -> None:
        """Add the mask specific control panel controls.

        Current controls are:
          - the mask type to edit
          - the size of brush to use
          - the cursor display color
        """
        masks = sorted(msk.title() for msk in list(self._det_faces.available_masks))
        if not masks:
            masks = ["None"]
        editable_masks = [mask for mask in masks if mask != "None"]
        default = masks[0] if len(masks) == 1 else (editable_masks[0] if editable_masks else masks[0])
        self._add_control(
            ControlPanelOption(
                "Mask type",
                str,
                group="Display",
                choices=masks,
                default=default,
                is_radio=True,
                helptext=_("Select which mask to edit"),
            )
        )
        self._add_control(
            ControlPanelOption(
                "Brush Size",
                int,
                group="Brush",
                min_max=(1, 100),
                default=10,
                rounding=1,
                helptext=_("Set the brush size. ([ - decrease, ] - increase)"),
            )
        )
        self._add_control(
            ControlPanelOption(
                "Cursor Color",
                str,
                group="Brush",
                choices="colorchooser",
                default="#ffffff",
                helptext=_("Select the brush cursor color."),
            )
        )
        self._add_control(
            ControlPanelOption(
                "Cursor Shape",
                str,
                group="Display",
                choices=["Circle", "Rectangle"],
                default="Circle",
                is_radio=True,
                helptext=_("Select a shape for masking cursor."),
            )
        )
        self._add_control(
            ControlPanelOption(
                "Delete Mask",
                str,
                group="Actions",
                button_command=self._delete_mask,
                helptext=_("Delete the selected mask entry from the target face. (Ctrl+Delete)"),
            )
        )

    def _control_click(self, event: tk.Event) -> None:
        """Handle control-click for reverse paint behaviour."""
        if self._is_lasso:
            self._lasso_click(event, mode="erase")
            return
        self._paint(event, mode="erase")

    def _get_cursor_shape(self) -> str:
        """Get the cursor shape based on the current selection."""
        return self._cursor_shape_name.lower()

    def _set_tk_mask_change_callback(self) -> tk.StringVar:
        """Set the callback for changing mask type."""
        tk_var = self._control_vars["display"]["MaskType"]
        tk_var.trace("w", lambda *e: self._mask_type_callback())
        return tk_var

    def _set_tk_cursor_shape_change_callback(self) -> tk.StringVar:
        """Set the callback for changing cursor shape."""
        tk_var = self._control_vars["display"]["CursorShape"]
        tk_var.trace("w", lambda *e: self._cursor_shape_callback())
        return tk_var

    def _mask_type_callback(self) -> None:
        """Callback for updating mask display when type changes."""
        self._globals.var_full_update.set(True)

    def _cursor_shape_callback(self) -> None:
        """Callback for updating cursor display when shape changes."""
        self._mouse_location = [self._get_cursor_shape(), False]
        self._globals.var_full_update.set(True)


__all__ = get_module_objects(__name__)
