#!/usr/bin/env python3
"""Action registry for the Qt Manual Tool."""

from __future__ import annotations

import typing as T


class ManualAction(T.NamedTuple):
    """Static metadata for one Manual Tool editor action."""

    key: str
    label: str
    handler: str
    shortcut: tuple[str, ...] = ()
    icon: str | None = None
    tooltip: str = ""
    separator_before: bool = False
    toolbar_visible: bool = True
    focus_scope: str = "window"


MANUAL_ACTIONS: tuple[ManualAction, ...] = (
    ManualAction(
        "save", "Save", "save", ("Ctrl+S",), "save", "Save edits to the alignments file (Ctrl+S)"
    ),
    ManualAction(
        "revert_frame",
        "Revert Frame",
        "revert_current_frame",
        ("R",),
        "reload",
        "Revert edits in the current frame (R)",
        True,
    ),
    ManualAction(
        "first_frame",
        "First Frame",
        "goto_first_frame",
        ("Home",),
        "beginning",
        "Jump to the first frame (Home)",
        True,
    ),
    ManualAction(
        "previous_frame", "Previous", "_previous_frame", ("Z",), "prev", "Previous frame (Z)"
    ),
    ManualAction("next_frame", "Next", "_next_frame", ("X",), "next", "Next frame (X)"),
    ManualAction(
        "last_frame",
        "Last Frame",
        "goto_last_frame",
        ("End",),
        "end",
        "Jump to the last frame (End)",
    ),
    ManualAction(
        "play_pause",
        "Play / Pause",
        "toggle_play",
        ("Space",),
        "play",
        "Play / pause playback (Space)",
    ),
    ManualAction(
        "copy_prev_face",
        "Copy From Previous",
        "copy_prev_face",
        ("C",),
        "copy_prev",
        "Copy alignment from the previous frame (C)",
        True,
    ),
    ManualAction(
        "copy_next_face",
        "Copy From Next",
        "copy_next_face",
        ("V",),
        "copy_next",
        "Copy alignment from the next frame (V)",
    ),
    ManualAction(
        "delete_face",
        "Delete Face",
        "delete_active_face",
        ("Delete", "Backspace"),
        "clear",
        "Delete the active face (Delete)",
    ),
    ManualAction(
        "add_face",
        "Add Face",
        "add_face_at_center",
        ("Ctrl+N",),
        "new",
        "Add a new face at the center of the current frame (Ctrl+N)",
    ),
    ManualAction(
        "undo_edit",
        "Undo",
        "undo_edit",
        ("Ctrl+Z",),
        "reload2",
        "Undo the last edit (Ctrl+Z)",
        True,
    ),
    ManualAction(
        "redo_edit",
        "Redo",
        "redo_edit",
        ("Ctrl+Shift+Z", "Ctrl+Y"),
        "reload3",
        "Redo the last undone edit (Ctrl+Shift+Z)",
    ),
    ManualAction(
        "cycle_filter",
        "Filter",
        "cycle_filter_mode",
        ("F",),
        "context",
        "Cycle the navigation filter mode (F)",
        True,
    ),
    ManualAction(
        "set_view_mode",
        "View",
        "set_editor_view",
        ("F1",),
        "view",
        "Switch to View editor (F1)",
        True,
    ),
    ManualAction(
        "set_boundingbox_mode",
        "Bounding Box",
        "set_editor_boundingbox",
        ("F2",),
        "boundingbox",
        "Switch to Bounding Box editor (F2)",
    ),
    ManualAction(
        "set_extractbox_mode",
        "Extract Box",
        "set_editor_extractbox",
        ("F3",),
        "extractbox",
        "Switch to Extract Box editor (F3)",
    ),
    ManualAction(
        "set_landmarks_mode",
        "Landmarks",
        "set_editor_landmarks",
        ("F4",),
        "landmarks",
        "Switch to Landmarks editor (F4)",
    ),
    ManualAction(
        "set_mask_mode", "Mask", "set_editor_mask", ("F5",), "mask", "Switch to Mask editor (F5)"
    ),
    ManualAction(
        "cycle_annotation",
        "Mesh",
        "toggle_mesh_annotation",
        ("F9",),
        None,
        "Toggle face-grid Mesh overlay (F9)",
        True,
    ),
    ManualAction(
        "toggle_mask_annotation",
        "Mask Overlay",
        "toggle_mask_annotation",
        ("F10",),
        None,
        "Toggle face-grid Mask overlay (F10)",
    ),
    ManualAction(
        "zoom_in", "Zoom In", "zoom_in", ("=", "+"), "zoom", "Zoom in on the frame view (+)", True
    ),
    ManualAction(
        "zoom_out", "Zoom Out", "zoom_out", ("-",), "zoom", "Zoom out of the frame view (-)"
    ),
    ManualAction(
        "reset_view", "Reset View", "reset_view", ("0",), "reload", "Reset zoom and pan (0)"
    ),
    ManualAction(
        "magnify_active_face",
        "Magnify Active Face",
        "magnify_active_face",
        ("M",),
        "zoom",
        "Fit the active face's bbox to the frame view (M)",
    ),
    ManualAction(
        "mask_draw",
        "Mask: Draw",
        "set_mask_draw_mode",
        ("D",),
        "draw",
        "Switch the Mask editor to Draw mode (D)",
    ),
    ManualAction(
        "mask_erase",
        "Mask: Erase",
        "set_mask_erase_mode",
        ("E",),
        "erase",
        "Switch the Mask editor to Erase mode (E)",
    ),
    ManualAction(
        "brush_size_increase",
        "Brush Size +",
        "increase_brush_size",
        ("]",),
        "zoom",
        "Increase Mask brush size (])",
    ),
    ManualAction(
        "brush_size_decrease",
        "Brush Size -",
        "decrease_brush_size",
        ("[",),
        "zoom",
        "Decrease Mask brush size ([)",
    ),
    ManualAction(
        "nudge_up",
        "Nudge Up",
        "nudge_up_one",
        ("Up",),
        None,
        "Nudge the active face up one source pixel (Up)",
        False,
        False,
        "frame_view",
    ),
    ManualAction(
        "nudge_down",
        "Nudge Down",
        "nudge_down_one",
        ("Down",),
        None,
        "Nudge the active face down one source pixel (Down)",
        False,
        False,
        "frame_view",
    ),
    ManualAction(
        "nudge_left",
        "Nudge Left",
        "nudge_left_one",
        ("Left",),
        None,
        "Nudge the active face left one source pixel (Left)",
        False,
        False,
        "frame_view",
    ),
    ManualAction(
        "nudge_right",
        "Nudge Right",
        "nudge_right_one",
        ("Right",),
        None,
        "Nudge the active face right one source pixel (Right)",
        False,
        False,
        "frame_view",
    ),
    ManualAction(
        "nudge_up_fast",
        "Nudge Up (10px)",
        "nudge_up_fast",
        ("Shift+Up",),
        None,
        "Nudge the active face up ten source pixels (Shift+Up)",
        False,
        False,
        "frame_view",
    ),
    ManualAction(
        "nudge_down_fast",
        "Nudge Down (10px)",
        "nudge_down_fast",
        ("Shift+Down",),
        None,
        "Nudge the active face down ten source pixels (Shift+Down)",
        False,
        False,
        "frame_view",
    ),
    ManualAction(
        "nudge_left_fast",
        "Nudge Left (10px)",
        "nudge_left_fast",
        ("Shift+Left",),
        None,
        "Nudge the active face left ten source pixels (Shift+Left)",
        False,
        False,
        "frame_view",
    ),
    ManualAction(
        "nudge_right_fast",
        "Nudge Right (10px)",
        "nudge_right_fast",
        ("Shift+Right",),
        None,
        "Nudge the active face right ten source pixels (Shift+Right)",
        False,
        False,
        "frame_view",
    ),
    ManualAction(
        "extract_faces",
        "Extract Faces",
        "extract_faces",
        ("Ctrl+E",),
        "task_save_as",
        "Extract aligned faces to a folder (Ctrl+E)",
        True,
    ),
    ManualAction(
        "legacy_tool",
        "Open Legacy Tool",
        "launch_legacy",
        (),
        None,
        "Launch the legacy Tk Manual Tool",
        True,
    ),
)
