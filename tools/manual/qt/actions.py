#!/usr/bin/env python3
"""Action registry for the Qt Manual Tool."""

from __future__ import annotations

import logging
import typing as T

logger = logging.getLogger(__name__)


class ManualAction(T.NamedTuple):
    """Static metadata for one Manual Tool editor action.

    The legacy Tk Manual Tool wires editor commands through Tk widgets and key
    bindings; the Qt shell mirrors the same surface through a registry so each
    action has a single source of truth for label, icon, shortcut, tooltip and
    availability semantics.  ``handler`` is the unbound method name on
    :class:`ManualToolWindow` that performs the action when triggered.

    Icons reuse the existing Qt theme icon cache where available.  Actions
    whose ``icon`` does not yet resolve are documented inline below.
    """

    key: str
    label: str
    handler: str
    shortcut: tuple[str, ...] = ()
    icon: str | None = None
    tooltip: str = ""
    separator_before: bool = False
    toolbar_visible: bool = True
    """When ``False``, the action is registered for its shortcut but not added
    to the toolbar — used by keyboard-only commands (nudge arrows, etc.)."""
    focus_scope: str = "window"
    """Where the shortcut listens. ``"window"`` (default) fires from anywhere
    in the Manual Tool. ``"frame_view"`` parents the action to the frame view
    with :data:`Qt.WidgetWithChildrenShortcut` so the panel can claim arrow
    keys for its own navigation when focused."""


# Manual Tool action registry. The first six actions cover the legacy file/
# navigation surface. The remaining entries register editor operations
# (add/delete/copy/revert), the filter cycle, the F1-F5 editor modes, the
# F9/F10 annotation toggles and the legacy-tool fallback.  Icons marked
# "(missing)" in the comments fall back to a null QIcon — replace them with
# real PNGs in lib/gui/.cache/icons/ when art lands.
MANUAL_ACTIONS: tuple[ManualAction, ...] = (
    ManualAction(
        key="save",
        label="Save",
        handler="save",
        shortcut=("Ctrl+S",),
        icon="save",
        tooltip="Save edits to the alignments file (Ctrl+S)",
    ),
    ManualAction(
        key="revert_frame",
        label="Revert Frame",
        handler="revert_current_frame",
        shortcut=("R",),
        icon="reload",
        tooltip="Revert edits in the current frame (R)",
        separator_before=True,
    ),
    ManualAction(
        key="first_frame",
        label="First Frame",
        handler="goto_first_frame",
        shortcut=("Home",),
        icon="beginning",
        tooltip="Jump to the first frame (Home)",
        separator_before=True,
    ),
    ManualAction(
        key="previous_frame",
        label="Previous",
        handler="_previous_frame",
        shortcut=("Z",),
        icon="prev",
        tooltip="Previous frame (Z)",
    ),
    ManualAction(
        key="next_frame",
        label="Next",
        handler="_next_frame",
        shortcut=("X",),
        icon="next",
        tooltip="Next frame (X)",
    ),
    ManualAction(
        key="last_frame",
        label="Last Frame",
        handler="goto_last_frame",
        shortcut=("End",),
        icon="end",
        tooltip="Jump to the last frame (End)",
    ),
    ManualAction(
        key="play_pause",
        label="Play / Pause",
        handler="toggle_play",
        shortcut=("Space",),
        icon="play",
        tooltip="Play / pause playback (Space)",
    ),
    ManualAction(
        key="copy_prev_face",
        label="Copy From Previous",
        handler="copy_prev_face",
        shortcut=("C",),
        icon="copy_prev",
        tooltip="Copy alignment from the previous frame (C)",
        separator_before=True,
    ),
    ManualAction(
        key="copy_next_face",
        label="Copy From Next",
        handler="copy_next_face",
        shortcut=("V",),
        icon="copy_next",
        tooltip="Copy alignment from the next frame (V)",
    ),
    ManualAction(
        key="delete_face",
        label="Delete Face",
        handler="delete_active_face",
        shortcut=("Delete", "Backspace"),
        icon="clear",  # No dedicated trash icon yet; reusing the "clear" glyph.
        tooltip="Delete the active face (Delete)",
    ),
    ManualAction(
        key="add_face",
        label="Add Face",
        handler="add_face_at_center",
        shortcut=("Ctrl+N",),
        icon=None,  # (missing) — no dedicated add-face icon in the cache.
        tooltip="Add a new face at the center of the current frame (Ctrl+N)",
    ),
    ManualAction(
        key="undo_edit",
        label="Undo",
        handler="undo_edit",
        shortcut=("Ctrl+Z",),
        icon=None,  # (missing) — no dedicated undo icon in the cache.
        tooltip="Undo the last edit (Ctrl+Z)",
        separator_before=True,
    ),
    ManualAction(
        key="redo_edit",
        label="Redo",
        handler="redo_edit",
        shortcut=("Ctrl+Shift+Z", "Ctrl+Y"),
        icon=None,  # (missing) — no dedicated redo icon in the cache.
        tooltip="Redo the last undone edit (Ctrl+Shift+Z)",
    ),
    ManualAction(
        key="cycle_filter",
        label="Filter",
        handler="cycle_filter_mode",
        shortcut=("F",),
        icon=None,  # (missing) — no filter icon in the cache.
        tooltip="Cycle the navigation filter mode (F)",
        separator_before=True,
    ),
    ManualAction(
        key="set_view_mode",
        label="View",
        handler="set_editor_view",
        shortcut=("F1",),
        icon="view",
        tooltip="Switch to View editor (F1)",
        separator_before=True,
    ),
    ManualAction(
        key="set_boundingbox_mode",
        label="Bounding Box",
        handler="set_editor_boundingbox",
        shortcut=("F2",),
        icon="boundingbox",
        tooltip="Switch to Bounding Box editor (F2)",
    ),
    ManualAction(
        key="set_extractbox_mode",
        label="Extract Box",
        handler="set_editor_extractbox",
        shortcut=("F3",),
        icon="extractbox",
        tooltip="Switch to Extract Box editor (F3)",
    ),
    ManualAction(
        key="set_landmarks_mode",
        label="Landmarks",
        handler="set_editor_landmarks",
        shortcut=("F4",),
        icon="landmarks",
        tooltip="Switch to Landmarks editor (F4)",
    ),
    ManualAction(
        key="set_mask_mode",
        label="Mask",
        handler="set_editor_mask",
        shortcut=("F5",),
        icon="mask",
        tooltip="Switch to Mask editor (F5)",
    ),
    ManualAction(
        key="cycle_annotation",
        label="Annotation",
        handler="cycle_annotation_display",
        shortcut=("F9",),
        icon=None,  # (missing) — no annotation icon in the cache.
        tooltip="Cycle annotation display (F9)",
        separator_before=True,
    ),
    ManualAction(
        key="zoom_in",
        label="Zoom In",
        handler="zoom_in",
        shortcut=("=", "+"),
        icon="zoom",
        tooltip="Zoom in on the frame view (+)",
        separator_before=True,
    ),
    ManualAction(
        key="zoom_out",
        label="Zoom Out",
        handler="zoom_out",
        shortcut=("-",),
        icon=None,  # (missing) — no dedicated zoom-out icon.
        tooltip="Zoom out of the frame view (-)",
    ),
    ManualAction(
        key="reset_view",
        label="Reset View",
        handler="reset_view",
        shortcut=("0",),
        icon=None,  # (missing) — reuse default until art lands.
        tooltip="Reset zoom and pan (0)",
    ),
    ManualAction(
        key="magnify_active_face",
        label="Magnify Active Face",
        handler="magnify_active_face",
        shortcut=("M",),
        icon=None,
        tooltip="Fit the active face's bbox to the frame view (M)",
    ),
    # Mask editor (F5, #101) — Draw / Erase actions + brush size shortcuts.
    ManualAction(
        key="mask_draw",
        label="Mask: Draw",
        handler="set_mask_draw_mode",
        shortcut=("B",),
        icon=None,
        tooltip="Switch the Mask editor to Draw mode (B)",
    ),
    ManualAction(
        key="mask_erase",
        label="Mask: Erase",
        handler="set_mask_erase_mode",
        shortcut=("D",),
        icon=None,
        tooltip="Switch the Mask editor to Erase mode (D)",
    ),
    ManualAction(
        key="brush_size_increase",
        label="Brush Size +",
        handler="increase_brush_size",
        shortcut=("]",),
        icon=None,
        tooltip="Increase Mask brush size (])",
    ),
    ManualAction(
        key="brush_size_decrease",
        label="Brush Size -",
        handler="decrease_brush_size",
        shortcut=("[",),
        icon=None,
        tooltip="Decrease Mask brush size ([)",
    ),
    ManualAction(
        key="nudge_up",
        label="Nudge Up",
        handler="nudge_up_one",
        shortcut=("Up",),
        icon=None,
        tooltip="Nudge the active face up one source pixel (Up)",
        toolbar_visible=False,
        focus_scope="frame_view",
    ),
    ManualAction(
        key="nudge_down",
        label="Nudge Down",
        handler="nudge_down_one",
        shortcut=("Down",),
        icon=None,
        tooltip="Nudge the active face down one source pixel (Down)",
        toolbar_visible=False,
        focus_scope="frame_view",
    ),
    ManualAction(
        key="nudge_left",
        label="Nudge Left",
        handler="nudge_left_one",
        shortcut=("Left",),
        icon=None,
        tooltip="Nudge the active face left one source pixel (Left)",
        toolbar_visible=False,
        focus_scope="frame_view",
    ),
    ManualAction(
        key="nudge_right",
        label="Nudge Right",
        handler="nudge_right_one",
        shortcut=("Right",),
        icon=None,
        tooltip="Nudge the active face right one source pixel (Right)",
        toolbar_visible=False,
        focus_scope="frame_view",
    ),
    ManualAction(
        key="nudge_up_fast",
        label="Nudge Up (10px)",
        handler="nudge_up_fast",
        shortcut=("Shift+Up",),
        icon=None,
        tooltip="Nudge the active face up ten source pixels (Shift+Up)",
        toolbar_visible=False,
        focus_scope="frame_view",
    ),
    ManualAction(
        key="nudge_down_fast",
        label="Nudge Down (10px)",
        handler="nudge_down_fast",
        shortcut=("Shift+Down",),
        icon=None,
        tooltip="Nudge the active face down ten source pixels (Shift+Down)",
        toolbar_visible=False,
        focus_scope="frame_view",
    ),
    ManualAction(
        key="nudge_left_fast",
        label="Nudge Left (10px)",
        handler="nudge_left_fast",
        shortcut=("Shift+Left",),
        icon=None,
        tooltip="Nudge the active face left ten source pixels (Shift+Left)",
        toolbar_visible=False,
        focus_scope="frame_view",
    ),
    ManualAction(
        key="nudge_right_fast",
        label="Nudge Right (10px)",
        handler="nudge_right_fast",
        shortcut=("Shift+Right",),
        icon=None,
        tooltip="Nudge the active face right ten source pixels (Shift+Right)",
        toolbar_visible=False,
        focus_scope="frame_view",
    ),
    ManualAction(
        key="extract_faces",
        label="Extract Faces",
        handler="extract_faces",
        shortcut=("Ctrl+E",),
        icon="task_save_as",
        tooltip="Extract aligned faces to a folder (Ctrl+E)",
        separator_before=True,
    ),
    ManualAction(
        key="legacy_tool",
        label="Open Legacy Tool",
        handler="launch_legacy",
        shortcut=(),
        icon=None,  # (missing) — no dedicated legacy icon.
        tooltip="Launch the legacy Tk Manual Tool",
        separator_before=True,
    ),
)
