# Qt Manual Tool — Behavior parity audit vs. Tk

Branch: `qt-manual-tool`. Snapshot date: 2026-05-22.

This document is a deliberate, behavior-by-behavior comparison of the legacy Tk
Manual Tool (`tools/manual/`) against the in-flight native Qt Manual Tool
(`lib/gui/qt_shell/manual_tool.py`). Each surface lists what Tk does, what Qt
does today, and the remaining work needed for parity. Use it as the single
source of truth for "what's left to ship" on the Qt path.

The summary at the bottom groups the gaps by ticket so triage can map each
remaining item directly onto #91 / #93 follow-ups or a dedicated mask /
extract-box / landmark ticket.

## Legend

- ✅ **Parity** — Qt matches Tk behavior in the relevant ways.
- ⚠️ **Partial** — present but missing one or more sub-behaviors documented inline.
- ❌ **Missing** — not implemented yet on the Qt side.
- ➖ **Intentionally deferred** — see the deferral note.

Each row cites the Tk source location (`file:line`) and the Qt source location
where applicable.

---

## 1. Startup & teardown

| Behavior | Tk | Qt | Status |
| --- | --- | --- | --- |
| CLI flags `-f`/`-a`/`-t`/`-s` accepted | `tools/manual/cli.py:ManualArgs` | `tools/manual/session.py:ManualSession.create` accepts the same set via `from_namespace` / `from_cli_values`. | ✅ |
| Background aligner initialization (HRNet / cv2-dnn) | `tools/manual/manual.py:Manual._wait_for_threads` | Qt has **no aligner** in process — bbox edits don't recompute landmarks. | ❌ |
| Background frame loader (image dir or video) | `tools/manual/manual.py:FrameLoader` | `lib/gui/qt_shell/manual_tool.py::_VideoFrameWorker` + `VideoFrameProvider` cover video; image-folder frames load on demand. | ✅ |
| Thumbnail cache check / regeneration | `tools/manual/manual.py:Manual` + `lib/align/thumbnails.py` | Qt checks `has_thumbnails()` in `_ManualStartupTask` but **does not regenerate** missing thumbnails (`-t` flag is a no-op). | ⚠️ |
| Startup progress feedback | None (busy cursor only) | Determinate status-bar progress bar (33/66/100 stages) + status text + console fanout. | ✅ (better than Tk) |
| Startup failure dialog | `sys.exit(1)` after log | `_on_startup_failed` shows status label, logs, hides progress, opens modal `QMessageBox.critical`. | ✅ |
| Window geometry restore / fullscreen on launch | `tools/manual/manual.py:_set_initial_layout` (940×600, fullscreen) | Fixed `resize(980, 680)`; no fullscreen, no `gui_config` integration. | ⚠️ |
| Dirty-close confirmation | Implicit (no prompt) | `closeEvent` prompts when there are unsaved edits. | ✅ (better than Tk) |

---

## 2. Frame navigation

| Behavior | Tk | Qt | Status |
| --- | --- | --- | --- |
| First / Prev / Next / Last buttons | `tools/manual/frame_viewer/control.py:Navigation` | `MANUAL_ACTIONS` registers `first_frame` / `previous_frame` / `next_frame` / `last_frame` handlers with `Home`/`Z`/`X`/`End` shortcuts. | ✅ |
| Play / Pause (Space) | `Navigation._play` (~60fps loop) | `MANUAL_ACTIONS["play_pause"]` toggles state but **no auto-advance timer**; it only flips the `is_playing` flag. | ⚠️ |
| Transport slider | `DisplayFrame._add_nav` (`Scale` + entry box, `0..filtered_max`) | **No transport slider**; users can't scrub. | ❌ |
| Jump-to-frame entry box | `DisplayFrame._add_nav` | **Missing.** | ❌ |
| Filtered frame counter label | `current / max` updates with filter | Status bar shows current frame; no filtered counter. | ⚠️ |
| Keyboard press auto-stops playback | `Manual._handle_key_press` | N/A (no playback loop). | ➖ |

---

## 3. View controls

| Behavior | Tk | Qt | Status |
| --- | --- | --- | --- |
| Zoom in / out | `BackgroundImage._switch_image` (editor-driven) | Mouse wheel + `+`/`-`/`0` shortcuts on the frame view; pure zoom on the source image. | ✅ |
| Reset view | implicit on editor change | `MANUAL_ACTIONS["reset_view"]` (`0`). | ✅ |
| Pan when zoomed | not pannable in Tk (re-fits) | `ManualFrameView` supports drag-pan with bounded offset + `OpenHand` cursor. | ✅ (better than Tk) |
| Auto-switch zoomed/full per editor | Mask + Landmarks force-zoom on enter | Qt does **not** auto-zoom on editor switch — the editor mode is just a label. | ❌ |
| Resize / fit-to-window on `<Configure>` | `DisplayFrame._resize` regenerates PhotoImage | Qt scales the pixmap on every paint via `_target_rect` (live resize). | ✅ |

---

## 4. Filter modes

| Behavior | Tk | Qt | Status |
| --- | --- | --- | --- |
| Six filter modes (All / Has / No / Single / Multiple / Misaligned) | `tools/manual/detected_faces.py:Filter` + `DisplayFrame._add_filter_section` | `cycle_filter_mode` rotates through the six names and writes to `editor_state.filter_mode`. | ⚠️ |
| Filter actually restricts frame navigation | yes — `Navigation` skips frames that don't match | **No.** Qt's filter mode is a label only; navigation still walks every source frame. | ❌ |
| Misaligned-distance slider (5–20) | shown only for Misaligned mode | **Missing.** | ❌ |
| Frame count label updates with filter | yes | No, no counter to update. | ❌ |
| Face panel reflects filter | `Grid.refresh_grid` reflows | Face panel always shows the current frame's faces; no filter awareness. | ❌ |

---

## 5. Editor modes (F1–F5)

| Behavior | Tk | Qt | Status |
| --- | --- | --- | --- |
| F1 — View | read-only annotations | F1 `set_view_mode` flips `editor_state.editor_mode = "View"`, **no annotation toggling** behind it. | ⚠️ |
| F2 — Bounding Box | drag bbox → aligner re-runs landmarks | Pointer move + resize via 8 handles **but no landmark recompute** (no aligner). | ⚠️ |
| F3 — Extract Box | translate/scale/rotate landmark cloud | **Not implemented.** Action exists, mode flag flips, no interaction surface. | ❌ |
| F4 — Landmarks | drag individual points, marquee select | **Not implemented.** Mode flag flips, no point drag. | ❌ |
| F5 — Mask | brush paint/erase, brush size + cursor shape | **Not implemented.** | ❌ |
| Editor-specific action buttons appear below toolbar | yes (Magnify for Landmarks, Draw/Erase for Mask) | No editor-specific button surface. | ❌ |
| Per-editor color controls (BoundingBox / Mesh / Landmarks / Mask colors) | `_base.Editor._default_colors` | **Missing.** Overlay uses hard-coded `#3aa0ff` / `#ffb000`. | ❌ |
| Per-editor annotation visibility rules | `FrameViewer.editor_display` per-editor dict | Overlay always renders bbox + landmarks; no per-mode visibility. | ❌ |
| Normalization-method radios (none/clahe/hist/mean) | `BoundingBox._add_controls` | **Missing.** | ❌ |
| Aligner-dropdown (HRNet / cv2-dnn / …) | `BoundingBox._add_controls` | **Missing.** | ❌ |

---

## 6. Annotation toggles (F9 / F10)

| Behavior | Tk | Qt | Status |
| --- | --- | --- | --- |
| F9 — toggle mesh overlay | `FacesActionsFrame.on_click` | `cycle_annotation` rotates through `("Mesh","Mask","")` but **does not render mesh**. | ⚠️ |
| F10 — toggle mask overlay | `FacesActionsFrame` | Same `cycle_annotation` rotation, mask not drawn. | ❌ |
| Mask-type dropdown (components / extended / …) | `DisplayFrame.tk_selected_mask` | **Missing.** | ❌ |
| Mask opacity slider 0–100% | yes | **Missing.** | ❌ |
| Mesh hidden if frame face count > previously-seen max | `FrameViewer.editor_display` | No such heuristic; not relevant until mesh draws. | ➖ |

---

## 7. Editing operations

| Behavior | Tk | Qt | Status |
| --- | --- | --- | --- |
| Click empty space → new face (BBox editor) | `BoundingBox._click` | **No pointer-add gesture.** The toolbar `add_face` button calls `add_face_at_center` (200-px centered box). | ⚠️ |
| Drag bbox interior to move | `BoundingBox._drag` | `ManualFrameView` emits `face_move_requested` → `editable.move_face`. | ✅ |
| Drag bbox corner to resize | `BoundingBox._drag` (4 corner anchors) | 8 handles (4 corners + 4 edges) → `face_resize_requested` → `editable.resize_face`. | ✅ (better than Tk) |
| Right-click face → delete | `BoundingBox._right_click` | **No context menu.** Delete is toolbar/`Del` only. | ⚠️ |
| `Del` key → delete active face | `Manual._handle_key_press` | `MANUAL_ACTIONS["delete_face"]` with `Del` shortcut. | ✅ |
| Copy Previous (`C`) | nearest earlier frame with faces | `MANUAL_ACTIONS["copy_prev_face"]`. | ✅ |
| Copy Next (`V`) | nearest later frame with faces | `MANUAL_ACTIONS["copy_next_face"]`. | ✅ |
| Revert frame (`R`) | reload from disk | `MANUAL_ACTIONS["revert_frame"]` → `editable.revert_frame`. | ✅ |
| Undo / Redo (`Ctrl+Z`/`Ctrl+Y`) | not present in Tk (Tk Manual has no global undo) | `editable.undo` / `editable.redo` stack + toolbar buttons. | ✅ (better than Tk) |
| Nudge active face by 1 / 10 px (Arrow / Shift+Arrow) | not present in Tk | New `nudge_*_one` / `nudge_*_fast` action set. | ✅ (better than Tk) |
| Extract-box translate / scale / rotate | `ExtractBox._drag` | **Missing.** | ❌ |
| Landmark drag single point | `Landmarks._drag` | **Missing.** | ❌ |
| Landmark marquee multi-select then drag | `Landmarks._drag` | **Missing.** | ❌ |
| Mask paint / erase brush | `Mask._drag` | **Missing.** | ❌ |
| Adjustable brush size (`[` / `]`) | yes | **Missing.** | ❌ |
| `Ctrl+click` inverts paint/erase | `Mask._drag` (modifier handling) | **Missing.** | ❌ |
| Magnify toggle (`M`) in Landmarks editor | `Landmarks._magnify` | **Missing.** | ❌ |

---

## 8. Pointer interactions

| Behavior | Tk | Qt | Status |
| --- | --- | --- | --- |
| Click empty space → select face by hit-test | `BoundingBox._click` | `ManualFrameView.clicked_at` → `editable.hit_test` → `editor_state.face_index`. | ✅ |
| Cursor change over bbox body | "fleur" (move) | `Qt.SizeAllCursor` on the active face's bbox interior. | ✅ |
| Cursor change over bbox handles | corner-specific sizing | `SizeFDiag` / `SizeBDiag` / `SizeVer` / `SizeHor` from `_cursor_for_handle`. | ✅ |
| Cursor change over pannable region | not relevant in Tk (no pan) | `Qt.OpenHandCursor` / `Qt.ClosedHandCursor`. | ✅ |
| Cursor change over landmark point | "hand2" | **Missing** (no landmark interaction yet). | ❌ |
| Cursor change over mask brush area | custom circle/square outline | **Missing.** | ❌ |
| Modifier-key gestures (Ctrl-paint, marquee select) | Tk Manual respects | **Missing.** | ❌ |
| Right-click context menu | per-editor | **Missing.** | ❌ |

---

## 9. Keyboard shortcuts

Coverage cross-checked against the Tk inventory in §`Keyboard Shortcuts`:

| Key | Tk action | Qt action | Status |
| --- | --- | --- | --- |
| `Z` | Previous frame | `previous_frame` | ✅ |
| `X` | Next frame | `next_frame` | ✅ |
| `Space` | Play/Pause | `play_pause` (state-flip only) | ⚠️ |
| `Home` | First frame | `first_frame` | ✅ |
| `End` | Last frame | `last_frame` | ✅ |
| `F` | Cycle filter mode | `cycle_filter` (state-flip only) | ⚠️ |
| `F1`–`F5` | Switch editor mode | `set_view_mode` / `set_boundingbox_mode` / `set_extractbox_mode` / `set_landmarks_mode` / `set_mask_mode` (state-flip only for F3–F5) | ⚠️ |
| `F9` | Toggle Mesh | `cycle_annotation` (no rendering) | ⚠️ |
| `F10` | Toggle Mask | `cycle_annotation` (no rendering) | ⚠️ |
| `C` | Copy prev | `copy_prev_face` | ✅ |
| `V` | Copy next | `copy_next_face` | ✅ |
| `R` | Revert | `revert_frame` | ✅ |
| `Ctrl+S` | Save | `save` | ✅ |
| `Delete` | Delete face | `delete_face` | ✅ |
| `[`/`]` | Brush size | — | ❌ |
| `M` | Magnify (Landmarks) | — | ❌ |
| `B` | Paint mode (Mask) | — | ❌ |
| `D` | Erase mode (Mask) | — | ❌ |
| `↑`/`↓` | Scroll face panel | bound to `nudge_up` / `nudge_down` (face move, not scroll) | ⚠️ — semantic mismatch |
| `PgUp`/`PgDn` | Scroll face panel page | — | ❌ |
| `Ctrl+Z` / `Ctrl+Y` / `Ctrl+Shift+Z` | — | Undo / Redo | Qt-only (better than Tk) |
| `Shift+↑`/`↓`/`←`/`→` | — | `nudge_*_fast` (10-px) | Qt-only (better than Tk) |

**Conflict to resolve:** Tk uses arrow keys to scroll the face panel; Qt
re-bound them to nudge the active face. We need to either (a) move nudge to a
different chord (e.g. `Alt+arrow`) and restore arrow-scroll on the face panel,
or (b) make nudge fire only when the frame view has keyboard focus and let the
face panel handle arrows when *it* has focus. **Recommend (b)** so neither
behavior is lost.

---

## 10. Face viewer (thumbnail panel)

| Behavior | Tk | Qt | Status |
| --- | --- | --- | --- |
| Thumbnail grid for full session (all frames) | `tools/manual/face_viewer/frame.py:FacesViewer` — every face in every frame | `FaceThumbnailPanel` only shows **the current frame's faces** as a horizontal strip. | ❌ — fundamental scope gap |
| Face-size dropdown (Tiny/Small/Med/Large/XL) | `FacesViewer.face_size` | Hard-coded 72-px icon. | ❌ |
| Hover preview with selection halo | `face_viewer/interact.py:HoverBox` | List widget's default highlight only. | ⚠️ |
| Click → set frame & face index | yes | Click sets `face_index` only; the panel doesn't navigate frames because it's frame-scoped. | ⚠️ |
| F9 / F10 annotations overlaid on thumbnails | `FacesActionsFrame` | **Missing.** | ❌ |
| Mouse wheel scroll | yes | List widget supports default scroll, but it's horizontal one-row. | ⚠️ |
| `Up`/`Down`/`PgUp`/`PgDn` scroll | yes | **Missing** (and arrow keys are claimed by nudge — see §9). | ❌ |
| Right-click context menu | `face_viewer/frame.py:ContextMenu` | **Missing.** | ❌ |
| Grid filter awareness | yes | **Missing** (no filter awareness anywhere). | ❌ |

The cross-frame thumbnail grid is the single largest missing surface — it's
roughly equivalent to a whole sub-application in Tk.

---

## 11. Persistence & save

| Behavior | Tk | Qt | Status |
| --- | --- | --- | --- |
| Save (`Ctrl+S`) | `DetectedFaces.save` | `ManualAlignmentsHandle.persist` writes the alignments file. | ✅ |
| Sparse-alignment frame mapping (image folder) | n/a in Tk | `frame_name_for_index` resolves by source-list ordering. | ✅ |
| Video-frame name synthesis for new alignments | n/a in Tk | `ManualSession.frame_name_for_index` derives `<basename>_<NNNNNN><ext>`. | ✅ |
| Save button enabled only when dirty | `tk_unsaved` flag | Qt tracks `editor_state.unsaved` + `dirty_changed` signal. | ✅ |
| Extract Faces button (folder picker + extraction) | `DetectedFaces.extract` | **Missing.** | ❌ |
| Auto-save on interval | absent in Tk | absent in Qt | ➖ |

---

## 12. Status / feedback

| Behavior | Tk | Qt | Status |
| --- | --- | --- | --- |
| Save button state | enabled when dirty | reflected via `dirty_changed` → status bar | ✅ |
| Copy-prev / copy-next button gating | disabled when no candidate | **Not gated** — buttons always enabled, action no-ops with status message. | ⚠️ |
| Revert button gating | disabled when frame unedited | **Not gated** — action no-ops. | ⚠️ |
| Play icon swap | `play` ↔ `pause` | Static icon (no play loop). | ⚠️ |
| Tooltip hover help | per-button | `MANUAL_ACTIONS.tooltip` populates QAction tooltips. | ✅ |
| Status-bar progress (startup) | none | Determinate progress (0 / 33 / 66 / 100). | ✅ (better than Tk) |
| Console fanout for INFO/WARN/ERROR | none — Tk has no console | Optional `console_logger` callback + Qt console router. | ✅ (better than Tk) |
| Modal error dialog on persist failure | `messagebox.showerror` (some paths) | `QMessageBox.critical` on save failure, startup failure. | ✅ |
| Action-failure status messages (move, resize, nudge with no selection) | n/a | "No active face to nudge" / "Resize failed" / "No frame to edit". | ✅ |

---

## 13. Optional features

| Behavior | Tk | Qt | Status |
| --- | --- | --- | --- |
| Multi-face per frame editing | yes | yes (face panel + `editor_state.face_index`) | ✅ |
| Video vs image-sequence parity | one loader handles both | Qt has separate paths but both work; video uses async worker. | ✅ |
| Legacy fallback / interop | n/a | `launch_legacy()` action spawns the Tk Manual via subprocess. | ✅ (better than Tk) |
| Per-user persisted GUI settings | tied to `lib/gui/utils/config.py` | Reads `lib.gui.gui_config` for theme/font/icon/layout (general Qt shell) but Manual Tool itself has no user prefs. | ⚠️ |
| Action-key debug logging | `logger.debug` / `trace` | Same coverage. | ✅ |

---

## 14. Summary — what's still outstanding

Grouped so each gap can be filed against the right ticket:

### Issue #91 — overlay editing interactions (still outstanding)

1. **Landmark editor (F4)** — single-point drag, marquee multi-select + drag.
2. **Extract-box editor (F3)** — translate / scale / rotate the landmark cloud
   via corner + interior + outside-corner drags.
3. **Mask editor (F5)** — brush paint + erase, brush-size keys, brush-shape
   cursor, `Ctrl+click` paint/erase invert, mask-type dropdown, opacity slider.
4. **Pointer-add gesture** — click empty space + drag to draw a new bbox
   (current toolbar `add_face` centers a 200-px box).
5. **Right-click context menus** — at least "Delete Face" on the frame view
   and face panel for Tk parity.
6. **Per-editor color controls + annotation visibility rules** — bbox, mesh,
   landmarks, mask color pickers; per-mode show/hide table.
7. **Aligner integration** — running an aligner after a bbox edit so landmarks
   stay valid (Normalization radios + aligner dropdown live here).
8. **Editor-specific action buttons** — Magnify (Landmarks), Draw/Erase
   (Mask), and any future editor-scoped buttons.
9. **Auto-zoom-to-face on editor enter** — Landmarks & Mask currently force
   the zoomed view in Tk; Qt does not.
10. **Arrow-key conflict** — let the face panel claim arrow keys when it has
    focus so panel scroll and face nudge can coexist.

### Issue #93 — progress / logging / startup feedback (still outstanding)

1. **Thumbnail-cache regeneration progress** — Qt currently only *checks*
   `has_thumbnails()`; honoring `-t` and regenerating with a determinate
   percent (per-frame increment) is still TODO.
2. **Save-operation progress** — long persists should show a busy/percent
   indicator and disable Save while in flight.
3. **Action progress for bulk operations** — copy-prev / copy-next from a
   far frame, future "extract faces" job, future bulk-revert.
4. **Determinate video decode queue feedback** — current `_VideoFrameWorker`
   has no UI surface beyond "Loading video frames…".
5. **End-to-end smoke for console fanout under a real Qt event loop** — only
   covered today by the unit-level dup-line regression test.

### Frame navigation (no parent ticket yet — file under #91 or a new ticket)

1. **Transport slider + frame-jump entry box** — currently missing entirely.
2. **Play/Pause auto-advance timer** — flag exists, loop does not.
3. **Filtered frame counter** — `current / max` label in the status bar.

### Filter modes (no parent ticket yet)

1. **Actually filter frame navigation** — Qt only flips the label.
2. **Misaligned-distance slider** for the Misaligned mode.
3. **Filter-aware face panel** — only show faces matching the active filter.

### Face viewer (no parent ticket yet — likely deserves its own)

1. **Cross-frame thumbnail grid** — Qt only shows the current frame's faces;
   Tk shows every face in the session, scrollable.
2. **Face-size dropdown** (Tiny/Small/Med/Large/XL).
3. **Hover preview with viewport** — Tk's HoverBox + expanded preview pane.
4. **F9 / F10 thumbnail annotations** (mesh / mask overlay on each thumb).
5. **Right-click context menu** on a thumbnail.
6. **`Up`/`Down`/`PgUp`/`PgDn` scroll** — colliding with the nudge bindings.
7. **Filter-aware face grid layout**.

### Persistence

1. **Extract Faces button** — folder picker + background extraction job +
   progress, matching Tk's `DetectedFaces.extract`.

### Status / feedback

1. **Action-availability gating** — disable Copy Prev / Copy Next / Revert
   when there's nothing to do (Tk does this; Qt no-ops with a status message).
2. **Play icon state** — once the play loop exists, swap the icon between
   "play" and "pause".

### Window / state

1. **Fullscreen-on-launch + window-geometry restore** to match Tk's initial
   layout behavior.
2. **Per-user Manual Tool preferences** — face size, default editor mode,
   default filter — these don't currently live in `gui_config`.

---

## Where Qt is already ahead of Tk

To balance the picture, these surfaces are net-better in the Qt build:

- Undo / redo stack (Tk Manual has no global undo).
- Keyboard nudge in 1-px and 10-px increments.
- Determinate startup progress + console fanout for INFO/WARN/ERROR.
- Pan + 8-handle bbox resize (Tk has only corner anchors and no pan).
- Sparse-alignment + video-frame name synthesis when persisting (Tk only
  handles dense image-folder + populated-video cases).
- Modal error dialogs on persist failure (Tk surfaces some errors only via
  log lines).
- Cross-shell legacy fallback launcher.

Keep these in mind when scoping Qt-side changes — none of them should
regress while filling the gaps above.

---

## Recommended ticket layout

The outstanding work is large enough that re-grouping #91 / #93 into
narrower follow-ups is probably worth doing. A reasonable cut:

1. **#91-a Mask editor** — biggest single missing surface; full brush stack.
2. **#91-b Extract-box editor** — translate/scale/rotate; needs ExtractBox UX.
3. **#91-c Landmark editor** — single + marquee drag; magnify toggle.
4. **#91-d Aligner integration** — re-run aligner on bbox edits; normalization
   + aligner dropdowns.
5. **#91-e Pointer-add face + right-click menus + arrow-key arbitration.**
6. **New: Frame navigation polish** — transport slider, frame entry box,
   playback loop.
7. **New: Filter awareness** — filter actually restricts navigation + face panel.
8. **New: Cross-frame face viewer** — Tk's bottom face grid is its own
   sub-app; rebuilding it in Qt is a chunky standalone ticket.
9. **#93-a Thumbnail regeneration progress.**
10. **#93-b Save-operation progress.**

Each of these can land independently because the GUI-neutral
`ManualEditableAlignments` and the `MANUAL_ACTIONS` registry already provide
the seams — the work is filling in the UX, not refactoring the spine.
