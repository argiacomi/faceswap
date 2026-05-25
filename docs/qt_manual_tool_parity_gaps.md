# Qt Manual Tool — Behavior parity audit vs. Tk

Branch: `qt-manual-tool`.
Updated snapshot: 2026-05-25.
Reviewed through #108 closure, #112 default/fallback policy work, #124 post-default polish closure, logger handler cleanup, persistence startup/save race stabilization, and a reported green full Qt Manual Tool test suite.

This document compares the legacy Tk Manual Tool (`tools/manual/`) against the native Qt Manual Tool (`lib/gui/qt_shell/manual_tool.py`). It tracks what is at parity, where Qt is already better, what remains open, and what has been explicitly deferred beyond the Qt default-switch target.

## Legend

- ✅ **Parity** — Qt matches Tk behavior in the relevant ways.
- ⚠️ **Partial / deferred polish** — present enough for the default switch, but a visual or interaction refinement remains tracked outside the closed parity tickets.
- ➖ **Deferred / not required** — intentionally not in the current default-switch target.

---

## 1. Startup & teardown

| Behavior                               | Tk                                             | Qt                                                                                                                                      | Status              |
| -------------------------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| CLI flags `-f`/`-a`/`-t`/`-s` accepted | Manual CLI parser accepts the legacy switches. | `ManualSession` supports namespace and command-panel value creation.                                                                    | ✅                  |
| Qt path avoids `tkinter` imports       | N/A                                            | Native Qt shell path is separate from legacy fallback.                                                                                  | ✅                  |
| Legacy fallback                        | N/A                                            | Qt action can launch the legacy Tk Manual Tool as a subprocess.                                                                         | ✅                  |
| Background frame loading               | Tk uses its frame-loader stack.                | Qt supports image-folder loading and async video-frame preparation.                                                                     | ✅                  |
| Existing alignments loading            | Tk loads through `DetectedFaces`.              | Qt seeds a GUI-neutral editable model from alignments.                                                                                  | ✅                  |
| Startup thumbnail check/regeneration   | Tk checks/regenerates thumbnails.              | Qt supports thumbnail regeneration with determinate progress.                                                                           | ✅                  |
| Thumbnail progress correctness         | N/A                                            | Per-event percent payloads avoid shared mutable progress state; #121 prevents the named `thumbs` stage from resetting the live percent. | ✅                  |
| Startup progress feedback              | Busy cursor/logging.                           | Determinate status-bar progress + status text + console fanout.                                                                         | ✅ (better than Tk) |
| Startup failure UX                     | Tk exits/logs.                                 | Qt reports status, logs, hides progress, and shows a modal error.                                                                       | ✅                  |
| Dirty-close confirmation               | Tk behavior is limited.                        | Qt prompts when unsaved edits exist and rejects unsafe close during stuck extract shutdown.                                             | ✅ (better than Tk) |
| Window geometry restore/fullscreen     | Tk has initial layout/fullscreen behavior.     | Qt persists Manual Tool geometry, window state, fullscreen/maximized state, and splitter sizes.                                         | ✅                  |

---

## 2. Frame navigation

| Behavior                        | Tk                                     | Qt                                                                                         | Status |
| ------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------ | ------ |
| First / Previous / Next / Last  | Buttons and shortcuts.                 | `Home` / `Z` / `X` / `End` actions exist and route through the active filtered frame list. | ✅     |
| Transport slider                | Tk scale widget.                       | Qt transport slider exists and reflects filtered totals.                                   | ✅     |
| Jump-to-frame entry             | Tk entry box.                          | Qt has a 1-based jump entry; Return emits one navigation request after #119 cleanup.       | ✅     |
| Play / Pause                    | Tk has playback loop.                  | Qt has play/pause state/timer plumbing and advances through filtered results.              | ✅     |
| Filtered frame counter          | Tk updates counter with active filter. | Qt updates transport/counter labels from the active filtered list.                         | ✅     |
| Navigation honors active filter | Tk skips frames according to filter.   | Qt restricts first/previous/next/last/playback to active filter results.                   | ✅     |

---

## 3. View controls

| Behavior                  | Tk                                                       | Qt                                                                            | Status              |
| ------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------- |
| Zoom in/out/reset         | Tk view updates through editor/viewer behavior.          | Qt supports wheel zoom plus shortcut reset.                                   | ✅                  |
| Pan when zoomed           | Tk is mostly fit-based.                                  | Qt supports bounded drag-pan.                                                 | ✅ (better than Tk) |
| Live resize/fit           | Tk regenerates image on configure.                       | Qt scales on paint.                                                           | ✅                  |
| Auto-zoom on editor enter | Tk Landmarks/Mask workflows force focused editing views. | Qt auto-fits the active face on Landmark/Mask entry and restores the prior view when leaving. | ✅                  |
| Per-user view preferences | Tk has broader GUI config integration.                   | Qt persists Manual Tool window/splitter state; broader app preference integration remains outside parity scope. | ✅                  |

---

## 4. Filter modes

| Behavior                    | Tk                                               | Qt                                                                                                                        | Status |
| --------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- | ------ |
| Filter mode names           | All / Has / No / Single / Multiple / Misaligned. | Qt cycles/stores the same modes.                                                                                          | ✅     |
| Filter restricts navigation | Yes.                                             | Qt transport/navigation/playback use `_filtered_frame_indices`.                                                           | ✅     |
| Misaligned threshold slider | Yes.                                             | Qt exposes a visible Misaligned Faces threshold slider with the Tk 5..20 range.                                           | ✅     |
| Filter-aware frame counter  | Yes.                                             | Qt counter/slider range reflects active filter results.                                                                   | ✅     |
| Empty-filter behavior       | Tk disables/skips navigation safely.             | Qt disables transport/navigation and surfaces a status message.                                                           | ✅     |
| Filter refresh after edits  | Tk filter results update as face data changes.   | Qt recomputes filters after editable model mutations, including non-current frames.                                       | ✅     |
| Filter-aware face grid      | Tk grid follows visible/filter state.            | Qt cross-frame face grid is populated from the active filtered frame list and refreshes on filter/edit/selection changes. | ✅     |

**Tickets:** #107 and #108 are implemented, covered by GUI regressions, and closed.

---

## 5. Editor modes (F1–F5)

| Mode              | Tk behavior                                                     | Qt behavior                                                                                                                                                       | Status |
| ----------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| F1 — View         | Read-only view/annotation mode.                                 | Qt mode exists and annotation visibility follows the editor/annotation matrix.                                                                                    | ✅     |
| F2 — Bounding Box | Move/resize bbox; optionally rerun aligner.                     | Qt supports move/resize, add/delete, undo/redo, nudge, copy/revert, aligner controls, normalization, auto-run, preload/rerun behavior, and safe failure handling. | ✅     |
| F3 — Extract Box  | Translate/scale/rotate landmark cloud.                          | Qt Extract Box editor is implemented and ticket #102 is closed.                                                                                                   | ✅     |
| F4 — Landmarks    | Drag points and selected groups.                                | Qt Landmark editor is implemented and ticket #103 is closed.                                                                                                      | ✅     |
| F5 — Mask         | Draw/erase brush, modifier invert, mask overlay, mask controls. | Qt supports painting, brush-size shortcuts, Ctrl invert, overlay preview, persisted mask decode/edit/save behavior, and touched-mask undo/revert tracking.        | ✅     |

---

## 6. Annotation toggles and overlays

| Behavior                    | Tk                            | Qt                                                                                                                                                                    | Status  |
| --------------------------- | ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| BBox overlay                | Yes.                          | Qt renders active/editable bboxes and handles.                                                                                                                        | ✅      |
| Landmark overlay            | Yes.                          | Qt renders landmarks and supports Landmark editor interactions.                                                                                                       | ✅      |
| Mesh overlay (`F9`)         | Yes.                          | Qt supports thumbnail mesh overlays in the cross-frame grid using `AlignedFace`/`LANDMARK_PARTS`; full-frame F9 exposes the landmark/mesh annotation visibility matrix. | ✅      |
| Mask overlay (`F10`)        | Yes.                          | Qt renders mask overlays for loaded/edited masks and cross-frame thumbnails use active mask type/opacity.                                                             | ✅      |
| Mask type dropdown          | Yes.                          | Qt has mask-type state/control behavior tied to loaded/edited masks.                                                                                                  | ✅      |
| Mask opacity control        | Yes.                          | State/render seam exists and remains covered by mask/grid tests.                                                                                                      | ✅      |
| Per-editor colors           | Tk editor defaults/config.    | Qt exposes per-editor overlay color override hooks used by the overlay painter.                                                                                         | ✅      |
| Per-editor visibility rules | Tk has editor display matrix. | Qt centralizes bbox/handle/landmark/mask visibility from editor mode plus annotation mode.                                                                              | ✅      |

---

## 7. Editing operations

| Behavior                           | Tk                                | Qt                                                                                                  | Status              |
| ---------------------------------- | --------------------------------- | --------------------------------------------------------------------------------------------------- | ------------------- |
| Click/select face by hit test      | Yes.                              | Qt hit-tests editable faces and updates active face.                                                | ✅                  |
| Add face from toolbar/action       | Yes.                              | Qt supports toolbar/action add and pointer-add in BBox mode.                                        | ✅                  |
| Drag bbox body to move             | Yes.                              | Implemented.                                                                                        | ✅                  |
| Drag bbox handles to resize        | Corner handles.                   | Qt supports 8 handles.                                                                              | ✅ (better than Tk) |
| Right-click delete/context menu    | Yes.                              | Qt has frame-view, current-frame panel, and cross-frame grid context-menu delete coverage.          | ✅                  |
| Delete key                         | Yes.                              | Implemented.                                                                                        | ✅                  |
| Copy Previous / Copy Next          | Yes.                              | Implemented.                                                                                        | ✅                  |
| Revert frame                       | Yes.                              | Implemented against editable history.                                                               | ✅                  |
| Undo / Redo                        | Tk has no global undo.            | Qt has undo/redo stacks.                                                                            | ✅ (better than Tk) |
| Keyboard nudge                     | Tk does not provide global nudge. | Qt supports 1px/10px nudge with focus/shortcut arbitration fixes.                                   | ✅ (better than Tk) |
| Extract-box translate/scale/rotate | Yes.                              | Implemented.                                                                                        | ✅                  |
| Landmark drag / group drag         | Yes.                              | Implemented.                                                                                        | ✅                  |
| Mask paint/erase                   | Yes.                              | Implemented with persistence/touched-state handling.                                                | ✅                  |
| Mask stroke as one undoable action | Expected editor feel.             | Touched-state regressions exist; grouping should remain covered if UX changes.                      | ✅                  |
| Magnify toggle                     | Tk Landmark editor has magnify.   | Qt `magnify_active_face` toggles between active-face fit and the previously captured view.          | ✅                  |

---

## 8. Pointer interactions

| Behavior                        | Tk                                            | Qt                                                                                         | Status  |
| ------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------ | ------- |
| Cursor over bbox body/handles   | Move/resize cursors.                          | Implemented.                                                                               | ✅      |
| Cursor over extract-box regions | Move/scale/rotate feedback.                   | Implemented with rotation-halo coverage.                                                   | ✅      |
| Cursor over landmarks           | Hand/select feedback.                         | Implemented for Landmark editor selection/move interactions.                               | ✅      |
| Mask brush cursor/preview       | Custom brush outline.                         | Qt has brush preview/paint feedback.                                                       | ✅      |
| Modifier gestures               | Tk supports Ctrl-paint and marquee selection. | Qt supports mask Ctrl-invert and landmark selection flows.                                 | ✅      |
| Right-click context menu        | Per-editor context menus.                     | Current-frame frame/panel and cross-frame grid context-menu paths are implemented/covered. | ✅      |

---

## 9. Keyboard shortcuts

| Key                 | Tk action              | Qt status                                                                                     |
| ------------------- | ---------------------- | --------------------------------------------------------------------------------------------- |
| `Z` / `X`           | Previous / Next frame  | ✅ filter-aware                                                                               |
| `Home` / `End`      | First / Last frame     | ✅ filter-aware                                                                               |
| `Space`             | Play/Pause             | ✅ filter-aware playback                                                                      |
| `F`                 | Cycle filter           | ✅ updates filtered navigation/counters                                                       |
| `F1`–`F5`           | Editor modes           | ✅                                                                                            |
| `F9` / `F10`        | Mesh / Mask annotation | ✅ thumbnail mask/mesh overlays plus frame-view annotation visibility state                 |
| `C` / `V`           | Copy previous / next   | ✅                                                                                            |
| `R`                 | Revert                 | ✅                                                                                            |
| `Ctrl+S`            | Save                   | ✅ async save path                                                                            |
| `Delete`            | Delete active face     | ✅                                                                                            |
| `[` / `]`           | Brush size             | ✅                                                                                            |
| `B` / `D`           | Mask draw / erase      | ✅                                                                                            |
| `M`                 | Magnify                | ✅ toggles active-face magnify/restore                                                        |
| Arrow keys          | Tk face-panel scroll   | ✅ Qt nudge/focus behavior is scoped; grid focus handles Arrow/Page scrolling                 |
| `Ctrl+Z` / `Ctrl+Y` | N/A in Tk              | ✅ Qt-only undo/redo                                                                          |

---

## 10. Face viewer / thumbnail panel

| Behavior                               | Tk                                      | Qt                                                                                        | Status |
| -------------------------------------- | --------------------------------------- | ----------------------------------------------------------------------------------------- | ------ |
| Current-frame thumbnail panel          | Tk shows/selects faces.                 | Qt current-frame panel selects faces and participates in context-menu coverage.           | ✅     |
| Cross-frame thumbnail grid             | Tk shows faces across the full session. | Qt shows a scrollable filtered-session grid with one item per visible face.               | ✅     |
| Face-size dropdown                     | Tiny/Small/Medium/Large/XL.             | Implemented.                                                                              | ✅     |
| Click thumbnail navigates frame + face | Yes.                                    | Implemented for the cross-frame grid.                                                     | ✅     |
| Hover preview                          | Tk has hover/preview behavior.          | Qt exposes visible hover state/status feedback.                                           | ✅     |
| Mesh/mask annotations on thumbnails    | Yes.                                    | Implemented with mask type/opacity and mesh groups from legacy landmark parts.            | ✅     |
| Mouse/keyboard scroll parity           | Yes.                                    | Grid focus supports wheel, Arrow, and Page Up/Down scrolling.                             | ✅     |
| Right-click thumbnail menu             | Yes.                                    | Grid context menu supports Delete Face and is covered.                                    | ✅     |
| Filter-aware grid                      | Yes.                                    | Grid contents derive from active filtered frame indices and refresh on filter/edit state. | ✅     |

**Ticket:** #108 is closed.

---

## 11. Persistence, save, and extraction

| Behavior                                    | Tk                                      | Qt                                                                                                                 | Status |
| ------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------ |
| Save alignments                             | Yes.                                    | Qt persists editable alignments and uses async save worker.                                                        | ✅     |
| Save progress/action gating                 | Limited.                                | Qt shows save busy state, blocks duplicate saves, and preserves dirty state on failure.                            | ✅     |
| Sparse image-folder alignment mapping       | N/A / less explicit.                    | Qt maps by source frame name.                                                                                      | ✅     |
| Video frame name synthesis                  | Tk dense video assumptions.             | Qt can synthesize video frame names for new alignments.                                                            | ✅     |
| Extract Faces workflow                      | Tk `DetectedFaces.extract`.             | Qt has folder picker, worker, progress, cancel, live editable targets, safe shutdown, and skipped-face accounting. | ✅     |
| Extract Faces dirty-state behavior          | Tk saves/uses current alignments state. | Qt chains save before extraction when dirty and extracts from live editable targets.                               | ✅     |
| Extract Faces cancellation                  | Tk path varies.                         | Qt Cancel button and close handling are present; #119 shutdown task closed.                                        | ✅     |
| Extract Faces skipped frame/face accounting | Tk summary style.                       | Qt distinguishes skipped source frames from skipped faces on readable frames.                                      | ✅     |
| Mask persistence                            | Tk saves masks in alignments.           | Qt loads/edit/saves touched masks without degrading untouched blobs.                                               | ✅     |
| Save/startup race safety                    | N/A.                                    | Persistence tests wait for startup/save workers before alignments-file assertions.                                 | ✅     |

---

## 12. Status / feedback

| Behavior                        | Tk                                   | Qt                                                                                                             | Status              |
| ------------------------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------- | ------------------- |
| Save button dirty-state gating  | Yes.                                 | Implemented.                                                                                                   | ✅                  |
| Startup progress                | Limited.                             | Implemented and more detailed.                                                                                 | ✅                  |
| Thumbnail regeneration progress | Present.                             | Implemented with live per-event percent and reset guard.                                                       | ✅                  |
| Extract Faces progress/cancel   | Present.                             | Implemented with worker, progress bar, Cancel button, safe close behavior.                                     | ✅                  |
| Copy/revert action availability | Tk disables some impossible actions. | Qt action gating and status feedback are present.                                                              | ✅                  |
| Play icon state                 | Tk swaps play/pause.                 | Qt swaps true play/pause icons, labels, tooltips, and status tips from playback state.                         | ✅                  |
| Console fanout                  | N/A.                                 | Qt console router/fanout exists; repeated `log_setup()` calls no longer duplicate alignment read/write output. | ✅ (better than Tk) |
| Modal errors                    | Partial.                             | Qt has modal startup/save/extract error surfaces.                                                              | ✅                  |

---

## 13. Remaining work grouped by ticket

### #112 — Migration docs/fallback policy

- Decide when Qt becomes default.
- Document legacy fallback expectations.
- Add release-note wording and final smoke checklist.
- Decide which remaining visual-polish gaps are explicit deferrals.

### #124 — Post-default visual/interactions polish

Closed in the Qt Manual Tool implementation:

- Manual Tool geometry, window state, fullscreen/maximized state, and splitter sizes persist through `QSettings`.
- Landmark and Mask editor entry auto-fits the active face and restores the prior view when leaving the detail editor.
- Per-editor overlay color override hooks feed bbox, active, landmark, selected-landmark, and mask tint rendering.
- Overlay visibility is centralized from editor mode plus annotation mode for bbox, handles, landmarks, and masks.
- Landmark magnify toggles between active-face fit and the previous view.
- Play/Pause uses the true play/pause icons and updates labels, tooltips, and status tips with playback state.

### #85 — Parent migration epic

- Keep open until #112 is closed and final Qt/fallback launch smoke is complete or recorded as the only remaining closure evidence.

---

## 14. Completed follow-up work since the original audit

- #101 Mask editor persistence/control work closed.
- #102 Extract Box editor parity closed.
- #103 Landmark editor parity closed.
- #104 Bounding Box aligner controls/preload/rerun behavior closed.
- #105 Pointer-add/context-menu/focus arbitration closed.
- #106 Frame navigation polish closed.
- #107 Filter-aware navigation, empty-filter handling, Misaligned threshold UI, transport/counter updates, and edit-triggered refresh closed.
- #108 Cross-frame face grid parity closed.
- #109 Thumbnail regeneration progress closed.
- #110 Save/progress parent scope covered by #115.
- #111 Extract Faces parent workflow closed.
- #113 Arrow-key/navigation shortcut collision closed.
- #114 Per-event thumbnail progress percent closed.
- #115 Async save worker/progress closed.
- #116 Extract Faces live editable state closed.
- #117 1-based jump entry closed.
- #118 Visible Extract Faces Cancel control closed.
- #119 Cleanup catchall closed.
- #120 Duplicate of #115 closed.
- #121 Thumbnail progress reset guard closed.
- #122 Persisted mask decode/edit path closed.
- #123 Mask touched-state undo/revert coverage closed.

---

## 15. Where Qt is already ahead of Tk

- Undo / redo stack.
- Keyboard nudge in 1px and 10px increments.
- Determinate startup and thumbnail progress.
- Async save and Extract Faces workers.
- Visible Extract Faces Cancel control and safe close handling.
- Console fanout for Manual Tool operations.
- Pan + 8-handle bbox resize.
- Sparse image-folder mapping and video frame-name synthesis.
- Better modal error handling for startup/save/extract failures.
- Legacy fallback launcher from the Qt shell.
- Filter-aware navigation with explicit empty-filter behavior.
- Cross-frame filtered face grid with mask/mesh thumbnail annotations.
- Non-blocking/explicit aligner preload and model-safe failure handling.
- Idempotent logging setup during repeated test/import initialization.

None of these should regress while #112 and #124 are completed.
