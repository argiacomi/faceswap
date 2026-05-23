# Qt Manual Tool — Behavior parity audit vs. Tk

Branch: `qt-manual-tool`.  
Updated snapshot: 2026-05-23.  
Reviewed through the follow-up work that closed #119 and #121, including `4bcbe843f6dded444d43a2c699b0882151504c92`.

This document compares the legacy Tk Manual Tool (`tools/manual/`) against the in-flight native Qt Manual Tool (`lib/gui/qt_shell/manual_tool.py`). It is intended to track what is already at parity, what is better in Qt, and what still blocks switching the default Manual Tool launch path.

## Legend

- ✅ **Parity** — Qt matches Tk behavior in the relevant ways.
- ⚠️ **Partial** — present, but missing one or more important sub-behaviors.
- ❌ **Missing** — not implemented yet on the Qt side.
- ➖ **Deferred / not required** — intentionally not in the current parity target.

---

## 1. Startup & teardown

| Behavior | Tk | Qt | Status |
| --- | --- | --- | --- |
| CLI flags `-f`/`-a`/`-t`/`-s` accepted | Manual CLI parser accepts the legacy switches. | `ManualSession` supports namespace and command-panel value creation. | ✅ |
| Qt path avoids `tkinter` imports | N/A | Native Qt shell path is separate from legacy fallback. | ✅ |
| Legacy fallback | N/A | Qt action can launch the legacy Tk Manual Tool as a subprocess. | ✅ |
| Background frame loading | Tk uses its frame-loader stack. | Qt supports image-folder loading and async video-frame preparation. | ✅ |
| Existing alignments loading | Tk loads through `DetectedFaces`. | Qt seeds a GUI-neutral editable model from alignments. | ✅ |
| Startup thumbnail check/regeneration | Tk checks/regenerates thumbnails. | Qt supports thumbnail regeneration with determinate progress. | ✅ |
| Thumbnail progress correctness | N/A | Per-event percent payloads avoid shared mutable progress state; #121 prevents the named `thumbs` stage from resetting the live percent. | ✅ |
| Startup progress feedback | Busy cursor/logging. | Determinate status-bar progress + status text + console fanout. | ✅ (better than Tk) |
| Startup failure UX | Tk exits/logs. | Qt reports status, logs, hides progress, and shows a modal error. | ✅ |
| Dirty-close confirmation | Tk behavior is limited. | Qt prompts when unsaved edits exist. | ✅ (better than Tk) |
| Window geometry restore/fullscreen | Tk has initial layout/fullscreen behavior. | Qt still uses a fixed initial size and lacks Manual Tool-specific persisted geometry. | ⚠️ |

---

## 2. Frame navigation

| Behavior | Tk | Qt | Status |
| --- | --- | --- | --- |
| First / Previous / Next / Last | Buttons and shortcuts. | `Home` / `Z` / `X` / `End` actions exist. | ✅ |
| Transport slider | Tk scale widget. | Qt transport slider exists. | ✅ |
| Jump-to-frame entry | Tk entry box. | Qt has a 1-based jump entry; Return emits one navigation request after #119 cleanup. | ✅ |
| Play / Pause | Tk has playback loop. | Qt has play/pause state/timer plumbing; verify final icon/counter polish before default switch. | ⚠️ |
| Filtered frame counter | Tk updates counter with active filter. | Qt counter/status is not fully filter-aware. | ⚠️ |
| Navigation honors active filter | Tk skips frames according to filter. | Qt filter state exists but navigation filtering remains a gap. | ❌ |

---

## 3. View controls

| Behavior | Tk | Qt | Status |
| --- | --- | --- | --- |
| Zoom in/out/reset | Tk view updates through editor/viewer behavior. | Qt supports wheel zoom plus shortcut reset. | ✅ |
| Pan when zoomed | Tk is mostly fit-based. | Qt supports bounded drag-pan. | ✅ (better than Tk) |
| Live resize/fit | Tk regenerates image on configure. | Qt scales on paint. | ✅ |
| Auto-zoom on editor enter | Tk Landmarks/Mask workflows force focused editing views. | Qt does not consistently auto-zoom on editor switch. | ⚠️ |
| Per-user view preferences | Tk has broader GUI config integration. | Qt Manual Tool-specific preferences are still limited. | ⚠️ |

---

## 4. Filter modes

| Behavior | Tk | Qt | Status |
| --- | --- | --- | --- |
| Filter mode names | All / Has / No / Single / Multiple / Misaligned. | Qt cycles and stores the same mode names. | ⚠️ |
| Filter restricts navigation | Yes. | Not yet. | ❌ |
| Misaligned threshold slider | Yes. | Missing. | ❌ |
| Filter-aware face panel | Yes. | Missing. | ❌ |
| Filter-aware frame count | Yes. | Missing/partial. | ❌ |

**Remaining ticket:** #107.

---

## 5. Editor modes (F1–F5)

| Mode | Tk behavior | Qt behavior | Status |
| --- | --- | --- | --- |
| F1 — View | Read-only view/annotation mode. | Qt mode exists; annotation rendering remains partial. | ⚠️ |
| F2 — Bounding Box | Move/resize bbox; optionally rerun aligner. | Qt supports move/resize, add/delete, undo/redo, nudge, copy/revert. Aligner service/hooks exist but visible controls/preload polish remain. | ⚠️ |
| F3 — Extract Box | Translate/scale/rotate landmark cloud. | Qt Extract Box editor is implemented and ticket #102 is closed. | ✅ |
| F4 — Landmarks | Drag points and selected groups. | Qt Landmark editor is implemented and ticket #103 is closed. | ✅ |
| F5 — Mask | Draw/erase brush, modifier invert, mask overlay, mask controls. | Qt supports in-memory paint/erase, brush-size shortcuts, Ctrl invert, and overlay preview. Persistence and loaded-mask controls remain open under #101. | ⚠️ |

---

## 6. Annotation toggles and overlays

| Behavior | Tk | Qt | Status |
| --- | --- | --- | --- |
| BBox overlay | Yes. | Qt renders active/editable bboxes and handles. | ✅ |
| Landmark overlay | Yes. | Qt renders landmarks and supports Landmark editor interactions. | ✅ |
| Mesh overlay (`F9`) | Yes. | Mesh rendering remains partial/missing. | ❌ |
| Mask overlay (`F10`) | Yes. | Qt can render in-memory mask overlays; loaded-mask type controls/persistence remain missing. | ⚠️ |
| Mask type dropdown | Yes. | Missing/partial; #101 remains open. | ❌ |
| Mask opacity control | Yes. | State/render seam exists, but visible control parity should be verified under #101. | ⚠️ |
| Per-editor colors | Tk editor defaults/config. | Qt still mostly uses fixed overlay colors. | ❌ |
| Per-editor visibility rules | Tk has editor display matrix. | Qt visibility is simpler and still needs parity review. | ⚠️ |

---

## 7. Editing operations

| Behavior | Tk | Qt | Status |
| --- | --- | --- | --- |
| Click/select face by hit test | Yes. | Qt hit-tests editable faces and updates active face. | ✅ |
| Add face from toolbar/action | Tk supports bbox creation gestures. | Qt can add a centered face; pointer-add drag parity still needs review. | ⚠️ |
| Drag bbox body to move | Yes. | Implemented. | ✅ |
| Drag bbox handles to resize | Corner handles. | Qt supports 8 handles. | ✅ (better than Tk) |
| Right-click delete/context menu | Yes. | Still missing/partial. | ⚠️ |
| Delete key | Yes. | Implemented. | ✅ |
| Copy Previous / Copy Next | Yes. | Implemented. | ✅ |
| Revert frame | Yes. | Implemented against editable history. | ✅ |
| Undo / Redo | Tk has no global undo. | Qt has undo/redo stacks. | ✅ (better than Tk) |
| Keyboard nudge | Tk does not provide global nudge. | Qt supports 1px/10px nudge with focus/shortcut arbitration fixes. | ✅ (better than Tk) |
| Extract-box translate/scale/rotate | Yes. | Implemented. | ✅ |
| Landmark drag / group drag | Yes. | Implemented. | ✅ |
| Mask paint/erase | Yes. | Implemented in memory; save/load persistence remains open. | ⚠️ |
| Mask stroke as one undoable action | Expected editor feel. | Needs review under #101 if one drag still creates multiple undo entries. | ⚠️ |
| Magnify toggle | Tk Landmark editor has magnify. | Not yet at parity. | ❌ |

---

## 8. Pointer interactions

| Behavior | Tk | Qt | Status |
| --- | --- | --- | --- |
| Cursor over bbox body/handles | Move/resize cursors. | Implemented. | ✅ |
| Cursor over extract-box regions | Move/scale/rotate feedback. | Implemented; rotation-halo coverage should remain in tests. | ✅ |
| Cursor over landmarks | Hand/select feedback. | Implemented enough for Landmark editor; hover polish may remain. | ⚠️ |
| Mask brush cursor/preview | Custom brush outline. | Mask painting works; full brush cursor preview remains a #101 polish item. | ⚠️ |
| Modifier gestures | Tk supports Ctrl-paint and marquee selection. | Qt supports mask Ctrl-invert and landmark selection flows. | ✅/⚠️ |
| Right-click context menu | Per-editor context menus. | Missing/partial. | ⚠️ |

---

## 9. Keyboard shortcuts

| Key | Tk action | Qt status |
| --- | --- | --- |
| `Z` / `X` | Previous / Next frame | ✅ |
| `Home` / `End` | First / Last frame | ✅ |
| `Space` | Play/Pause | ⚠️ present, but verify final playback/icon polish |
| `F` | Cycle filter | ⚠️ state only until filter-aware navigation lands |
| `F1`–`F5` | Editor modes | ✅/⚠️ modes exist; Mask persistence still partial |
| `F9` / `F10` | Mesh / Mask annotation | ⚠️ mask partial, mesh missing |
| `C` / `V` | Copy previous / next | ✅ |
| `R` | Revert | ✅ |
| `Ctrl+S` | Save | ✅ async save path |
| `Delete` | Delete active face | ✅ |
| `[` / `]` | Brush size | ✅ |
| `B` / `D` | Mask draw / erase | ✅ |
| `M` | Magnify | ❌ |
| Arrow keys | Tk face-panel scroll | ⚠️ Qt nudge/focus behavior should stay scoped so face-panel scroll can coexist |
| `Ctrl+Z` / `Ctrl+Y` | N/A in Tk | ✅ Qt-only undo/redo |

---

## 10. Face viewer / thumbnail panel

| Behavior | Tk | Qt | Status |
| --- | --- | --- | --- |
| Cross-frame thumbnail grid | Tk shows faces across the full session. | Qt panel is still current-frame scoped. | ❌ |
| Face-size dropdown | Tiny/Small/Medium/Large/XL. | Missing. | ❌ |
| Click thumbnail navigates frame + face | Yes. | Current panel selects face; full cross-frame navigation is missing. | ⚠️ |
| Hover preview | Tk has hover/preview behavior. | Missing/partial. | ⚠️ |
| Mesh/mask annotations on thumbnails | Yes. | Missing. | ❌ |
| Mouse/keyboard scroll parity | Yes. | Partial; full grid not built yet. | ⚠️ |
| Right-click thumbnail menu | Yes. | Missing. | ❌ |
| Filter-aware grid | Yes. | Missing. | ❌ |

**Remaining ticket:** #108.

---

## 11. Persistence, save, and extraction

| Behavior | Tk | Qt | Status |
| --- | --- | --- | --- |
| Save alignments | Yes. | Qt persists editable alignments and uses async save worker. | ✅ |
| Save progress/action gating | Limited. | Qt shows save busy state, blocks duplicate saves, and preserves dirty state on failure. | ✅ |
| Sparse image-folder alignment mapping | N/A / less explicit. | Qt maps by source frame name. | ✅ |
| Video frame name synthesis | Tk dense video assumptions. | Qt can synthesize video frame names for new alignments. | ✅ |
| Extract Faces workflow | Tk `DetectedFaces.extract`. | Qt has folder picker, worker, progress, cancel, live editable targets, safe shutdown, and skipped-face accounting. | ✅ |
| Extract Faces cancellation | Tk path varies. | Qt Cancel button and close handling are present; #119 shutdown task closed. | ✅ |
| Extract Faces skipped frame/face accounting | Tk summary style. | Qt distinguishes skipped source frames from skipped faces on readable frames. | ✅ |
| Mask persistence | Tk saves masks in alignments. | Qt Mask editor currently needs loaded-mask seeding and mask blob writeback. | ❌ |

---

## 12. Status / feedback

| Behavior | Tk | Qt | Status |
| --- | --- | --- | --- |
| Save button dirty-state gating | Yes. | Implemented. | ✅ |
| Startup progress | Limited. | Implemented and more detailed. | ✅ |
| Thumbnail regeneration progress | Present. | Implemented with live per-event percent and reset guard. | ✅ |
| Extract Faces progress/cancel | Present. | Implemented with worker, progress bar, Cancel button, safe close behavior. | ✅ |
| Copy/revert action availability | Tk disables some impossible actions. | Qt still often allows action then no-ops with status text. | ⚠️ |
| Play icon state | Tk swaps play/pause. | Verify final Qt icon polish once playback is final. | ⚠️ |
| Console fanout | N/A. | Qt console router/fanout exists. | ✅ (better than Tk) |
| Modal errors | Partial. | Qt has modal startup/save/extract error surfaces. | ✅ |

---

## 13. Remaining work grouped by ticket

### #101 — Mask editor persistence and controls

- Seed editable mask state from existing on-disk alignments masks.
- Persist edited masks back into the alignments mask blob format.
- Populate mask type controls from loaded alignments.
- Add save/reopen coverage for painted masks.
- Finish brush cursor/preview and stroke undo grouping if still needed.
- Verify opacity/type controls are visible and not just editor-state fields.

### #104 — Aligner integration polish

- Show Bounding Box editor controls for aligner selection, normalization, and auto-run.
- Wire non-blocking preload from the host or explicitly document first-run blocking.
- Verify the editable model method surface matches host/tests.
- Keep failure handling model-safe.

### #105 — Pointer-add / context menus / focus arbitration

- Add pointer-add drag gesture for new bboxes if still missing.
- Add right-click context menus on frame and face panels.
- Keep face-panel keyboard navigation from conflicting with frame-view nudge shortcuts.

### #106 — Frame navigation polish

- Verify playback loop, icon state, and frame counter behavior.
- Confirm filter-aware counters move to #107 if not part of this ticket.

### #107 — Filter awareness

- Make filter mode restrict frame navigation.
- Add Misaligned threshold slider.
- Make face panel/grid reflect active filter.
- Update current/max frame counter with active filter.

### #108 — Cross-frame face viewer

- Build full-session face grid.
- Add face-size dropdown.
- Add hover preview and annotations.
- Add context menus and keyboard scroll.
- Make it filter-aware.

### #111 — Extract Faces parent

- Core workflow appears implemented.
- Verify whether the parent issue can close now that cancellation, live editable targets, and result accounting are done.

### #112 — Migration docs/fallback policy

- Decide when Qt becomes default.
- Document legacy fallback expectations.
- Add release notes and manual test checklist.

---

## 14. Completed follow-up work since the original audit

- #102 Extract Box editor parity closed.
- #103 Landmark editor parity closed.
- #109 Thumbnail regeneration progress closed.
- #113 Arrow-key/navigation shortcut collision closed.
- #114 Per-event thumbnail progress percent closed.
- #115 Async save worker/progress closed.
- #116 Extract Faces live editable state closed.
- #117 1-based jump entry closed.
- #118 Visible Extract Faces Cancel control closed.
- #119 Cleanup catchall closed.
- #120 Duplicate of #115 closed.
- #121 Thumbnail progress reset guard closed.

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

None of these should regress while the remaining Tk parity gaps are filled.
