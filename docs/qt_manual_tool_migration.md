# Qt Manual Tool migration checklist

Updated: 2026-05-24.
Scope: `qt-manual-tool`, reviewed through the #107 filter GUI regressions, edit-triggered filter-refresh patch, logger handler cleanup, persistence startup/save race stabilization, and a reported green full test suite.

Tracks the native Qt Manual Tool work for the main migration epic (#85) and the follow-up tickets split out of the legacy Tk parity audit.

## Current migration state

The Qt Manual Tool is now a functional native editor path rather than a launchable shell. It has a GUI-neutral session/edit model, frame viewer, current-frame face panel, filter-aware transport, background startup/save/extract workers, native implementations for the major editor surfaces, and regression coverage for the highest-risk Manual Tool workflows.

#107 is now completed and closed. The remaining default-switch work is concentrated around the broad cross-frame face viewer/grid surface (#108), migration documentation/fallback/default-launch policy (#112), and final launch smoke from the Qt Tools tab plus the legacy fallback.

## Implemented in this branch

- GUI-neutral `ManualSession`, `ManualAlignmentsHandle`, `ManualEditableAlignments`, frame metadata, thumbnail, extraction, frame-filter, and aligner-service seams.
- Qt-native `ManualToolWindow` with no `tkinter` import on the native Qt path.
- Startup validation, status/error reporting, determinate startup progress, console fanout, background startup worker teardown, and test-stable startup/save sequencing.
- Image-folder and video-frame loading paths, including async video-frame preparation.
- Frame viewer with fit-to-window rendering, wheel zoom, reset view, pan, active-face overlays, edit hit testing, pointer-add behavior, context-menu hooks, and focused nudge shortcuts.
- Current-frame face thumbnail panel seeded from editable/persisted alignments.
- Toolbar/action registry for save, navigation, playback, filter mode, annotation mode, editor mode, edit operations, aligner controls, and legacy fallback.
- Transport controls for first/previous/next/last, play/pause state, slider navigation, and 1-based frame jump entry.
- Filter-aware navigation for All Frames, Has Face(s), No Faces, Single Face, Multiple Faces, and Misaligned Faces, including empty-filter handling, Misaligned threshold controls, transport/counter updates, and edit-triggered refresh for current and non-current frames.
- Dirty-state lifecycle with close confirmation and action gating.
- Async save through `ManualSaveWorker`, with dirty-state success/failure handling, duplicate-save blocking, and persistence tests that wait for startup/save workers before alignments-file assertions.
- Editable alignment operations for add, remove, move, resize, copy previous/next, revert current frame, undo, redo, nudge, landmark updates, extract-box transforms, and in-memory mask painting.
- Bounding Box overlay editing plus visible aligner selection/normalization/auto-run controls and explicit preload behavior.
- Extract Box editor (`F3`) for translate/scale/rotate of the landmark cloud.
- Landmark editor (`F4`) for point and group landmark movement.
- Mask editor (`F5`) for mask seeding, in-memory draw/erase painting, brush-size shortcuts, Ctrl-invert gesture, overlay preview, touched-mask persistence, undo/revert tracking, and persisted-mask edit coverage.
- Extract Faces workflow with folder picker, worker-thread execution, progress, visible Cancel control, live editable-target extraction, safe shutdown handling, and distinct skipped-frame/skipped-face accounting.
- Thumbnail regeneration workflow and progress, including immutable per-event percentages and protection against resetting live progress back to the static `thumbs` stage.
- Idempotent Faceswap logging setup so repeated `log_setup()` calls do not duplicate later alignment read/write output.
- Legacy Tk subprocess fallback command support.
- Regression coverage for Manual Tool startup, navigation, frame view, edit interactions, filters, extraction, save worker behavior, persistence, thumbnail progress, logger handler setup, and editor-specific surfaces.

## Remaining work before switching the default

- Complete or explicitly defer #108 cross-frame face viewer/grid parity:
  - full-session scrollable face grid,
  - face-size selector,
  - thumbnail click navigation to frame and face,
  - hover/active styling,
  - mesh/mask thumbnail overlays,
  - thumbnail context actions,
  - keyboard/mouse scroll parity,
  - filter-aware grid updates.
- Complete #112 migration docs/fallback/default policy:
  - decide when Qt becomes the default launch path,
  - document legacy fallback expectations,
  - update release notes and manual QA checklist,
  - record any explicitly deferred Tk-only polish.
- Smoke-test Qt launch from the Tools tab and legacy fallback launch.
- Decide whether remaining visual polish such as persisted Manual Tool window geometry, mesh overlay rendering, magnify toggle, per-editor colors, and additional annotation visibility details are default blockers or deferred follow-ups.

## Completed follow-up tickets

| Ticket | Completed scope |
| --- | --- |
| #101 | Mask editor surface, persisted mask decode/edit/save behavior, touched-mask tracking, and save/reopen regressions. |
| #102 | Extract Box editor parity (`F3`). |
| #103 | Landmark editor parity (`F4`). |
| #104 | Bounding Box aligner selection/normalization/auto-run controls and preload/rerun behavior. |
| #105 | Pointer-add, context menus, and frame-view/focus shortcut arbitration. |
| #106 | Frame navigation transport polish, playback/counter behavior, and 1-based jump behavior. |
| #107 | Filter-aware navigation, empty-filter handling, Misaligned threshold UI, transport/counter updates, and edit-triggered refresh. |
| #109 | Thumbnail regeneration workflow/progress. |
| #110 | Save/progress parent scope covered by async save worker behavior and #115 regressions. |
| #111 | Extract Faces workflow with dirty-save chaining, live editable targets, cancel, progress, and result accounting. |
| #113 | Arrow-key/navigation shortcut collision. |
| #114 | Immutable per-event thumbnail progress percent. |
| #115 | Async `ManualSaveWorker` save progress. |
| #116 | Extract Faces live editable state. |
| #117 | 1-based transport jump entry. |
| #118 | Visible Extract Faces Cancel control. |
| #119 | Catchall cleanup after thumbnail reset, extract shutdown, jump Return dedup, and skipped-face accounting. |
| #120 | Duplicate of #115; closed as duplicate/not planned. |
| #121 | Thumbnail progress reset guard. |
| #122 | Persisted masks decoded from saved blobs before editing. |
| #123 | Mask undo/revert touched-state tracking. |

## Open follow-up tickets

| Ticket | Current status |
| --- | --- |
| #85 | Main Qt Manual Tool migration epic remains open until #108/#112/default-switch readiness are settled and final smoke passes. |
| #108 | Cross-frame face viewer/grid parity remains open. |
| #112 | Migration docs/fallback/default policy remains open until the docs, release notes, launch-default decision, and smoke checklist are finalized. |

## Original issue coverage (#85-#94)

| Issue | Updated status |
| --- | --- |
| #85 | Qt-native Manual Tool path is functional but the epic remains open until remaining default-switch work closes. |
| #86 | GUI-neutral `ManualSession` service exists. |
| #87 | Qt Manual Tool window and legacy fallback seam exist; default-launch policy remains a release decision. |
| #88 | Qt frame viewer with scaling, zoom, pan, overlays, and edit hit testing exists. |
| #89 | Current-frame face/thumbnail panel exists; full cross-frame face viewer is #108. |
| #90 | Toolbar/actions, state management, async save, and navigation controls exist. |
| #91 | BBox, Extract Box, Landmark, Mask, aligner controls, pointer interactions, and filter-aware navigation are implemented; remaining broad editor-adjacent parity is the #108 cross-frame grid plus any explicitly deferred polish. |
| #92 | Dirty state, save seam, close confirmation, and async save are implemented. |
| #93 | Startup/status progress is implemented; thumbnail progress follow-ups #114/#121 are closed. |
| #94 | Migration checklist exists and is current as of this update. |

## Default-switch readiness checklist

- [ ] Close or explicitly defer #108 cross-frame face viewer/grid parity.
- [ ] Update release notes, migration docs, and fallback policy (#112).
- [ ] Decide whether geometry restore, mesh overlay, magnify toggle, per-editor colors, and annotation visibility polish are default blockers or deferred follow-ups.
- [ ] Smoke-test Qt launch from the Tools tab.
- [ ] Smoke-test legacy fallback launch from the Qt Manual Tool action.
- [ ] Update #85 with final closure evidence and close it only after the above items are complete.
