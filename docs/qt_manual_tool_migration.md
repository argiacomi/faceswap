# Qt Manual Tool migration checklist

Updated: 2026-05-25.
Scope: `qt-manual-tool`, reviewed through #108 closure, #112 default/fallback policy work, #124 post-default polish tracking, logger handler cleanup, persistence startup/save race stabilization, and a reported green full test suite.

Tracks the native Qt Manual Tool work for the main migration epic (#85) and the follow-up tickets split out of the legacy Tk parity audit.

## Current migration state

The Qt Manual Tool is now the complete native editor path targeted by the migration epic. It has a GUI-neutral session/edit model, frame viewer, current-frame face panel, cross-frame filtered face grid, filter-aware transport, background startup/save/extract workers, native implementations for the major editor surfaces, and regression coverage for the highest-risk Manual Tool workflows.

The major parity follow-ups split out of #85 are complete through #108. The only remaining default-switch work tracked by #112 is documentation, fallback/default-launch policy, release-note wording, and final manual smoke evidence. Non-blocking visual/interactions polish has been split to #124 so it remains visible after default switch without blocking #85/#112 closure.

## Default and fallback policy

- The native Qt Manual Tool is the intended default Manual Tool launch path from the Qt shell once final smoke is recorded.
- The legacy Tk Manual Tool should remain available as a compatibility/debug fallback for this migration cycle.
- The fallback should stay discoverable from the native Qt Manual Tool action/toolbar while users validate the native path against real projects.
- The fallback is not a sign that the Qt path is incomplete; it is an escape hatch for regression triage, environment-specific Tk/Qt differences, and user workflows not yet covered by automated tests.
- Removing or hiding the fallback should be a later product decision after at least one release cycle or after explicit signoff that no compatibility escape hatch is needed.

## Implemented in this branch

- GUI-neutral `ManualSession`, `ManualAlignmentsHandle`, `ManualEditableAlignments`, frame metadata, thumbnail, extraction, frame-filter, and aligner-service seams.
- Qt-native `ManualToolWindow` with no `tkinter` import on the native Qt path.
- Startup validation, status/error reporting, determinate startup progress, console fanout, background startup worker teardown, and test-stable startup/save sequencing.
- Image-folder and video-frame loading paths, including async video-frame preparation.
- Frame viewer with fit-to-window rendering, wheel zoom, reset view, pan, active-face overlays, edit hit testing, pointer-add behavior, context-menu hooks, and focused nudge shortcuts.
- Current-frame face thumbnail panel seeded from editable/persisted alignments.
- Cross-frame filtered face grid with legacy size choices, hover/active state, click-to-frame/face navigation, keyboard/mouse scrolling, context delete, mask/mesh thumbnail overlay support, layout-width wrapping coverage, and filter/edit refresh coverage.
- Toolbar/action registry for save, navigation, playback, filter mode, annotation mode, editor mode, edit operations, aligner controls, Extract Faces, and legacy fallback.
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
- Regression coverage for Manual Tool startup, navigation, frame view, edit interactions, filters, cross-frame face grid, extraction, save worker behavior, persistence, thumbnail progress, logger handler setup, and editor-specific surfaces.

## Remaining work before switching the default

- Record final manual smoke evidence for:
  - Qt launch from the Tools tab,
  - image-folder input with existing alignments,
  - sparse alignments save/reopen,
  - fresh video input and video-frame naming,
  - Mask edit/save/reopen,
  - Extract Faces with dirty editable state,
  - thumbnail regeneration progress,
  - legacy Tk fallback launch from the native window.
- Update #85 with the final closure evidence once smoke is recorded.
- Keep #124 open for post-default visual/interactions polish that is explicitly not a default-switch blocker.

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
| #108 | Cross-frame face viewer/grid parity: filtered full-session grid, face-size control, click navigation, hover/active state, mask/mesh overlays, context delete, scrolling, layout-width wrapping, and GUI coverage. |
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
| #85 | Main Qt Manual Tool migration epic remains open until #112 is closed and final smoke evidence is recorded. |
| #112 | Migration docs/fallback/default policy remains open until this docs update is reviewed and final smoke evidence is either recorded or explicitly left to #85. |
| #124 | Post-default visual/interactions polish bucket; does not block the Qt default switch. |

## Original issue coverage (#85-#94)

| Issue | Updated status |
| --- | --- |
| #85 | Qt-native Manual Tool path is functionally complete; final epic close needs #112/default-policy closure and final smoke evidence. |
| #86 | GUI-neutral `ManualSession` service exists. |
| #87 | Qt Manual Tool window and legacy fallback seam exist; Qt is the intended default after final smoke, with Tk fallback retained as compatibility/debug escape hatch for this migration cycle. |
| #88 | Qt frame viewer with scaling, zoom, pan, overlays, and edit hit testing exists. |
| #89 | Current-frame face panel and cross-frame filtered face grid exist. |
| #90 | Toolbar/actions, state management, async save, and navigation controls exist. |
| #91 | BBox, Extract Box, Landmark, Mask, aligner controls, pointer interactions, filter-aware navigation, and cross-frame face grid are implemented; post-default polish is tracked by #124. |
| #92 | Dirty state, save seam, close confirmation, and async save are implemented. |
| #93 | Startup/status progress is implemented; thumbnail progress follow-ups #114/#121 are closed. |
| #94 | Migration checklist exists and is current as of this update. |

## Release-note draft

The Manual Tool migration now has a native Qt editor path with GUI-neutral session/edit state, filter-aware navigation, current-frame and cross-frame face browsing, BBox/Extract Box/Landmark/Mask editing, async startup/save/extract workers, thumbnail regeneration progress, and legacy Tk fallback access. The Qt path is intended to become the default Manual Tool launch path after final smoke validation. The legacy Tk Manual Tool remains available as a compatibility/debug fallback during the migration cycle.

## Final smoke checklist

- [ ] Launch Qt Manual Tool from the Tools tab.
- [ ] Open image-folder input with existing alignments.
- [ ] Open sparse alignments and confirm frame/thumbnail mapping.
- [ ] Save edits, close, reopen, and confirm persisted bbox/landmark/mask state.
- [ ] Open fresh video input and confirm video-frame naming/persistence.
- [ ] Add/delete/copy/revert/undo/redo faces and confirm filter/grid refresh.
- [ ] Verify BBox aligner controls and explicit preload/rerun path.
- [ ] Verify Extract Box translate/scale/rotate path.
- [ ] Verify Landmark point/group movement path.
- [ ] Verify Mask draw/erase, opacity/type, touched-mask save/revert path.
- [ ] Verify cross-frame face grid size control, click navigation, filter awareness, and context delete.
- [ ] Run Extract Faces with unsaved editable state and confirm output/accounting.
- [ ] Regenerate thumbnails and confirm progress behavior.
- [ ] Launch the legacy Tk fallback from the native Qt Manual Tool window.

## Explicit non-blocking deferrals

The following are tracked in #124 and should not block the Qt Manual Tool default switch:

- persisted Manual Tool window geometry / fullscreen state,
- editor-specific auto-zoom refinements,
- per-editor configurable overlay colors,
- stricter annotation visibility matrix parity,
- Landmark magnify-toggle parity if still needed,
- final play/pause icon and toolbar visual polish.
