# Qt Manual Tool migration checklist

Updated: 2026-05-24.
Scope: `qt-manual-tool`, reviewed through the current Manual Tool follow-up work after #119/#121, #101/#104/#105/#106/#111, the #107 GUI regressions, and the edit-triggered filter-refresh patch (`Manual Tool: refresh filters after editable changes`).

Tracks the native Qt Manual Tool work for the main migration epic (#85) and the follow-up tickets split out of the legacy Tk parity audit.

## Current migration state

The Qt Manual Tool is now a functional native editor path rather than a launchable shell. It has a GUI-neutral session/edit model, frame viewer, current-frame face panel, filter-aware transport, background startup/save/extract workers, native implementations for the major editor surfaces, and regression coverage for the highest-risk Manual Tool workflows.

The remaining default-switch work is now concentrated around the broad cross-frame face viewer/grid surface (#108), migration documentation/fallback/default-launch policy (#112), final targeted/full test passes, and launch smoke from the Qt Tools tab. #107 is implemented by the current filter-aware navigation work plus the edit-triggered refresh patch, but should only be closed after the targeted GUI filter tests pass in the target environment.

## Implemented in this branch

- GUI-neutral `ManualSession`, `ManualAlignmentsHandle`, `ManualEditableAlignments`, frame metadata, thumbnail, extraction, and aligner-service seams.
- Qt-native `ManualToolWindow` with no `tkinter` import on the Qt path.
- Startup validation, status/error reporting, determinate startup progress, console fanout, and background startup worker teardown.
- Image-folder and video-frame loading paths, including async video-frame preparation.
- Frame viewer widget with fit-to-window rendering, mouse-wheel zoom, reset view, pan, active-face overlays, and edit hit testing.
- Current-frame face thumbnail panel seeded from editable/persisted alignments.
- Toolbar/action registry for save, navigation, playback, filter mode, annotation mode, editor mode, edit operations, and legacy fallback.
- Transport controls for first/previous/next/last, play/pause state, slider navigation, and 1-based frame jump entry.
- Filter-aware frame list, transport counters, navigation/playback routing, empty-filter handling, and Misaligned Faces threshold control.
- Dirty-state lifecycle with close confirmation and action gating.
- Async save path through `ManualSaveWorker`, with dirty-state success/failure handling, duplicate-save blocking, and visible save busy state before persistence completes.
- Editable alignment operations for add, delete, move, resize, copy previous/next, revert current frame, undo, redo, nudge, landmark updates, extract-box transforms, and mask painting/persistence state.
- Bounding Box overlay editing with move/resize handles.
- Bounding Box aligner controls for aligner selection, normalization, auto-run behavior, explicit preload/rerun paths, and model-safe failure handling.
- Extract Box editor (`F3`) for translate/scale/rotate of the landmark cloud.
- Landmark editor (`F4`) for point and group landmark movement.
- Mask editor (`F5`) for draw/erase painting, brush-size shortcuts, Ctrl-invert gesture, live overlay preview, persisted mask loading/editing, and touched-mask save/revert behavior.
- Pointer-add/context-menu/focus-scoped shortcut behavior for current-frame interactions.
- Extract Faces workflow with folder picker, worker-thread execution, progress, visible Cancel control, live editable-target extraction, dirty-save chaining, safe shutdown handling, and distinct skipped-frame/skipped-face accounting.
- Thumbnail regeneration workflow and progress, including immutable per-event percentages and protection against resetting live progress back to the static `thumbs` stage.
- Legacy Tk subprocess fallback command support.
- Regression coverage for Manual Tool startup, navigation, frame view, edit interactions, filtering, extraction, save worker behavior, thumbnail progress, and editor-specific surfaces.
- Non-suppressing pytest diagnostics for unraisable temporary-file cleanup warnings (`tracemalloc` allocation traces plus teardown-time GC attribution).

## Remaining work before switching the default

- Complete #108 cross-frame face viewer/grid parity:
    - full-session scrollable face grid,
    - face-size selector,
    - thumbnail click navigation to frame and face,
    - hover/active styling,
    - mesh/mask thumbnail overlays,
    - right-click delete/context menu,
    - keyboard/mouse scroll parity,
    - filter-aware grid updates.
- Complete #112 migration docs/fallback/default policy:
    - decide when Qt becomes the default launch path,
    - document legacy fallback expectations,
    - update release notes and manual QA checklist,
    - record any explicitly deferred Tk-only polish.
- Verify and close #107 after the filter GUI tests pass with the edit-triggered refresh patch applied.
- Run the full Manual Tool GUI/model test subset on the target environments.
- Smoke-test Qt launch from the Tools tab and legacy fallback launch.
- Decide whether remaining visual polish such as persisted Manual Tool window geometry, mesh overlay rendering, magnify toggle, per-editor colors, and any additional annotation visibility details are default blockers or explicitly deferred follow-ups.

## Completed follow-up tickets

| Ticket | Completed scope                                                                                                              |
| ------ | ---------------------------------------------------------------------------------------------------------------------------- |
| #101   | Mask editor surface, persisted mask decode/edit/save behavior, touched-mask tracking, and save/reopen regressions.           |
| #102   | Extract Box editor parity (`F3`) implemented and closed.                                                                     |
| #103   | Landmark editor parity (`F4`) implemented and closed.                                                                        |
| #104   | Bounding Box aligner selection/normalization/auto-run controls and preload/rerun behavior implemented.                       |
| #105   | Pointer-add, context menus, and frame-view/focus shortcut arbitration covered.                                               |
| #106   | Frame navigation transport polish, playback/counter behavior, and 1-based jump behavior implemented.                         |
| #109   | Thumbnail regeneration workflow/progress implemented and closed.                                                             |
| #110   | Save/progress parent scope covered by async save worker behavior and #115 regressions.                                       |
| #111   | Extract Faces workflow implemented with dirty-save chaining, live editable targets, cancel, progress, and result accounting. |
| #113   | Arrow-key/navigation shortcut collision fixed and closed.                                                                    |
| #114   | Thumbnail progress carries immutable per-event percent; closed.                                                              |
| #115   | Save progress moved to async `ManualSaveWorker`; closed.                                                                     |
| #116   | Extract Faces uses live editable state / handles dirty extraction path; closed.                                              |
| #117   | Transport jump entry is 1-based; closed.                                                                                     |
| #118   | Extract Faces visible Cancel control; closed.                                                                                |
| #119   | Catchall cleanup closed after thumbnail reset, extract shutdown, jump Return dedup, and skipped-face accounting.             |
| #120   | Duplicate of #115; closed as duplicate/not planned.                                                                          |
| #121   | Thumbnail progress no longer resets live percent to the static `thumbs` anchor; closed.                                      |
| #122   | Persisted masks are decoded from saved blobs before editing; closed.                                                         |
| #123   | Mask undo/revert touched-state tracking is implemented and covered; closed.                                                  |

## Open or not-yet-closed follow-up tickets

| Ticket | Current status                                                                                                                                                                                                                        |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| #85    | Main Qt Manual Tool migration epic remains open until #108/#112/default-switch readiness are settled and final smoke passes.                                                                                                          |
| #107   | Filter-aware navigation is implemented in code/tests, including empty-filter handling, Misaligned threshold UI, and edit-triggered refresh patch. Keep open until the targeted GUI filter tests pass and the issue is updated/closed. |
| #108   | Cross-frame face viewer/grid parity remains open.                                                                                                                                                                                     |
| #112   | Migration docs/fallback/default policy remains open until the docs, release notes, and launch-default decision are current.                                                                                                           |

## Original issue coverage (#85–#94)

| Issue | Updated status                                                                                                                                                                                                                   |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| #85   | Qt-native Manual Tool path is functional but the epic remains open until remaining default-switch work closes.                                                                                                                   |
| #86   | GUI-neutral `ManualSession` service exists.                                                                                                                                                                                      |
| #87   | Qt Manual Tool window and legacy fallback seam exist; default-launch policy remains a release decision.                                                                                                                          |
| #88   | Qt frame viewer with scaling, zoom, pan, overlays, and edit hit testing exists.                                                                                                                                                  |
| #89   | Current-frame face/thumbnail panel exists; full cross-frame face viewer is #108.                                                                                                                                                 |
| #90   | Toolbar/actions, state management, async save, and navigation controls exist.                                                                                                                                                    |
| #91   | BBox, Extract Box, Landmark, Mask, aligner controls, pointer interactions, and filter-aware navigation are implemented; remaining broad editor-adjacent parity is the #108 cross-frame grid plus any explicitly deferred polish. |
| #92   | Dirty state, save seam, close confirmation, and async save are implemented.                                                                                                                                                      |
| #93   | Startup/status progress is implemented; thumbnail progress follow-ups #114/#121 are closed.                                                                                                                                      |
| #94   | Migration checklist exists; this file is the refreshed version.                                                                                                                                                                  |

## Default-switch readiness checklist

Before making the Qt Manual Tool the default launch path:

- [ ] Close #107 after the edit-triggered filter refresh patch and `tests/lib/gui/test_qt_shell_manual_filters.py` pass.
- [ ] Close or explicitly defer #108 cross-frame face viewer/grid parity.
- [ ] Update release notes, migration docs, and fallback policy (#112).
- [ ] Decide whether geometry restore, mesh overlay, magnify toggle, per-editor colors, and annotation visibility polish are default blockers or deferred follow-ups.
- [ ] Run the full Manual Tool GUI/model test subset on the target platforms.
- [ ] Smoke-test Qt launch from the Tools tab.
- [ ] Smoke-test legacy fallback launch from the Qt Manual Tool action.
- [ ] Update #85 with final closure evidence and close it only after the above items are complete.
