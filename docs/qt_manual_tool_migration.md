# Qt Manual Tool migration checklist

Updated: 2026-05-23.  
Scope: `qt-manual-tool`, reviewed through the Manual Tool follow-up commits that closed #119 and #121, including `4bcbe843f6dded444d43a2c699b0882151504c92`.

Tracks the native Qt Manual Tool work for the main migration epic (#85) and the follow-up tickets split out of the legacy Tk parity audit.

## Current migration state

The Qt Manual Tool is no longer just a launchable shell. It now has a working editable model, frame viewer, current-frame face panel, navigation transport, background startup/save/extract workers, and native implementations for the major editor surfaces. The default-switch blockers are now concentrated around the remaining broad parity surfaces: mask persistence and controls, full aligner UI/preload polish, filter-aware navigation, the cross-frame face viewer, pointer/context-menu polish, and migration/fallback policy.

## Implemented in this branch

- GUI-neutral `ManualSession`, `ManualAlignmentsHandle`, `ManualEditableAlignments`, frame metadata, thumbnail, extraction, and aligner-service seams.
- Qt-native `ManualToolWindow` with no `tkinter` import on the Qt path.
- Startup validation, status/error reporting, determinate startup progress, console fanout, and background startup worker.
- Image-folder and video-frame loading paths, including async video-frame preparation.
- Frame viewer widget with fit-to-window rendering, mouse-wheel zoom, reset view, pan, active-face overlays, and edit hit testing.
- Current-frame face thumbnail panel seeded from editable/persisted alignments.
- Toolbar/action registry for save, navigation, playback, filter mode, annotation mode, editor mode, edit operations, and legacy fallback.
- Transport controls for first/previous/next/last, play/pause state, slider navigation, and 1-based frame jump entry.
- Dirty-state lifecycle with close confirmation and action gating.
- Async save path through `ManualSaveWorker`, with dirty-state success/failure handling and duplicate-save blocking.
- Editable alignment operations for add, delete, move, resize, copy previous/next, revert current frame, undo, redo, nudge, landmark updates, extract-box transforms, and in-memory mask painting.
- Bounding Box overlay editing with move/resize handles.
- Extract Box editor (`F3`) for translate/scale/rotate of the landmark cloud.
- Landmark editor (`F4`) for point and group landmark movement.
- Mask editor (`F5`) for in-memory draw/erase painting, brush-size shortcuts, Ctrl-invert gesture, and live overlay preview.
- Extract Faces workflow with folder picker, worker-thread execution, progress, visible Cancel control, live editable-target extraction, safe shutdown handling, and distinct skipped-frame/skipped-face accounting.
- Thumbnail regeneration workflow and progress, including immutable per-event percentages and protection against resetting live progress back to the static `thumbs` stage.
- Legacy Tk subprocess fallback command support.
- Regression coverage for Manual Tool startup, navigation, frame view, edit interactions, extraction, save worker behavior, thumbnail progress, and editor-specific surfaces.

## Remaining parity blockers before switching the default

- Finish Mask editor persistence and controls:
  - seed editable masks from existing on-disk alignments,
  - save edited masks back to the alignments mask blob format,
  - populate mask type controls from loaded masks,
  - add save/reopen coverage.
- Finish aligner integration polish:
  - visible Bounding Box controls for aligner selection, normalization, and auto-run,
  - non-blocking host preload or explicit documented first-run behavior,
  - verify the editable model API and host/test call sites remain aligned.
- Implement filter-aware navigation and face-panel filtering.
- Build the cross-frame face viewer/grid surface that exists in Tk.
- Add pointer-add drag and right-click/context-menu parity where still missing.
- Add or confirm per-editor color controls, annotation visibility rules, and mesh overlay rendering.
- Complete migration docs/fallback policy and any launch-default decision notes.
- Add any remaining end-to-end smoke coverage for launching Manual from the Qt shell Tools tab.

## Completed follow-up tickets

| Ticket | Completed scope |
| --- | --- |
| #102 | Extract Box editor parity (`F3`) implemented and closed. |
| #103 | Landmark editor parity (`F4`) implemented and closed. |
| #109 | Thumbnail regeneration workflow/progress implemented and closed. |
| #113 | Arrow-key/navigation shortcut collision fixed and closed. |
| #114 | Thumbnail progress carries immutable per-event percent; closed. |
| #115 | Save progress moved to async `ManualSaveWorker`; closed. |
| #116 | Extract Faces uses live editable state / handles dirty extraction path; closed. |
| #117 | Transport jump entry is 1-based; closed. |
| #118 | Extract Faces visible Cancel control; closed. |
| #119 | Catchall cleanup closed after thumbnail reset, extract shutdown, jump Return dedup, and skipped-face accounting. |
| #120 | Duplicate of #115; closed as duplicate/not planned. |
| #121 | Thumbnail progress no longer resets live percent to the static `thumbs` anchor; closed. |

## Open or partially complete follow-up tickets

| Ticket | Current status |
| --- | --- |
| #85 | Main Qt Manual Tool migration epic remains open until the default can switch safely. |
| #101 | Mask editor main surface exists, but persistence, loaded-mask controls, and save/reopen parity remain open. |
| #104 | Aligner service and host hooks exist, but visible controls and preload/UI polish remain open. |
| #105 | Pointer-add, context menus, and shortcut/focus arbitration should be reviewed against current behavior. |
| #106 | Frame navigation polish is mostly implemented; verify any remaining play icon/counter/filter-counter items. |
| #107 | Filter-aware navigation and face visibility remain open. |
| #108 | Cross-frame face viewer/grid parity remains open. |
| #110 | Parent save/bulk-progress ticket mostly addressed by #115; verify whether any bulk-operation progress remains. |
| #111 | Extract Faces workflow is implemented; verify the parent ticket can close if no save-policy edge cases remain. |
| #112 | Migration docs/fallback policy remains a documentation/release-readiness item. |

## Original issue coverage (#85–#94)

| Issue | Updated status |
| --- | --- |
| #85 | Qt-native Manual Tool path is functional but not yet ready as the default until remaining parity blockers close. |
| #86 | GUI-neutral `ManualSession` service exists. |
| #87 | Qt Manual Tool window and legacy fallback seam exist; default-launch policy remains a release decision. |
| #88 | Qt frame viewer with scaling, zoom, pan, overlays, and edit hit testing exists. |
| #89 | Face/thumbnail panel exists for current-frame faces; full cross-frame face viewer is still #108. |
| #90 | Toolbar/actions and state management exist; some action-availability polish remains. |
| #91 | BBox, Extract Box, Landmark, and in-memory Mask editing are implemented; remaining gaps are split across #101, #104, #105, #107, and #108. |
| #92 | Dirty state, save seam, close confirmation, and async save are implemented. |
| #93 | Startup/status progress is implemented; thumbnail progress follow-ups #114/#121 are closed. |
| #94 | Migration checklist exists; this file is the refreshed version. |

## Default-switch readiness checklist

Before making the Qt Manual Tool the default launch path:

- [ ] Close or explicitly defer #101 Mask persistence/control gaps.
- [ ] Close or explicitly defer #104 aligner UI/preload gaps.
- [ ] Close or explicitly defer #105 pointer-add/context-menu/focus gaps.
- [ ] Close or explicitly defer #107 filter-awareness gaps.
- [ ] Close or explicitly defer #108 cross-frame face viewer gaps.
- [ ] Verify #111 Extract Faces parent ticket is fully satisfied.
- [ ] Update release notes and fallback policy (#112).
- [ ] Run the full Manual Tool GUI/model test subset on the target platforms.
- [ ] Smoke-test Qt launch from the Tools tab and legacy fallback launch.
