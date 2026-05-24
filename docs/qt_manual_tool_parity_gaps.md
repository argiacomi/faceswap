# Qt Manual Tool — Behavior parity audit vs. Tk

Branch: `qt-manual-tool`.
Updated snapshot: 2026-05-24.
Reviewed through the #107 GUI regressions, edit-triggered filter-refresh patch, logger handler cleanup, persistence startup/save race stabilization, and a reported green full test suite.

This document compares the legacy Tk Manual Tool (`tools/manual/`) against the native Qt Manual Tool (`lib/gui/qt_shell/manual_tool.py`). It tracks what is at parity, where Qt is already better, what remains open, and what can be explicitly deferred before switching the default Manual Tool launch path.

## Current parity summary

The Qt Manual Tool now covers the major native-editor path:

- native Qt launch path with no `tkinter` import on the Qt path;
- legacy Tk fallback action;
- GUI-neutral Manual session/editable alignment model;
- image-folder and async video-frame loading;
- startup/status/progress/error handling;
- current-frame face panel;
- frame viewer with zoom, pan, overlays, edit hit testing, pointer-add behavior, context-menu hooks, and scoped nudge shortcuts;
- async save worker with busy state, failure handling, duplicate-save blocking, and stable persistence tests;
- thumbnail regeneration progress with immutable per-event percent;
- Extract Faces workflow with progress, cancel, live editable targets, dirty-save chaining, safe shutdown handling, and skipped-frame/skipped-face accounting;
- Bounding Box, Extract Box, Landmark, and Mask editor surfaces;
- Bounding Box aligner controls for aligner selection, normalization, auto-run, explicit preload/rerun behavior, and safe failure handling;
- filter-aware navigation, transport counters, playback routing, empty-filter handling, Misaligned Faces threshold UI, and edit-triggered filter refresh;
- idempotent Faceswap logging setup for repeated `log_setup()` initialization in tests.

## At parity or better than Tk

| Area | Qt status |
| --- | --- |
| Startup and teardown | Parity or better: determinate progress, modal errors, dirty-close confirmation, and safe worker teardown. |
| Frame navigation | Parity: first/previous/next/last, play/pause, 1-based jump, slider, counters, and active-filter routing. |
| Filter modes | Parity for navigation: All, Has Face(s), No Faces, Single Face, Multiple Faces, Misaligned Faces, threshold control, empty-filter handling, and edit-triggered refresh. |
| Bounding Box editor | Parity or better: move/resize, add, context hooks, nudge, undo/redo, copy/revert, and aligner controls. |
| Extract Box editor | Parity: translate, scale, and rotate. |
| Landmark editor | Parity: point and group movement. |
| Mask editor | Parity for current target: persisted mask decode/edit/save behavior, painting, brush shortcuts, Ctrl invert, overlay preview, and touched-state tracking. |
| Save/persistence | Parity or better: async save worker, busy state, failure handling, duplicate-save guard, and save/startup race coverage. |
| Extract Faces | Parity or better: worker execution, visible cancel, live editable targets, dirty-save chaining, safe close behavior, and clearer skipped accounting. |
| Logging/test stability | Better than previous branch state: repeated `log_setup()` calls no longer duplicate alignment read/write log output. |

## Remaining default-switch blockers

### #108 — Cross-frame face viewer/grid

Still open. The current Qt face panel is current-frame scoped. The remaining Tk parity surface is the full-session face grid:

- scrollable grid across all visible faces in the session;
- face-size selector matching the legacy size choices;
- thumbnail click navigation to frame and face;
- hover and active-thumbnail feedback;
- mesh/mask annotations on grid thumbnails where enabled;
- keyboard and mouse scroll parity;
- thumbnail context actions;
- active-filter-aware grid updates.

The #107 filter source of truth is ready for this work through `ManualToolWindow.filtered_frame_indices()` and edit-triggered filter refresh.

### #112 — Migration docs/fallback/default policy

Still open. Remaining work:

- decide when Qt becomes the default Manual Tool launch path;
- document legacy fallback expectations;
- update release notes;
- create the final manual QA/smoke checklist;
- explicitly defer any visual-polish items that should not block the default switch.

### #85 — Parent migration epic

Still open until #108 and #112 are closed or explicitly deferred and final Qt/fallback launch smoke is recorded.

## Explicit defer-decision candidates

These are not currently open child tickets, but should be decided under #112 before default switch:

- persisted Manual Tool window geometry/fullscreen state;
- mesh overlay rendering parity;
- Landmark magnify toggle;
- per-editor configurable overlay colors;
- any additional annotation visibility matrix details;
- final play/pause icon polish.

## Completed follow-up work since the original audit

- #101 Mask editor persistence/control work closed.
- #102 Extract Box editor parity closed.
- #103 Landmark editor parity closed.
- #104 Bounding Box aligner controls/preload/rerun behavior closed.
- #105 Pointer-add/context-menu/focus arbitration closed.
- #106 Frame navigation polish closed.
- #107 Filter-aware navigation, empty-filter handling, Misaligned threshold UI, transport/counter updates, and edit-triggered refresh closed.
- #109 Thumbnail regeneration progress closed.
- #110 Save/progress parent scope covered by async save work.
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

## Where Qt is already ahead of Tk

- Undo/redo stack.
- Keyboard nudge in 1px and 10px increments.
- Determinate startup and thumbnail progress.
- Async save and Extract Faces workers.
- Visible Extract Faces Cancel control and safe close handling.
- Console fanout for Manual Tool operations.
- Pan plus 8-handle bbox resize.
- Sparse image-folder mapping and video frame-name synthesis.
- Better modal error handling for startup/save/extract failures.
- Legacy fallback launcher from the Qt shell.
- Filter-aware navigation with explicit empty-filter behavior.
- Explicit aligner preload/rerun behavior with model-safe failure handling.
- Idempotent logging setup during repeated test/import initialization.

None of these should regress while #108 and #112 are completed.
