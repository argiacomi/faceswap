# Qt Manual Tool migration checklist

Tracks the native Qt Manual Tool work for issues #85 through #94.

## Implemented in this branch

- GUI-neutral `ManualSession` service for Manual Tool startup metadata.
- Qt-native `ManualToolWindow` shell with no `tkinter` import on the Qt path.
- Frame viewer widget with scaling, mouse-wheel zoom and reset behavior.
- Thumbnail/selection panel for directly discoverable image-frame folders.
- Toolbar actions for save, frame navigation, zoom and legacy fallback.
- Dirty-state lifecycle with close confirmation.
- Startup validation and user-facing status/error seams.
- Legacy Tk subprocess fallback command support.
- Unit coverage for session creation, validation and command-value normalization.

## Remaining parity blockers before switching the default

- Extract video frames asynchronously into the shared session layer.
- Load existing alignments data into a GUI-neutral editable model.
- Render detected-face thumbnails rather than frame-name placeholders.
- Wire add, delete, move, copy, revert and landmark edit operations.
- Persist edits back to the alignments file and refresh thumbnails after save.
- Route long-running thumbnail and metadata preparation through Qt workers.
- Add widget tests for frame view, thumbnail selection, action state and dirty close behavior.
- Add smoke coverage for launching Manual from the Qt shell Tools tab.

## Issue coverage

| Issue | Status in this branch |
| --- | --- |
| #85 | Adds a Qt-native Manual Tool shell and documents remaining parity gaps. |
| #86 | Introduces a GUI-neutral `ManualSession` service. |
| #87 | Adds a launchable Qt Manual Tool window and legacy fallback seam. Full Tools-tab wiring remains a follow-up. |
| #88 | Adds the first Qt frame viewer panel with scaling and zoom. |
| #89 | Adds a Qt thumbnail/selection panel placeholder backed by session frames. |
| #90 | Adds toolbar actions and basic state management. |
| #91 | Leaves overlay editing explicitly blocked on the editable alignment model. |
| #92 | Adds dirty-state tracking, save seam and close confirmation. |
| #93 | Adds startup validation, status messages and error seams. |
| #94 | Adds migration checklist and session tests. |
