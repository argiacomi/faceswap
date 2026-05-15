# GUI Test Fixtures

Real `.fsw` (project) and `.fst` (task) files produced by, or shaped to match,
the legacy Tk Faceswap GUI. Used by Qt shell parity tests to verify that the
versioned `ProjectFile` loader, `ProjectStore`, `RecentFilesStore`, and
`LastSessionStore` correctly handle on-disk shapes written by Tk.

## Fixtures

| File | Source | Schema | Notes |
| --- | --- | --- | --- |
| `tk_generated_project_v1.fsw` | Copied from `test.fsw` at the repo root, written by the Tk GUI. | Legacy v1 (flat: one key per command + top-level `tab_name`). | Active tab is `extract` with real local paths from a Tk session. |
| `tk_generated_project_v1_alt.fsw` | Copied from `../faceswap_arch/proj.fsw`, written by an older Tk GUI build. | Legacy v1. | Active tab is `train`. Different command set (no `automask`/`faceqa_*`). Confirms the loader does not require a fixed command list. |
| `tk_generated_task_v1.fst` | Hand-extracted from the `extract` block of `tk_generated_project_v1.fsw`. Matches the legacy on-disk shape Tk writes for `.fst` task files (single command key + `tab_name`). | Legacy v1. | Used to verify task-file migration to v2 + Qt task loading. |
| `qt_generated_task_v2.fst` | Authored to match the current versioned shape produced by `ProjectStore.save`. | v2 (`{"version": 2, "tab_name": ..., "tasks": {...}}`). | Round-trip baseline for the modern format. |
| `tk_recent_files.json` | Authored to match the on-disk shape `RecentFilesStore` writes (list of `[filename, kind]` pairs). Includes a legacy `"extract"` kind to verify normalization. | Legacy / current mixed. | Targets the Tk-era recent-files JSON layout. |

A real Tk-written `last-session.json` is not currently checked in; the
`LastSessionStore` shape is exercised via in-test serializer writes.

## Conventions

- Resolve fixtures with `Path(__file__).parent / "data" / "<name>"`.
- All fixtures are small JSON; keep them committed (no large binaries).
- Do not edit these files to fix test failures — they represent real on-disk
  shapes. If the loader changes, update the loader (or the v2 fixture), not
  the legacy fixtures.
