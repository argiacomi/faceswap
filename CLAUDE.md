# CLAUDE.md

Guidance for when working in `argiacomi/faceswap`.

## Project Summary

FaceSwap is a Python deep-learning app for extracting, training, and swapping faces in images and videos.

Main entry points:

- `faceswap.py`
    - `extract`: extract faces from images or video
    - `train`: train a model from two face sets
    - `convert`: apply a trained model to images or video
    - `gui`: launch the GUI
- `tools.py`: dynamically loads tools from `tools/<tool>/cli.py`

The project is cross-platform and backend-sensitive. Account for Linux, macOS, CPU, Nvidia CUDA, ROCm, and Apple Silicon where relevant.

## Non-Negotiable Rules

1. Run Python commands through the `faceswap` mamba/conda environment unless told otherwise.
2. Keep CLI behavior stable for `faceswap.py extract`, `train`, `convert`, and `gui`.
3. Do not break CPU mode. CPU tests are the reliable local and CI validation path.
4. Do not assume GPU availability in tests.
5. Ask before making broad dependency changes. Backend requirements vary by hardware and Python version.

## Local Environment

Use:

```bash
mamba run -n faceswap rtk <command>
```

## Validation

### Ruff lint and format

```bash
mamba run -n faceswap rtk ruff check . --fix
mamba run -n faceswap rtk ruff format .
```

### Type check

```bash
mamba run -n faceswap rtk mypy .
```

Mypy may be allowed to fail in CI. Inspect and report relevant new errors.

### Unit tests

Per-commit / fast tier (no `slow` or `e2e` markers - matches the CI default):

```bash
mamba run -n faceswap env KERAS_BACKEND=torch KERAS_TORCH_DEVICE=CPU FACESWAP_BACKEND=cpu rtk pytest -q tests/ -m "not slow and not e2e"
```

Full local coverage (includes the heavyweight model / landmark matrices and GUI suites):

```bash
mamba run -n faceswap env KERAS_BACKEND=torch KERAS_TORCH_DEVICE=CPU FACESWAP_BACKEND=cpu rtk pytest -q tests/
```

Just the slow tier:

```bash
mamba run -n faceswap env KERAS_BACKEND=torch KERAS_TORCH_DEVICE=CPU FACESWAP_BACKEND=cpu rtk pytest -q tests/ -m slow
```

Test tier markers (declared in `pyproject.toml`):

- `slow`: heavy model / landmark / parametrized matrices - excluded from per-commit CI.
- `e2e`: end-to-end shell-out coverage in `tests/simple_tests.py` - runs nightly via `.github/workflows/pytest-full.yml`.
- `gui`: Qt or Tk display-dependent tests - auto-applied to `tests/lib/gui/`, `tests/tools/manual/qt/`, and `tests/tools/preview/`.
- `regression`: targeted regression tests; stay on the fast tier unless additionally marked `slow`.

### End-to-end smoke tests

```bash
mamba run -n faceswap env KERAS_BACKEND=torch KERAS_TORCH_DEVICE=CPU FACESWAP_BACKEND=cpu rtk python tests/simple_tests.py
```

The smoke script is no longer part of the per-commit CI matrix.  Trigger it on demand via the `ci/full-suite` workflow (`workflow_dispatch`) or run it locally before merging changes that touch extract / sort / alignments / train / convert flows.

If a full test run is too expensive, run targeted tests and state exactly what was and was not run.

## Repository Map

```text
faceswap.py                 Main CLI entrypoint: extract, train, convert, gui
tools.py                    Dynamic tools CLI loader
setup.py                    Environment/backend setup and dependency installation
update_deps.py              Dependency update helper
scripts/                    Runtime command implementations
scripts/fs_media.py         Shared media/alignment handling for extract and convert
lib/                        Shared application infrastructure
lib/align/                  Alignment data, masks, thumbnails, poses, alignment files
lib/cli/                    CLI args, actions, launch helpers
lib/config/                 Config generation, INI parsing, config models
lib/convert.py              Conversion pipeline orchestration
lib/gui/                    Tk GUI, panels, display pages, config popups, session/project UI
lib/image.py                Image loading, encoding, batching, media utilities
lib/infer/                  Extraction inference, runners, handlers, profiling, precision policy
lib/model/                  Layers, losses, optimizers, normalization, backup/restore helpers
lib/training/               Training loop, data loading, augmentation, previews, TensorBoard
lib/utils.py                Shared utilities
plugins/                    Extract, train, and convert plugins
plugins/plugin_loader.py    Plugin discovery/loading
plugins/extract/            Detection, alignment, mask, identity, analysis plugins
plugins/convert/            Color, mask, scaling, writer plugins
plugins/train/              Train models, trainers, train config
tests/                      Pytest suite and smoke tests
```

## Before File Operations

Always confirm the working directory and branch:

```bash
pwd
git branch
```

Verify you are in the correct project root before making changes.

## Scope Control

Stay focused on the task.

Do not over-research external APIs, documentation, or edge cases unless asked. If verification needs more than 3 to 4 exploratory commands, stop and ask whether to continue.

Avoid:

- Excessive regex edge-case testing
- Deep dives into external command docs
- Over-testing cross-platform behavior
- Checking API signatures across many package versions

## Numbered Plans

When given a numbered plan:

1. Execute steps in order.
2. Commit after each logical phase.
3. Do not skip or reorder steps.
4. Report blockers before continuing.
5. Use a task list for plans with 3 or more steps.
6. Verify referenced paths before starting.
