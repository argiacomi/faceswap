# Landmark pipeline tools

Ticket: Epic 1, Ticket 1.1 plus cleanup follow-ups

This document inventories the current landmark pipeline command surface, records deleted legacy CLIs, and maps ownership after the scorer helper migration.

## Current command surface

`tools/landmarks/` contains pipeline entrypoints and thin compatibility wrappers. Shared scorer implementation code now lives under `lib.landmarks.ensemble`; new code should import scorer helpers from `lib.landmarks.ensemble.*`, not from `tools.landmarks.*`.

Run command tools directly, for example:

```bash
python tools/landmarks/run_landmark_resolver_pipeline.py --help
python tools/landmarks/search_ensemble_setup.py --help
python tools/landmarks/evaluate_runtime_resolver_scorer.py --help
```

## Status legend

- Active: supported command or helper in the current workflow.
- Compatibility shim: retained only to avoid breaking older imports while callers migrate.
- Deleted after merge: removed because replacement flow is now canonical.

## Pipeline stage map

| Stage | Primary tools/modules | Notes |
| --- | --- | --- |
| Dataset build | `build_quality_dataset.py`, `run_landmark_resolver_pipeline.py` | Dataset normalization and manifest generation. The resolver wrapper now exposes explicit dataset/cache/split/static-weight stages instead of hiding a legacy static pipeline run. |
| Candidate generation | `cache_predictions.py`, `search_ensemble_setup.py`, `run_landmark_resolver_pipeline.py` | Prediction cache population and candidate enumeration. The resolver wrapper defaults cache stages to model execution (`--prediction-cache-mode run-models`) and writes per-source completion sentinels for resume safety. |
| Static weight search | `search_ensemble_setup.py`, `run_landmark_resolver_pipeline.py` | `search_ensemble_setup.py` is the promoted setup/weight search path. `compute_static_weights.py` and the legacy static pipeline orchestrator were deleted after merge. |
| Production manifest build | `build_faceswap_validated_manifest.py`, `run_landmark_resolver_pipeline.py` | Production-validated manifests and resolver metadata are built from reviewed Faceswap alignments before promotion/eval. |
| Hard slice validation | `build_hard_alignment_validation.py`, `evaluate_alignment_geometry.py`, `run_landmark_resolver_pipeline.py` | GT geometry and hard-case validation. Signal/profile/worst-case standalone CLIs were deleted after merge. |
| Scorer training | `train_runtime_resolver_scorer.py`, `lib.landmarks.ensemble.scorer_training`, `lib.landmarks.ensemble.scorer_dataset`, `lib.landmarks.ensemble.scorer_contexts` | `scorer_training` is the canonical scorer-suite stage. It builds one canonical scorer dataset and trains v1, v1.1, and v2 scorers from the same split. Legacy binary/continuous/v2 scorer stages are aliases during migration. |
| Scorer evaluation | `evaluate_runtime_resolver_scorer.py`, `lib.landmarks.ensemble.scorer_eval`, `lib.landmarks.ensemble.scorer_reports` | Evaluation consumes canonical `scorer_dataset/rows.csv` when available and falls back to context rebuilding only for legacy/manual runs. |
| Production promotion | `run_landmark_resolver_pipeline.py`, `search_ensemble_setup.py`, `production_promotion_gate.py`, `evaluate_runtime_resolver_scorer.py` | Full wrapper owns end-to-end promotion, summaries, artifact export, and config preview/write. |
| Metadata and debug sidecars | `evaluate_alignment_geometry.py`, `run_landmark_resolver_pipeline.py`, `lib.landmarks.ensemble.runtime_resolver_scorer_data` | Debug JSON/JSONL and scorer sidecar handling are centralized through shared helpers/conventions. |

## Active tool inventory

| Tool | Owning stage | Purpose | Required inputs | Outputs | Current status |
| --- | --- | --- | --- | --- | --- |
| `__init__.py` | Package integration | Keeps `tools.landmarks` importable. | None. | None. | Active helper. |
| `build_quality_dataset.py` | Dataset build | Builds normalized landmark quality manifests for supported datasets and fixtures. | Dataset selection and source inputs. | `manifest.json`, normalized image/landmark artifacts, `dataset_audit.json`, optional overlays. | Active. |
| `cache_predictions.py` | Candidate generation | Writes imported fixture predictions or model-run predictions into `DiskPredictionCache`. | `--manifest`, `--models`, `--cache-dir`, prediction roots or `--run-models`. | Per-sample, per-model cache entries. | Active. |
| `run_landmark_resolver_pipeline.py` | Full pipeline orchestration | Canonical end-to-end resolver pipeline wrapper for dataset/cache/split/static-weight generation, production manifest export, candidate search, scorer v1/v1.1 comparison, promotion, artifact export, and config update. | `--run-root`, `--production-root`, `--output-root`, dataset/model inputs, production images/alignments or existing production artifacts, frozen GT-hard sidecar when needed. | `pipeline_summary.json`, `pipeline_summary.md`, `pipeline_progress.jsonl`, stable `current/` artifact export, config patch/preview. | Active canonical wrapper. |
| `search_ensemble_setup.py` | Candidate generation, static weight search, production promotion | Enumerates model subsets, weight generators, strategies, thresholds, crop scales, and writes promoted setup artifacts. | Manifest, cache, splits, models, output dir, optional production manifest/cache. | `candidate_results.csv`, `candidate_results.json`, `best_setup.json`, `best_weights.json`, `promotion_report.md`, optional production gate artifacts. | Active preferred setup/weight promotion path. |
| `evaluate_alignment_geometry.py` | Hard slice validation, diagnostics | Cache-only GT-derived alignment geometry evaluation for models and optional ensemble variants. | Manifest, cache, models, optional variants/weights/thresholds. | `geometry_metrics.json`, `geometry_metrics.csv`, `per_region_geometry.csv`, catastrophic/worst-sample reports. | Active. |
| `build_hard_alignment_validation.py` | Hard slice validation | Builds hard-case validation manifest and can run geometry evaluation over the slice. | Source manifest, output dir, hard-slice thresholds, optional cache/models/weights. | Filtered `manifest.json`, `hard_slice_summary.json`, optional geometry outputs. | Active. |
| `train_runtime_resolver_scorer.py` | Scorer training | Thin CLI for canonical scorer-suite training. | Weights, output dir, GT and/or production manifest/cache, optional frozen GT-hard sidecar. | `scorer_dataset/rows.csv`, scorer dataset manifest, v1 binary scorer, v1.1 scorer, v2 scorer, suite metrics, compatibility aliases. | Active CLI wrapper. |
| `evaluate_runtime_resolver_scorer.py` | Scorer evaluation, promotion | Thin CLI for scorer policy evaluation and promotion gates. | Scorer artifacts, candidates, canonical scorer rows or legacy manifest/cache inputs. | Scorer metrics/report CSV/JSON, worst samples, feature importances. | Active CLI wrapper. |
| `production_promotion_gate.py` | Production promotion | Standalone repeatable production validation gate. | Production manifest/cache, weights, output dir, policy/gate thresholds. | Production promotion report JSON/MD, per-bucket metrics, policy failures, worst samples. | Active. |
| `pipeline_conventions.py` | Shared tool conventions | Shared source labels, filenames, sidecar validation, and artifact writers. | Imported by tools and library-facing scorer modules. | Constants/helpers only. | Active helper. |
| `pipeline_progress.py` | Shared progress/logging | TRACE logging, structured stage progress, progress bars. | Imported by wrapper/tools. | Progress JSONL events and logging helpers. | Active helper. |
| `scorer_contexts.py` | Compatibility shim | Re-exports `lib.landmarks.ensemble.scorer_contexts`. | Import only. | Import compatibility. | Compatibility shim. |
| `scorer_targets.py` | Compatibility shim | Re-exports `lib.landmarks.ensemble.scorer_targets`. | Import only. | Import compatibility. | Compatibility shim. |
| `scorer_reports.py` | Compatibility shim | Re-exports `lib.landmarks.ensemble.scorer_reports`. | Import only. | Import compatibility. | Compatibility shim. |

## Library-facing scorer modules

The scorer implementation is moving out of `tools/landmarks` and into `lib.landmarks.ensemble`:

| Module | Purpose | Status |
| --- | --- | --- |
| `lib.landmarks.ensemble.scorer_contexts` | Centralized scorer context loading and GT-hard sidecar validation. | Active library-facing helper. |
| `lib.landmarks.ensemble.scorer_targets` | Centralized scorer target/row generation. | Active library-facing helper. |
| `lib.landmarks.ensemble.scorer_reports` | Centralized scorer report writers. | Active library-facing helper. |
| `lib.landmarks.ensemble.runtime_resolver_scorer_data` | Runtime resolver scorer data builders. | Library-facing re-export during migration. |
| `lib.landmarks.ensemble.scorer_training` | Runtime resolver scorer training implementation. | Library-facing re-export during migration. |
| `lib.landmarks.ensemble.scorer_eval` | Runtime resolver scorer evaluation implementation. | Library-facing re-export during migration. |

## Deleted after merge

These CLIs were removed after their replacement flows were confirmed as canonical:

| Deleted CLI | Replacement path | Reason |
| --- | --- | --- |
| `compute_static_weights.py` | `search_ensemble_setup.py` and `run_landmark_resolver_pipeline.py` | Promoted static weight search now happens through candidate search and the full wrapper. |
| `run_static_weight_pipeline.py` | `build_quality_dataset.py`, `cache_predictions.py`, `search_ensemble_setup.py`, and `run_landmark_resolver_pipeline.py` | Legacy orchestration was replaced by explicit dataset/cache/search stages plus the resolver wrapper. |
| `validate_geometry_signals.py` | `evaluate_alignment_geometry.py`, `search_ensemble_setup.py`, and wrapper summaries | Signal diagnostics are folded into the active geometry/promotion reporting surface. |
| `evaluate_aflw_profile.py` | `evaluate_alignment_geometry.py`, `build_hard_alignment_validation.py`, and wrapper summaries | Profile diagnostics are no longer a standalone promotion gate. |
| `failure_viewer.py` | `evaluate_alignment_geometry.py` debug outputs and wrapper artifact/report outputs | Worst-case/debug sidecars are handled by active reporting flows. |

## Issue #206 scorer-suite migration

`scorer_training` is now the canonical scorer stage for runtime resolver scorer artifacts. It loads scorer contexts once, writes a canonical scorer dataset, and trains the v1 binary scorer, v1.1 selection-cost scorer, and v2 scorer from the same split.

The old stage names remain compatibility aliases:

- `binary_scorer_training`
- `continuous_scorer_training`
- `v2_scorer_training`

Evaluation should consume `scorer_dataset/rows.csv` through `--scorer-rows` or `--scorer-dataset` whenever that dataset is available. Legacy manifest/cache rebuilding remains available for manual diagnostics and old runs.

## Remaining cleanup

1. Finish physical migration of the large scorer implementation bodies out of `tools/landmarks` once all imports are migrated and CI is green.
2. Remove scorer compatibility shims from `tools/landmarks` after no callers import them.
3. Consider moving `pipeline_conventions.py` to `lib.landmarks.pipeline_conventions` if the lib/tool dependency boundary needs to be strict.
