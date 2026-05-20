# Landmark pipeline tools

Ticket: Epic 1, Ticket 1.1 - Audit current pipeline tools and ownership

This document inventories the landmark pipeline scripts under `tools/landmarks/`, maps each script to its owning pipeline stage, and calls out overlap and unclear contracts to resolve before consolidation.

## How the tools are wired

`tools/landmarks/` currently has no package-level `cli.py`, so these scripts are not registered as a top-level `python tools.py landmarks ...` command. Run them directly, for example:

```bash
python tools/landmarks/run_static_weight_pipeline.py --help
python tools/landmarks/search_ensemble_setup.py --help
```

`tools/landmarks/__init__.py` intentionally keeps the package import-only until CLI integration is designed.

## Status legend

- Active: used by the current workflow or by tests as a supported entrypoint.
- Candidate for merge: still valid, but overlaps another tool enough that it should be folded into a clearer command surface.
- Deprecated: should remain in place for compatibility, but should be marked clearly in its docstring or comments before removal.

No full tool is marked deprecated in this audit. Some individual CLI flags are already deprecated, and those should remain documented in the affected parser help until the next cleanup pass.

## Pipeline stage map

| Stage | Primary tools | Notes |
| --- | --- | --- |
| Dataset build | `build_quality_dataset.py`, `run_static_weight_pipeline.py` | Dataset normalization and manifest generation. |
| Candidate generation | `cache_predictions.py`, `search_ensemble_setup.py` | Prediction cache population plus candidate enumeration over models, strategies, weights, and crop scales. |
| Static weight search | `compute_static_weights.py`, `search_ensemble_setup.py`, `run_static_weight_pipeline.py` | Legacy per-landmark weight fitting plus newer candidate search and promotion artifacts. |
| Hard slice validation | `build_hard_alignment_validation.py`, `evaluate_alignment_geometry.py`, `validate_geometry_signals.py`, `evaluate_aflw_profile.py` | GT geometry, profile, signal, and hard-case validation. |
| Scorer training | `runtime_resolver_scorer_data.py`, `train_runtime_resolver_scorer.py` | Runtime resolver scorer row construction and logistic scorer artifact training. |
| Scorer evaluation | `evaluate_runtime_resolver_scorer.py`, `runtime_resolver_scorer_data.py` | Learned scorer policy evaluation against production and GT-hard baselines. |
| Production promotion | `search_ensemble_setup.py`, `production_promotion_gate.py`, `evaluate_runtime_resolver_scorer.py` | Best setup artifacts, production validation gates, and scorer promotion gates. |
| Metadata and debug sidecars | `failure_viewer.py`, `evaluate_alignment_geometry.py`, `validate_geometry_signals.py`, `runtime_resolver_scorer_data.py` | Worst-case overlays, CSV diagnostics, scorer rows, candidate tables, and resolver metadata ingestion. |

## Tool inventory

| Tool | Owning stage | Purpose | Required inputs | Outputs | Current status |
| --- | --- | --- | --- | --- | --- |
| `__init__.py` | Package integration | Keeps `tools.landmarks` importable while intentionally avoiding `tools.py` auto-registration. | None. | None. | Active helper. |
| `build_quality_dataset.py` | Dataset build | Builds normalized landmark quality manifests for supported datasets such as WFLW, COFW, AFLW2000-3D, MERL-RAV, 300W, and directory fixtures. Can audit existing manifests and write overlays. | `--dataset`, `--output-dir`, plus dataset source inputs such as `--source-dir`, `--source-zip`, `--download-url`, `--wflw-annotations`, `--cofw-json`, or directory inputs. | `manifest.json`, copied or normalized image and landmark artifacts, `dataset_audit.json`, optional GT/indexed/region overlays. | Active. |
| `cache_predictions.py` | Candidate generation | Writes imported fixture predictions or model-run predictions into `DiskPredictionCache`. | `--manifest`, `--models`, `--cache-dir`, and either `--prediction-root model=path` inputs or model-run options. | Per-sample, per-model prediction cache entries with checkpoint/config metadata. | Active. |
| `compute_static_weights.py` | Static weight search | Fits per-landmark static ensemble weights from cached predictions and truth landmarks. | `--manifest`, `--cache-dir`, `--models`, optional `--generator` and `--generator-params`. | Static weight JSON, optional report with mean errors and generator diagnostics. | Active, candidate for merge into the unified search/promotion flow once the legacy static pipeline is retired. |
| `run_static_weight_pipeline.py` | Dataset build, candidate generation, static weight search, debug | End-to-end static-weight validation orchestrator. Runs dataset build, prediction cache population, baseline harness, split creation, static weight fitting, weighted harness, and optional failure viewer. | `--output-root`, `--datasets`, `--models`, build/run/skip flags, prediction roots or model-run options, split settings, thresholds. | `run_summary.json`, dataset manifest, optional `splits.json` plus fit/select/report manifests, prediction cache, baseline metrics, static weights/report, weighted metrics, detector bbox report, debug artifacts. | Active orchestrator. |
| `search_ensemble_setup.py` | Candidate generation, static weight search, production promotion | Cache-only ensemble candidate search and promoted setup writer. Enumerates model subsets, weight generators, strategies, outlier thresholds, and crop scales, then applies NME, geometry, profile, effective-ensemble, and optional production gates. | `--manifest`, `--cache-dir`, `--splits`, `--models`, candidate search dimensions, gate flags, `--output-dir`, optional production manifest/cache. | `candidate_results.csv`, `candidate_results.json`, `best_setup.json`, `best_weights.json`, `promotion_report.md`, `no_promotion.json`, optional production promotion artifacts. | Active. This is the preferred promotion path for ensemble setup artifacts. |
| `evaluate_alignment_geometry.py` | Hard slice validation, scorer diagnostics | Cache-only GT-derived alignment geometry evaluation for single models and optional ensemble variants. | `--manifest`, `--cache-dir`, `--models`, optional variants, weights, outlier threshold, aligned size, region failure threshold. | `geometry_metrics.json`, `geometry_metrics.csv`, `per_region_geometry.csv`, `catastrophic_geometry_failures.csv`, `worst_geometry_failures/worst_samples.json`. | Active. |
| `build_hard_alignment_validation.py` | Hard slice validation | Builds a hard-case alignment validation manifest from an existing manifest, usually AFLW2000-3D, and can run geometry evaluation over the slice. | Source manifest, output directory, hard-slice thresholds, include/unposed/hard-only flags, optional cache/models/weights for geometry evaluation. | Filtered `manifest.json`, `hard_slice_summary.json`, optional geometry outputs. | Active. |
| `validate_geometry_signals.py` | Hard slice validation, scorer diagnostics | Evaluates candidate geometry signals, oracle labels, and selector ablations over a manifest/cache. | `--manifest`, `--cache-dir`, `--models`, optional variants, weights, outlier threshold, aligned size, signal thresholds. | `candidate_index.csv`, `signal_validation_report.json`, `signal_validation_report.csv`, `selector_ablations.json`, `selector_ablations.csv`. | Candidate for merge into `evaluate_alignment_geometry.py` or `search_ensemble_setup.py` reporting. |
| `evaluate_aflw_profile.py` | Hard slice validation | Cache-only profile-aware AFLW/profile diagnostic metrics. This is diagnostic after geometry became the primary promotion objective. | `--manifest`, `--cache-dir`, `--models`, optional variants, weights or promoted setup, outlier threshold, normalizer, region weights, PCK thresholds. | `aflw_profile_metrics.json`, `aflw_profile_metrics.csv`, `aflw_region_failures.csv`. | Candidate for merge into geometry/profile reporting. Not deprecated because it still provides profile-specific diagnostics. |
| `failure_viewer.py` | Metadata and debug sidecars | Creates worst-first overlays, debug records, and contact sheets from quality metrics and cached predictions. | Metrics JSON, manifest, cache dir, models, optional weights, output/debug dirs, limit, outlier threshold. | Worst-case JSON/CSV records, overlays, contact sheets. | Active debug helper, candidate for merge into the main reporting surface. |
| `runtime_resolver_scorer_data.py` | Scorer training, scorer evaluation, metadata sidecars | Shared data builders for runtime resolver scorer training/evaluation. Loads manifests, caches, weights, candidates, optional resolver metadata, and creates candidate-quality contexts and rows. | Called by scorer train/eval tools with manifest/cache/weights/candidates and optional resolver metadata. | In-memory `SampleCandidateContext` and candidate-quality rows, plus CSV writer helpers for candidate tables. | Active helper, candidate for move from `tools/` into `lib.landmarks` if it remains shared by multiple CLIs. |
| `train_runtime_resolver_scorer.py` | Scorer training | Trains a portable logistic runtime resolver candidate-quality scorer. | `--weights`, `--output-dir`, at least one manifest/cache pair from `--gt-manifest` plus `--gt-cache-dir` or `--production-manifest` plus `--production-cache-dir`, optional candidates and training hyperparameters. | `runtime_resolver_scorer.json`, `runtime_resolver_scorer_training_rows.csv`, `runtime_resolver_scorer_eval_rows.csv`, `candidate_table.csv`, `runtime_resolver_scorer_training_metrics.json`. | Active. |
| `evaluate_runtime_resolver_scorer.py` | Scorer evaluation, production promotion | Evaluates learned runtime resolver scorer policy against resolver baselines, static alternatives, production data, and GT-hard data. | Scorer artifact, weights, candidates, output dir, GT and/or production manifest/cache inputs, optional eval split and resolver metadata sidecar. | `scorer_metrics.json`, `scorer_policy_report.json`, optional held-out eval report, `scorer_policy_report.csv`, `scorer_worst_samples.json`, `scorer_feature_importance.csv`. | Active. |
| `production_promotion_gate.py` | Production promotion | Standalone production validation gate for ensemble policies or explicit candidates. Compares chosen policy against best single, oracle, and static ensemble summaries. | Production manifest, prediction cache, weights, output dir, policy, candidate list, gate thresholds. | `production_promotion_report.json`, `production_promotion_report.md`, `production_per_bucket_metrics.csv`, `production_policy_failures.csv`, `production_worst_samples.json`. | Active. |

## Overlap and unclear contracts

1. `run_static_weight_pipeline.py` and the leaf tools overlap by design. The orchestrator owns flow control and standard paths, while `build_quality_dataset.py`, `cache_predictions.py`, `compute_static_weights.py`, and `failure_viewer.py` own individual steps. Keep this split, but document that orchestration owns path layout and leaf tools own pure step behavior.

2. `compute_static_weights.py` and `search_ensemble_setup.py` both fit weights. `compute_static_weights.py` fits one generator for the static-weight validation path. `search_ensemble_setup.py` searches many generators and strategies and writes promoted artifacts. Consolidation should make `search_ensemble_setup.py` the promotion path and keep `compute_static_weights.py` only as a simple diagnostic/legacy step until retired.

3. `evaluate_alignment_geometry.py`, `validate_geometry_signals.py`, and `evaluate_aflw_profile.py` all consume manifest/cache/model/weights inputs and emit per-candidate diagnostics. Fusion and bbox helpers are already centralized, but CLI contracts remain separate. The likely merge target is a single evaluation command with selectable report modules: geometry metrics, signal validation, profile metrics, and worst samples.

4. `search_ensemble_setup.py` imports and runs the production promotion gate when production inputs are present. `production_promotion_gate.py` also works as a standalone gate for explicit weights or policies. Keep the standalone tool for repeatable gate checks, but make the search tool's production-gate wrapper clearly subordinate.

5. `runtime_resolver_scorer_data.py` is a shared helper under `tools/`. Because both scorer training and evaluation import it, it behaves more like library code than a CLI. Move it to `lib.landmarks.ensemble` or `lib.landmarks.evaluation` during consolidation.

6. Runtime metadata sidecar handling is split between manifest metadata and optional resolver metadata JSONL loading in `runtime_resolver_scorer_data.py` and `evaluate_runtime_resolver_scorer.py`. The expected schema for sidecar rows should be promoted to a stable doc before more tools consume it.

7. Deprecated flag aliases exist in `search_ensemble_setup.py`, including single-model baseline and explicit production weights aliases. These are not full deprecated tools, but the comments/help text should remain until the aliases are removed.

## Recommended consolidation order

1. Define one public workflow command for promotion: dataset or manifest in, cache in, splits in, candidate search out, production gate out.
2. Fold signal/profile diagnostics behind `evaluate_alignment_geometry.py` or a new `evaluate_pipeline.py` facade.
3. Move `runtime_resolver_scorer_data.py` into `lib.landmarks` and leave a temporary import shim only if external callers exist.
4. Keep `run_static_weight_pipeline.py` as the legacy/static validation orchestrator until `search_ensemble_setup.py` fully covers the same reporting and promotion needs.
5. Remove no files in this ticket. Only mark true deprecated tools in docstrings once a replacement command and migration path are explicit.
