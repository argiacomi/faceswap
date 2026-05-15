# Landmark Ensemble Pipeline

Last reviewed against the current implementation: 2026-05-15.

This repo ships a landmark ensemble as both:

- a Faceswap extract aligner plugin at `plugins/extract/align/ensemble.py`
- standalone evaluation and promotion tooling under `lib/landmarks` and `tools/landmarks`

The runtime path is intentionally small: run several landmark adapters on the same crop, convert every prediction to canonical 68-point frame-space landmarks, fuse there, then convert the fused result back to Faceswap's normalized crop coordinates.

## Runtime Use In Faceswap

Select the ensemble aligner during extraction:

```bash
python faceswap.py extract --aligner ensemble
```

Runtime defaults live in `plugins/extract/align/ensemble_defaults.py`. The plugin currently supports these adapter names:

- `hrnet`
- `spiga`
- `orformer`

Adapters are built around the existing Faceswap aligner plugins. Missing optional plugin modules are skipped at load time. If no enabled adapters are available, the ensemble fails fast.

## Runtime Flow

1. Faceswap passes detector boxes into the aligner.
2. `Ensemble.pre_process()` expands each box into a shared square crop using `crop_scale`.
3. Each enabled adapter predicts landmarks on that crop.
4. Adapter outputs are converted to canonical `2d_68` original-frame pixels.
5. The configured fusion strategy combines predictions in frame space.
6. The fused landmarks are converted back to normalized crop coordinates for Faceswap post-processing.

This coordinate round trip matters. Fusion never mixes model-specific crop spaces. The common fusion space is always canonical 68-point frame pixels.

## Runtime Options

| Option | Default | Notes |
| --- | ---: | --- |
| `batch_size` | `8` | Ensemble wrapper batch size. |
| `models` | `hrnet,spiga,orformer` | Adapter list to try. |
| `crop_scale` | `1.6` | Square crop scale relative to the longest detector-box side. |
| `strategy` | `static_weighted` | Canonical fusion strategy. See below. |
| `reject_outliers` | `true` | Deprecated compatibility flag. With `strategy=static_weighted`, this maps to `static_weighted_hard_drop`. Leave disabled for new configs and set `strategy` directly. |
| `outlier_threshold` | `3.5` | Robust z-score threshold used only by threshold-aware strategies. |
| `min_models` | `1` | Minimum successful adapter predictions required per face. |
| `setup_path` | empty | Optional path to a promoted `best_setup.json`. |
| `setup_mode` | `off` | `off`, `strict`, or `fallback`. Empty `setup_path` always means `off`. |
| `fallback_strategy` | `plain_average` | Used only when `setup_mode=fallback` and setup loading fails. Use `adapter_config` to fall back to the configured `strategy`. |

## Fusion Strategies

Canonical strategy names are centralized in `lib/landmarks/ensemble/strategies.py`:

| Strategy | Weights | Threshold | Behavior |
| --- | --- | --- | --- |
| `plain_average` | No | No | Equal average across successful predictions. |
| `static_weighted` | Yes | No | Weighted average using adapter weights or promoted per-landmark weights. |
| `static_weighted_hard_drop` | Yes | Yes | Per-landmark robust outliers are zeroed, then weights are renormalized. |
| `static_weighted_downweight` | Yes | Yes | Per-landmark robust outliers are multiplied by `0.25`, then weights are renormalized. |
| `weighted_median` | Yes | No | Weighted median per landmark coordinate. |

`static_weighted_outliers` is accepted as a legacy alias for `static_weighted_hard_drop`, but promoted artifacts must store canonical names only.

## Promoted Setup Runtime

A candidate search can promote a setup bundle containing:

- `best_setup.json`
- `best_weights.json`
- `promotion_report.md`

At runtime, set `setup_path` to the generated `best_setup.json`.

Modes:

- `setup_mode=off`: ignore `setup_path`.
- `setup_mode=strict`: load and validate the setup, and fail fast on any mismatch.
- `setup_mode=fallback`: log setup validation errors and use `fallback_strategy`.

When a setup is loaded successfully, it overrides the configured runtime strategy with the promoted strategy, threshold, model list, and per-landmark weights. The runtime validates that every model required by the setup is available in the loaded adapters.

The promoted artifact schema is versioned as `artifact_schema_version = 1` and is restricted to canonical `2d_68` landmarks.

## Recommended Offline Flow

### 1. Build or reuse a quality dataset manifest

For WFLW annotations:

```bash
python tools/landmarks/build_quality_dataset.py \
  --dataset wflw \
  --source-dir path/to/images \
  --wflw-annotations path/to/list_98pt_rect_attr_train_test.txt \
  --output-dir outputs/landmark_dataset
```

For COFW JSON:

```bash
python tools/landmarks/build_quality_dataset.py \
  --dataset cofw \
  --source-dir path/to/images \
  --cofw-json path/to/cofw.json \
  --output-dir outputs/landmark_dataset
```

The broader pipeline also supports WFLW, COFW, 300W, directory fixtures, MERL-RAV, and AFLW2000-3D through `run_static_weight_pipeline.py`.

### 2. Populate the prediction cache

Import existing `.npy` predictions:

```bash
python tools/landmarks/cache_predictions.py \
  --manifest outputs/landmark_dataset/manifest.json \
  --models hrnet,spiga,orformer \
  --prediction-root hrnet=path/to/hrnet_predictions \
  --prediction-root spiga=path/to/spiga_predictions \
  --prediction-root orformer=path/to/orformer_predictions \
  --cache-dir outputs/landmark_predictions \
  --checkpoint validation
```

Or run the supported model adapters directly:

```bash
python tools/landmarks/cache_predictions.py \
  --manifest outputs/landmark_dataset/manifest.json \
  --models hrnet,spiga,orformer \
  --cache-dir outputs/landmark_predictions \
  --checkpoint-tag validation \
  --run-models \
  --batch-size 8 \
  --device auto
```

Prediction cache entries are stored in frame-space canonical 68-point format with metadata for freshness checks.

### 3. Run the quality harness

```bash
python tools/landmarks/run_quality_harness.py \
  --manifest outputs/landmark_dataset/manifest.json \
  --cache-dir outputs/landmark_predictions \
  --models hrnet,spiga,orformer \
  --variants plain_average,static_weighted,static_weighted_hard_drop,static_weighted_downweight,weighted_median \
  --output-dir outputs/landmark_quality
```

The harness writes:

- `metrics.json`
- `metrics.csv`
- `per_landmark_error.csv`
- `per_region_error.csv`
- `per_condition_error.csv`
- `per_scenario_bucket_error.csv`
- `ensemble_regressions.csv`
- `best_variant_summary.json`

Overall NME is aggregated with equal weighting across dataset-condition scenario buckets so large easy buckets do not hide smaller hard buckets.

### 4. Fit static weights

```bash
python tools/landmarks/compute_static_weights.py \
  --manifest outputs/landmark_dataset/manifest.json \
  --cache-dir outputs/landmark_predictions \
  --models hrnet,spiga,orformer \
  --generator regularized_inverse_error \
  --output configs/ensemble/static_landmark_weights.json \
  --report outputs/landmark_weights/static_weight_report.json
```

Supported weight generators:

- `equal`
- `inverse_mean_error`
- `inverse_median_error`
- `region_inverse_error`
- `regularized_inverse_error`
- `winner_take_landmark`
- `softmax_negative_error`

Generator params can be passed as comma-separated `key=value` pairs:

```bash
python tools/landmarks/compute_static_weights.py \
  --manifest outputs/landmark_dataset/manifest.json \
  --cache-dir outputs/landmark_predictions \
  --models hrnet,spiga,orformer \
  --generator softmax_negative_error \
  --generator-params temperature=2.0
```

### 5. Run the end-to-end static-weight pipeline

This is the main validation runner for building data, caching predictions, creating fit/select/report splits, deriving weights, evaluating variants, and writing debug outputs.

```bash
python tools/landmarks/run_static_weight_pipeline.py \
  --datasets wflw,cofw \
  --output-root outputs/static_weight_validation \
  --models hrnet,spiga,orformer \
  --run-predictions \
  --split-mode scenario-stratified
```

Split behavior:

- Default: `scenario-stratified`
- Other modes: `random`, `file`, `none`
- Default ratios: fit `0.6`, select `0.2`, report `0.2`
- The generated split file is written to `<output-root>/splits/splits.json`
- Fit weights are derived from the fit manifest, not the full manifest, unless `--split-mode none` is used

The runner writes a `run_summary.json` with dataset counts, cache counts, split provenance, best baseline model, best weighted variant, deltas vs best single model, and stage status.

## Candidate Search And Promotion

After you have a manifest, populated cache, and splits file, search the setup space:

```bash
python tools/landmarks/search_ensemble_setup.py \
  --manifest outputs/static_weight_validation/dataset/manifest.json \
  --cache-dir outputs/static_weight_validation/cache \
  --splits outputs/static_weight_validation/splits/splits.json \
  --models hrnet,spiga,orformer \
  --model-subsets all,pairs,triples \
  --weight-generators equal,inverse_mean_error,regularized_inverse_error \
  --strategies static_weighted,static_weighted_downweight,weighted_median \
  --outlier-thresholds 2.5,3.5,4.5 \
  --output-dir outputs/landmark_setup_search
```

The search is cache-only. It does not run adapters.

It evaluates candidates across:

- model subsets
- weight generators
- canonical fusion strategies
- outlier thresholds for threshold-aware strategies only
- crop scale
- bbox source

Selection uses `extract_alignment_v1`:

```text
score = overall_nme
      + 0.5 * failure_rate
      + 0.25 * regression_rate_vs_best_single
      + 0.25 * bucket_regression_rate_vs_best_single
```

The lowest score wins, with lower overall NME as the tie-breaker.

Outputs:

- `candidate_results.csv`
- `candidate_results.json`
- `best_setup.json`
- `best_weights.json`
- `promotion_report.md`

Use the generated `best_setup.json` as runtime `setup_path`.

## Debugging And Failure Review

Generate debug outputs and inspect failures:

```bash
python tools/landmarks/debug_output.py \
  --manifest outputs/landmark_dataset/manifest.json \
  --models hrnet,spiga,orformer

python tools/landmarks/failure_viewer.py \
  --metrics outputs/landmark_quality/metrics.json \
  --manifest outputs/landmark_dataset/manifest.json \
  --cache-dir outputs/landmark_predictions \
  --debug-dir outputs/landmark_debug
```

Runtime `Ensemble.last_debug_metadata` records per-face fusion metadata, including active models, source models, weights, kept and rejected indices, rejected landmark count, adapter errors, strategy, threshold, setup path, setup mode, promoted candidate ID, and weight source.

## Core Modules

| Path | Purpose |
| --- | --- |
| `plugins/extract/align/ensemble.py` | Faceswap runtime aligner wrapper. |
| `plugins/extract/align/ensemble_defaults.py` | Runtime config defaults. |
| `lib/landmarks/schema.py` | Canonical `2d_68` schema and schema conversion helpers. |
| `lib/landmarks/coordinates.py` | Normalized-crop and frame-space coordinate transforms. |
| `lib/landmarks/adapters.py` | Adapter interfaces and Faceswap aligner wrapper. |
| `lib/landmarks/fusion.py` | FusionResult, plain average, static weighted fusion, and weight normalization. |
| `lib/landmarks/rejection.py` | Per-landmark robust outlier handling and weighted median. |
| `lib/landmarks/ensemble/strategies.py` | Canonical fusion strategy registry. |
| `lib/landmarks/ensemble/weights.py` | Per-landmark static weight normalization and loading. |
| `lib/landmarks/ensemble/weight_generators.py` | Pluggable static weight generators. |
| `lib/landmarks/ensemble/promoted_setup.py` | Promoted setup writer, loader, and strict artifact validators. |
| `lib/landmarks/eval/harness.py` | Cache-driven quality harness and metrics output. |
| `lib/landmarks/eval/candidate_search.py` | Cache-only candidate enumeration, scoring, and selection. |
| `lib/landmarks/eval/splits.py` | Fit/select/report split creation and provenance hashes. |

## Compatibility Notes

- Runtime fusion output remains a normal Faceswap aligner result.
- Promoted setup artifacts must use canonical strategy names, not aliases.
- `reject_outliers` remains only to preserve older config behavior.
- The runtime can continue without a promoted setup by leaving `setup_path` empty.
- The candidate search and quality harness are cache-driven. Build or import predictions before running them.
