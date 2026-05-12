# Landmark Ensemble Quickstart

This repo includes a standalone landmark ensemble package under `lib/landmarks`,
plus a thin Faceswap aligner wrapper at `plugins/extract/align/ensemble.py`.

## Use In Faceswap

Select the ensemble aligner from extract:

```bash
python faceswap.py extract --aligner ensemble
```

The plugin runs the configured HRNet/SPIGA/ORFormer adapters, normalizes outputs
to canonical 68-point landmarks, fuses them, and returns normal Faceswap aligner
output. Options live in `plugins/extract/align/ensemble_defaults.py`.

## Evaluation Flow

Build a quality dataset manifest:

```bash
python tools/landmarks/build_quality_dataset.py \
  --dataset wflw \
  --source-dir path/to/prepared_samples \
  --output-dir outputs/landmark_dataset
```

For WFLW annotation text or COFW JSON exports:

```bash
python tools/landmarks/build_quality_dataset.py \
  --dataset wflw \
  --source-dir path/to/images \
  --wflw-annotations path/to/list_98pt_rect_attr_train_test.txt \
  --output-dir outputs/landmark_dataset

python tools/landmarks/build_quality_dataset.py \
  --dataset cofw \
  --source-dir path/to/images \
  --cofw-json path/to/cofw.json \
  --output-dir outputs/landmark_dataset
```

Cache predictions once per model/sample:

```bash
python tools/landmarks/cache_predictions.py \
  --sample-id sample_001 \
  --model hrnet \
  --prediction path/to/hrnet_landmarks.npy
```

Run metrics from the cache:

```bash
python tools/landmarks/run_quality_harness.py \
  --manifest outputs/landmark_dataset/manifest.json \
  --cache-dir outputs/landmark_predictions \
  --variants plain_average,static_weighted,static_weighted_outliers
```

Generate static weights and debug outputs:

```bash
python tools/landmarks/compute_static_weights.py \
  --manifest outputs/landmark_dataset/manifest.json \
  --models hrnet,spiga,orformer

python tools/landmarks/debug_output.py \
  --manifest outputs/landmark_dataset/manifest.json \
  --models hrnet,spiga,orformer

python tools/landmarks/failure_viewer.py \
  --metrics outputs/landmark_quality/metrics.json \
  --debug-dir outputs/landmark_debug
```

## Core Modules

- `lib/landmarks/schema.py`: canonical 68-point schema and 98-to-68 mapping.
- `lib/landmarks/adapters.py`: shared adapter interface.
- `lib/landmarks/fusion.py`: average and static weighted fusion.
- `lib/landmarks/ensemble/outliers.py`: per-landmark outlier rejection.
- `lib/landmarks/eval/`: prediction cache, metrics, harness, visualization.
