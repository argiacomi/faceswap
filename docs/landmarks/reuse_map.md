# Landmark Ensemble Reuse Map

This map keeps the landmark ensemble work standalone-first. The local
`faceswap` repository is the implementation target; `faceswap-test-dev` is a
read-only reference for ports and tooling.

## Summary

| Candidate | Source path | Classification | Notes |
| --- | --- | --- | --- |
| HRNet aligner | `plugins/extract/align/hrnet.py` | reuse as-is | Existing Faceswap aligner. Emits `(N, 68, 2)` normalized crop-space landmarks via `post_process()`. Good first adapter target. |
| HRNet defaults | `plugins/extract/align/hrnet_defaults.py` | reuse as-is | Existing config/defaults and model cache behavior. |
| FAN aligner | `plugins/extract/align/fan.py` | use as reference only | Useful as another aligner contract example, not part of the MVP ensemble. |
| cv2-dnn aligner | `plugins/extract/align/cv2_dnn.py` | use as reference only | Useful low-dependency smoke/reference aligner, not part of MVP ensemble. |
| Extract plugin base | `plugins/extract/base.py` | reuse as-is | Defines the aligner contract: align plugins return float32 normalized `0.0-1.0` points. Current docstring says `(N, 68, 2)`, but runtime accepts 68 or 98 through `LandmarkType.from_shape()`. |
| Align handler | `lib/infer/align.py` | reuse as-is | Converts normalized plugin output to original frame pixels with `batch_transform(batch.matrices, result, in_place=True)`, records `batch.landmark_type`, handles re-feed/re-align/filtering. |
| Landmark constants | `lib/align/constants.py` | reuse as-is | Supports `LM_2D_68` and `LM_2D_98`, region ranges in `LANDMARK_PARTS`, and `MAP_2D_68` 98-to-68 mapping. |
| Landmark utilities | `lib/align/aligned_utils.py` | reuse as-is | Provides `points_to_68()`, `batch_transform()`, `batch_adjust_matrices()`, `batch_align()`, `batch_resize()`, and `sub_crop()`. |
| Plugin loader | `plugins/plugin_loader.py` | reuse as-is | SPIGA/ORFormer plugin files become selectable by normal plugin discovery once ported with defaults. |
| Extract config | `config/extract.ini` | port with small changes | Add SPIGA/ORFormer/ensemble sections after plugin/defaults are ported or regenerated. |
| SPIGA plugin | `faceswap-test-dev/plugins/extract/align/spiga.py` | port with small changes | Main SPIGA source. Includes 300W/MERL-RAV 68 and WFLW 98 variants, model URL/SHA256 registry, crop scaling, and Faceswap-compatible `post_process()`. |
| SPIGA defaults | `faceswap-test-dev/plugins/extract/align/spiga_defaults.py` | port with small changes | Defines `model`, `crop_scale`, and `batch_size`. Defaults to WFLW 98 and crop scale 1.60. |
| SPIGA vendored code | `faceswap-test-dev/plugins/extract/align/_spiga/` | port with small changes | Required model implementation and 3D mean-face files. Preserve upstream BSD-3-Clause provenance notes. |
| ORFormer plugin | `faceswap-test-dev/plugins/extract/align/orformer.py` | port with small changes | Main ORFormer source. Includes WFLW 98 and 300W 68 variants, Google Drive archive download/extract/validate logic, and Faceswap-compatible normalized output. |
| ORFormer defaults | `faceswap-test-dev/plugins/extract/align/orformer_defaults.py` | port with small changes | Defines `model`, `crop_scale`, and `batch_size`. Defaults to WFLW 98 and crop scale 1.20. |
| ORFormer vendored code | `faceswap-test-dev/plugins/extract/align/_orformer/` | use as reference only until license review | Required model wrapper exists, but the reference plugin notes upstream lacks an explicit license. Keep this explicit before porting beyond private use. |
| SPIGA tests | `faceswap-test-dev/tests/plugins/extract/align/spiga_test.py` | port with small changes | Good smoke coverage for ROI geometry, channel order, model registry, SHA256 shape, and `post_process()`. |
| ORFormer tests | `faceswap-test-dev/tests/plugins/extract/align/orformer_*test.py` | port with small changes | Good smoke coverage for ROI geometry, archive extraction, checkpoint mismatch validation, parity/reference resize behavior. |
| Automask quality metrics | `faceswap-test-dev/lib/auto_mask/quality_metrics.py` | use as reference only | Pure metric functions and scenario aggregation patterns. Landmark metrics will need NME/AUC/failure-rate equivalents rather than mask IoU. |
| Automask quality harness | `faceswap-test-dev/lib/auto_mask/quality_harness.py` | use as reference only | Manifest runner, scenario metrics, and cache orchestration patterns. Keep landmark harness independent. |
| Automask quality dataset tooling | `faceswap-test-dev/tools/automask/build_quality_dataset.py`, `automask_dataset_common.py`, dataset adapters | use as reference only | Useful manifest/audit/overlay/scenario patterns for landmark dataset tooling. Do not couple landmark evaluation to automask packages. |
| Automask harness/reporting | `faceswap-test-dev/tools/automask/run_quality_harness.py`, `debug_output.py`, `autotune_quality_thresholds.py`, `tune_runtime_settings.py`, `tune_binary_threshold.py` | use as reference only | Reuse report shape, threshold proposal, and worst-first visual patterns conceptually for `tools/landmarks/`. |

## Required Source Paths

SPIGA source path:

- `faceswap-test-dev/plugins/extract/align/spiga.py`
- `faceswap-test-dev/plugins/extract/align/spiga_defaults.py`
- `faceswap-test-dev/plugins/extract/align/_spiga/`

ORFormer source path:

- `faceswap-test-dev/plugins/extract/align/orformer.py`
- `faceswap-test-dev/plugins/extract/align/orformer_defaults.py`
- `faceswap-test-dev/plugins/extract/align/_orformer/`

## HRNet Fork Deltas

The local `faceswap/plugins/extract/align/hrnet.py` already contains the production HRNet aligner:

- `HRNet.__init__()` configures input size 256, RGB input, float32 `0-1` scale, and legacy realign centering.
- `HRNet.load_model()` loads `hrnet_landmark_v2.pth` through `GetModel`.
- `HRNet.post_process()` decodes heatmaps to normalized crop-space `(N, 68, 2)` landmarks.
- The non-DARK branch appears to have a typo: `pnt_y = np.clip(pnt_x, ...)` should likely clip `pnt_y`. The default DARK path bypasses this, but the issue should be fixed before using non-DARK HRNet for evaluation.

The test-dev SPIGA/ORFormer paths are additions rather than HRNet replacements. HRNet can be the first adapter and baseline model without porting deltas.

## 68/98 Normalization Deltas

Faceswap already has the canonical mapping primitives:

- `LandmarkType.LM_2D_68` and `LandmarkType.LM_2D_98` in `lib/align/constants.py`.
- `LANDMARK_PARTS` region ranges for both schemas.
- `MAP_2D_68[LandmarkType.LM_2D_98]` maps WFLW-style 98 landmarks to canonical 68.
- `points_to_68()` in `lib/align/aligned_utils.py` handles batched and unbatched inputs.

The standalone package should wrap these as `normalize_landmarks()` so adapters and evaluation code do not import low-level Faceswap utilities directly everywhere. Document that indices follow Faceswap/iBUG convention where subject-left facial features appear under `left_*` regions and image x increases left-to-right.

## Coordinate Conventions

Aligner plugin output is normalized crop space. `Align._matrices_from_roi()` builds matrices that scale normalized unit-square landmarks to the plugin ROI in original frame pixels. `Align.post_process()` applies:

```python
batch_transform(batch.matrices, result, in_place=True)
```

After that call, `batch.landmarks` are original frame pixel coordinates. Fusion must use comparable frame-space points unless explicitly operating on synthetic normalized test data. No production fusion should mix crop-space outputs from different model ROIs.

## Model Weight And Config Requirements

SPIGA:

- 300W: `spiga_300wprivate.pt`, 68 landmarks, 13 edges, SHA256 recorded in `spiga.py`.
- MERL-RAV: `spiga_merlrav.pt`, 68 landmarks, 13 edges, SHA256 recorded in `spiga.py`.
- WFLW: `spiga_wflw.pt`, 98 landmarks, 15 edges, SHA256 recorded in `spiga.py`.
- Downloads from Hugging Face URL template `https://huggingface.co/aprados/spiga/resolve/main/{filename}?download=true`.

ORFormer:

- Archive: `orformer_weights_20240704.zip`, SHA256 recorded in `orformer.py`.
- 300W: paired HGNet and ORFormer/VQ-VAE checkpoints, 68 landmarks, 13 edges.
- WFLW: paired HGNet and ORFormer/VQ-VAE checkpoints, 98 landmarks, 15 edges.
- Downloads official Google Drive archive and extracts validated members into Faceswap cache.
- Upstream code/weights licensing gap is documented in the reference plugin and should remain visible.
- ORFormer avoids an `einops` runtime dependency by using local tensor reshaping helpers in `_orformer/model.py`.

## Automask Quality Tooling Candidates

Use these as patterns for landmark-specific tooling:

- Manifest and audit output: `tools/automask/build_quality_dataset.py`, `automask_dataset_common.py`.
- Pure quality metric and harness patterns: `lib/auto_mask/quality_metrics.py`, `lib/auto_mask/quality_harness.py`.
- Dataset adapters: `lapa_adapter.py`, `celebamask_hq_adapter.py`, `facesynthetics_adapter.py`, `portrait_matting_adapter.py`.
- Harness/reporting: `tools/automask/run_quality_harness.py`.
- Debug output: `tools/automask/debug_output.py`.
- Tuning patterns: `autotune_quality_thresholds.py`, `tune_runtime_settings.py`, `tune_binary_threshold.py`.

## Test Coverage Candidates

Port or mirror these tests into local `faceswap` as each ticket lands:

- `tests/plugins/extract/align/spiga_test.py`
- `tests/plugins/extract/align/orformer_test.py`
- `tests/plugins/extract/align/orformer_preprocess_test.py`
- `tests/plugins/extract/align/orformer_checkpoint_test.py`
- `tests/plugins/extract/align/orformer_attention_test.py`
- `tests/plugins/extract/align/orformer_parity_test.py`
- `tests/plugins/extract/align/orformer_reference_resize_test.py`
- `tests/lib/infer/aligner_contract_test.py`
- `tests/lib/infer/mixed_landmarks_test.py`
- Automask quality tests under `tests/tools/automask_*` for CLI, dataset, report, debug, and tuning patterns.

## Initial Port Order

1. Add standalone `lib/landmarks` schema, adapters, cache, metrics, fusion, and outlier modules.
2. Port SPIGA plugin/defaults/vendor/tests.
3. Port ORFormer plugin/defaults/vendor/tests after explicitly preserving the license warning.
4. Add model-backed adapters that normalize every prediction to canonical `[68, 2]`.
5. Add cache, dataset, harness, weights, and failure viewer tools.
6. Add the thin production `plugins/extract/align/ensemble.py` wrapper only after standalone evaluation has a baseline.
