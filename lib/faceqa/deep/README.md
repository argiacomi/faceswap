# FaceQA Deep Audit (DECA)

Optional deep dataset-quality analysis for the FaceQA `coverage` workflow
(issue #156). It predicts training-readiness *before* training by measuring how
well a faceset spans 3D **expression**, **pose**, and **lighting** space, using
coefficients regressed by the [DECA](https://github.com/yfeng95/DECA) encoder
instead of landmark/thumbnail heuristics alone.

The default FaceQA workflow is unchanged and lightweight: nothing in this
package is imported unless `--deep-analysis deca` is passed.

## Usage

```bash
faceswap tools faceqa \
  --mode coverage \
  -a alignments.fsa \
  -r /path/to/source/frames \
  -o /path/to/output \
  --deep-analysis deca
```

Results are written to the standard coverage report under a new `deep_audit`
block (both the JSON and, when present, summarized in the Markdown report).

## Weights

The DECA encoder weights (`deca_model.tar`) are downloaded on first use through
the standard faceswap model cache and SHA256-validated before loading. FaceQA
does not need the FLAME mesh decoder or PyTorch3D rasterizer because it consumes
encoder coefficients only, not rendered meshes.

The checkpoint lands at `.fs_cache/deca_model.tar`.

No new pip dependency is introduced: the encoder uses `torch` + `torchvision`
and the metrics use `numpy` + `scikit-learn`, all already pinned by faceswap.
The checkpoint is loaded as a full pickle, as required by the official PyTorch
archive format.

## What it reports (`deep_audit`)

| Field | Meaning |
| --- | --- |
| `expression_space_coverage` | Occupancy + entropy of DECA expression coefficients (absolute binning). |
| `pose_space_coverage` | Occupancy + entropy of DECA global/head and jaw pose coefficients. |
| `lighting_space_coverage` | Occupancy + entropy of DECA spherical-harmonic lighting coefficients. |
| `latent_entropy` | Normalized Shannon entropy of a joint coarse binning of the dominant latent dims. |
| `cluster_coverage` | k-means partition `balance` plus an absolute `mean_dispersion` collapse signal. |
| `device` / `device_auto_selected` | Torch device used for DECA inference (`auto` selects CUDA, then Apple MPS, then CPU). |
| `missing_keys_count` / `unexpected_keys_count` / `matched_key_ratio` | Real-weight load diagnostics; the default loader refuses partial state-dict matches. |
| `deca_readiness` | 0–100 sub-score derived from entropy, occupied coverage, latent entropy, cluster balance, and absolute cluster dispersion (independent of the landmark readiness score). |
| `landmark_vs_deca` | Side-by-side `entropy_coverage` / `occupied_coverage` for the landmark vs DECA views. |
| `pruning_signals` | Per-face DECA cells used to keep learned-space expression / pose / lighting / latent variation from being auto-pruned. |

Coverage is measured in **absolute** coefficient space (fixed reference scale),
not per-faceset z-scores, so a faceset collapsed onto a single
expression/lighting scores low rather than being normalized to look diverse.

When `--suggest-pruning` is combined with `--deep-analysis deca`, FaceQA passes
the per-face DECA cells into the tight prune-subgroup splitter. Faces that look
redundant under landmark/thumbnail metrics but differ in DECA expression, pose,
lighting, or latent cluster stay in separate prune groups instead of being
automatically collapsed.

## Validation status

The pure-numpy metrics, the coefficient `decode_parameters` layout, the
state-dict remapper, and the orchestration are unit-tested with a synthetic
encoder and do not require the weights. The **real-weights** path (loading
`deca_model.tar` and running the encoder) must be validated once on hardware
with the licensed weights — see the issue #156 validation checklist. In
particular, confirm against the real checkpoint:

- the loaded state-dict reports few/zero missing/unexpected keys,
- the input preprocessing (crop centering, `[0,1]` scaling) matches upstream,
- the per-group latent scales (`EXPRESSION_SCALE`, `POSE_SCALE`, `LIGHT_SCALE`
  in `audit.py`) are sensible for the real coefficient ranges.
