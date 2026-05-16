# Alignment Geometry Roadmap for Faceswap Extract

## Summary

The landmark ensemble work has built a solid evaluation and promotion foundation: datasets can be built and split, landmark predictions can be cached, multiple models can be compared, static weights can be generated, ensemble variants can be searched, and promoted setups can be written for runtime use.

That foundation is useful, but the current optimization target is still too benchmark-shaped. It primarily tells us which model has better landmark point accuracy. Faceswap extract quality depends on something more specific: whether the predicted landmarks produce stable, usable face geometry for alignment, masks, warping, and swap placement.

This roadmap shifts the product goal from:

```text
lowest average landmark error
```

to:

```text
lowest hard-case geometry failure rate with acceptable landmark accuracy
```

The objective is not to win generic 300W/WFLW-style NME. The objective is to produce aligned faces that are stable and usable in the extract pipeline, especially for profile faces, occlusions, blur, truncation, detector noise, and other real Faceswap failure cases.

## North star

A landmark setup is better only if it improves downstream extract geometry.

That means better:

- crop center, scale, and rotation
- visible face hull for masks
- eye, nose, and mouth placement for swap alignment
- robustness on profile, occlusion, blur, truncation, and bad detector boxes
- catastrophic failure rate on hard slices
- temporal stability on video sequences

NME remains useful as a diagnostic, but it should not be the primary promotion criterion.

## Why the current objective is not enough

The current validation flow compares HRNet, SPIGA, ORFormer, and ensemble variants using NME, failure rate, AUC, and per-landmark/per-region error. That is a good model-comparison layer.

It does not directly answer the product question:

```text
Will this landmark output produce a better extracted face?
```

A model can have slightly worse point error but produce a better alignment transform. Another model can have decent NME but create a bad crop, bad roll, invalid mask hull, or unusable mouth/eye placement. For Faceswap, the second model is worse even if the benchmark says it is better.

The repo's extract path makes this dependency direct. `AlignedFace` uses landmarks to compute the alignment matrix, transformed landmarks, pose, ROI, crop, and mask-related geometry. Therefore, transform quality and crop/mask stability are first-order output metrics, not secondary diagnostics.

## Product principle

Optimize for stable visible face geometry, not hallucinated full-face landmarks.

For profile and occluded faces, the best output may not be the landmark set with the best full 68-point average error. The best output is the one that preserves usable visible geometry without letting ambiguous or invisible points distort the crop, mask, or swap placement.

This also means one 68-point output may eventually be the wrong internal contract for every downstream task. We should move toward separate geometry products:

```text
alignment_landmarks: stable interior anchors for crop, scale, and rotation
visible_mask_landmarks: visible face hull for mask construction
canonical_landmarks: full 68-point compatibility output
geometry_flags: profile, occluded, truncated, low confidence, etc.
geometry_confidence: confidence in the chosen geometry
```

That split should come after we have the metrics and evidence to justify it.

## Roadmap

### Phase 1: Measure extract geometry directly

Ticket: [#76 Add alignment geometry evaluation metrics](https://github.com/argiacomi/faceswap/issues/76)

This is the foundation. Before changing runtime behavior, we need a geometry evaluation layer that measures the objects Faceswap actually uses.

The evaluator should compute:

- affine/similarity transform error
- crop center error
- scale error
- roll/rotation error
- aligned ROI coverage and miss rate
- visible-face hull IoU
- landmark/mask hull quality
- eye-mouth triangle validity
- relative eye-mouth position validity
- landmark cloud collapse or flip flags
- points outside expanded detector bbox
- per-region geometry errors for eyes, nose, mouth, and jaw/silhouette
- catastrophic geometry failure flags
- legacy NME, AUC, and failure rate as diagnostics

Expected output:

```text
geometry_metrics.json
geometry_metrics.csv
per_region_geometry.csv
catastrophic_geometry_failures.csv
worst_geometry_failures/
```

This phase lets us compare models and ensemble variants by extract-relevant geometry, not just landmark error.

### Phase 2: Build a product-grounded hard validation set

Ticket: [#82 Build Faceswap-hard validation set with extract-quality labels](https://github.com/argiacomi/faceswap/issues/82)

Public landmark datasets are useful, but they do not directly encode Faceswap output quality. We need a small, reviewable validation set made from real extraction failure modes.

Required slices:

- profile left
- profile right
- heavy occlusion
- partial face at image edge
- motion blur
- low-res face
- strong yaw plus open mouth
- sunglasses
- hands over face
- bad detector box but salvageable landmarks
- truncation/cropped head
- detector ambiguity where relevant

Each sample should support product labels such as:

```text
good
usable
bad_crop
bad_rotation
bad_scale
bad_mask_hull
bad_mouth_eye_placement
catastrophic
```

For video/sequence samples, add:

```text
stable
minor_jitter
bad_jitter
```

This validation set is what should eventually gate extract alignment changes. Public datasets remain calibration sources; this is the product-quality target.

### Phase 3: Prove which geometry signals predict better output

Ticket: [#80 Add geometry-signal validation reports for alignment optimization](https://github.com/argiacomi/faceswap/issues/80)

We should not assume which geometry signals matter. We need to prove it.

For each hard validation sample, generate multiple candidate outputs:

- SPIGA
- HRNet
- ORFormer
- plain average
- static weighted variants
- weighted median
- promoted setup candidates
- crop-scale variants where available

For every sample/candidate pair, save:

- aligned crop
- mask hull overlay
- detector bbox overlay
- geometry metrics
- model disagreement
- bbox scale, IoU, and confidence
- hard-slice tags
- human/product label

Then measure which signals predict bad output:

- precision and recall
- AUC
- false positive rate
- false negative rate
- selector ablations

Selector ablations should compare:

```text
lowest NME
lowest transform error
lowest crop + roll error
lowest hull error
composite geometry score
human/oracle best
```

A signal should not enter promotion or runtime routing unless it demonstrates value on held-out hard cases.

### Phase 4: Validate crop and ROI before blaming models

Ticket: [#81 Add profile-aware crop and ROI validation for landmark adapters](https://github.com/argiacomi/faceswap/issues/81)

If the crop is wrong, the model cannot recover. Profile/full-yaw faces may require different crop expansion or centering than frontal faces. Before building a smarter runtime selector, we need to know whether failures are caused by crop/ROI problems.

Diagnostics should include:

- bbox source
- bbox size and aspect
- model ROI
- crop scale
- crop center
- crop coverage of visible face/landmark hull
- landmark coverage inside crop
- predicted landmarks outside crop/frame/bbox bounds
- aligned crop misses visible face flag

This phase should also enable crop-scale experiments for hard/profile buckets. If profile failures correlate strongly with crop scale or bbox aspect, that is a higher-ROI fix than another fusion strategy.

### Phase 5: Promote only geometry-improving setups

Ticket: [#77 Add geometry-first promotion gates to ensemble setup search](https://github.com/argiacomi/faceswap/issues/77)

Once geometry metrics and signal validation exist, candidate search should stop promoting setups based primarily on NME. Promotion should require geometry improvement or at least geometry preservation.

Suggested gates:

- geometry score improves vs best single
- catastrophic geometry failure rate does not increase
- P95 transform error does not regress
- P95 crop center error does not regress
- P95 roll/scale error does not regress
- visible hull IoU improves or stays flat
- hard-slice regression rate stays under threshold
- NME does not regress badly, but acts as a tie-breaker or diagnostic

If no candidate passes, the system should write `no_promotion.json` instead of producing a misleading `best_setup.json`.

This phase changes promotion from:

```text
best landmark benchmark result
```

to:

```text
best extract-geometry candidate that passes safety gates
```

### Phase 6: Make ensemble promotion honest

Ticket: [#79 Add effective-ensemble diagnostics and single-model baselines to promotion](https://github.com/argiacomi/faceswap/issues/79)

The current promoted setup can look like an ensemble while functionally collapsing to one model. For example, a two-model weighted median where one model has majority weight for every landmark is effectively a single-model proxy with extra runtime cost.

Candidate search should report:

- single-model baseline candidates
- majority model by landmark
- percent of landmarks controlled by each model
- weighted-median collapse detection
- per-landmark entropy/effective model count for weighted averages

Promotion should state whether the selected setup is a meaningful ensemble. If it is not, we should either promote the single model directly or reject the ensemble candidate when `--require-effective-ensemble` is enabled.

### Phase 7: Add adaptive alignment geometry resolver

Ticket: [#78 Add adaptive alignment geometry resolver for extract alignment](https://github.com/argiacomi/faceswap/issues/78)

This is the likely runtime endgame, but it should not happen first. Runtime adaptation should use only signals proven by Phase 3.

The resolver should expose:

```python
resolve_alignment_geometry(predictions, detector_bbox, image_shape, config) -> AlignmentGeometryResult
```

The result should include:

```text
alignment_landmarks
visible_mask_landmarks
canonical_landmarks
geometry_confidence
geometry_flags
chosen_strategy
rejected_models
debug_metadata
```

Initial behavior:

1. Run candidate landmark models.
2. Compute validated geometry sanity checks.
3. Reject obviously bad candidates.
4. Select or fuse by semantic region where supported.
5. Prefer stable interior anchors for crop, scale, and rotation.
6. Downweight or isolate jaw/silhouette under profile/occlusion when it harms geometry.
7. Enforce detector-bbox sanity constraints.
8. Fall back to static weights or best single when confidence is low.

This should land behind a config flag so it can be compared against the current aligner, static weighted ensemble, and promoted setups.

## Recommended implementation order

```text
#76 Geometry metrics
#82 Faceswap-hard validation set
#80 Geometry-signal validation reports
#81 Crop / ROI validation
#77 Geometry-first promotion gates
#79 Effective-ensemble diagnostics
#78 Adaptive geometry resolver
```

This order avoids building runtime logic before we know which signals predict better output.

## Release gates

A change should not be promoted for extract alignment unless it passes geometry-first gates:

- catastrophic geometry failures do not increase
- hard-slice P95 crop error improves or stays flat
- visible mask hull IoU improves or stays flat
- transform error improves or stays flat
- normal-case failures do not increase
- video jitter improves or stays flat when sequence data is available
- NME does not regress beyond an accepted tolerance

The key gate is not:

```text
Did average NME go down?
```

The key gate is:

```text
Does this produce a stable, usable aligned face for Faceswap?
```

## Non-goals for the first iteration

- Do not replace the extract runtime path immediately.
- Do not train a learned selector before collecting labeled geometry-output data.
- Do not remove NME reports; demote them to diagnostics.
- Do not block geometry evaluation on full visibility labels across every dataset.
- Do not assume AFLW/profile cases are noise; they are target hard cases.

## Expected outcome

By the end of this roadmap, the extract pipeline should be able to:

1. evaluate landmark outputs by downstream geometry quality
2. identify why hard-case extraction fails
3. prove which geometry signals predict usable output
4. reject benchmark-good but geometry-bad promotions
5. promote or route aligner setups based on extract usefulness
6. reduce hard-case catastrophic alignment failures without hurting normal cases

That is the path from model comparison to product-quality optimization.
