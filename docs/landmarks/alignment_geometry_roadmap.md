# Alignment Geometry Roadmap for Faceswap Extract

## Summary

The landmark ensemble work has built a solid evaluation and promotion foundation: datasets can be built and split, landmark predictions can be cached, multiple models can be compared, static weights can be generated, ensemble variants can be searched, and promoted setups can be written for runtime use.

That foundation is useful, but the current optimization target is still too benchmark-shaped. It primarily tells us which model has better landmark point accuracy. Faceswap extract quality depends on something more specific: whether predicted landmarks produce the same stable, usable extraction geometry that the ground-truth landmarks would produce.

This roadmap shifts the product goal from:

```text
lowest average landmark error
```

to:

```text
lowest hard-case GT-geometry deviation with acceptable landmark accuracy
```

The objective is not to win generic 300W/WFLW-style NME. The objective is to produce aligned faces that match the GT-derived aligned-face geometry and remain stable in the extract pipeline, especially for profile faces, occlusions, blur, truncation, detector noise, and other hard cases.

## North star

A landmark setup is better only if it improves downstream extract geometry.

The v1 supervised target is simple:

```text
GT landmarks -> reference AlignedFace geometry
Predicted landmarks -> candidate AlignedFace geometry
Score the candidate by deviation from the GT-derived reference geometry.
```

That means better:

- crop center, scale, and rotation vs GT-derived geometry
- aligned ROI/crop coverage vs GT-derived geometry
- visible face / landmark hull behavior vs GT-derived hull behavior
- eye, nose, and mouth placement after alignment
- robustness on profile, occlusion, blur, truncation, and bad detector boxes
- lower catastrophic geometry failure rate on hard slices

NME remains useful as a diagnostic, but it should not be the primary promotion criterion.

## Why the current objective is not enough

The current validation flow compares HRNet, SPIGA, ORFormer, and ensemble variants using NME, failure rate, AUC, and per-landmark/per-region error. That is a good model-comparison layer.

It does not directly answer the product question:

```text
Will this landmark output produce the same usable aligned face that the GT landmarks would produce?
```

A model can have slightly worse point error but produce a better alignment transform. Another model can have decent NME but create a bad crop, bad roll, invalid hull, or unusable mouth/eye placement. For Faceswap, the second model is worse even if the point benchmark says it is better.

The repo's extract path makes this dependency direct. `AlignedFace` uses landmarks to compute the alignment matrix, transformed landmarks, pose, ROI, crop, and mask-related geometry. Therefore, transform quality and crop/mask stability are first-order output metrics, not secondary diagnostics.

## Product principle

Use dataset landmarks as ground truth. Do not use mean full-68 NME as the only success definition.

For profile and occluded faces, some landmarks may be projected, canonical, compressed, or not visually visible. Those points are not automatically bad. If the dataset ground truth projects far-side features into compressed or overlapping locations, a good prediction should match that convention.

The failure condition is narrower:

```text
Predicted landmarks diverge from GT enough to distort crop, scale, roll, hull, warp, or placement.
```

This means the first milestone is not a broad manual review dataset. It is a GT-derived geometry benchmark from landmarked hard-case datasets, starting with AFLW/AFLW2000.

Over time, we may still move toward separate downstream geometry products:

```text
alignment_landmarks: stable interior anchors for crop, scale, and rotation
visible_mask_landmarks: visible face hull for mask construction
canonical_landmarks: full 68-point compatibility output
geometry_flags: profile, occluded, truncated, low confidence, etc.
geometry_confidence: confidence in the chosen geometry
```

That split should come after we have the metrics and evidence to justify it.

## Roadmap

### Phase 1: Measure GT-derived extract geometry directly

Ticket: [#76 Add GT-derived alignment geometry evaluation metrics](https://github.com/argiacomi/faceswap/issues/76)

This is the foundation. Before changing runtime behavior, we need a geometry evaluation layer that measures the objects Faceswap actually uses.

For each sample:

1. Build reference geometry from ground-truth landmarks.
2. Build candidate geometry from predicted landmarks.
3. Compare candidate geometry against the reference geometry.
4. Report legacy point metrics as diagnostics.

The evaluator should compute prediction-derived geometry vs GT-derived geometry:

- affine/similarity transform delta
- crop center error
- scale error
- roll/rotation error
- aligned ROI/crop coverage difference
- predicted aligned landmarks vs GT aligned landmarks
- predicted hull or mask hull vs GT-derived hull
- landmark/mask hull validity
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

### Phase 2: Build a landmarked hard-case validation split

Ticket: [#82 Build landmarked hard-case alignment validation set](https://github.com/argiacomi/faceswap/issues/82)

The hard-case validation set should be grounded in existing landmark annotations, not broad manual product labels.

Start with AFLW/AFLW2000 hard/profile cases:

- profile left
- profile right
- large yaw
- self-occluded face contours
- projected or non-visible landmark stress cases
- truncation or difficult pose when available

The v1 target is:

```text
A good alignment output matches the hard-case landmark annotations well enough to preserve visible face geometry, and any projected / non-visible landmarks do not distort crop, mask hull, warp, or placement.
```

For each selected sample/candidate output, report both:

```text
landmark correctness against GT
extraction geometry deviation from GT-derived geometry
```

Manual review labels are optional later for ambiguous cases or sanity checks. They are not required for v1.

### Phase 3: Prove which geometry signals identify better GT alignment behavior

Ticket: [#80 Add GT-geometry signal validation reports for alignment optimization](https://github.com/argiacomi/faceswap/issues/80)

We should not assume which geometry signals matter. We need to prove which signals identify candidates that are closest to the GT-derived reference geometry.

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
- GT-derived aligned crop
- overlay path
- geometry metrics
- model disagreement
- bbox scale, IoU, and confidence
- hard-slice tags
- optional review label fields when available

Then measure which signals predict high GT-geometry deviation:

- precision and recall
- AUC
- false positive rate
- false negative rate
- selector ablations

Selector ablations should compare:

```text
lowest NME
lowest transform error vs GT
lowest crop + roll error vs GT
lowest hull error vs GT
composite geometry score
oracle lowest GT-geometry deviation
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

### Phase 5: Promote only GT-geometry-improving setups

Ticket: [#77 Add GT-geometry-first promotion gates to ensemble setup search](https://github.com/argiacomi/faceswap/issues/77)

Once geometry metrics and signal validation exist, candidate search should stop promoting setups based primarily on NME. Promotion should require improvement or preservation against GT-derived alignment geometry.

Suggested gates:

- geometry score vs GT improves vs best single
- catastrophic geometry failure rate does not increase
- P95 transform error vs GT does not regress
- P95 crop center error vs GT does not regress
- P95 roll/scale error vs GT does not regress
- hull IoU vs GT-derived hull improves or stays flat
- hard-slice regression rate stays under threshold
- NME does not regress badly, but acts as a tie-breaker or diagnostic

If no candidate passes, the system should write `no_promotion.json` instead of producing a misleading `best_setup.json`.

This phase changes promotion from:

```text
best landmark benchmark result
```

to:

```text
best extract-geometry candidate that matches GT-derived geometry and passes safety gates
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

### Phase 7: Add geometry-risk alignment resolver

Ticket: [#78 Add geometry-risk alignment resolver for extract alignment](https://github.com/argiacomi/faceswap/issues/78)

This is the likely runtime endgame, but it should not happen first. Runtime adaptation should use only signals proven by Phase 3.

The resolver should not depend on a binary frontal/profile classifier. The V1 production question is:

```text
is the general alignment path low-risk for this face, or do the candidate outputs show geometry risk that needs hard-case handling or fallback?
```

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
risk_score
risk_route
chosen_strategy
rejected_models
debug_metadata
```

V1 routing should be based on validated geometry risk:

```text
low risk -> general setup / best single
high geometry risk -> hard-case setup
invalid geometry -> reject candidate or safe fallback
high disagreement -> region-wise resolver or safer fallback
```

Candidate risk signals should come from #80, not guesses. Candidate signals include:

- inter-model landmark disagreement
- candidate transform disagreement
- crop center / scale / roll disagreement
- points outside expanded detector bbox
- eye-mouth geometry validity
- relative eye-mouth position
- hull collapse / flip / extreme skew
- detector bbox aspect ratio
- detector confidence / bbox size where available
- model confidence where available
- optional yaw or pose proxy as one feature, not the primary route

Initial behavior:

1. Run candidate landmark models.
2. Compute validated geometry-risk signals.
3. Reject obviously invalid geometry candidates.
4. Route low-risk faces to the general setup or best single.
5. Route high-risk faces to the hard-case setup or hard-case fusion strategy.
6. Select or fuse by semantic region only where supported by validation.
7. Prefer stable interior anchors for crop, scale, and rotation.
8. Downweight or isolate jaw/silhouette under high-risk geometry when it harms alignment.
9. Enforce detector-bbox sanity constraints.
10. Fall back to static weights or best single when confidence is low.

This should land behind a config flag so it can be compared against the current aligner, static weighted ensemble, and promoted setups.

## Recommended implementation order

```text
#76 GT-derived geometry metrics
#82 Landmarked hard-case validation split
#80 GT-geometry signal validation reports
#81 Crop / ROI validation
#77 GT-geometry-first promotion gates
#79 Effective-ensemble diagnostics
#78 Geometry-risk alignment resolver
```

This order avoids building runtime logic before we know which signals identify better GT-derived extract geometry.

## Release gates

A change should not be promoted for extract alignment unless it passes GT-geometry-first gates:

- catastrophic geometry failures do not increase
- hard-slice P95 crop error vs GT improves or stays flat
- hull IoU vs GT-derived hull improves or stays flat
- transform error vs GT improves or stays flat
- normal-case failures do not increase
- video jitter improves or stays flat when sequence data is available
- NME does not regress beyond an accepted tolerance

The key gate is not:

```text
Did average NME go down?
```

The key gate is:

```text
Does this produce the same stable, usable aligned face geometry that the GT landmarks would produce?
```

## Non-goals for the first iteration

- Do not replace the extract runtime path immediately.
- Do not train a learned selector before GT-geometry metrics are validated.
- Do not require manual product labels for v1.
- Do not require real Faceswap failure collection for v1.
- Do not remove NME reports; demote them to diagnostics.
- Do not assume AFLW/profile cases are noise; they are target hard cases.
- Do not build V1 runtime routing around a brittle binary frontal/profile classifier.

## Expected outcome

By the end of this roadmap, the extract pipeline should be able to:

1. evaluate landmark outputs by deviation from GT-derived extraction geometry
2. identify why hard-case extraction geometry fails
3. prove which geometry signals identify candidates closest to GT-derived aligned geometry
4. reject benchmark-good but geometry-bad promotions
5. promote or route aligner setups based on validated geometry risk
6. reduce hard-case catastrophic alignment failures without hurting normal cases

That is the path from model comparison to product-quality optimization.
