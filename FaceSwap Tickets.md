FaceSwap Tickets

## ~~1. Add alignments schema support for AutoMask metadata~~

**~~Goal:~~** ~~Add first-class storage for AutoMask metadata without breaking alignments serialization.~~

**~~Scope~~**

- ~~Extend alignment dataclasses to support `auto_mask` metadata per face.~~
- ~~Include:~~
    - ~~`schema_version`~~
    - ~~`status`~~
    - ~~`score`~~
    - ~~`route`~~
    - ~~`flags`~~
    - ~~`features`~~
    - ~~`parser_used`~~
    - ~~`aux_parsers_used`~~
    - ~~`obstacle_detectors_used`~~
    - ~~`refinement_used`~~
    - ~~`semantic_layers_available`~~
    - ~~`confidence_heatmap_available`~~

- ~~Add migration/backward compatibility for alignments without metadata.~~
- ~~Ensure `DetectedFace.to_alignment()` / `from_alignment()` round-trips metadata.~~
- ~~Decide whether PNG headers include full metadata or only lightweight mask metadata.~~

**~~Acceptance~~**

- ~~Existing alignments load unchanged.~~
- ~~New alignments round-trip AutoMask metadata.~~
- ~~Manual tool can read metadata from loaded `DetectedFace` objects.~~
- ~~Mask tool and Extract do not drop metadata during save.~~

---

## ~~2. Add FaceQA metadata schema~~

**~~Goal:~~** ~~Add a separate structured schema for training faceset QA.~~

**~~Scope~~**

- ~~Prefer sidecar `faceset_qa.json` unless alignments integration proves cleaner.~~
- ~~Store per face:~~
    - ~~identity score~~
    - ~~identity cluster~~
    - ~~duplicate cluster~~
    - ~~recommended keep~~
    - ~~yaw/pitch/roll~~
    - ~~blur/sharpness~~
    - ~~exposure~~
    - ~~resolution~~
    - ~~occlusion~~
    - ~~expression bucket~~
    - ~~mask QA reference~~

- ~~Support frame filename + face index addressing.~~
- ~~Add schema versioning.~~

**Acceptance**

- ~~QA metadata can be loaded independently of masks.~~
- ~~Existing Extract/Manual/Train flows ignore it safely if absent.~~
- ~~Manual/Sort tooling can consume it later.~~

---

## ~~3. Create shared AutoMask backfill runner using Mask tool conventions~~

**~~Goal:~~** ~~Avoid a parallel loader/writeback implementation.~~

**~~Scope~~**

- ~~Review `tools.mask.mask_generate.MaskGenerator`.~~
- ~~Extract or share common behavior for:~~
    - ~~frames/video/faces input~~
    - ~~alignments loading~~
    - ~~PNG header updates~~
    - ~~mask writeback~~
    - ~~output previews~~
    - ~~batch mode~~

- ~~Implement `AutoMaskGenerator` or `BaseMaskBackfillRunner`.~~
- ~~Keep AutoMask orchestration separate from single-mask `MaskGenerator` where composition/QA requires it.~~

**~~Acceptance~~**

- ~~Decision documented in code comments/docs.~~
- ~~AutoMask frames/video/faces modes behave like Mask tool.~~
- ~~No duplicate alignments discovery logic unless justified.~~
- ~~No duplicate PNG header logic unless justified.~~~

---

## ~~4. Implement canonical `AutoMaskService`~~

**~~Goal:~~** ~~Central service used by Tool mode, Extract integration, Manual editor preload, cascade/refinement, and tests.~~

**~~Scope~~**

- ~~Input:~~
    - ~~frame image~~
    - ~~face object~~
    - ~~landmarks~~
    - ~~alignment matrix~~
    - ~~optional existing masks~~
    - ~~optional temporal context~~

- ~~Output:~~
    - ~~final binary mask~~
    - ~~semantic mask layer references~~
    - ~~per-class probability summaries~~
    - ~~obstacle masks~~
    - ~~confidence data~~
    - ~~QA features~~
    - ~~score~~
    - ~~route~~
    - ~~metadata payload~~

- ~~Separate generation from persistence.~~
- ~~No CLI/GUI-specific logic inside service.~~

**Acceptance**

- ~~Service can be called from a unit test with synthetic frame/face input.~~
- ~~Service returns mask + metadata without writing files.~~
- ~~Tool/Extract wrappers handle persistence.~~

---

## ~~5. Add `tools/automask/cli.py` and `tools/automask/automask.py`~~

**~~Goal:~~** ~~Register AutoMask as a normal FaceSwap Tool.~~

**~~Scope~~**

- ~~Add `AutomaskArgs` in `tools/automask/cli.py`.~~
- ~~Add `Automask` process class in `tools/automask/automask.py`.~~
- ~~Let `tools.py` and GUI auto-discover it.~~
- ~~CLI options should mirror Mask tool where possible:~~
    - ~~`--alignments`~~
    - ~~`--input`~~
    - ~~`--input-type faces|frames`~~
    - ~~`--batch-mode`~~
    - ~~`--output-folder`~~

- ~~AutoMask-specific options:~~
    - ~~primary parser~~
    - ~~EasyPortrait enable~~
    - ~~hand detector enable~~
    - ~~QA preset/config~~
    - ~~output debug overlays~~
    - ~~update all/missing/metadata-only if useful~~

**~~Acceptance~~**

- ~~`python tools.py automask -h` works.~~
- ~~AutoMask appears under GUI Tools automatically.~~
- ~~No custom top-level GUI tab is required.~~
- ~~GUI-generated command works through existing `CommandNotebook`.~~

---

## ~~6. Align AutoMask settings with existing config system~~

**~~Goal:~~** ~~Put persistent plugin-style options in config, not a custom settings store.~~

**~~Scope~~**

- ~~Add config defaults through existing `lib.config` pattern.~~
- ~~Main CLI/GUI options:~~
    - ~~input/output~~
    - ~~enable/disable high-level features~~
    - ~~primary parser choice~~

- ~~Config/settings:~~
    - ~~morphology radius~~
    - ~~confidence thresholds~~
    - ~~parser-specific thresholds~~
    - ~~semantic class inclusion/exclusion~~
    - ~~storage size if not purely persistence-level~~
    - ~~SAM/refiner defaults~~

- ~~Avoid duplicate config state inside orchestrator.~~

**~~Acceptance~~**

- ~~Settings appear through existing config infrastructure.~~
- ~~GUI controls are generated from CLI/config patterns.~~
- ~~Defaults are safe and documented.~~
- ~~No AutoMask-only config loader.~~

~~Use mamba env faceswap for any necessary execution~~

---

## ~~7. Build AutoMask MVP composition and QA~~

**~~Goal:~~** ~~Implement primary parser + optional aux parser + optional obstacle detector + QA scoring.~~

**~~Scope~~**

- ~~Primary parser:~~
    - ~~BiSeNet-FP~~
    - ~~SegNeXt-FP~~

- ~~Auxiliary:~~
    - ~~EasyPortrait~~

- ~~Obstacle:~~
    - ~~hand_yolov9c~~

- ~~Combination:~~
    - ~~primary face components~~
    - ~~EasyPortrait hair boundary adjustment~~
    - ~~subtract hand/obstacle mask~~
    - ~~morphology~~
    - ~~subtract hand/obstacle mask again~~

- ~~QA features:~~
    - ~~parser argmax margin~~
    - ~~landmark/mask IoU~~
    - ~~mask area percentile~~
    - ~~hand central-face overlap~~

- ~~Flags:~~
    - ~~`low_parser_confidence`~~
    - ~~`landmark_mask_disagreement`~~
    - ~~`mask_area_outlier`~~
    - ~~`hand_central_overlap`~~
    - ~~`overexpanded_mask`~~
    - ~~`underfilled_mask`~~
    - ~~`obstacle_reintroduced_after_morphology`~~

**~~Acceptance~~**

- ~~200-frame sample populates masks and AutoMask metadata.~~
- ~~Obvious occlusions score lower.~~
- ~~Clean front-facing faces score high.~~
- ~~Raw QA features are persisted, not just scalar score.~~

~~Use mamba env faceswap for any necessary execution~~

---

## ~~8. Implement dataset-relative AutoMask finalization~~

**~~Goal:~~** ~~Preserve percentile-based QA while supporting Tool and Extract modes.~~

**~~Scope~~**

- ~~During per-frame processing, store temporary QA inputs.~~
- ~~After all frames/faces:~~
    - ~~compute mask-area percentiles~~
    - ~~finalize score~~
    - ~~finalize flags~~
    - ~~mark metadata `complete`~~

- ~~Add recovery path for `pending_finalization`.~~

**~~Acceptance~~**

- ~~Standalone AutoMask finalizes metadata.~~
- ~~Extract AutoMask finalizes after Extract completes.~~
- ~~Interrupted runs can be finalized without regenerating masks.~~
- ~~Percentile scoring is preserved.~~

~~Use mamba env faceswap for any necessary execution~~

---

## ~~9. Implement extracted-faces input mode~~

**~~Goal:~~** ~~Match Mask tool behavior for Faceswap extracted PNGs.~~

**~~Scope~~**

- ~~`--input-type faces`.~~
- ~~Read source filename and face index from PNG metadata.~~
- ~~Update corresponding alignments entry when alignments file is provided.~~
- ~~Update PNG header if FaceSwap convention expects it.~~
- ~~Warn clearly if only PNG headers can be updated.~~

**~~Acceptance~~**

- ~~Extracted face PNG folder works.~~
- ~~Correct frame/face index is updated.~~
- ~~Behavior matches Mask tool where feasible.~~
- ~~Missing alignments file behavior is documented and tested.~~

~~Use mamba env faceswap for any necessary execution~~

---

## ~~10. Add AutoMask smoke tests~~

**~~Goal:~~** ~~Cover the surfaces most likely to regress.~~

**~~Scope~~**

- ~~Frames input.~~
- ~~Video input.~~
- ~~Faces input.~~
- ~~GUI-generated command construction.~~
- ~~EasyPortrait + hand detector.~~
- ~~Alignments writeback.~~
- ~~PNG header writeback.~~
- ~~Invalid input mode.~~
- ~~Morphology + obstacle exclusion.~~
- ~~Pending finalization recovery.~~

**~~Acceptance~~**

- ~~All smoke cases pass in mamba env `faceswap`.~~
- ~~Failure messages are clear.~~
- ~~Existing Mask tool tests still pass.~~

~~Use mamba env faceswap for any necessary execution~~

---

## ~~11. Extend Manual tool filter/navigation for AutoMask QA~~

**~~Goal:~~** ~~Reuse existing Manual filter UI.~~

**~~Scope~~**

- ~~Extend filter modes beyond:~~
    - ~~All Frames~~
    - ~~Has Face(s)~~
    - ~~No Faces~~
    - ~~Multiple Faces~~
    - ~~Misaligned Faces~~

- ~~Add:~~
    - ~~AutoMask: Low QA~~
    - ~~AutoMask: Flagged~~
    - ~~AutoMask: Flag Reason~~
    - ~~AutoMask: Needs Review~~

- ~~Add optional flag dropdown only when AutoMask filter is active.~~
- ~~Sort filtered frame list by QA score ascending.~~
- ~~Preserve existing keyboard cycling behavior where possible.~~

**~~Acceptance~~**

- ~~Manual tool can filter to low-score/flagged frames.~~
- ~~Default behavior unchanged for alignments without AutoMask metadata.~~
- ~~Existing `F` filter cycling still works or has documented updated order.~~
- ~~No new Manual window.~~

~~Use mamba env faceswap for any necessary execution~~

---

## ~~12. Extend Manual mask editor display for AutoMask metadata~~

**~~Goal:~~** ~~Make AutoMask masks and QA actionable in the existing Mask editor.~~

**~~Scope~~**

- ~~AutoMask appears as a normal mask type.~~
- ~~Display score and active flag chips near frame/face context.~~
- ~~Show raw QA feature values in a compact panel.~~
- ~~Add overlay toggles:~~
    - ~~final mask~~
    - ~~parser confidence~~
    - ~~semantic layers~~
    - ~~obstacle/hand~~
    - ~~landmark hull~~
    - ~~disagreement heatmap~~

- ~~Reuse existing `Mask` editor and `ControlPanelOption`.~~

**~~Acceptance~~**

- ~~Existing mask painting still works.~~
- ~~AutoMask mask preloads as editable mask.~~
- ~~QA overlays do not require separate GUI.~~
- ~~No regression for masks without AutoMask metadata.~~

~~Use mamba env faceswap for any necessary execution~~

---

## ~~13. Add AutoMask debug output mode~~

**~~Goal:~~** ~~Support review outside the Manual editor.~~

**~~Scope~~**

- ~~Optional output folder.~~
- ~~Contact sheet of worst frames.~~
- ~~Per-frame overlay images.~~
- ~~CSV/JSON report:~~
    - ~~frame~~
    - ~~face index~~
    - ~~score~~
    - ~~route~~
    - ~~flags~~
    - ~~raw features~~
    - ~~parsers/refiners used~~

**~~Acceptance~~**

- ~~Debug output generated from Tool mode.~~
- ~~Output does not modify alignments unless processing mode requests it.~~
- ~~Contact sheet ranks worst frames first.~~

~~Use mamba env faceswap for any necessary execution~~

---

## ~~14. Integrate AutoMask into Extract pipeline~~

**~~Goal:~~** ~~Enable first-pass extraction with AutoMask without a second frame load.~~

**~~Scope~~**

- ~~Add Extract option:~~
    - ~~`--auto-mask`~~

- ~~Use existing Extract runner/output flow.~~
- ~~Reuse `AutoMaskService`.~~
- ~~Avoid reloading video/frames.~~
- ~~During Extract:~~
    - ~~generate mask~~
    - ~~store temporary QA inputs~~
    - ~~write mask under expected key~~

- ~~After Extract:~~
    - ~~finalize dataset-relative QA~~

- ~~UI:~~
    - ~~main Extract tab only gets enable AutoMask~~
    - ~~detailed options live in settings/config.~~

**~~Acceptance~~**

- ~~User can enable AutoMask during Extract.~~
- ~~AutoMask masks and metadata are written.~~
- ~~Dataset-relative percentile scoring is preserved.~~
- ~~Existing standalone AutoMask still works.~~
- ~~Extract batch mode works or explicitly warns if unsupported.~~

~~Use mamba env faceswap for any necessary execution~~

---

## ~~15. Add Extract finalization recovery command~~

**~~Goal:** Cleanly recover interrupted Extract AutoMask runs.~~

**~~Scope~~**

- ~~Detect `pending_finalization`.~~
- ~~Add Tool mode operation:~~
    - ~~finalize pending AutoMask metadata.~~

- ~~Recompute dataset-relative percentiles.~~
- ~~Mark failed entries with reason.~~

**~~Acceptance~~**

- ~~Interrupted run can be finalized.~~
- ~~No masks are regenerated.~~
- ~~Completed metadata is not overwritten unless requested.~~

~~Use mamba env faceswap for any necessary execution~~

---

## ~~16. Add parser cascade and parser disagreement QA~~

**~~Goal:** Route hard frames to secondary parser/ensemble.~~

**~~Scope~~**

- ~~Parser A / Parser B config.~~
- ~~Default:~~
    - ~~BiSeNet-FP first~~
    - ~~SegNeXt-FP if flagged~~

- ~~Ensemble strategies:~~
    - ~~confidence-weighted union~~
    - ~~confidence-weighted intersection~~
    - ~~class-specific voting~~

- ~~Persist:~~
    - ~~parsers run~~
    - ~~disagreement score~~
    - ~~ensemble strategy~~

- ~~Add flag:~~
    - ~~`parser_disagreement`~~

**~~Acceptance~~**

- ~~Easy frames use one parser.~~
- ~~Flagged frames use cascade.~~
- ~~Disagreement contributes to QA.~~
- ~~Flat pipeline works when cascade disabled.~~

~~Use mamba env faceswap for any necessary execution~~

---

## ~~17. Add MediaPipe Hands fallback plugin~~

**~~Goal:** CPU-only obstacle detector fallback.~~

**~~Scope~~**

- ~~New obstacle plugin with same contract as YOLO hand plugin.~~
- ~~Convert keypoints to convex hull mask.~~
- ~~Add dilation setting.~~
- ~~Selectable via config.~~
- ~~Not default if YOLO is available.~~

**~~Acceptance~~**

- ~~Works without ultralytics.~~
- ~~Produces usable coarse exclusion mask.~~
- ~~Metadata records detector used.~~

~~Use mamba env faceswap for any necessary execution~~

---

## ~~18. Add FaceXFormer visibility QA analysis plugin~~

**~~Goal:** Add direct landmark visibility/occlusion signal.~~

**~~Scope~~**

- ~~Analysis plugin, not mask plugin.~~
- ~~Outputs:~~
    - ~~`visibility[]`~~
    - ~~`occluded_regions[]`~~
    - ~~`central_region_occlusion`~~

- ~~Persist under AutoMask metadata.~~
- ~~Add features:~~
    - ~~`visibility_occlusion_ratio`~~
    - ~~`central_region_occlusion`~~

- ~~Add flag:~~
    - ~~`visibility_high_occlusion`~~

**~~Acceptance~~**

- ~~Visibility metadata persists.~~
- ~~Hand/hair occlusions lower QA score.~~
- ~~Disabled behavior has no regression.~~

~~Use mamba env faceswap for any necessary execution~~

---

## ~~19. Add SAM2 / MobileSAM / EfficientSAM refinement provider~~

**~~Goal:** Add promptable refinement without hardcoding one SAM backend.~~

**~~Scope~~**

- ~~Provider interface:~~
    - ~~SAM2~~
    - ~~MobileSAM~~
    - ~~EfficientSAM~~

- ~~Prompt generation:~~
    - ~~landmark hull positives~~
    - ~~high-confidence face positives~~
    - ~~far-background negatives~~
    - ~~obstacle negatives~~

- ~~Reapply obstacle exclusion after refinement.~~
- ~~Persist backend, prompt strategy, and `refinement_used`.~~
- ~~Refinement provider should accept:~~
    - ~~source frame~~
    - ~~aligned face crop~~
    - ~~coarse mask~~
    - ~~semantic layer masks~~
    - ~~obstacle masks~~
    - ~~landmarks~~
    - ~~AutoMask QA metadata~~
- ~~Add mask sanity checks after refinement:~~
    - ~~max expansion ratio~~
    - ~~minimum overlap with coarse mask~~
    - ~~no large disconnected islands~~
    - ~~obstacle regions remain excluded~~
- ~~Add config:~~

```YAML
promptable_refiner:
  backend: sam2 | mobilesam | efficientsam
  enabled: true | false
  run_policy: flagged_only | all | manual_selected
```

**~~Acceptance~~**

- ~~Backend selectable.~~
- ~~Refined mask respects plausible face bounds.~~
- ~~Non-face areas are not pulled into mask.~~
- ~~Metadata records backend.~~
- ~~Refiner output is rejected or flagged if it diverges too far from parser mask.~~
- ~~Obstacle exclusion is preserved after refinement.~~
- ~~Disabled behavior has no regression.~~

**~~Main point**: SAM-style refinement should be bounded by parser/mask QA, not blindly trusted.~~

~~Use mamba env faceswap for any necessary execution~~

---

## ~~20. Add SAM2 video-memory temporal refinement~~

**~~Goal:** Use SAM2 temporal propagation for video mask consistency.~~

**~~Scope~~**

- ~~Initialize from high-confidence masks.~~
- ~~Propagate over segments.~~
- ~~Compare SAM2 output to parser output.~~
- ~~Flag divergence.~~
- ~~Persist temporal refinement metadata.~~
- ~~Segment selection from temporal mask QA.~~
- ~~Start propagation from high-confidence anchor frames.~~
- ~~Use low-confidence frames as correction targets.~~
- ~~Persist:~~
    - ~~sam2_video_segment_id~~
    - ~~anchor_frames~~
    - ~~propagation_confidence~~
    - ~~sam2_parser_divergence~~

**~~Acceptance~~**

- ~~Video segment refinement works.~~
- ~~Flicker decreases on test clips.~~
- ~~Divergence is flagged.~~
- ~~High-confidence anchor frame selection is deterministic.~~
- ~~Bad propagation is detected and not silently accepted.~~
- ~~Segment-level metadata is written, not only per-frame metadata.~~

~~Use mamba env faceswap for any necessary execution~~

---

## ~~21. Add MODNet / RVM hair alpha refinement~~

**~~Goal:** Improve hair-edge cases that SAM handles poorly.~~

**~~Scope**~~

- ~~Trigger on fragmented/high-perimeter hair near face boundary.~~
- ~~Preserve face skin.~~
- ~~Avoid portrait-wide leakage.~~
- ~~Persist `hair_matting` refinement use.~~
- ~~Use only for hair-boundary refinement.~~
- ~~Trigger using:~~
    - ~~EasyPortrait hair class~~
    - ~~fragmented hair boundary~~
    - ~~high-perimeter hair region~~
    - ~~hair crossing face boundary~~
- ~~Clamp/refit output to plausible face/hair boundary area.~~
- ~~Do not let matte pull in shoulders, background, or full head silhouette unless explicitly configured.~~

**~~Acceptance**~~

- ~~Bangs/flyaways improve on test frames.~~
- ~~Leakage is detected/flagged.~~
- ~~Disabled behavior has no regression.~~
- ~~Full portrait/background leakage is flagged.~~
- ~~Face skin region is preserved.~~
- ~~Hair matte does not override confident eye/mouth/skin semantic regions.~~

~~Use mamba env faceswap for any necessary execution~~

---

## ~~22. Add temporal mask stability scoring~~

**~~Goal:** Detect mask flicker and unstable frame-to-frame behavior.~~

**~~Scope~~**

- ~~Score:~~
    - ~~mask area delta~~
    - ~~centroid delta normalized by face motion~~
    - ~~boundary jitter~~
    - ~~semantic instability~~
    - ~~sudden class-area changes~~
    - ~~obstacle-mask temporal jumps~~
- ~~Add flags:~~
    - ~~temporal_flicker~~
    - ~~temporal_area_jump~~
    - ~~temporal_boundary_jitter~~
    - ~~temporal_semantic_instability~~
- ~~Persist temporal scores.~~

**~~Acceptance~~**

- ~~Flickering masks are flagged.~~
- ~~Stable clips score high.~~
- ~~Metadata records temporal features.~~
- ~~No mask pixels are changed by this ticket.~~

~~Use mamba env faceswap for any necessary execution~~

---

## ~~23. Add temporal mask smoothing~~

**~~Goal:** Reduce flicker after temporal instability is detected.~~

**~~Scope~~**

- ~~Optical-flow-guided smoothing.~~
- ~~Neighbor-frame consistency.~~
- ~~Segment-level stabilization.~~
- ~~Preserve obstacle exclusions.~~
- ~~Preserve confident semantic regions.~~
- ~~Avoid lag during fast motion.~~
- ~~Re-score masks before/after smoothing.~~

**~~Acceptance~~**

- ~~Flicker decreases on test clips.~~
- ~~Fast motion does not create delayed/lagging masks.~~
- ~~Obstacle exclusions survive smoothing.~~
- ~~Before/after QA deltas are reported.~~

~~Use mamba env faceswap for any necessary execution~~

---

# Extract Inference Performanc Tickets

## ~~24. Batch AutoMask inference~~

**~~Goal:** Replace per-face AutoMask execution with batched inference.~~

**~~Scope~~**

- ~~Add batch API:~~

```python
AutoMaskService.analyze_faces(frame, faces)
```

- ~~Keep existing single-face API for manual/debug paths.~~
- ~~Batch:~~
    - ~~aligned crop generation~~
    - ~~primary parser inference~~
    - ~~secondary parser inference~~
    - ~~EasyPortrait auxiliary parser inference~~
    - ~~obstacle detector inference where feasible~~
    - ~~optional analysis/refinement plugins where feasible~~
- ~~Compose and score results per face after batched model inference.~~
- ~~Ensure batched and single-face results match within tolerance.~~
- ~~Add benchmark comparison:~~
    - ~~single-face AutoMask~~
    - ~~batched AutoMask~~
    - ~~multi-face frames~~
    - ~~long video sample~~

**~~Acceptance~~**

- ~~Batched path produces masks and metadata equivalent to single-face path within tolerance.~~
- ~~Throughput improves when AutoMask is enabled.~~
- ~~Existing single-face call path remains available.~~
- ~~Multi-face frames process correctly.~~
- ~~Debug output still maps results to the correct frame/face index.~~

~~Use mamba env faceswap for any necessary execution~~

---

## ~~25. Add controlled compile modes~~

**~~Goal:** Replace one aggressive global `--compile` path with backend-aware compile modes and safe fallback.~~

**~~Scope~~**

- ~~Keep existing compile support, but change from boolean-style behavior to controlled modes:~~

```bash
--compile off|default|reduce-overhead|max-autotune|max-autotune-no-cudagraphs
```

- ~~Preserve backward compatibility for existing boolean-style `--compile` usage where feasible.~~
- ~~Add backend gating:~~
    - ~~CUDA: allow all supported modes.~~
    - ~~ROCm: allow only validated modes.~~
    - ~~MPS: allow default mode only.~~
    - ~~CPU: default off.~~
- ~~Add fullgraph fallback:~~
    - ~~try `fullgraph=True`~~
    - ~~on failure retry `fullgraph=False`~~
    - ~~on failure fall back to eager with warning~~
- ~~Add logging:~~
    - ~~backend~~
    - ~~compile mode~~
    - ~~fullgraph setting~~
    - ~~dynamic/static setting~~
    - ~~fallback status~~
    - ~~compile time~~
- ~~Avoid CUDA-graph modes on non-CUDA backends.~~

**~~Acceptance~~**

- ~~Existing `--compile` behavior maps to a documented mode.~~
- ~~Compile failure falls back cleanly.~~
- ~~CUDA graph modes do not run on MPS/CPU.~~
- ~~Logs show selected compile mode and fallback result.~~
- ~~Small extraction runs can use `off` without compile startup overhead.~~
- ~~Long CUDA extraction runs can use `max-autotune`.~~

~~Use mamba env faceswap for any necessary execution~~

---

## ~~26. SCRFD Torch/GPU postprocess and NMS~~

**~~Goal:** Keep SCRFD decode, score thresholding, and NMS on GPU to reduce Python/NumPy overhead and GPU→CPU transfer volume.~~

**~~Scope~~**

- ~~Keep SCRFD model outputs as Torch tensors.~~
- ~~Decode boxes/landmarks on GPU.~~
- ~~Apply score thresholding on GPU.~~
- ~~Use `torchvision.ops.nms` / `batched_nms` or custom Torch NMS.~~
- ~~Copy only compact final detections to CPU.~~
- ~~Preserve existing NumPy/Python postprocess as fallback.~~
- ~~Add tolerance tests against current implementation.~~

**~~Acceptance~~**

- ~~Detection results match current NumPy implementation within tolerance.~~
- ~~Throughput improves on detector-heavy samples.~~
- ~~CPU transfer volume decreases.~~
- ~~Fallback path works when TorchVision ops are unavailable.~~
- ~~Existing SCRFD behavior remains available via config/debug fallback.~~

~~Use mamba env faceswap for any necessary execution~~

---

## ~~27. Add Torch GPU preprocessing / crop / warp primitives~~

**~~Goal:** Reduce NumPy/OpenCV ↔ Torch roundtrips across Extract and AutoMask.~~

**~~Scope~~**

- ~~Add shared Torch utilities for:~~
    - ~~resize~~
    - ~~normalization~~
    - ~~affine warp~~
    - ~~face crop~~
    - ~~batched crop/align via `grid_sample`~~
- ~~Make utilities backend-aware:~~
    - ~~CUDA~~
    - ~~ROCm~~
    - ~~MPS~~
    - ~~CPU fallback~~
- ~~Use where feasible in:~~
    - ~~detector preprocessing~~
    - ~~aligner preprocessing~~
    - ~~mask/AutoMask crop generation~~
    - ~~identity encoder preprocessing~~
- ~~Keep OpenCV/NumPy fallback path.~~
- ~~Add output tolerance tests against existing OpenCV behavior.~~

**~~Acceptance~~**

- ~~GPU path outputs match existing OpenCV/NumPy path within tolerance.~~
- ~~CPU fallback remains available.~~
- ~~AutoMask batching can use shared crop/warp path.~~
- ~~Benchmarks report reduced transfer/preprocess cost.~~
- ~~No regression for plugins that still require NumPy input.~~

~~Use mamba env faceswap for any necessary execution~~

---

## ~~28. Add selective mixed precision inference~~

**~~Goal:** Add FP16/BF16 inference selectively by plugin type without destabilizing detection, alignment, identity, or QA.~~

**~~Scope~~**

- ~~Add precision config per plugin type:~~

```ini
detect_precision = fp32 | fp16 | bf16 | auto
align_precision = fp32 | fp16 | bf16 | auto
mask_precision = fp32 | fp16 | bf16 | auto
identity_precision = fp32 | fp16 | bf16 | auto
```

- ~~Validate drift by plugin type:~~
    - ~~detector box/landmark drift~~
    - ~~aligner landmark drift~~
    - ~~mask IoU drift~~
    - ~~identity cosine-similarity drift~~
- ~~Keep QA scoring math in FP32 where possible.~~
- ~~Do not blanket-enable half precision.~~
- ~~Default conservative behavior:~~
    - ~~maskers: auto/FP16 where validated~~
    - ~~detector: auto only after validation~~
    - ~~aligner: FP32 unless validated~~
    - ~~identity: FP32 unless cosine drift validation passes~~

**~~Acceptance~~**

- ~~Mixed precision can be enabled per plugin type.~~
- ~~Precision drift report is generated.~~
- ~~Identity defaults do not silently change similarity behavior.~~
- ~~Masker mixed precision works on a validation fixture.~~
- ~~Logs show selected precision per plugin.~~

~~Use mamba env faceswap for any necessary execution~~

---

## ~~29. Build queue-based Extract benchmark with transfer/copy timing~~

**~~Goal:** Profile real Extract runner queues and expose model-forward, preprocessing, postprocess, memory, and CPU↔GPU transfer bottlenecks.~~

**~~Scope~~**

- ~~Add:~~

```bash
--profile-mode plugin|pipeline
```

- ~~Synthetic deterministic frames.~~
- ~~Real runner scheduling.~~
- ~~Include AutoMask path when enabled.~~
- ~~Include batched AutoMask vs single-face AutoMask comparison when available.~~
- ~~Metrics:~~
    - ~~frames/sec~~
    - ~~faces/sec~~
    - ~~queue occupancy~~
    - ~~starved time~~
    - ~~backpressure time~~
    - ~~memory~~
    - ~~bottleneck stage~~
    - ~~recommended batch sizes~~
    - ~~CPU↔GPU transfer time~~
    - ~~tensor residency percentage~~
    - ~~preprocessing time~~
    - ~~model-forward time~~
    - ~~postprocess time~~
    - ~~copy-back time~~
    - ~~MPS/CUDA memory pressure~~
- ~~Preserve existing plugin benchmark mode.~~

**~~Acceptance~~**

- ~~200-frame synthetic workload runs without real images.~~
- ~~Reports bottleneck and recommended batch sizes.~~
- ~~Reports transfer/copy timing.~~
- ~~Reports model-forward vs preprocessing/postprocess breakdown.~~
- ~~Existing plugin benchmark unchanged.~~
- ~~Benchmark can include AutoMask when enabled.~~

~~Use mamba env faceswap for any necessary execution~~

---

## ~~30. Add MPS performance path~~

**~~Goal:** Improve Apple Silicon extraction performance without overcommitting to immature or unsuitable compile behavio~~r.

**~~Scope~~**

- ~~Prevent CUDA-specific compile options from running on MPS.~~
- ~~Add MPS benchmark reporting:~~
    - ~~throughput~~
    - ~~memory~~
    - ~~transfer/copy timing~~
    - ~~MPS fallback ops if detectable~~
    - ~~compile fallback status if experimental compile is enabled~~
- ~~Prioritize:~~
    - ~~batching~~
    - ~~fewer CPU/GPU roundtrips~~
    - ~~MPS-safe Torch ops~~
    - ~~queue benchmark visibility~~
- ~~Add feasibility benchmark for static-model ONNX/CoreML path.~~

**~~Acceptance~~**

- ~~MPS does not accidentally use CUDA compile modes.~~
- ~~Compile failures fall back to eager.~~
- ~~MPS benchmark reports transfer and memory hotspots.~~
- ~~MPS speedup path is benchmark-driven.~~
- ~~Existing MPS extraction continues working without compile.~~

~~Use mamba env faceswap for any necessary execution~~

---

## ~~31. Add optional ONNX / TensorRT / CoreML static inference backend~~

**~~Goal:** Add optional production inference engines for static Extract models after the optimized PyTorch path exist~~s.

**~~Scope~~**

- ~~Keep PyTorch implementation canonical.~~
- ~~Add optional export/cache path per:~~
    - ~~plugin~~
    - ~~model checkpoint~~
    - ~~input size~~
    - ~~batch size~~
    - ~~precision~~
    - ~~backend~~
- ~~Candidate backends:~~
    - ~~TensorRT for NVIDIA~~
    - ~~ONNX Runtime providers~~
    - ~~CoreML Execution Provider for Apple feasibility~~
- ~~Candidate plugins:~~
    - ~~SCRFD~~
    - ~~mask parsers~~
    - ~~identity encoders~~
    - ~~ORFormer/SPIGA only if export path is stable~~
- ~~Benchmark against:~~
    - ~~PyTorch eager~~
    - ~~PyTorch compile~~
    - ~~optimized PyTorch GPU postprocess path~~
- ~~Add engine invalidation when plugin/config/input shape changes.~~

**~~Acceptance~~**

- ~~Engine cache works for at least one static plugin.~~
- ~~PyTorch path remains canonical and unchanged when disabled.~~
- ~~Benchmark compares engine path against optimized PyTorch, not eager only.~~
- ~~Engine mismatch invalidates cleanly.~~
- ~~Fallback to PyTorch works.~~

~~Use mamba env faceswap for any necessary execution~~

---

# Faceset QA / cleanup tickets

## ~~32. Add modern identity backends and calibration for FaceQA~~

**Goal:** Upgrade identity outlier detection from existing embeddings only to a calibrated, pluggable identity QA system using modern face-recognition backends.

**Scope**

- Keep existing identity plugins working.
- Add new identity plugins:
    - `insightface-buffalo-l`
    - `insightface-antelopev2`
    - `insightface-buffalo-s or buffalo-sc`
    - `magface`
    - `adaface`
- Use existing Extract identity plugin infrastructure.
- Store embeddings through existing DetectedFace.identity.
- Add FaceQA metadata:
    - `identity_model`
    - `identity_embedding_dim`
    - `identity_score`
    - `identity_quality`
    - `identity_cluster_id`
    - `identity_outlier`
    - `identity_threshold_used`
- Add per-model threshold calibration.
- Add cosine similarity normalization checks.
- Add cluster-based outlier detection:
    - main cluster
    - secondary clusters
    - singletons
    - low-confidence faces
- Add multi-model agreement mode:
    - primary model decides
    - secondary model verifies uncertain cases
- Add licensing note/help text for models with non-commercial restrictions.

**Acceptance**

- Existing `t-face` / `vggface2` behavior remains unchanged.
- `insightface-buffalo-l` can generate embeddings during Extract.
- FaceQA can cluster using `buffalo_l` embeddings.
- Obvious wrong-person faces are flagged.
- Low-quality same-person faces are not over-rejected if quality score is low.
- Thresholds are model-specific, not shared blindly.
- Manual/Sort can consume `identity_outlier`, `identity_cluster_id`, and `identity_quality`.

Use mamba env faceswap for any necessary execution

---

## ~~33. Identity benchmark and threshold calibration~~

**Goal:** Prevent bad default thresholds from creating false outliers.

**Scope**

- Build small benchmark harness for identity models.
- Evaluate:
    - same-person similarity distribution
    - different-person similarity distribution
    - low-quality same-person cases
    - profile/occlusion cases
    - duplicate adjacent frames
- Produce recommended thresholds per model.
- Store threshold defaults in config.

**Acceptance**

- Each identity model has documented default threshold.
- QA report shows distance-to-threshold.
- Thresholds can be adjusted in config.
- False rejection cases are visible in report.

Use mamba env faceswap for any necessary execution

---

## ~~34. Identity consensus mode~~

**Goal:** Reduce false positives by using multiple identity models only when needed.

**Scope**

- Primary model: `insightface-buffalo-l`.
- Optional verifier:
    - `magface`
    - `adaface`
    - `antelopev2`
- Run verifier only on uncertain faces:
    - `near threshold`
    - `low quality`
    - `singleton cluster`
    - `conflicting with main cluster`
- Persist:
    - model scores
    - agreement/disagreement
    - final decision reason

**Acceptance**

- Uncertain faces get second-pass identity check.
- Obvious outliers are still flagged.
- Low-quality true positives are less likely to be rejected.
- Metadata explains which model made the decision.

Use mamba env faceswap for any necessary execution

---

## ~~35. Implement duplicate / near-duplicate clustering~~

**Goal:** Reduce redundant adjacent video frames.

**Scope**

- Use face embeddings and/or visual embeddings.
- Cluster near duplicates.
- Rank best representative by:
    - blur
    - resolution
    - pose diversity
    - occlusion
    - mask QA

- Persist duplicate cluster and keep recommendation.

**Acceptance**

- Adjacent near-identical video frames cluster.
- Best representative is recommended.
- User can review before pruning.

Use mamba env faceswap for any necessary execution

---

## ~~36. Implement faceset coverage profiler~~

**Goal:** Report trainability and dataset balance.

**Scope**

- Compute:
    - yaw/pitch/roll
    - expression
    - mouth/eyes state
    - lighting
    - blur
    - resolution
    - occlusion
    - duplicate ratio
    - identity outlier ratio
    - mask QA distribution

- Output readiness report.

**Acceptance**

- Coverage report generated.
- Underrepresented buckets are flagged.
- JSON + readable summary output.

Use mamba env faceswap for any necessary execution

---

## ~~37. Extend existing Manual/Sort tooling for faceset cleanup~~

**Goal:** Reuse existing visual tooling for active-learning cleanup.

**Scope**

- Add filters/sorts:
    - identity outlier
    - duplicate cluster
    - blur
    - pose bucket
    - occlusion
    - expression
    - mask QA

- Show cluster representatives.
- Bulk accept/reject.
- Track reviewed state.
- Avoid permanent deletes by default.

**Acceptance**

- User can clean faceset without inspecting every face.
- Actions update FaceQA metadata.
- Existing Manual/Sort behavior unchanged when QA absent.

Use mamba env faceswap for any necessary execution

---

# Training Tickets

---

## 38. ~~Add decoder ResidualBlock configurable normalization~~

**Goal:** Make decoder residual blocks configurable without changing existing behavior.

**Problem:** `dec_norm` is configurable elsewhere, but decoder residual blocks currently do not have their own normalization control.

**Scope**

- Add decoder-only setting:

```ini
dec_res_block_norm = none | instance | group | rms | layer
```

- Default:

```ini
dec_res_block_norm = none
```

- Apply only inside decoder residual blocks.
- Do not change encoder residual blocks unless explicitly configured elsewhere.
- Ensure normalization layer selection matches existing model/config naming conventions.
- Add compatibility handling for older saved configs/checkpoints.

**Acceptance**

- Existing models behave unchanged with `dec_res_block_norm=none`.
- Each supported normalization option builds successfully.
- Save/resume works with the selected norm.
- Config persists and reloads correctly.
- Unit/model-construction tests cover all options.

Use mamba env faceswap for any necessary execution

---

## 39. ~~Add residual scaling inside decoder residual blocks~~

**Goal:** Prevent decoder residual paths from becoming too dominant at higher resolutions or with stacked residual blocks.

**Scope**

- Add decoder-only residual branch scaling:

```python
x = input + residual_scale * F(input)
```

- Add config:

```ini
dec_residual_scale = 0.1 | 0.2 | ... | 1.0
```

- Default recommendation: `0.1` or `0.2`.
- Do not globally change `ResidualBlock` behavior unless every caller explicitly opts in.
- Ensure scaling applies to the residual branch only, not the entire block output.

**Acceptance**

- Existing behavior can be reproduced with `dec_residual_scale=1.0` if needed.
- Default setting trains without NaNs on a small fixture.
- Decoder-only behavior is verified.
- Unit/model-construction tests confirm residual scaling math.

Use mamba env faceswap for any necessary execution

---

## 40. ~~Make activation configurable by stage and add Swish/SiLU~~

**Goal:** Allow targeted activation experiments without disturbing pretrained Keras encoder paths.

**Scope**

- Add config:

```ini
enc_activation = leakyrelu | swish
dec_activation = leakyrelu | swish
fc_activation = leakyrelu | swish
```

- Use Swish/SiLU implementation consistent with the backend/Keras version.
- Keep `leakyrelu` as default for backward compatibility.
- Do not touch pretrained Keras encoder activations.
- Apply encoder activation only to `_EncoderFaceswap`.
- Apply decoder activation only to configurable decoder blocks.
- Apply FC activation only to bottleneck/dense stack where appropriate.

**Acceptance**

- Existing default behavior unchanged.
- Swish model builds and trains on a small fixture.
- Pretrained Keras encoder activations are not modified.
- Config persists/reloads correctly.
- Model summary reflects selected activations.

Use mamba env faceswap for any necessary execution

---

## 41. ~~Move bottleneck normalization to post-flatten latent normalization~~

**Goal:** Normalize the actual latent vector/tensor representation rather than normalizing inconsistently before flatten/pooling/dense behavior.

**Problem:** Current `_bottleneck()` normalization happens before flatten/dense/global pooling, so the meaning of “norm” changes depending on encoder output structure.

**Scope**

- Refactor bottleneck flow toward:

```python
x = encoder_output
x = flatten_or_pool_or_dense(x)
x = latent_norm(x)
```

- Add/confirm config:

```ini
latent_norm = none | layer | rms | group | instance
```

- Keep backward-compatible default behavior unless changing default is explicitly accepted.
- Ensure old checkpoints/configs still load or fail with clear migration guidance.
- Log selected latent norm in model setup.

**Acceptance**

- Existing default behavior remains compatible.
- Post-flatten latent norm works for flatten, dense, and pooling paths.
- Save/resume works.
- Unit/model-construction tests cover each bottleneck mode.

Use mamba env faceswap for any necessary execution

---

## 42. ~~Add optional high-frequency refinement tail before `face_out`~~

**Goal:** Improve fine detail cautiously without creating fake texture, halos, or over-sharpened artifacts.

**Scope**

- Add config:

```ini
refinement_tail = none | light
refinement_blocks = 1 | 2
refinement_filters = min(dec_min_filters, 64)
```

- Use small residual conv blocks.
- No attention in the refinement tail.
- No aggressive sharpening.
- No large filter counts.
- Apply immediately before `Conv2DOutput(..., name="face_out")`.
- Keep default:

```ini
refinement_tail = none
```

**Acceptance**

- Existing behavior unchanged when disabled.
- Light tail builds and trains on a small fixture.
- Output shape unchanged.
- No additional conversion special-casing required.
- Validation dashboard can compare enabled vs disabled detail metrics.

Use mamba env faceswap for any necessary execution

---

## 43. ~~Add optional anti-alias blur after learned upscales~~

**Goal:** Reduce checkerboard/stair-step artifacts from learned upscaling paths.

**Scope**

- Add config:

```ini
dec_antialias = none | light
```

- Default:

```ini
dec_antialias = none
```

- Apply after learned upscales, not after final output.
- Use fixed depthwise blur kernel:

```text
[1, 2, 1] x [1, 2, 1] / 16
```

- Keep effect subtle.
- Avoid destroying skin/eye/mouth detail.

**Acceptance**

- Existing behavior unchanged when disabled.
- Light anti-alias builds and trains.
- Checkerboard artifacts decrease on targeted validation samples.
- No blur is applied after final `face_out`.

Use mamba env faceswap for any necessary execution

---

# Stop Here

## 44. Add mask-conditioned training inputs

**Goal:** Let eligible model backbones use mask/semantic information during training rather than relying only on RGB reconstruction.

**Scope**

- Add optional training input channels:
    - binary face mask
    - semantic face-region masks where available
    - boundary band mask
    - occlusion/obstacle mask
- Support semantic regions:
    - eyes
    - brows
    - mouth
    - lips
    - teeth
    - skin
    - hair boundary
    - glasses/obstacles
- Add config:

```ini
mask_conditioning = none | binary | semantic
boundary_conditioning = true | false
```

- Ensure existing models can ignore these channels.
- Add compatibility checks so unsupported models fail clearly.

**Acceptance**

- Existing training works unchanged with `mask_conditioning=none`.
- Supported model variant can train with binary mask conditioning.
- Supported model variant can train with semantic conditioning when metadata exists.
- Poor/missing masks are handled gracefully.
- Training logs show conditioning mode.

Use mamba env faceswap for any necessary execution

---

## 52. Add training metadata ingestion and curriculum sampler

**Goal:** Use FaceQA/AutoMask metadata during training.

**Scope**

- Load metadata when available:
    - FaceQA
    - AutoMask QA
    - identity outlier flags
    - duplicate cluster metadata
    - pose/expression/occlusion buckets
    - mask confidence/flags
- Exclude:
    - rejected faces
    - identity outliers
    - duplicate non-keeps
    - low-quality masks
- Balance:
    - pose
    - expression
    - occlusion
    - lighting/domain where available
- Curriculum:
    - clean/frontal/high-confidence first
    - broader pose/expression next
    - harder occlusions later
- Report sampling distribution.

**Acceptance**

- Existing training works without QA metadata.
- QA-aware sampler works when metadata exists.
- Exclusion and weighting behavior is configurable.
- Sampling distribution is reportable.
- Duplicate clusters can be deduped at sample time.

Use mamba env faceswap for any necessary execution

---

## 54. Add 3D facial-prior extraction for training metadata

**Goal:** Persist geometry metadata for training, QA, validation, and later refinement.

**Scope**

- Estimate:
    - yaw/pitch/roll
    - expression coefficients
    - optional 3DMM/FLAME-like params
    - camera/projection params
    - landmark visibility if available
- Feed:
    - FaceQA
    - curriculum sampler
    - validation dashboard
    - conversion QA
    - ControlNet conditioning
    - Gaussian avatar path if implemented
- Persist per face/frame with schema versioning.

**Acceptance**

- Metadata written per face/frame.
- Severe profile/extreme pose detected.
- Sampler and validation dashboard can consume pose buckets.
- Existing training ignores metadata safely if absent.

Use mamba env faceswap for any necessary execution

---

# Stop Here

## 45. Add region-weighted perceptual loss

**Goal:** Improve perceptual fidelity in the regions users notice most: eyes, mouth, skin texture, and boundaries.

**Scope**

- Use frozen perceptual network features.
- Region masks from parser/AutoMask metadata:
    - eyes
    - brows
    - mouth
    - teeth
    - lips
    - skin
    - boundary band
- Add configurable region weights:

```ini
loss_perceptual_weight = 0.0
loss_perceptual_eye_weight = ...
loss_perceptual_mouth_weight = ...
loss_perceptual_skin_weight = ...
loss_perceptual_boundary_weight = ...
```

- Higher default experimental weights for eyes and mouth.
- Separate boundary perceptual term.
- Optional skin-texture patch term.
- Disable by default unless this model family explicitly opts in.

**Acceptance**

- Disabled behavior unchanged.
- Loss components are logged separately.
- Region weighting can be enabled/disabled.
- Small fixture trains without NaNs.
- Missing region masks fall back safely.

Use mamba env faceswap for any necessary execution

---

## 46. Add identity embedding loss using frozen face-recognition model

**Goal:** Improve identity preservation without training the identity model.

**Scope**

- Use frozen face-recognition encoder, preferably ArcFace/InsightFace-family backend.
- Do not backprop through identity model weights.
- Loss form:

```python
identity_loss = 1 - cosine_similarity(E(generated_face), E(source_identity_reference))
```

- Apply after face alignment into the identity model’s expected input format.
- Ignore or downweight heavily occluded regions.
- Add configurable weight:

```ini
loss_identity_weight = 0.0
loss_identity_model = insightface-buffalo-l | antelopev2 | existing
```

- Use with caution: too much identity loss can crush target expression.

**Acceptance**

- Disabled behavior unchanged.
- Frozen identity model is not updated during training.
- Identity loss logs separately.
- Small fixture trains without NaNs.
- Identity loss can be computed on validation samples.
- Clear warning/help text describes expression tradeoff.

Use mamba env faceswap for any necessary execution

---

## 47. Add mask-aware, boundary-aware, and occlusion-excluded reconstruction losses

**Goal:** Make reconstruction loss respect the mask/occlusion structure instead of penalizing invalid regions equally.

**Scope**

- Add optional losses:
    - mask-weighted reconstruction
    - boundary-band reconstruction
    - occlusion-excluded reconstruction
    - semantic-region weighted reconstruction
- Use AutoMask/semantic masks when available.
- Exclude:
    - hands/obstacles
    - low-confidence parser regions
    - occluded central regions where appropriate
- Add configurable weights.
- Log each component.

**Acceptance**

- Disabled behavior unchanged.
- Training works when masks are missing.
- Loss excludes obstacle regions correctly.
- Boundary loss can be enabled independently.
- Component losses appear in logs/dashboard.

Use mamba env faceswap for any necessary execution

---

# Stop Here

## 50. Modernize autoencoder backbone as an experimental model plugin

**Goal:** Add a stronger classic FaceSwap model while preserving existing model compatibility.

**Scope**

- Add new experimental model/trainer option rather than rewriting existing models.
- Architecture options:
    - residual encoder/decoder blocks
    - U-Net-style skip connections where appropriate
    - attention at mid/high resolutions only
    - configurable normalization
    - mask/semantic conditioning support
    - optional refinement tail
    - optional anti-alias upscaling
- Avoid attention in final refinement tail.
- Preserve conversion compatibility.
- Include stronger augmentations:
    - color jitter
    - blur/compression
    - pose perturbation
    - mask perturbation
    - occlusion simulation
    - boundary erosion/dilation augmentation

**Acceptance**

- Existing models unchanged.
- New model builds and trains on fixture.
- Save/resume works.
- Mixed precision works or fails clearly.
- Conversion path consumes output without special casing.
- Validation dashboard can compare old vs new backbone.

Use mamba env faceswap for any necessary execution

---

# Stop Here

## 48. Add low-weight patch/multi-scale adversarial loss

**Goal:** Improve local realism carefully without destabilizing training or sacrificing identity/expression.

**Scope**

- Add optional discriminator path:
    - PatchGAN or multi-scale patch discriminator.
- Low default experimental weight.
- Region-aware crops:
    - full face
    - eyes
    - mouth
    - skin patches
    - boundary regions
- Stabilization options:
    - spectral normalization or gradient penalty if feasible
    - lazy discriminator updates
    - discriminator warmup delay
- Disabled by default.
- Explicitly mark experimental.

**Acceptance**

- Existing trainers unchanged when disabled.
- Small fixture trains without collapse or NaNs.
- Generator/discriminator losses are logged separately.
- Save/resume preserves discriminator state if enabled.
- Mixed precision either works or fails clearly.
- Documentation warns against high adversarial weights.

Use mamba env faceswap for any necessary execution

---

## 49. Add training-time temporal consistency losses

**Goal:** Reduce video flicker at the model-output level, not only in post-processing.

**Scope**

- Add optional video/window training mode.
- Frame-window sampler for adjacent or nearby frames.
- Losses:
    - landmark-consistent reconstruction
    - optical-flow consistency
    - embedding consistency across adjacent frames
    - mask-boundary temporal consistency
    - optional recurrent/state consistency if model supports it
- Detect/downweight unreliable optical flow.
- Do not require temporal training for normal image-pair training.

**Acceptance**

- Existing image training unchanged.
- Video-window training fixture runs.
- Temporal losses log separately.
- Temporal validation clip shows reduced flicker.
- Optical-flow failures are skipped or downweighted.

Use mamba env faceswap for any necessary execution

---

# Stop Here

## 51. Build training batch-size profiler

**Goal:** Find stable and practical batch size.

**Scope**

- Add train option:

```bash
faceswap.py train --profile-training-batch
```

or equivalent GUI option.

- Candidate search:
    - double until OOM/NaN/memory threshold
    - binary search between last-good and first-bad
- Backend-aware memory:
    - CUDA
    - ROCm
    - MPS
- Report:
    - images/sec
    - steps/sec
    - loader wait
    - train step time
    - CPU sync/bookkeeping time
    - memory allocated/in use
- Recommend smallest batch within ~95% peak throughput.
- No saves/previews/TensorBoard writes during profiling.
- Restore original model/optimizer state unless user applies recommendation.

**Acceptance**

- Multiple batch sizes are tested safely.
- Original training state is restored unless user applies recommendation.
- Works on CUDA/ROCm/MPS through shared backend abstraction.
- Output table is clear.
- Does not recommend a batch size above configured memory threshold.

Use mamba env faceswap for any necessary execution

---

## 53. Add validation dashboard

**Goal:** Measure quality beyond scalar training loss.

**Scope**

- Fixed validation grid.
- Validation buckets:
    - identity
    - pose
    - expression
    - occlusion
    - mask quality
    - duplicate cluster representative
- Metrics:
    - reconstruction loss
    - identity similarity
    - perceptual loss
    - mask-weighted loss
    - pose-bucket performance
    - expression-bucket performance
    - temporal sample consistency
    - boundary artifact score
- CLI and GUI report where feasible.
- Save validation metadata with model snapshots when enabled.

**Acceptance**

- Stable validation set across checkpoints.
- Metrics are reported per bucket.
- CLI/GUI report available.
- No change to training behavior unless enabled.
- Dashboard can compare architecture/loss experiments.

Use mamba env faceswap for any necessary execution

---

## 60. Add loss/architecture ablation report

**Goal:** Make the training roadmap measurable instead of accumulating knobs.

**Scope**

- Add report comparing enabled features:
    - residual scaling
    - decoder norm
    - Swish
    - latent norm
    - refinement tail
    - anti-alias
    - region perceptual loss
    - identity loss
    - mask-aware losses
    - adversarial loss
    - temporal losses
    - modern backbone
- Report:
    - training throughput
    - memory use
    - validation metrics
    - artifact scores
    - identity similarity
    - temporal consistency
- Export markdown/JSON summary.

**Acceptance**

- Ablation report runs on a small controlled fixture.
- Report identifies enabled training features.
- Results are comparable across runs.
- No training behavior changes unless report mode is enabled.

Use mamba env faceswap for any necessary execution

---

# Stop Here

## 56. Add optimizer evaluation harness

**Goal:** Prevent optimizer work from becoming untestable.

**Scope**

- Deterministic training fixture.
- Common metrics:
    - loss decrease
    - NaN/OOM
    - save/resume
    - mixed precision
    - throughput
    - memory usage
- Covers:
    - existing optimizers
    - AdamW
    - Lion
    - AdamW/Lion + CWD
    - Schedule-Free AdamW/Lion
    - Simplified-AdEMAMix
- CI/smoke mode with small fixture.

**Acceptance**

- All optimizers run through same harness.
- Failure reports are clear.
- Save/resume and mixed precision are tested.
- Harness can be run without long training jobs.

Use mamba env faceswap for any necessary execution

---

## 57. Add Cautious Weight Decay to AdamW/Lion

**Goal:** Optional CWD for decoupled weight decay.

**Scope**

- Shared CWD utility.
- Respect no-decay param groups:
    - bias
    - norm params
    - gamma/beta
    - other configured exclusions
- Config:

```ini
cautious_weight_decay = true | false
```

- Existing behavior unchanged when disabled.
- Log whether standard or cautious weight decay is active.

**Acceptance**

- AdamW/Lion train with CWD enabled and disabled.
- No-decay params remain excluded.
- Unit tests cover CWD masking and standard fallback.
- Config persists and reloads.

Use mamba env faceswap for any necessary execution

---

## 58. Add Schedule-Free AdamW/Lion

**Goal:** Add schedule-free optimizers safely.

**Scope**

- Optimizers:
    - `schedule-free-adamw`
    - `schedule-free-lion`
- Keras-compatible implementations.
- Train/eval mode hooks:
    - `optimizer.train()` before training batches
    - `optimizer.eval()` before preview/save/validation/checkpoint
- Mixed precision compatibility.
- Save/resume support.
- LR finder warning/disable unless explicitly tested.

**Acceptance**

- Loss decreases on fixture.
- Preview/save/checkpoint use eval weights.
- Resume restores optimizer state.
- Existing optimizers unchanged.
- Unit tests cover train/eval switching and serialization.

Use mamba env faceswap for any necessary execution

---

## 59. Add Simplified-AdEMAMix

**Goal:** Add experimental optimizer option.

**Scope**

- Keras-compatible implementation.
- Config:
    - learning rate
    - beta1
    - beta2
    - long-momentum coefficient or simplified equivalent
    - epsilon
    - weight decay
- CWD support.
- Mixed precision support.
- Serialization/resume support.

**Acceptance**

- Small fixture trains without NaNs.
- Loss decreases.
- Save/resume preserves optimizer state.
- Existing optimizers unchanged.
- Unit tests cover update step, weight decay, serialization, and config loading.

Use mamba env faceswap for any necessary execution

---

# Stop Here

## ~~55. Add adapter registry, not full LoRA for current classic trainers~~

**Goal:** Prepare for adapters where they fit: diffusion/refinement/future models, without forcing LoRA into current autoencoder trainers.

**Scope**

- Registry metadata:
    - adapter type
    - base model hash
    - faceset hash
    - subject/subject-pair/domain label
    - training config
    - optimizer config
    - consent/reference metadata
- Compatibility checks.
- UI/CLI selection hooks.
- Explicitly document that LoRA/adapters are not required for current classic trainers unless a future model architecture supports them.

**Acceptance**

- Adapter metadata persists.
- Wrong base/adapter pairing fails clearly.
- Existing training does not require adapters.
- Diffusion/refinement/future model plugins can use adapter registry later.

Use mamba env faceswap for any necessary execution

---

# Conversion Tickets

---

## 61. Add conversion artifact QA

**Goal:** Detect bad converted frames.

**Scope**

- Scores:
    - seam/boundary artifacts
    - mask leak
    - color mismatch
    - identity drift
    - eye/mouth artifacts
    - blur mismatch
    - temporal flicker

- Output:
    - bad-frame list
    - contact sheet
    - JSON report.

**Acceptance**

- Obvious artifacts are flagged.
- Clean frames score high.
- Report links to frame/mask metadata.

Use mamba env faceswap for any necessary execution

---

## 62. Add ControlNet-style conditional post-conversion refinement

**Goal:** Spatially constrained diffusion repair.

**Scope**

- Inputs:
    - converted patch
    - target frame
    - final mask
    - semantic layers
    - landmarks
    - edge map
    - depth/normal/pose hints

- Restrict to masked face region.
- Preserve non-face pixels.
- Record conditioning metadata.

**Acceptance**

- Non-face areas unchanged.
- Pose/expression preserved.
- Identity does not regress beyond threshold.

Use mamba env faceswap for any necessary execution

---

## 63. Add ID-constrained diffusion refinement

**Goal:** Improve hard-frame identity preservation.

**Scope**

- Source identity embedding condition.
- Identity strength control.
- Compare pre/post:
    - identity score
    - expression score
    - artifact score.

- Prevent over-preserving identity at cost of expression.

**Acceptance**

- Identity improves on hard frames.
- Expression remains within threshold.
- Metadata records identity conditioning.

Use mamba env faceswap for any necessary execution

---

## 64. Add temporal conversion stabilization

**Goal:** Reduce output flicker.

**Scope**

- Optical-flow consistency.
- Embedding consistency.
- Boundary consistency.
- Reprocess unstable frames/segments only.

**Acceptance**

- Flicker decreases on test clips.
- No identity lag or over-smoothing.

Use mamba env faceswap for any necessary execution

---

## 65. Add conversion manifest and provenance metadata

**Goal:** Make outputs auditable.

**Scope**

- Emit manifest:
    - generated media flag
    - source hash
    - model hash
    - faceset hashes
    - alignments hash
    - plugin versions
    - AutoMask schema version
    - conversion config
    - optional consent/reference metadata.

- Optional generated-media metadata tag/label.

**Acceptance**

- Conversion can emit manifest.
- Existing conversion output unchanged when disabled.
- Manifest links source/model/faceset/config.

Use mamba env faceswap for any necessary execution
