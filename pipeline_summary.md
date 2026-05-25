# Landmark resolver pipeline summary

Status: **planned**
Promotion status: ``
Promotion scope: `production`
Selected runtime policy: `learned_quality_v1`
Promoted scorer version: `continuous_regret_v1_1`
Promoted scorer target: `selection_cost`
Best weights: `current/candidate_search/best_weights.json`
Scorer: `current/scorer_training/runtime_resolver_scorer.json`
Eval report: `scorer_evaluation/scorer_policy_report.json`
Failed gates: `[]`
Fallback counts: `{'fallback_count': 0, 'safe_fallback_count': 0, 'hard_slice_fallback_count': 0, 'consensus_collapse_fallback_count': 0}`
Config patch: `config_update_patch.ini`
Progress log: `pipeline_progress.jsonl`

## Stages

| Stage | Status | Duration | Outputs | Notes |
| --- | --- | ---: | --- | --- |
| `build_dataset_manifest` | planned | 0.0s | dataset/manifest.json<br>dataset/.base_dataset_manifest_complete.json | would run 1 dataset build command(s) |
| `build_hard_source_manifest` | planned | 0.0s | aflw2000_3d/manifest.json |  |
| `build_prediction_cache` | planned | 0.0s | cache<br>cache/.base_prediction_cache_complete.json |  |
| `build_hard_source_prediction_cache` | planned | 0.0s | cache<br>cache/.hard_source_prediction_cache_complete.json |  |
| `build_splits` | planned | 0.0s | splits/splits.json<br>dataset/fit_manifest.json<br>dataset/select_manifest.json<br>dataset/report_manifest.json<br>run_summary.json | would write split assignment and split manifests |
| `fit_static_weights` | planned | 0.0s | weights/static_landmark_weights.json | would fit static landmark weights from fit split |
| `build_production_manifest` | planned | 0.0s | manifest.json<br>resolver_metadata.jsonl<br>audit.json |  |
| `build_production_prediction_cache` | planned | 0.0s | cache<br>cache/.production_prediction_cache_complete.json |  |
| `build_production_resolver_metadata` | planned | 0.0s | resolver_metadata.jsonl<br>.production_resolver_metadata_complete.json |  |
| `candidate_search` | planned | 0.0s | candidate_search/best_setup.json<br>candidate_search/best_weights.json |  |
| `hard_alignment_validation` | planned | 0.0s | gt_hard_validation/manifest.json |  |
| `build_gt_hard_resolver_metadata` | planned | 0.0s | resolver_metadata.jsonl | would run runtime resolver to write GT-hard metadata sidecar |
| `freeze_resolver_metadata` | planned | 0.0s | resolver_metadata.jsonl | would validate, copy, or reuse frozen GT-hard resolver metadata |
| `binary_scorer_training` | planned | 0.0s | scorer_training/v1_binary/runtime_resolver_scorer.json |  |
| `continuous_scorer_training` | planned | 0.0s | scorer_training/v1_1_selection_cost/runtime_resolver_scorer.json<br>scorer_training/v1_1_selection_cost/runtime_resolver_scorer_eval_rows.csv |  |
| `v2_scorer_training` | planned | 0.0s | scorer_training/v2_lambdarank/runtime_resolver_scorer_v2.json<br>scorer_training/v2_lambdarank/.v2_scorer_training_complete.json |  |
| `scorer_evaluation` | planned | 0.0s | scorer_evaluation/scorer_policy_report.json |  |
| `production_promotion_check` | planned | 0.0s | scorer_evaluation/scorer_policy_report.json | would require promotion_status=pass |
| `artifact_export` | planned | 0.0s | current/candidate_search/best_setup.json<br>current/candidate_search/best_weights.json<br>current/scorer_training/runtime_resolver_scorer.json<br>current/artifacts/artifacts_manifest.json | would copy promoted artifacts |
| `config_update` | planned | 0.0s | config_update_preview.json<br>config_update_patch.ini | would write config preview/update |

## Config fields changed

- `batch_size`
- `crop_scale`
- `fallback_model`
- `fallback_strategy`
- `hard_case_strategy`
- `hard_disagreement_px`
- `hard_roll_degrees`
- `min_models`
- `models`
- `outlier_threshold`
- `reject_outliers`
- `resolver_policy`
- `resolver_scorer_path`
- `roll_veto_degrees`
- `secondary_hard_case_strategy`
- `setup_mode`
- `setup_path`
- `strategy`
- `strict`
- `use_alignment_resolver`
- `weights_path`
