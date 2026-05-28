# Phaze-A Decoder Architecture Controls Coordination

## Assigned subagents

- Coordinator: shared contracts, integration, compatibility review, final verification
- Subagent A: Unit 0 shared decoder contracts
- Subagent B: Unit 1 decoder residual normalization
- Subagent C: Unit 2 decoder residual scaling
- Subagent D: Unit 3 stage-specific activation config
- Subagent E: Unit 4 decoder anti-alias after learned upscales
- Subagent F: Unit 5 refinement tail before `face_out`

## Unit boundaries

- Unit 0: shared helper contracts only, no user-visible behavior change beyond scaffolding
- Unit 1: `dec_res_block_norm` in decoder residual blocks only
- Unit 2: `dec_residual_scale` in decoder residual branch only
- Unit 3: `enc_activation`, `dec_activation`, `fc_activation` for Phaze-A controlled paths only
- Unit 4: optional fixed anti-alias blur after learned decoder upscales only
- Unit 5: optional light refinement tail immediately before `face_out`

## Dependency order

- Unit 0 first
- Units 1, 3, 4 after Unit 0
- Unit 2 after Unit 1 if decoder residual wrapper changes overlap
- Unit 5 after Units 1 through 3, preferably after Unit 4

## Shared contracts

- Normalization resolver for decoder and latent-stage helper use
- Activation resolver for Phaze-A controlled encoder, FC, decoder, and optional helper stages
- Decoder residual block wrapper that keeps behavior stable by default
- Residual branch scaling semantics: `inputs + scale * F(inputs)`
- Config defaults that preserve existing behavior for old configs and checkpoints
- CPU-safe Phaze-A model-construction helper coverage

## Approved edit scope

- Coordinator / Unit 0: `plugins/train/model/phaze_a.py`, `plugins/train/model/phaze_a_defaults.py`, `tests/plugins/train/`
- Unit 1: `plugins/train/model/phaze_a.py`, `plugins/train/model/phaze_a_defaults.py`, `tests/plugins/train/`
- Unit 2: `plugins/train/model/phaze_a.py`, `plugins/train/model/phaze_a_defaults.py`, `tests/plugins/train/`
- Unit 3: `plugins/train/model/phaze_a.py`, `plugins/train/model/phaze_a_defaults.py`, `tests/plugins/train/`
- Unit 4: `plugins/train/model/phaze_a.py`, `plugins/train/model/phaze_a_defaults.py`, `tests/plugins/train/`, `lib/model/nn_blocks.py` only if a reusable fixed blur layer is needed
- Unit 5: `plugins/train/model/phaze_a.py`, `plugins/train/model/phaze_a_defaults.py`, `tests/plugins/train/`

## Required tests before merge

- Unit 0: `tests/plugins/train/test_phaze_a_decoder_contracts.py`
- Unit 1: `tests/plugins/train/test_phaze_a_decoder_norm.py`
- Unit 2: `tests/plugins/train/test_phaze_a_residual_scale.py`
- Unit 3: `tests/plugins/train/test_phaze_a_activation_config.py`
- Unit 4: `tests/plugins/train/test_phaze_a_antialias.py`
- Unit 5: `tests/plugins/train/test_phaze_a_refinement_tail.py`
- Broad smoke after integration: `tests/lib/model/` and `tests/plugins/train/`
