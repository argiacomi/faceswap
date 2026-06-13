#!/usr/bin/env python3
"""Unit tests for CD-ViT checkpoint validation and schema-aware key handling.

Covers:
  - ``_strip_training_only_keys`` drops only the schema-aware training head families
  - ``_validate_checkpoint`` accepts plain staged checkpoints
  - ``_validate_checkpoint`` accepts schema-aware staged checkpoints
  - ``_validate_checkpoint`` unwraps ``state_dict`` nesting and wrapper prefixes
  - ``_validate_checkpoint`` rejects UNet-backed checkpoints
  - ``_validate_checkpoint`` rejects non-staged checkpoints
  - ``_validate_checkpoint`` rejects non-dict payloads
  - Real schema-aware checkpoint loads strictly after filtering and produces
    (N, 68, 2) normalized coordinates (slow; skipped if weights absent)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from plugins.extract.align.cd_vit import (
    Cd_Vit,
    CDViTStage,
    _strip_training_only_keys,
)

# Path to the Faceswap model cache relative to the repo root
_CACHE_DIR = Path(__file__).resolve().parents[4] / ".fs_cache"
_CD_VIT_WEIGHTS = _CACHE_DIR / "cd_vit_v1.pth"

_TINY = torch.zeros(1)


def _staged_keys() -> dict[str, torch.Tensor]:
    """Minimal key layout of a plain staged CD-ViT checkpoint."""
    return {
        "pre.body.0.res_layer.1.conv.weight": _TINY,
        "stages.0.0.conv.weight": _TINY,
        "output_layers.0.weight": _TINY,
        "merge.0.double_conv.0.weight": _TINY,
    }


def _schema_aware_keys() -> dict[str, torch.Tensor]:
    """Key layout of a schema-aware training checkpoint."""
    return _staged_keys() | {
        "schema_output_layers.landmarks_98.0.weight": _TINY,
        "schema_output_layers.profile39.0.bias": _TINY,
        "visibility_output_layers.landmarks_68.0.feature_proj.weight": _TINY,
        "auxiliary_output_layers.pose_bucket.weight": _TINY,
    }


# ---------------------------------------------------------------------------
# _strip_training_only_keys
# ---------------------------------------------------------------------------


def test_strip_training_only_keys_drops_only_training_heads() -> None:
    state = _schema_aware_keys()
    filtered = _strip_training_only_keys(state)
    assert set(filtered) == set(_staged_keys())


def test_strip_training_only_keys_is_noop_for_plain_checkpoints() -> None:
    state = _staged_keys()
    assert _strip_training_only_keys(state) == state


# ---------------------------------------------------------------------------
# _validate_checkpoint
# ---------------------------------------------------------------------------


def _save(tmp_path: Path, payload: object) -> str:
    path = tmp_path / "checkpoint.pth"
    torch.save(payload, path)
    return str(path)


def test_validate_accepts_plain_staged_checkpoint(tmp_path: Path) -> None:
    Cd_Vit._validate_checkpoint(_save(tmp_path, _staged_keys()))


def test_validate_accepts_schema_aware_checkpoint(tmp_path: Path) -> None:
    Cd_Vit._validate_checkpoint(_save(tmp_path, _schema_aware_keys()))


def test_validate_unwraps_state_dict_and_module_prefix(tmp_path: Path) -> None:
    wrapped = {"state_dict": {f"module.{key}": value for key, value in _staged_keys().items()}}
    Cd_Vit._validate_checkpoint(_save(tmp_path, wrapped))


def test_validate_rejects_unet_backed_checkpoint(tmp_path: Path) -> None:
    state = _staged_keys() | {"unet.down1.weight": _TINY}
    with pytest.raises(ValueError, match="UNet-backed"):
        Cd_Vit._validate_checkpoint(_save(tmp_path, state))


def test_validate_rejects_non_staged_checkpoint(tmp_path: Path) -> None:
    state = {"backbone.layer1.weight": _TINY}
    with pytest.raises(ValueError, match="does not look like staged CD-ViT"):
        Cd_Vit._validate_checkpoint(_save(tmp_path, state))


def test_validate_rejects_non_dict_payload(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="state dictionary"):
        Cd_Vit._validate_checkpoint(_save(tmp_path, [1, 2, 3]))


# ---------------------------------------------------------------------------
# Real checkpoint integration
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not _CD_VIT_WEIGHTS.is_file(), reason="CD-ViT weights not present")
def test_real_checkpoint_loads_and_predicts() -> None:
    """The shipped schema-aware checkpoint loads strictly after filtering."""
    Cd_Vit._validate_checkpoint(str(_CD_VIT_WEIGHTS))

    state = torch.load(_CD_VIT_WEIGHTS, map_location="cpu")
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    state = {Cd_Vit._canonical_key(key): value for key, value in state.items()}

    model = CDViTStage()
    model.load_state_dict(state)  # strict for all non-training-head keys
    model.eval()

    with torch.inference_mode():
        coords = model(torch.zeros((1, 3, 256, 256)))

    assert coords.shape == (1, 68, 2)
    assert torch.isfinite(coords).all()
    assert coords.min() >= 0.0
    assert coords.max() <= 1.0
