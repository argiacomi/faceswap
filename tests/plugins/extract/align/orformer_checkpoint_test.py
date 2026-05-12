#!/usr/bin/env python3
"""Unit tests for ORFormer checkpoint mismatch validation.

Covers:
  - No mismatch: no warning, no error
  - Allowlisted missing key: warning logged, no error
  - Allowlisted unexpected key: warning logged, no error
  - Non-allowlisted missing key: RuntimeError raised
  - Non-allowlisted unexpected key: RuntimeError raised
  - Mixed allowlisted + bad keys: RuntimeError raised, bad keys listed
  - Official WFLW VQ-VAE checkpoint loads with zero mismatch (skipped if weights absent)
  - Official 300W VQ-VAE checkpoint loads with zero mismatch (skipped if weights absent)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from lib.infer.align import Align  # noqa: F401  (import for side-effect)

# Path to the Faceswap model cache relative to the repo root
_CACHE_DIR = Path(__file__).resolve().parents[4] / ".fs_cache"

_WFLW_VQVAE = _CACHE_DIR / "orformer_vqvae_wflw_20240704.pt"
_W300_VQVAE = _CACHE_DIR / "orformer_vqvae_300w_20240704.pt"


# ---------------------------------------------------------------------------
# Minimal stub for the load_state_dict IncompatibleKeys result
# ---------------------------------------------------------------------------


class _IncompatibleKeys:
    def __init__(self, missing: list[str], unexpected: list[str]) -> None:
        self.missing_keys = missing
        self.unexpected_keys = unexpected


# ---------------------------------------------------------------------------
# Helper: import the private function under test
# ---------------------------------------------------------------------------


def _fn():
    from plugins.extract.align._orformer.model import _validate_orformer_checkpoint

    return _validate_orformer_checkpoint


def _allowlists():
    from plugins.extract.align._orformer import model as m

    return m._ALLOWED_ORFORMER_MISSING, m._ALLOWED_ORFORMER_UNEXPECTED


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_mismatch_does_not_warn_or_raise(caplog) -> None:
    """Clean checkpoint load produces no warning and no exception."""
    validate = _fn()
    with caplog.at_level(logging.WARNING):
        validate(_IncompatibleKeys([], []))
    assert not caplog.records


def test_allowlisted_missing_key_warns_not_raises(caplog, monkeypatch) -> None:
    """An allowlisted missing key logs a warning but does not raise."""
    from plugins.extract.align._orformer import model as m

    monkeypatch.setattr(m, "_ALLOWED_ORFORMER_MISSING", frozenset({"training.loss_scale"}))
    monkeypatch.setattr(m, "_ALLOWED_ORFORMER_UNEXPECTED", frozenset())
    validate = _fn()
    with caplog.at_level(logging.WARNING):
        validate(_IncompatibleKeys(["training.loss_scale"], []))
    assert any("allowlisted mismatch" in r.message for r in caplog.records)


def test_allowlisted_unexpected_key_warns_not_raises(caplog, monkeypatch) -> None:
    """An allowlisted unexpected key logs a warning but does not raise."""
    from plugins.extract.align._orformer import model as m

    monkeypatch.setattr(m, "_ALLOWED_ORFORMER_MISSING", frozenset())
    monkeypatch.setattr(m, "_ALLOWED_ORFORMER_UNEXPECTED", frozenset({"vq_loss_weight"}))
    validate = _fn()
    with caplog.at_level(logging.WARNING):
        validate(_IncompatibleKeys([], ["vq_loss_weight"]))
    assert any("allowlisted mismatch" in r.message for r in caplog.records)


def test_unapproved_missing_key_raises() -> None:
    """A missing key not in the allowlist raises RuntimeError."""
    validate = _fn()
    with pytest.raises(RuntimeError, match="missing=\\['encoder.conv_stack.0.weight'\\]"):
        validate(_IncompatibleKeys(["encoder.conv_stack.0.weight"], []))


def test_unapproved_unexpected_key_raises() -> None:
    """An unexpected key not in the allowlist raises RuntimeError."""
    validate = _fn()
    with pytest.raises(RuntimeError, match="unexpected=\\['unknown.layer.bias'\\]"):
        validate(_IncompatibleKeys([], ["unknown.layer.bias"]))


def test_mixed_allowlisted_and_bad_raises(monkeypatch) -> None:
    """When bad keys coexist with allowlisted keys, only bad keys appear in the error."""
    from plugins.extract.align._orformer import model as m

    monkeypatch.setattr(m, "_ALLOWED_ORFORMER_MISSING", frozenset({"training.loss_scale"}))
    monkeypatch.setattr(m, "_ALLOWED_ORFORMER_UNEXPECTED", frozenset())
    validate = _fn()
    with pytest.raises(RuntimeError) as exc_info:
        validate(
            _IncompatibleKeys(
                ["training.loss_scale", "encoder.conv_stack.0.weight"],
                ["rogue.extra.bias"],
            )
        )
    msg = str(exc_info.value)
    assert "encoder.conv_stack.0.weight" in msg
    assert "rogue.extra.bias" in msg
    assert "training.loss_scale" not in msg


def test_error_message_includes_update_instructions() -> None:
    """RuntimeError text instructs the developer to update the allowlists."""
    validate = _fn()
    with pytest.raises(RuntimeError, match="_ALLOWED_ORFORMER"):
        validate(_IncompatibleKeys(["some.bad.key"], []))


# ---------------------------------------------------------------------------
# Weight-backed integration tests
# These load the official pinned checkpoints and confirm the allowlists remain
# correct.  They are skipped when the weights are not in the local cache.
# ---------------------------------------------------------------------------


def _load_vqvae_result(weights_path: Path, num_edges: int):
    """Load a VQVAE state dict and return the IncompatibleKeys result."""
    import torch

    from plugins.extract.align._orformer.model import VQVAE, ORFormerTokens

    vit = ORFormerTokens()
    model = VQVAE(output_dim=num_edges, vit=vit)
    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    return model.load_state_dict(state, strict=False)


@pytest.mark.skipif(
    not _WFLW_VQVAE.exists(), reason="Official WFLW VQ-VAE weights not in .fs_cache"
)
def test_official_wflw_vqvae_loads_with_zero_mismatch() -> None:
    """Official WFLW VQ-VAE checkpoint produces no missing or unexpected keys.

    Confirms that _ALLOWED_ORFORMER_MISSING and _ALLOWED_ORFORMER_UNEXPECTED
    being empty frozensets is correct for the pinned WFLW weights
    (sha256 prefix: 77b1646c).
    """
    result = _load_vqvae_result(_WFLW_VQVAE, num_edges=15)
    assert result.missing_keys == [], (
        f"WFLW VQ-VAE has unexpected missing keys: {result.missing_keys}. "
        "Update _ALLOWED_ORFORMER_MISSING if these are confirmed training-only."
    )
    assert result.unexpected_keys == [], (
        f"WFLW VQ-VAE has unexpected extra keys: {result.unexpected_keys}. "
        "Update _ALLOWED_ORFORMER_UNEXPECTED if these are confirmed training-only."
    )


@pytest.mark.skipif(
    not _W300_VQVAE.exists(), reason="Official 300W VQ-VAE weights not in .fs_cache"
)
def test_official_300w_vqvae_loads_with_zero_mismatch() -> None:
    """Official 300W VQ-VAE checkpoint produces no missing or unexpected keys.

    Confirms that _ALLOWED_ORFORMER_MISSING and _ALLOWED_ORFORMER_UNEXPECTED
    being empty frozensets is correct for the pinned 300W weights
    (sha256 prefix: 147b16fc).
    """
    result = _load_vqvae_result(_W300_VQVAE, num_edges=13)
    assert result.missing_keys == [], (
        f"300W VQ-VAE has unexpected missing keys: {result.missing_keys}. "
        "Update _ALLOWED_ORFORMER_MISSING if these are confirmed training-only."
    )
    assert result.unexpected_keys == [], (
        f"300W VQ-VAE has unexpected extra keys: {result.unexpected_keys}. "
        "Update _ALLOWED_ORFORMER_UNEXPECTED if these are confirmed training-only."
    )
