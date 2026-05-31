#!/usr/bin/env python3
"""Tests for the DECA encoder param layout, torch module, and weights loader.

The pure-numpy ``decode_parameters`` layout and the state-dict remapper are
validated without the research weights; the torch module is exercised with
random weights (the real ``deca_model.tar`` path is validated on hardware per
the issue #156 checklist).
"""

from __future__ import annotations

import numpy as np
import pytest

from lib.faceqa.deep import deca_encoder as de
from lib.faceqa.deep import weights as w
from lib.utils import FaceswapError

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Parameter layout / decode
# ---------------------------------------------------------------------------


def test_param_dim_matches_layout() -> None:
    assert de.DECA_PARAM_DIM == 236
    assert sum(count for _, count in de.DECA_PARAM_LAYOUT) == de.DECA_PARAM_DIM


def test_decode_parameters_slices_at_canonical_offsets() -> None:
    params: np.ndarray = np.arange(de.DECA_PARAM_DIM, dtype=np.float32).reshape(1, -1)
    coef = de.decode_parameters(params)
    assert coef.expression.shape == (1, 50)
    assert coef.pose.shape == (1, 6)
    assert coef.light.shape == (1, 27)
    # exp at 150, pose at 200, light at 209 (shape 100 + tex 50, then +exp +pose).
    assert coef.expression[0, 0] == 150.0
    assert coef.pose[0, 0] == 200.0
    assert coef.light[0, 0] == 209.0


def test_decode_parameters_accepts_single_vector() -> None:
    coef = de.decode_parameters(np.zeros(de.DECA_PARAM_DIM, dtype=np.float32))
    assert len(coef) == 1


def test_decode_parameters_rejects_wrong_width() -> None:
    with pytest.raises(ValueError, match="DECA parameters"):
        de.decode_parameters(np.zeros((2, 10)))


# ---------------------------------------------------------------------------
# Torch encoder module
# ---------------------------------------------------------------------------


def test_build_module_forward_shape() -> None:
    module = de.TorchDecaEncoder.build_module()
    encoder = de.TorchDecaEncoder(module)
    crops: np.ndarray = np.zeros((3, 256, 256, 3), dtype=np.uint8)
    out = encoder.encode(crops)
    assert out.shape == (3, de.DECA_PARAM_DIM)
    assert out.dtype == np.float32


def test_encoder_empty_batch() -> None:
    encoder = de.TorchDecaEncoder(de.TorchDecaEncoder.build_module())
    assert encoder.encode(np.empty((0, 224, 224, 3), dtype=np.uint8)).shape == (
        0,
        de.DECA_PARAM_DIM,
    )


def test_encoder_satisfies_protocol() -> None:
    encoder = de.TorchDecaEncoder(de.TorchDecaEncoder.build_module())
    assert isinstance(encoder, de.DecaEncoder)


def test_encoder_rejects_non_rgb_crops() -> None:
    encoder = de.TorchDecaEncoder(de.TorchDecaEncoder.build_module())
    with pytest.raises(ValueError, match=r"\(n, H, W, 3\)"):
        encoder.encode(np.zeros((2, 224, 224), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Weights remapper / loader
# ---------------------------------------------------------------------------


def test_remap_state_dict_round_trips_all_keys() -> None:
    module = de.TorchDecaEncoder.build_module()
    state = module.state_dict()
    deca_style: dict[str, object] = {}
    for key, value in state.items():
        if key.startswith("backbone."):
            deca_style[f"encoder.{key[len('backbone.') :]}"] = value
        elif key.startswith("head."):
            deca_style[f"layers.{key[len('head.') :]}"] = value
        else:
            deca_style[key] = value
    remapped = w.remap_deca_state_dict(deca_style)
    assert set(remapped) == set(state)


def test_load_deca_encoder_missing_weights_raises_with_guidance(monkeypatch) -> None:
    class _MissingModel:
        def __init__(self, *_args, **_kwargs) -> None:
            raise FaceswapError("failed to download deca_model.tar")

    monkeypatch.setattr(w, "GetModelFromUrl", _MissingModel)
    with pytest.raises(FaceswapError, match="deca_model.tar"):
        w.load_deca_encoder()


def test_load_deca_encoder_from_checkpoint(tmp_path, monkeypatch) -> None:
    module = de.TorchDecaEncoder.build_module()
    state = module.state_dict()
    deca_style: dict[str, object] = {}
    for key, value in state.items():
        if key.startswith("backbone."):
            deca_style[f"encoder.{key[len('backbone.') :]}"] = value
        elif key.startswith("head."):
            deca_style[f"layers.{key[len('head.') :]}"] = value
        else:
            deca_style[key] = value
    checkpoint = tmp_path / "deca_model.tar"
    torch.save({"E_flame": deca_style, "E_detail": {"junk": torch.zeros(1)}}, checkpoint)

    monkeypatch.setattr(w, "resolve_weights_path", lambda: str(checkpoint))
    encoder = w.load_deca_encoder()
    out = encoder.encode(np.zeros((2, 224, 224, 3), dtype=np.uint8))
    assert out.shape == (2, de.DECA_PARAM_DIM)


def test_resolve_weights_path_uses_standard_model_cache(monkeypatch) -> None:
    class _CachedModel:
        def __init__(self, filename: str, url: str, sha256: str) -> None:
            assert filename == w.DECA_WEIGHTS_FILENAME
            assert w.DECA_WEIGHTS_FILE_ID in url
            assert sha256 == w.DECA_WEIGHTS_SHA256
            self.model_path = f"/cache/{filename}"

    monkeypatch.setattr(w, "GetModelFromUrl", _CachedModel)
    assert w.resolve_weights_path() == "/cache/deca_model.tar"
