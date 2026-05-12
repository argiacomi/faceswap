#!/usr/bin/env python3
"""Targeted ORFormer attention regressions."""

from __future__ import annotations

import torch

from plugins.extract.align._orformer.model import ORAttention


def test_orattention_avoids_bool_eye_mask(monkeypatch) -> None:
    """The attention mask should not request a boolean ``torch.eye`` inside forward."""
    attention = ORAttention(dim=8, heads=2, dim_head=4)
    inputs = torch.randn(1, 4, 8, dtype=torch.float32)
    or_query = torch.randn(1, 4, 8, dtype=torch.float32)
    original_eye = torch.eye

    def _guarded_eye(*args, **kwargs):
        assert kwargs.get("dtype") is not torch.bool
        return original_eye(*args, **kwargs)

    monkeypatch.setattr(torch, "eye", _guarded_eye)

    out, or_out = attention(inputs, or_query)

    assert out.shape == (1, 4, 8)
    assert or_out.shape == (1, 4, 8)
