#!/usr/bin/env python3
"""Tests for manual tool aligner plugin discovery."""

from tools.manual import aligners


def test_available_aligners_uses_all_align_plugins(monkeypatch) -> None:
    """Manual aligner labels are generated from plugin discovery."""
    monkeypatch.setattr(
        aligners.PluginLoader,
        "get_available_extractors",
        lambda plugin_type: ["cv2-dnn", "ensemble", "fan", "hrnet", "orformer", "spiga"],
    )

    assert aligners.available_aligners() == [
        "cv2-dnn",
        "Ensemble",
        "FAN",
        "HRNet",
        "ORFormer",
        "SPIGA",
    ]
    assert aligners.default_aligner() == "HRNet"
