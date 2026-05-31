#!/usr/bin/env python3
"""Tests for the FaceQA --deep-analysis (DECA) tool wiring.

Verifies the CLI surface and the ``Faceqa._run_deep_analysis`` orchestration
hook in isolation (synthetic encoder + stub loader), without the DECA weights.
"""

from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest

import lib.faceqa.deep.audit as audit_mod
from lib.faceqa.deep.deca_encoder import DECA_PARAM_DIM
from lib.faceqa.readiness import ReadinessReport
from lib.utils import FaceswapError
from tools.faceqa.cli import FaceqaArgs
from tools.faceqa.faceqa import Faceqa


class _Entry:
    def __init__(self, n_faces: int) -> None:
        self.faces = [object() for _ in range(n_faces)]


class _Loader:
    def load_image(self, frame: str) -> np.ndarray:
        return np.zeros((64, 64, 3), dtype=np.uint8)


class _SynthEncoder:
    def encode(self, crops: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(len(crops))
        return rng.normal(size=(len(crops), DECA_PARAM_DIM)).astype(np.float32)


class _FakeExtractor:
    def __init__(self, loader, **_kwargs) -> None:
        self.disabled_reason = None

    def crop(self, face, frame, face_index):
        return np.zeros((224, 224, 3), dtype=np.uint8)


def _entries() -> dict[str, _Entry]:
    return {f"f{i}.png": _Entry(2) for i in range(10)}


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_exposes_deep_analysis_flag() -> None:
    args = {a["dest"]: a for a in FaceqaArgs.get_argument_list()}
    assert args["deep_analysis"]["choices"] == ("none", "deca")
    assert args["deep_analysis"]["default"] == "none"
    assert "deca_weights" not in args


# ---------------------------------------------------------------------------
# Orchestration hook
# ---------------------------------------------------------------------------


def test_deep_analysis_none_is_noop() -> None:
    faceqa = Faceqa(Namespace(deep_analysis="none"))
    report = ReadinessReport()
    assert faceqa._run_deep_analysis(_entries(), report, total_faces=20) is None
    assert report.deep_audit == {}


def test_deep_analysis_deca_populates_report(monkeypatch) -> None:
    monkeypatch.setattr(audit_mod, "DecaCropExtractor", _FakeExtractor)
    monkeypatch.setattr(audit_mod, "build_encoder", lambda **_k: _SynthEncoder())

    faceqa = Faceqa(Namespace(deep_analysis="deca"))
    monkeypatch.setattr(faceqa, "_frames_loader", lambda: _Loader())

    report = ReadinessReport()
    report.readiness_scores = {
        "overall_readiness_score": 40.0,
        "components": {
            "expression": {
                "score": 30.0,
                "weight": 0.25,
                "entropy_coverage": 0.3,
                "occupied_coverage": 0.4,
            },
            "lighting": {
                "score": 50.0,
                "weight": 0.20,
                "entropy_coverage": 0.5,
                "occupied_coverage": 0.6,
            },
        },
    }
    deep_pruning_signals = faceqa._run_deep_analysis(_entries(), report, total_faces=20)

    assert report.deep_audit["mode"] == "deca"
    assert report.deep_audit["faces_encoded"] == 20
    assert deep_pruning_signals is not None
    assert len(deep_pruning_signals) == 20
    assert report.deep_audit["status"] == "ok"
    assert "deca_readiness" in report.deep_audit
    assert report.readiness_scores["landmark_readiness_score"] == 40.0
    assert report.readiness_scores["deep_analysis_mode"] == "deca"
    assert "deca_deep" in report.readiness_scores["components"]
    assert report.readiness_scores["overall_readiness_score"] != 40.0
    assert (
        report.deep_audit["landmark_vs_deca"]["expression"]["landmark"]["entropy_coverage"] == 0.3
    )
    markdown = report.to_markdown()
    assert "## DECA Deep Audit" in markdown
    assert "### Landmark vs DECA" in markdown


def test_deep_analysis_deca_requires_frames_loader(monkeypatch) -> None:
    monkeypatch.setattr(audit_mod, "build_encoder", lambda **_k: _SynthEncoder())
    faceqa = Faceqa(Namespace(deep_analysis="deca"))
    monkeypatch.setattr(faceqa, "_frames_loader", lambda: None)
    with pytest.raises(FaceswapError, match="frames-dir"):
        faceqa._run_deep_analysis(_entries(), ReadinessReport(), total_faces=20)


def test_deep_analysis_deca_propagates_missing_weights(monkeypatch) -> None:
    """A missing-weights FaceswapError from build_encoder must surface."""

    def _raise(*_a, **_k):
        raise FaceswapError("failed to download deca_model.tar")

    monkeypatch.setattr(audit_mod, "build_encoder", _raise)
    faceqa = Faceqa(Namespace(deep_analysis="deca"))
    monkeypatch.setattr(faceqa, "_frames_loader", lambda: _Loader())
    with pytest.raises(FaceswapError, match="deca_model.tar"):
        faceqa._run_deep_analysis(_entries(), ReadinessReport(), total_faces=20)
