#!/usr/bin/env python3
"""Tests for the FaceQA DECA deep-audit orchestration.

A synthetic encoder and a stub crop extractor drive ``run_deep_audit`` so the
full pipeline (encode -> decode -> metrics -> readiness -> comparison) is
validated without torch or the DECA weights. The real crop reconstruction is
covered by the existing coverage tests' alignment fixtures.
"""

from __future__ import annotations

import json

import numpy as np

from lib.faceqa.deep import audit as audit_mod
from lib.faceqa.deep.deca_encoder import DECA_PARAM_DIM


class _Entry:
    def __init__(self, n_faces: int) -> None:
        self.faces = [object() for _ in range(n_faces)]


class _Loader:
    def load_image(self, frame: str) -> np.ndarray:
        return np.zeros((64, 64, 3), dtype=np.uint8)


class _DiverseEncoder:
    """Returns well-spread coefficients regardless of input crop."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def encode(self, crops: np.ndarray) -> np.ndarray:
        return self._rng.normal(size=(len(crops), DECA_PARAM_DIM)).astype(np.float32)


class _CollapsedEncoder:
    """Returns near-identical coefficients (degenerate faceset)."""

    def __init__(self) -> None:
        self._point = np.random.default_rng(1).normal(size=(1, DECA_PARAM_DIM))

    def encode(self, crops: np.ndarray) -> np.ndarray:
        noise = np.random.default_rng(2).normal(scale=1e-3, size=(len(crops), DECA_PARAM_DIM))
        return (self._point + noise).astype(np.float32)


def _fake_extractor_factory(*, fail: bool = False, disabled: bool = False):
    rng = np.random.default_rng(7)

    class _FakeExtractor:
        def __init__(self, loader, **_kwargs) -> None:
            self.disabled_reason = "loader dead" if disabled else None

        def crop(self, face, frame, face_index):
            if fail or disabled:
                return None
            return rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint16).astype(np.uint8)

    return _FakeExtractor


def _entries(n_frames: int = 40, per_frame: int = 3) -> dict[str, _Entry]:
    return {f"frame_{i}.png": _Entry(per_frame) for i in range(n_frames)}


def _patch_extractor(monkeypatch, **kwargs) -> None:
    monkeypatch.setattr(audit_mod, "DecaCropExtractor", _fake_extractor_factory(**kwargs))


def test_run_deep_audit_happy_path(monkeypatch) -> None:
    _patch_extractor(monkeypatch)
    entries = _entries()
    scores = {
        "components": {
            "expression": {
                "entropy_coverage": 0.4,
                "occupied_coverage": 0.5,
                "classified_faces": 120,
            },
            "lighting": {
                "entropy_coverage": 0.6,
                "occupied_coverage": 0.7,
                "classified_faces": 120,
            },
        }
    }
    report = audit_mod.run_deep_audit(
        entries,
        encoder=_DiverseEncoder(),
        frames_loader=_Loader(),
        readiness_scores=scores,
        weights_path="/cache/deca_model.tar",
    )
    data = report.to_dict()

    assert data["status"] == "ok"
    assert data["faces_total"] == 120
    assert data["faces_encoded"] == 120
    assert data["faces_skipped"] == 0
    assert data["expression_space_coverage"]["samples"] == 120
    assert data["pose_space_coverage"]["samples"] == 120
    assert data["lighting_space_coverage"]["samples"] == 120
    assert 0.0 <= data["deca_readiness"]["score"] <= 100.0
    # Comparison carries both views with comparable keys.
    expr = data["landmark_vs_deca"]["expression"]
    assert expr["landmark"]["entropy_coverage"] == 0.4
    assert "entropy_coverage" in expr["deca"]
    # Fully JSON serializable (it lands in the coverage report).
    assert json.loads(json.dumps(data))["mode"] == "deca"


def test_run_deep_audit_diverse_scores_higher_than_collapsed(monkeypatch) -> None:
    _patch_extractor(monkeypatch)
    entries = _entries()
    diverse = audit_mod.run_deep_audit(entries, encoder=_DiverseEncoder(), frames_loader=_Loader())
    collapsed = audit_mod.run_deep_audit(
        entries, encoder=_CollapsedEncoder(), frames_loader=_Loader()
    )
    assert diverse.deca_readiness["score"] > collapsed.deca_readiness["score"]
    assert collapsed.cluster_coverage["mean_dispersion"] < 0.05
    assert collapsed.notes == [] or "skipped" not in " ".join(collapsed.notes)


def test_run_deep_audit_no_faces_encoded(monkeypatch) -> None:
    _patch_extractor(monkeypatch, disabled=True)
    report = audit_mod.run_deep_audit(
        _entries(), encoder=_DiverseEncoder(), frames_loader=_Loader()
    )
    assert report.status == "no_faces_encoded"
    assert report.faces_encoded == 0
    assert report.faces_skipped == 120
    assert report.skipped_reasons["frame_loader_disabled"] == 120


def test_run_deep_audit_partial_skips_are_tallied(monkeypatch) -> None:
    rng = np.random.default_rng(9)

    class _FlakyExtractor:
        def __init__(self, loader, **_kwargs) -> None:
            self.disabled_reason = None

        def crop(self, face, frame, face_index):
            if rng.random() < 0.5:
                return None
            return rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint16).astype(np.uint8)

    monkeypatch.setattr(audit_mod, "DecaCropExtractor", _FlakyExtractor)
    report = audit_mod.run_deep_audit(
        _entries(), encoder=_DiverseEncoder(), frames_loader=_Loader()
    )
    assert report.faces_encoded + report.faces_skipped == report.faces_total
    if report.faces_skipped:
        assert report.skipped_reasons.get("crop_failed", 0) > 0
        assert any("skipped" in note for note in report.notes)


def test_progress_callback_fires_per_face(monkeypatch) -> None:
    _patch_extractor(monkeypatch)
    ticks = []
    audit_mod.run_deep_audit(
        _entries(n_frames=5, per_frame=2),
        encoder=_DiverseEncoder(),
        frames_loader=_Loader(),
        progress_callback=lambda n=1: ticks.append(n),
    )
    assert sum(ticks) == 10
