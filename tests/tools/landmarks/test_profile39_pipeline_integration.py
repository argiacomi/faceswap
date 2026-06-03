#!/usr/bin/env python3
"""Integration: profile39 args reach the suite, and the artifact loads at runtime."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

import tools.landmarks.train_runtime_resolver_scorer as cli
from lib.landmarks.ensemble.profile39_dataset import Profile39Row
from lib.landmarks.ensemble.profile39_training import (
    PROFILE39_SCORER_ARTIFACT,
    train_profile39_specialist,
)
from lib.landmarks.ensemble.runtime_resolver_scorer import load_runtime_resolver_scorer
from lib.landmarks.ensemble.weights import save_weights


def test_cli_passes_profile39_args_to_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def _fake_suite(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"artifact": str(tmp_path / "artifact.json")}

    monkeypatch.setattr(cli, "train_runtime_resolver_scorer_suite", _fake_suite)

    weights_path = tmp_path / "weights.json"
    save_weights(weights_path, {"hrnet": [1.0] * 68})
    profile39_manifest = tmp_path / "p39_manifest.json"
    profile39_manifest.write_text("{}", encoding="utf-8")
    profile39_cache = tmp_path / "p39_cache"

    rc = cli.main(
        [
            "--weights",
            str(weights_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--profile39-manifest",
            str(profile39_manifest),
            "--profile39-cache-dir",
            str(profile39_cache),
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0
    assert captured["profile39_manifest"] == profile39_manifest
    assert captured["profile39_cache_dir"] == profile39_cache


def _install_fake_lightgbm(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeBooster:
        def __init__(self, feature_count: int = 1) -> None:
            self.feature_count = feature_count

        def feature_importance(self, *, importance_type: str = "gain") -> np.ndarray:
            return np.arange(self.feature_count, 0, -1, dtype="float64")

        def model_to_string(self) -> str:
            return "tree\nfake-profile39\n"

    class FakeRanker:
        def __init__(self, **_params: object) -> None:
            self.booster_ = FakeBooster()

        def fit(self, matrix, labels, *, group, **_kwargs):  # noqa: ANN001
            self.booster_ = FakeBooster(feature_count=matrix.shape[1])
            return self

    module = types.ModuleType("lightgbm")
    module.LGBMRanker = FakeRanker  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lightgbm", module)


def _p39_rows() -> list[Profile39Row]:
    rows: list[Profile39Row] = []
    for sample_id in ("s0", "s1"):
        for name, regret in (("hrnet", 0.0), ("spiga", 0.2), ("plain_avg", 0.4)):
            rows.append(
                Profile39Row(
                    sample_id=sample_id,
                    dataset="multipie",
                    side="left",
                    candidate_name=name,
                    profile39_transform_cost=regret,
                    profile39_transform_regret=regret,
                    profile39_visible_side_error=regret,
                    profile39_oracle_candidate="hrnet",
                    profile39_is_oracle=name == "hrnet",
                    profile39_rankable=True,
                    feature_values={f"name={name}": 1.0, "candidate_is_fusion": 0.0},
                )
            )
    return rows


def test_produced_profile39_artifact_loads_via_runtime_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_lightgbm(monkeypatch)
    train_profile39_specialist(
        _p39_rows(),
        output_dir=tmp_path,
        candidates=("hrnet", "spiga", "plain_avg"),
        iterations=4,
    )
    scorer = load_runtime_resolver_scorer(tmp_path / PROFILE39_SCORER_ARTIFACT)
    assert scorer.runtime_policy == "learned_quality_v3_profile"
    assert scorer.version == "learned_quality_v3_profile"
    assert scorer.target == "profile39_transform_regret"
    assert scorer.features
