#!/usr/bin/env python3
"""Tests for the partial-schema 39-point profile specialist trainer."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.ensemble.profile39_dataset import Profile39Row
from lib.landmarks.ensemble.profile39_training import (
    PROFILE39_METRICS_JSON,
    PROFILE39_SCORER_ARTIFACT,
    grouped_profile39_rows,
    train_profile39_specialist,
)


def _install_fake_lightgbm(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeBooster:
        def __init__(self, feature_count: int = 1) -> None:
            self.feature_count = feature_count

        def feature_importance(self, *, importance_type: str = "gain") -> np.ndarray:
            return np.arange(self.feature_count, 0, -1, dtype="float64")

        def model_to_string(self) -> str:
            return "fake-profile39-model"

    class FakeRanker:
        def __init__(self, **params: object) -> None:
            self.params = params
            self.booster_ = FakeBooster()

        def fit(self, matrix, labels, *, group, **_kwargs):  # noqa: ANN001
            assert sum(group) == matrix.shape[0] == labels.shape[0]
            self.booster_ = FakeBooster(feature_count=matrix.shape[1])
            return self

    module = types.ModuleType("lightgbm")
    module.LGBMRanker = FakeRanker  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lightgbm", module)


def _row(sample_id: str, name: str, regret: float, *, oracle: str) -> Profile39Row:
    return Profile39Row(
        sample_id=sample_id,
        dataset="multipie",
        side="left",
        candidate_name=name,
        profile39_transform_cost=regret,
        profile39_transform_regret=regret,
        profile39_visible_side_error=regret,
        profile39_oracle_candidate=oracle,
        profile39_is_oracle=name == oracle,
        profile39_rankable=True,
        feature_values={"candidate_is_fusion": 1.0 if "avg" in name else 0.0, f"name={name}": 1.0},
    )


def _rows() -> list[Profile39Row]:
    rows: list[Profile39Row] = []
    for sample_id in ("s0", "s1"):
        rows.append(_row(sample_id, "hrnet", 0.0, oracle="hrnet"))
        rows.append(_row(sample_id, "spiga", 0.2, oracle="hrnet"))
        rows.append(_row(sample_id, "plain_avg", 0.4, oracle="hrnet"))
    return rows


def test_grouped_rows_filter_single_and_flat() -> None:
    rows = [
        _row("solo", "hrnet", 0.0, oracle="hrnet"),  # single-candidate group -> dropped
        _row("flat", "hrnet", 0.1, oracle="hrnet"),
        _row("flat", "spiga", 0.1, oracle="hrnet"),  # flat regret -> dropped
        *_rows(),
    ]
    groups, stats = grouped_profile39_rows(rows)
    assert stats["rankable_group_count"] == 2  # only s0 and s1
    assert stats["single_candidate_group_count"] == 1
    assert stats["flat_regret_group_count"] == 1
    assert all(len(group) >= 2 for group in groups)


def test_train_profile39_specialist_writes_artifact_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_lightgbm(monkeypatch)
    metrics = train_profile39_specialist(
        _rows(), output_dir=tmp_path, candidates=("hrnet", "spiga", "plain_avg"), iterations=4
    )
    artifact_path = tmp_path / PROFILE39_SCORER_ARTIFACT
    assert artifact_path.is_file()
    assert (tmp_path / PROFILE39_METRICS_JSON).is_file()

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["runtime_policy"] == "learned_quality_v3_profile"
    assert artifact["target"] == "profile39_transform_regret"
    assert artifact["schema"] == "2d_39_partial"
    assert artifact["features"]

    assert metrics["objective"] == "lambdarank_profile39_transform_regret"
    assert metrics["profile39_report"]["sample_count"] == 2
    assert metrics["profile39_report"]["oracle_row_count"] == 2


def test_train_raises_without_rankable_groups(tmp_path: Path) -> None:
    flat = [
        _row("flat", "hrnet", 0.1, oracle="hrnet"),
        _row("flat", "spiga", 0.1, oracle="hrnet"),
    ]
    with pytest.raises(ValueError):
        train_profile39_specialist(flat, output_dir=tmp_path, candidates=("hrnet", "spiga"))
