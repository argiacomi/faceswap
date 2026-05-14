#!/usr/bin/env python3
"""Tests for scenario-bucket weighted landmark metrics."""

from __future__ import annotations

import pytest

from lib.landmarks.eval.harness import (
    _best_variant_summary,
    _summarize_bucket_weighted,
)


def test_scenario_bucket_weighted_nme_does_not_pool_by_sample_count() -> None:
    """Each dataset/condition bucket contributes equally to overall_nme."""
    metrics = _summarize_bucket_weighted(
        {
            "300w:common": {"model": [0.01] * 100},
            "wflw:pose": {"model": [0.09]},
        },
        failure_threshold=0.08,
    )

    assert metrics["model"]["count"] == 101.0
    assert metrics["model"]["scenario_bucket_count"] == 2.0
    assert metrics["model"]["overall_nme"] == pytest.approx(0.05)
    assert metrics["model"]["nme"] == pytest.approx(0.05)
    assert metrics["model"]["aggregation"] == "equal_weighted_scenario_buckets"


def test_scenario_bucket_weighted_handles_missing_labels_per_bucket() -> None:
    """Labels are averaged over only the scenario buckets where they exist."""
    metrics = _summarize_bucket_weighted(
        {
            "wflw:pose": {"hrnet": [0.10], "ensemble": [0.08]},
            "cofw:occlusion": {"hrnet": [0.04]},
        },
        failure_threshold=0.08,
    )

    assert metrics["hrnet"]["scenario_bucket_count"] == 2.0
    assert metrics["hrnet"]["overall_nme"] == pytest.approx(0.07)
    assert metrics["ensemble"]["scenario_bucket_count"] == 1.0
    assert metrics["ensemble"]["overall_nme"] == pytest.approx(0.08)


def test_best_variant_summary_exposes_scenario_bucket_diagnostics() -> None:
    """Best summary includes the dataset/condition buckets used for ranking."""
    scenario_buckets = {
        "300w:common": {
            "orformer": {"nme": 0.04, "failure_rate": 0.0, "auc": 0.6, "count": 100.0},
            "static_weighted": {"nme": 0.03, "failure_rate": 0.0, "auc": 0.7, "count": 100.0},
        },
        "wflw:pose": {
            "orformer": {"nme": 0.10, "failure_rate": 0.2, "auc": 0.2, "count": 10.0},
            "static_weighted": {"nme": 0.09, "failure_rate": 0.1, "auc": 0.3, "count": 10.0},
        },
    }
    overall_metrics = {
        "orformer": {"nme": 0.07, "failure_rate": 0.1, "auc": 0.4, "count": 110.0},
        "static_weighted": {"nme": 0.06, "failure_rate": 0.05, "auc": 0.5, "count": 110.0},
    }

    summary = _best_variant_summary(
        {"orformer": [0.04, 0.10], "static_weighted": [0.03, 0.09]},
        variants=("static_weighted",),
        failure_threshold=0.08,
        conditions={},
        scenario_buckets=scenario_buckets,
        threshold_failed=True,
        overall_metrics=overall_metrics,
    )

    assert summary["label"] == "static_weighted"
    assert summary["failure_rate_by_scenario_bucket"] == scenario_buckets
    assert summary["worst_scenario_buckets"][0]["scenario_bucket"] == "wflw:pose"
    assert summary["worst_best_single_scenario_buckets"][0]["scenario_bucket"] == "wflw:pose"
    assert summary["worst_best_variant_scenario_buckets"][0]["scenario_bucket"] == "wflw:pose"
