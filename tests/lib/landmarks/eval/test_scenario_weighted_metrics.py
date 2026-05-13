#!/usr/bin/env python3
"""Tests for scenario-bucket weighted landmark metrics."""

from __future__ import annotations

import pytest

from lib.landmarks.eval.harness import _summarize_bucket_weighted


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
