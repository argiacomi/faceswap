#!/usr/bin/env python3
"""Regression tests for landmark artifact path conventions."""

from __future__ import annotations

from lib.landmarks.pipeline_conventions import (
    BEST_SETUP_FILENAME,
    BEST_WEIGHTS_FILENAME,
    CANDIDATE_RESULTS_CSV,
    CANDIDATE_RESULTS_JSON,
    PRODUCTION_PER_BUCKET_CSV,
    PRODUCTION_POLICY_FAILURES_CSV,
    PRODUCTION_PROMOTION_REPORT_JSON,
    PRODUCTION_PROMOTION_REPORT_MD,
    PRODUCTION_WORST_SAMPLES_JSON,
    STATIC_WEIGHTS_FILENAME,
)
from tools.landmarks import production_promotion_gate


def test_promoted_search_outputs_use_shared_filenames() -> None:
    assert STATIC_WEIGHTS_FILENAME == "static_landmark_weights.json"
    assert BEST_SETUP_FILENAME == "best_setup.json"
    assert BEST_WEIGHTS_FILENAME == "best_weights.json"
    assert CANDIDATE_RESULTS_CSV == "candidate_results.csv"
    assert CANDIDATE_RESULTS_JSON == "candidate_results.json"


def test_production_gate_outputs_use_shared_filenames() -> None:
    assert production_promotion_gate.PRODUCTION_REPORT_JSON == PRODUCTION_PROMOTION_REPORT_JSON
    assert production_promotion_gate.PRODUCTION_REPORT_MD == PRODUCTION_PROMOTION_REPORT_MD
    assert production_promotion_gate.PRODUCTION_PER_BUCKET_CSV == PRODUCTION_PER_BUCKET_CSV
    assert (
        production_promotion_gate.PRODUCTION_POLICY_FAILURES_CSV == PRODUCTION_POLICY_FAILURES_CSV
    )
    assert production_promotion_gate.PRODUCTION_WORST_SAMPLES_JSON == PRODUCTION_WORST_SAMPLES_JSON
