#!/usr/bin/env python3
"""Regression tests for legacy landmark CLI path conventions."""

from __future__ import annotations

from tools.landmarks import compute_static_weights, failure_viewer, production_promotion_gate
from tools.landmarks.pipeline_conventions import (
    DEBUG_DIRNAME,
    FAILURE_ENSEMBLE_REGRESSIONS_CONTACT_SHEET,
    FAILURE_ENSEMBLE_REGRESSIONS_JSON,
    FAILURE_WORST_CASES_JSON,
    FAILURE_WORST_CONTACT_SHEET,
    PRODUCTION_PER_BUCKET_CSV,
    PRODUCTION_POLICY_FAILURES_CSV,
    PRODUCTION_PROMOTION_REPORT_JSON,
    PRODUCTION_PROMOTION_REPORT_MD,
    PRODUCTION_WORST_SAMPLES_JSON,
    STATIC_WEIGHTS_FILENAME,
)


def test_static_weights_default_output_uses_shared_filename() -> None:
    assert f"configs/ensemble/{STATIC_WEIGHTS_FILENAME}" == compute_static_weights.DEFAULT_OUTPUT


def test_failure_viewer_defaults_use_shared_debug_names() -> None:
    assert f"outputs/landmark_{DEBUG_DIRNAME}" == failure_viewer.DEFAULT_DEBUG_DIR
    assert failure_viewer.FAILURE_WORST_CASES_JSON == "worst_cases.json"
    assert failure_viewer.FAILURE_WORST_CASES_JSON == FAILURE_WORST_CASES_JSON
    assert failure_viewer.FAILURE_ENSEMBLE_REGRESSIONS_JSON == FAILURE_ENSEMBLE_REGRESSIONS_JSON
    assert failure_viewer.FAILURE_WORST_CONTACT_SHEET == FAILURE_WORST_CONTACT_SHEET
    assert (
        failure_viewer.FAILURE_ENSEMBLE_REGRESSIONS_CONTACT_SHEET
        == FAILURE_ENSEMBLE_REGRESSIONS_CONTACT_SHEET
    )


def test_production_gate_outputs_use_shared_filenames() -> None:
    assert production_promotion_gate.PRODUCTION_REPORT_JSON == PRODUCTION_PROMOTION_REPORT_JSON
    assert production_promotion_gate.PRODUCTION_REPORT_MD == PRODUCTION_PROMOTION_REPORT_MD
    assert production_promotion_gate.PRODUCTION_PER_BUCKET_CSV == PRODUCTION_PER_BUCKET_CSV
    assert (
        production_promotion_gate.PRODUCTION_POLICY_FAILURES_CSV == PRODUCTION_POLICY_FAILURES_CSV
    )
    assert production_promotion_gate.PRODUCTION_WORST_SAMPLES_JSON == PRODUCTION_WORST_SAMPLES_JSON
