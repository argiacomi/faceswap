#!/usr/bin/env python3
"""Tests for static weight pipeline progress/final reporting helpers."""

from __future__ import annotations

from tools.landmarks.run_static_weight_pipeline import _format_run_report


def test_format_run_report_summarizes_key_results() -> None:
    """Final report includes dataset/cache counts, metrics, findings, and paths."""
    report = _format_run_report(
        {
            "output_root": "outputs/run",
            "dataset_counts": {"total": 4, "wflw": 2, "cofw": 2},
            "cache_counts": {
                "samples": 4,
                "predictions": 12,
                "models": {"hrnet": 4, "spiga": 4, "orformer": 4},
            },
            "bbox_source": "manifest",
            "baseline_best_model": "spiga",
            "baseline_best_single_metrics": {
                "overall_nme": 0.040,
                "failure_rate": 0.05,
                "auc": 0.50,
            },
            "weighted_best_variant": "static_weighted_downweight",
            "weighted_best_variant_metrics": {
                "overall_nme": 0.036,
                "failure_rate": 0.03,
                "auc": 0.56,
            },
            "ensemble_deltas_vs_best_single": {
                "static_weighted_downweight": -0.004,
            },
            "ensemble_improved_over_best_single": True,
            "threshold_failed": False,
            "generated_weight_path": "outputs/run/weights/static_landmark_weights.json",
        },
        exit_code=0,
    )

    assert "Pipeline COMPLETE" in report
    assert "Datasets: total=4" in report
    assert "wflw=2" in report
    assert "Prediction cache: samples=4, predictions=12" in report
    assert "Best single (spiga): NME 4.00%" in report
    assert "Best weighted (static_weighted_downweight): NME 3.60%" in report
    assert "Weighted delta vs best single: -0.400 pp NME" in report
    assert "Finding: ensemble improved over best single" in report
    assert "Weights: outputs/run/weights/static_landmark_weights.json" in report


def test_format_run_report_surfaces_failure() -> None:
    """Failed reports include the failing exception string."""
    report = _format_run_report(
        {
            "output_root": "outputs/run",
            "dataset_counts": {},
            "cache_counts": {},
            "bbox_source": "faceswap-detector",
            "failed_stage_error": "ValueError: bad source",
        },
        exit_code=1,
    )

    assert "Pipeline FAILED" in report
    assert "Failure: ValueError: bad source" in report
    assert "BBox source: faceswap-detector" in report
