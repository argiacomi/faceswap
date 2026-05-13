#!/usr/bin/env python3
"""Tests for Analysis summary metrics helpers."""

from __future__ import annotations

from pathlib import Path

from lib.gui.services.analysis_session_service import AnalysisSessionSource, AnalysisTableRow
from lib.gui.services.analysis_summary_service import (
    AnalysisSummaryMetrics,
    AnalysisSummaryService,
)


class _SessionServiceDouble:
    """Small AnalysisSessionService stand-in."""

    def __init__(self) -> None:
        self.source = AnalysisSessionSource(
            state_file=Path("/models/model_state.json"),
            model_dir=Path("/models"),
            model_name="model",
        )
        self.table_rows = (
            AnalysisTableRow(True, 1, "start", "end", "1h", 4, "1,000", 12.5),
            AnalysisTableRow(False, 2, "start", "end", "1h", 0, 0, 0),
        )
        self.is_loaded = True
        self.is_training = False


def test_analysis_summary_metrics_counts_rows_graphs_and_iterations() -> None:
    """Metrics should summarize rows, graph availability and iteration totals."""
    metrics = AnalysisSummaryMetrics(
        source=None,
        rows=(
            AnalysisTableRow(True, 1, "", "", "", 4, "1,000", ""),
            AnalysisTableRow(False, 2, "", "", "", 0, 250, ""),
        ),
        is_loaded=True,
        is_training=False,
    )

    assert metrics.row_count == 2
    assert metrics.graph_count == 1
    assert metrics.total_iterations == 1250
    assert metrics.has_rows is True
    assert metrics.status_text == "Loaded session: 2 rows, 1 graph, 1250 iterations"


def test_analysis_summary_metrics_marks_training_sessions() -> None:
    """Status text should distinguish active training sessions."""
    metrics = AnalysisSummaryMetrics(
        source=None,
        rows=(AnalysisTableRow(True, 1, "", "", "", 1, 10, ""),),
        is_loaded=True,
        is_training=True,
    )

    assert metrics.status_text == "Training session: 1 row, 1 graph, 10 iterations"


def test_analysis_summary_metrics_empty_states() -> None:
    """Status text should distinguish unloaded and empty-loaded states."""
    unloaded = AnalysisSummaryMetrics(None, (), is_loaded=False, is_training=False)
    loaded_empty = AnalysisSummaryMetrics(None, (), is_loaded=True, is_training=False)

    assert unloaded.status_text == "No session data loaded"
    assert loaded_empty.status_text == "Session loaded with no summary rows"


def test_analysis_summary_service_builds_metrics_from_session_service() -> None:
    """AnalysisSummaryService should adapt an AnalysisSessionService-like object."""
    session_service = _SessionServiceDouble()

    metrics = AnalysisSummaryService.from_session(session_service)

    assert metrics.source == session_service.source
    assert metrics.rows == session_service.table_rows
    assert metrics.is_loaded is True
    assert metrics.is_training is False
    assert metrics.status_text == "Loaded session: 2 rows, 1 graph, 1000 iterations"
