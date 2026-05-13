#!/usr/bin/env python3
"""Summary metrics helpers for GUI Analysis session rows."""

from __future__ import annotations

import typing as T
from dataclasses import dataclass

from lib.utils import get_module_objects

from .analysis_session_service import AnalysisSessionSource, AnalysisTableRow


@dataclass(frozen=True)
class AnalysisSummaryMetrics:
    """Display-ready metrics for Analysis summary rows."""

    source: AnalysisSessionSource | None
    rows: tuple[AnalysisTableRow, ...]
    is_loaded: bool
    is_training: bool

    @property
    def row_count(self) -> int:
        """Return number of loaded summary rows."""
        return len(self.rows)

    @property
    def graph_count(self) -> int:
        """Return number of summary rows that have graphable data."""
        return sum(1 for row in self.rows if row.graph_available)

    @property
    def total_iterations(self) -> int:
        """Return total parsed iterations across all rows."""
        return sum(_to_int(row.iterations) for row in self.rows)

    @property
    def has_rows(self) -> bool:
        """Return whether any rows are present."""
        return bool(self.rows)

    @property
    def status_text(self) -> str:
        """Return concise status text for Analysis panel display."""
        if not self.is_loaded:
            return "No session data loaded"
        if not self.rows:
            return "Session loaded with no summary rows"
        row_word = "row" if self.row_count == 1 else "rows"
        graph_word = "graph" if self.graph_count == 1 else "graphs"
        prefix = "Training session" if self.is_training else "Loaded session"
        return (
            f"{prefix}: {self.row_count} {row_word}, "
            f"{self.graph_count} {graph_word}, "
            f"{self.total_iterations} iterations"
        )


class AnalysisSummaryService:
    """Build display metrics from an AnalysisSessionService instance."""

    @staticmethod
    def from_session(session_service: T.Any) -> AnalysisSummaryMetrics:
        """Return summary metrics for the provided Analysis session service."""
        return AnalysisSummaryMetrics(
            source=session_service.source,
            rows=session_service.table_rows,
            is_loaded=bool(session_service.is_loaded),
            is_training=bool(getattr(session_service, "is_training", False)),
        )



def _to_int(value: T.Any) -> int:
    """Best-effort conversion of legacy summary values to integers."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        normalized = value.replace(",", "").strip()
        return int(normalized) if normalized.isdigit() else 0
    return 0


__all__ = get_module_objects(__name__)
