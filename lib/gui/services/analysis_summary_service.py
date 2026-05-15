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
    def total_batch(self) -> int:
        """Return total parsed batch values across all rows."""
        return sum(_to_int(row.batch) for row in self.rows)

    @property
    def average_rate(self) -> float:
        """Return average parsed examples-per-second for rows with numeric rate values."""
        rates = [rate for rate in (_to_float(row.rate) for row in self.rows) if rate is not None]
        return 0.0 if not rates else sum(rates) / len(rates)

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

    @property
    def detail_text(self) -> str:
        """Return richer display metrics for the selected filter/group view."""
        if not self.rows:
            return "Rows: 0 | Graphs: 0 | Iterations: 0 | Avg EGs/sec: 0.00"
        return (
            f"Rows: {self.row_count} | Graphs: {self.graph_count} | "
            f"Iterations: {self.total_iterations} | "
            f"Avg EGs/sec: {self.average_rate:.2f}"
        )


class AnalysisSummaryService:
    """Build display metrics from an AnalysisSessionService instance."""

    FILTER_ALL = "All sessions"
    FILTER_GRAPHABLE = "Graphable only"
    FILTER_TOTAL = "Totals only"
    FILTER_SESSIONS = "Sessions only"

    GROUP_NONE = "None"
    GROUP_GRAPHABLE = "Graph availability"
    GROUP_TOTAL = "Total vs sessions"

    FILTERS = (FILTER_ALL, FILTER_GRAPHABLE, FILTER_TOTAL, FILTER_SESSIONS)
    GROUPS = (GROUP_NONE, GROUP_GRAPHABLE, GROUP_TOTAL)

    @classmethod
    def from_session(
        cls,
        session_service: T.Any,
        *,
        rows: tuple[AnalysisTableRow, ...] | None = None,
    ) -> AnalysisSummaryMetrics:
        """Return summary metrics for the provided Analysis session service."""
        selected_rows = session_service.table_rows if rows is None else rows
        return AnalysisSummaryMetrics(
            source=session_service.source,
            rows=tuple(selected_rows),
            is_loaded=bool(session_service.is_loaded),
            is_training=bool(getattr(session_service, "is_training", False)),
        )

    @classmethod
    def filtered_rows(
        cls,
        rows: tuple[AnalysisTableRow, ...],
        filter_name: str,
    ) -> tuple[AnalysisTableRow, ...]:
        """Return rows filtered to match the selected Analysis view."""
        if filter_name == cls.FILTER_GRAPHABLE:
            return tuple(row for row in rows if row.graph_available)
        if filter_name == cls.FILTER_TOTAL:
            return tuple(row for row in rows if row.is_total)
        if filter_name == cls.FILTER_SESSIONS:
            return tuple(row for row in rows if not row.is_total)
        return rows

    @classmethod
    def grouped_rows(
        cls,
        rows: tuple[AnalysisTableRow, ...],
        group_name: str,
    ) -> tuple[AnalysisTableRow, ...]:
        """Return rows ordered to match the selected grouping."""
        if group_name == cls.GROUP_GRAPHABLE:
            return tuple(sorted(rows, key=lambda row: (not row.graph_available, str(row.session))))
        if group_name == cls.GROUP_TOTAL:
            return tuple(sorted(rows, key=lambda row: (not row.is_total, str(row.session))))
        return rows

    @classmethod
    def display_rows(
        cls,
        rows: tuple[AnalysisTableRow, ...],
        *,
        filter_name: str = FILTER_ALL,
        group_name: str = GROUP_NONE,
    ) -> tuple[AnalysisTableRow, ...]:
        """Return rows after applying the selected filter then grouping."""
        return cls.grouped_rows(cls.filtered_rows(rows, filter_name), group_name)

    @staticmethod
    def row_detail(row: AnalysisTableRow | None) -> str:
        """Return a selected-row detail line matching Tk's selected session behavior."""
        if row is None:
            return "No session selected"
        graph = "graph available" if row.graph_available else "no graph data"
        return (
            f"Selected session {row.session}: {row.iterations} iterations, "
            f"batch {row.batch}, {graph}"
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


def _to_float(value: T.Any) -> float | None:
    """Best-effort conversion of legacy numeric rate values."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        normalized = value.replace(",", "").strip()
        try:
            return float(normalized)
        except ValueError:
            return None
    return None


__all__ = get_module_objects(__name__)
