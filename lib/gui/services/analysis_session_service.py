#!/usr/bin/env python3
"""Service helpers for loading and summarizing Analysis sessions."""

from __future__ import annotations

import csv
import typing as T
from dataclasses import dataclass
from pathlib import Path

from lib.utils import get_module_objects


class AnalysisSessionError(ValueError):
    """Raised when analysis session data cannot be resolved or loaded."""


class AnalysisSessionProtocol(T.Protocol):
    """Small protocol for the legacy ``lib.gui.analysis.Session`` singleton."""

    @property
    def is_loaded(self) -> bool:
        """Return whether a session is currently loaded."""

    @property
    def is_training(self) -> bool:
        """Return whether the loaded session is currently training."""

    @property
    def full_summary(self) -> list[dict[str, T.Any]]:
        """Return formatted summary dictionaries for the loaded session."""

    def initialize_session(
        self, model_folder: str, model_name: str, is_training: bool = False
    ) -> None:
        """Initialize session state for a model folder/name pair."""

    def clear(self) -> None:
        """Clear the currently loaded session."""


@dataclass(frozen=True)
class AnalysisSessionSource:
    """Resolved model state file details for an Analysis session."""

    state_file: Path
    model_dir: Path
    model_name: str


@dataclass(frozen=True)
class AnalysisTableRow:
    """One Analysis summary row shaped for a Qt table model or widget."""

    graph_available: bool
    session: T.Any
    start: T.Any
    end: T.Any
    elapsed: T.Any
    batch: T.Any
    iterations: T.Any
    rate: T.Any

    @property
    def values(self) -> tuple[T.Any, ...]:
        """Return row values in ``AnalysisSessionService.TABLE_HEADERS`` order."""
        return (
            self.graph_available,
            self.session,
            self.start,
            self.end,
            self.elapsed,
            self.batch,
            self.iterations,
            self.rate,
        )

    @classmethod
    def from_summary(cls, summary: T.Mapping[str, T.Any]) -> AnalysisTableRow:
        """Create a table row from a legacy Analysis summary dictionary."""
        return cls(
            graph_available=_has_graph_data(summary.get("batch"), summary.get("iterations")),
            session=summary.get("session", ""),
            start=summary.get("start", ""),
            end=summary.get("end", ""),
            elapsed=summary.get("elapsed", ""),
            batch=summary.get("batch", ""),
            iterations=summary.get("iterations", ""),
            rate=summary.get("rate", ""),
        )


class AnalysisSessionService:
    """Service-first API for Analysis session summary workflows.

    The default session adapter is the legacy ``lib.gui.analysis.Session`` singleton. Tests and
    alternate shells can inject a small compatible object to avoid loading real TensorBoard logs.
    """

    SUMMARY_COLUMNS = ("session", "start", "end", "elapsed", "batch", "iterations", "rate")
    TABLE_HEADERS = ("Graphs", "#", "Start", "End", "Elapsed", "Batch", "Iterations", "EGs/sec")
    STATE_SUFFIX = "_state.json"

    def __init__(
        self,
        session: AnalysisSessionProtocol | None = None,
        *,
        require_logs: bool = True,
    ) -> None:
        self._session = session
        self._require_logs = require_logs
        self._source: AnalysisSessionSource | None = None
        self._summary: list[dict[str, T.Any]] = []

    @property
    def source(self) -> AnalysisSessionSource | None:
        """Return the currently loaded source details."""
        return self._source

    @property
    def summary(self) -> tuple[dict[str, T.Any], ...]:
        """Return the currently loaded summary dictionaries."""
        return tuple(dict(row) for row in self._summary)

    @property
    def table_rows(self) -> tuple[AnalysisTableRow, ...]:
        """Return summary rows shaped for a Qt table."""
        return tuple(AnalysisTableRow.from_summary(row) for row in self._summary)

    @property
    def table_values(self) -> tuple[tuple[T.Any, ...], ...]:
        """Return raw table values in ``TABLE_HEADERS`` order."""
        return tuple(row.values for row in self.table_rows)

    @property
    def is_loaded(self) -> bool:
        """Return whether the service has loaded summary data."""
        session = self._session_or_default(import_default=False)
        return False if session is None else session.is_loaded

    @property
    def is_training(self) -> bool:
        """Return whether the loaded session is currently training."""
        session = self._session_or_default(import_default=False)
        return False if session is None else session.is_training

    def load_session(
        self, state_file_or_folder: str | Path, *, is_training: bool = False
    ) -> tuple[AnalysisTableRow, ...]:
        """Load Analysis data from a model state file or model folder.

        Parameters
        ----------
        state_file_or_folder: str or :class:`pathlib.Path`
            A ``*_state.json`` file, or a folder containing one.
        is_training: bool, optional
            Whether the loaded session is the currently running training session.
        """
        source = self.resolve_source(state_file_or_folder)
        self.clear_session()
        session = self._session_or_default(import_default=True)
        assert session is not None
        session.initialize_session(
            str(source.model_dir), source.model_name, is_training=is_training
        )
        self._source = source
        return self.refresh_summaries()

    def refresh_summaries(self) -> tuple[AnalysisTableRow, ...]:
        """Refresh cached summaries from the loaded session."""
        session = self._session_or_default(import_default=False)
        if session is None or not session.is_loaded:
            self._summary = []
            return ()
        self._summary = [dict(row) for row in session.full_summary]
        return self.table_rows

    def save_csv(self, filename: str | Path) -> int:
        """Save current summaries to a CSV file and return the number of rows written."""
        if not self._summary:
            return 0

        fieldnames = sorted(self._summary[0])
        with Path(filename).open("w", encoding="utf-8", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._summary)
        return len(self._summary)

    def clear_session(self) -> None:
        """Clear cached summaries and the injected legacy session."""
        self._summary = []
        self._source = None
        session = self._session_or_default(import_default=False)
        if session is not None and session.is_loaded:
            session.clear()

    def resolve_source(self, state_file_or_folder: str | Path) -> AnalysisSessionSource:
        """Resolve a model state file from a file path or folder."""
        path = Path(state_file_or_folder)
        if path.is_dir():
            path = self._state_file_from_folder(path)
        elif not path.is_file():
            raise AnalysisSessionError(f"Analysis source does not exist: {path}")

        if not path.name.endswith(self.STATE_SUFFIX):
            raise AnalysisSessionError(f"Analysis source is not a model state file: {path}")

        model_name = path.name[: -len(self.STATE_SUFFIX)]
        source = AnalysisSessionSource(
            state_file=path,
            model_dir=path.parent,
            model_name=model_name,
        )
        self._validate_logs(source)
        return source

    def _state_file_from_folder(self, folder: Path) -> Path:
        """Return the first model state file from a folder."""
        state_files = sorted(
            path for path in folder.glob(f"*{self.STATE_SUFFIX}") if path.is_file()
        )
        if not state_files:
            raise AnalysisSessionError(f"No model state file found in folder: {folder}")
        return state_files[0]

    def _validate_logs(self, source: AnalysisSessionSource) -> None:
        """Validate the TensorBoard log folder expected by the legacy Analysis session."""
        if not self._require_logs:
            return
        logs_dir = source.model_dir / f"{source.model_name}_logs"
        if not logs_dir.is_dir():
            raise AnalysisSessionError(f"No logs folder found for analysis session: {logs_dir}")

    def _session_or_default(self, *, import_default: bool) -> AnalysisSessionProtocol | None:
        """Return the injected Analysis session or lazy-load the legacy singleton."""
        if self._session is not None or not import_default:
            return self._session
        from lib.gui.analysis import Session

        self._session = Session
        return self._session


def _has_graph_data(batch: T.Any, iterations: T.Any) -> bool:
    """Return whether a summary row has enough data to graph."""
    return not (_is_zero(batch) or _is_zero(iterations))


def _is_zero(value: T.Any) -> bool:
    """Return whether a legacy summary value represents integer zero."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value == 0
    if isinstance(value, str) and value.isdigit():
        return int(value) == 0
    return False


__all__ = get_module_objects(__name__)
