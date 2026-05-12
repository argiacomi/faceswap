#!/usr/bin/env python3
"""Tests for the Analysis session service."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from lib.gui.services.analysis_session_service import (
    AnalysisSessionError,
    AnalysisSessionService,
)


class _SessionDouble:
    """Small legacy Analysis Session test double."""

    def __init__(self, summary: list[dict[str, object]] | None = None) -> None:
        self.is_loaded = False
        self.is_training = False
        self.full_summary = [] if summary is None else summary
        self.initialized: list[tuple[str, str, bool]] = []
        self.clear_count = 0

    def initialize_session(
        self, model_folder: str, model_name: str, is_training: bool = False
    ) -> None:
        """Capture session initialization calls."""
        self.is_loaded = True
        self.is_training = is_training
        self.initialized.append((model_folder, model_name, is_training))

    def clear(self) -> None:
        """Capture clear calls."""
        self.is_loaded = False
        self.clear_count += 1


def _summary_row(
    session: object = 1,
    *,
    batch: object = 16,
    iterations: object = 100,
) -> dict[str, object]:
    """Return a complete legacy Analysis summary row."""
    return {
        "session": session,
        "start": "01/01/26 00:00:00",
        "end": "01/01/26 00:10:00",
        "elapsed": "00:10:00",
        "batch": batch,
        "iterations": iterations,
        "rate": "53.3",
    }


def test_load_session_from_state_file_uses_injected_session(tmp_path: Path) -> None:
    """A state file should initialize the injected session and cache table rows."""
    state_file = tmp_path / "model_state.json"
    state_file.write_text("{}", encoding="utf-8")
    session = _SessionDouble([_summary_row()])
    service = AnalysisSessionService(session, require_logs=False)

    rows = service.load_session(state_file)

    assert session.initialized == [(str(tmp_path), "model", False)]
    assert service.source is not None
    assert service.source.state_file == state_file
    assert service.summary == (_summary_row(),)
    assert rows[0].values == (
        True,
        1,
        "01/01/26 00:00:00",
        "01/01/26 00:10:00",
        "00:10:00",
        16,
        100,
        "53.3",
    )


def test_load_session_from_folder_uses_first_state_file(tmp_path: Path) -> None:
    """A folder source should resolve the first state file deterministically."""
    (tmp_path / "zeta_state.json").write_text("{}", encoding="utf-8")
    (tmp_path / "alpha_state.json").write_text("{}", encoding="utf-8")
    session = _SessionDouble([_summary_row()])
    service = AnalysisSessionService(session, require_logs=False)

    service.load_session(tmp_path, is_training=True)

    assert session.initialized == [(str(tmp_path), "alpha", True)]
    assert service.source is not None
    assert service.source.state_file == tmp_path / "alpha_state.json"


def test_load_requires_logs_by_default(tmp_path: Path) -> None:
    """Default behavior should match the legacy Analysis requirement for logs."""
    state_file = tmp_path / "model_state.json"
    state_file.write_text("{}", encoding="utf-8")
    service = AnalysisSessionService(_SessionDouble())

    with pytest.raises(AnalysisSessionError, match="No logs folder"):
        service.load_session(state_file)


def test_resolve_source_rejects_missing_or_invalid_sources(tmp_path: Path) -> None:
    """Missing files, empty folders and non-state files should be rejected."""
    service = AnalysisSessionService(_SessionDouble(), require_logs=False)
    invalid_file = tmp_path / "model.json"
    empty_folder = tmp_path / "empty"
    invalid_file.write_text("{}", encoding="utf-8")
    empty_folder.mkdir()

    with pytest.raises(AnalysisSessionError, match="does not exist"):
        service.resolve_source(tmp_path / "missing_state.json")
    with pytest.raises(AnalysisSessionError, match="No model state file"):
        service.resolve_source(empty_folder)
    with pytest.raises(AnalysisSessionError, match="not a model state file"):
        service.resolve_source(invalid_file)


def test_table_rows_mark_graph_availability() -> None:
    """Rows with zero batch or iterations should not be marked graphable."""
    session = _SessionDouble(
        [
            _summary_row(1),
            _summary_row(2, iterations=0),
            _summary_row("Total", batch="0"),
        ]
    )
    session.is_loaded = True
    service = AnalysisSessionService(
        session,
        require_logs=False,
    )

    rows = service.refresh_summaries()

    assert AnalysisSessionService.TABLE_HEADERS == (
        "Graphs",
        "#",
        "Start",
        "End",
        "Elapsed",
        "Batch",
        "Iterations",
        "EGs/sec",
    )
    assert [row.graph_available for row in rows] == [True, False, False]
    assert service.table_values[0][1:] == rows[0].values[1:]


def test_save_csv_writes_current_summary(tmp_path: Path) -> None:
    """CSV saving should preserve legacy sorted fieldname behavior."""
    output = tmp_path / "summary.csv"
    session = _SessionDouble([_summary_row()])
    session.is_loaded = True
    service = AnalysisSessionService(session, require_logs=False)
    service.refresh_summaries()

    written = service.save_csv(output)

    assert written == 1
    with output.open(encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        assert reader.fieldnames == sorted(_summary_row())
        assert list(reader) == [
            {
                "batch": "16",
                "elapsed": "00:10:00",
                "end": "01/01/26 00:10:00",
                "iterations": "100",
                "rate": "53.3",
                "session": "1",
                "start": "01/01/26 00:00:00",
            }
        ]


def test_save_csv_without_summary_is_noop(tmp_path: Path) -> None:
    """Saving with no summary should not create an empty CSV."""
    output = tmp_path / "summary.csv"
    service = AnalysisSessionService(_SessionDouble(), require_logs=False)

    assert service.save_csv(output) == 0
    assert output.exists() is False


def test_refresh_summaries_replaces_cached_rows() -> None:
    """Refreshing should pull the latest summary from the injected session."""
    session = _SessionDouble([_summary_row(1)])
    session.is_loaded = True
    service = AnalysisSessionService(session, require_logs=False)

    assert [row.session for row in service.refresh_summaries()] == [1]

    session.full_summary = [_summary_row(2)]

    assert [row.session for row in service.refresh_summaries()] == [2]


def test_clear_session_clears_cache_and_legacy_session(tmp_path: Path) -> None:
    """Clearing should reset cached data and delegate to the session when loaded."""
    state_file = tmp_path / "model_state.json"
    state_file.write_text("{}", encoding="utf-8")
    session = _SessionDouble([_summary_row()])
    service = AnalysisSessionService(session, require_logs=False)
    service.load_session(state_file)

    service.clear_session()

    assert session.clear_count == 1
    assert service.source is None
    assert service.summary == ()
    assert service.table_rows == ()
