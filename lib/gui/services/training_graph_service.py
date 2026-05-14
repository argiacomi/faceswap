#!/usr/bin/env python3
"""Service helpers for extracting graph series from Analysis sessions."""

from __future__ import annotations

import csv
import typing as T
from dataclasses import dataclass
from pathlib import Path

from lib.utils import get_module_objects


class TrainingGraphError(ValueError):
    """Raised when graph data cannot be resolved or loaded."""


class TrainingGraphSessionProtocol(T.Protocol):
    """Small protocol for the legacy ``lib.gui.analysis.Session`` singleton."""

    @property
    def is_loaded(self) -> bool:
        """Return whether a session is currently loaded."""

    @property
    def session_ids(self) -> list[int]:
        """Return available session ids."""

    def initialize_session(
        self, model_folder: str, model_name: str, is_training: bool = False
    ) -> None:
        """Initialize session state for a model folder/name pair."""

    def get_loss(self, session_id: int | None) -> dict[str, T.Any]:
        """Return loss arrays for a session or all sessions."""

    def get_loss_keys(self, session_id: int | None) -> list[str]:
        """Return available loss keys."""


@dataclass(frozen=True)
class TrainingGraphSource:
    """Resolved model source details for graph data."""

    model_dir: Path
    model_name: str

    @property
    def state_file(self) -> Path:
        """Return the expected model state file."""
        return self.model_dir / f"{self.model_name}_state.json"


@dataclass(frozen=True)
class TrainingGraphSeries:
    """One loss series shaped for a graph widget."""

    name: str
    values: tuple[float, ...]

    @property
    def count(self) -> int:
        """Return number of plotted points."""
        return len(self.values)

    @property
    def minimum(self) -> float | None:
        """Return the smallest finite value, if available."""
        return min(self.values) if self.values else None

    @property
    def maximum(self) -> float | None:
        """Return the largest finite value, if available."""
        return max(self.values) if self.values else None


@dataclass(frozen=True)
class TrainingGraphSnapshot:
    """A refreshed set of graph series."""

    source: TrainingGraphSource | None
    session_id: int | None
    series: tuple[TrainingGraphSeries, ...]

    @property
    def point_count(self) -> int:
        """Return the largest series length."""
        return max((item.count for item in self.series), default=0)

    @property
    def is_empty(self) -> bool:
        """Return whether there are no points to plot."""
        return self.point_count == 0

    def series_for_keys(self, selected_keys: tuple[str, ...] = ()) -> tuple[TrainingGraphSeries, ...]:
        """Return selected series or all series when no keys are supplied."""
        if not selected_keys:
            return self.series
        selected = set(selected_keys)
        return tuple(series for series in self.series if series.name in selected)


class TrainingGraphService:
    """Read loss graph data from the legacy Analysis session adapter."""

    STATE_SUFFIX = "_state.json"

    def __init__(self, session: TrainingGraphSessionProtocol | None = None) -> None:
        self._session = session
        self._source: TrainingGraphSource | None = None
        self._session_id: int | None = None
        self._loss_keys: tuple[str, ...] = ()
        self._snapshot = TrainingGraphSnapshot(None, None, ())

    @property
    def source(self) -> TrainingGraphSource | None:
        """Return the currently configured source."""
        return self._source

    @property
    def session_id(self) -> int | None:
        """Return the currently selected session id, or ``None`` for all sessions."""
        return self._session_id

    @property
    def session_ids(self) -> tuple[int, ...]:
        """Return available session ids from the loaded Analysis session."""
        session = self._session_or_default(import_default=False)
        return tuple(session.session_ids) if session is not None and session.is_loaded else ()

    @property
    def loss_keys(self) -> tuple[str, ...]:
        """Return available loss keys for the current session selection."""
        return self._loss_keys

    @property
    def snapshot(self) -> TrainingGraphSnapshot:
        """Return the latest graph snapshot."""
        return self._snapshot

    @property
    def is_loaded(self) -> bool:
        """Return whether the backing session is loaded."""
        session = self._session_or_default(import_default=False)
        return False if session is None else session.is_loaded

    def configure(
        self,
        *,
        model_folder: str | Path | None,
        model_name: str | None,
    ) -> TrainingGraphSource | None:
        """Configure a graph source without requiring it to exist yet."""
        if model_folder is None or model_name is None:
            self._source = None
            self._loss_keys = ()
            self._snapshot = TrainingGraphSnapshot(None, self._session_id, ())
            return None
        self._source = TrainingGraphSource(Path(model_folder), model_name)
        self._loss_keys = ()
        self._snapshot = TrainingGraphSnapshot(self._source, self._session_id, ())
        return self._source

    def load_source(
        self,
        state_file_or_folder: str | Path,
        *,
        is_training: bool = False,
    ) -> TrainingGraphSnapshot:
        """Load a graph source from a state file or model folder and refresh graph data."""
        source = self.resolve_source(state_file_or_folder)
        self._source = source
        session = self._session_or_default(import_default=True)
        assert session is not None
        session.initialize_session(str(source.model_dir), source.model_name, is_training)
        return self.refresh()

    def load_configured_source(self, *, is_training: bool = False) -> TrainingGraphSnapshot:
        """Load the configured source when its state file exists."""
        if self._source is None:
            raise TrainingGraphError("No graph source configured")
        if not self._source.state_file.is_file():
            raise TrainingGraphError(f"Graph state file does not exist: {self._source.state_file}")
        session = self._session_or_default(import_default=True)
        assert session is not None
        session.initialize_session(
            str(self._source.model_dir), self._source.model_name, is_training
        )
        return self.refresh()

    def set_session_id(self, session_id: int | None) -> TrainingGraphSnapshot:
        """Select a session id and refresh graph data."""
        self._session_id = session_id
        return self.refresh()

    def refresh(self) -> TrainingGraphSnapshot:
        """Refresh loss graph data from the loaded Analysis session."""
        session = self._session_or_default(import_default=False)
        if session is None or not session.is_loaded:
            self._loss_keys = ()
            self._snapshot = TrainingGraphSnapshot(self._source, self._session_id, ())
            return self._snapshot
        keys = tuple(sorted(session.get_loss_keys(self._session_id)))
        self._loss_keys = keys
        loss = session.get_loss(self._session_id)
        graph_series: list[TrainingGraphSeries] = []
        for name, values in sorted(loss.items()):
            if name not in keys:
                continue
            converted = self._to_float_tuple(values)
            if converted:
                graph_series.append(TrainingGraphSeries(name, converted))
        self._snapshot = TrainingGraphSnapshot(self._source, self._session_id, tuple(graph_series))
        return self._snapshot

    def save_csv(
        self,
        filename: str | Path,
        *,
        selected_keys: tuple[str, ...] = (),
    ) -> int:
        """Save the current snapshot to CSV and return the number of data rows written."""
        series = self._snapshot.series_for_keys(selected_keys)
        if not series:
            return 0
        max_count = max(item.count for item in series)
        fieldnames = ("iteration", *(item.name for item in series))
        with Path(filename).open("w", encoding="utf-8", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for index in range(max_count):
                row: dict[str, int | float | str] = {"iteration": index + 1}
                for item in series:
                    row[item.name] = item.values[index] if index < item.count else ""
                writer.writerow(row)
        return max_count

    def clear(self) -> None:
        """Clear graph source and cached graph data."""
        self._source = None
        self._session_id = None
        self._loss_keys = ()
        self._snapshot = TrainingGraphSnapshot(None, None, ())

    def resolve_source(self, state_file_or_folder: str | Path) -> TrainingGraphSource:
        """Resolve model folder/name from a state file or model folder."""
        path = Path(state_file_or_folder)
        if path.is_dir():
            state_file = self._state_file_from_folder(path)
        elif path.is_file():
            state_file = path
        else:
            raise TrainingGraphError(f"Graph source does not exist: {path}")
        if not state_file.name.endswith(self.STATE_SUFFIX):
            raise TrainingGraphError(f"Graph source is not a model state file: {state_file}")
        model_name = state_file.name[: -len(self.STATE_SUFFIX)]
        return TrainingGraphSource(state_file.parent, model_name)

    def _state_file_from_folder(self, folder: Path) -> Path:
        """Return the first model state file from a folder."""
        state_files = sorted(
            path for path in folder.glob(f"*{self.STATE_SUFFIX}") if path.is_file()
        )
        if not state_files:
            raise TrainingGraphError(f"No model state file found in folder: {folder}")
        return state_files[0]

    def _session_or_default(self, *, import_default: bool) -> TrainingGraphSessionProtocol | None:
        """Return the injected Analysis session or lazy-load the legacy singleton."""
        if self._session is not None or not import_default:
            return self._session
        from lib.gui.analysis import Session

        self._session = Session
        return self._session

    @staticmethod
    def _to_float_tuple(values: T.Any) -> tuple[float, ...]:
        """Convert numpy arrays or sequences into finite float tuples."""
        try:
            iterable = values.tolist()
        except AttributeError:
            iterable = values
        converted = []
        for value in iterable:
            if isinstance(value, bool):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number == number:
                converted.append(number)
        return tuple(converted)


__all__ = get_module_objects(__name__)
