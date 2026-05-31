#!/usr/bin/env python3
"""Service helpers for loading preview diagnostics from TensorBoard logs."""

from __future__ import annotations

import struct
import typing as T
from dataclasses import dataclass
from pathlib import Path

from tensorboard.compat.proto import event_pb2

from lib.training.tensorboard import RecordIterator
from lib.utils import get_module_objects


class PreviewDiagnosticsError(ValueError):
    """Raised when preview diagnostics cannot be resolved or loaded."""


@dataclass(frozen=True)
class PreviewDiagnosticsSource:
    """Resolved model source details for preview diagnostics."""

    model_dir: Path
    model_name: str

    @property
    def state_file(self) -> Path:
        """Return the expected model state file."""
        return self.model_dir / f"{self.model_name}_state.json"

    @property
    def logs_dir(self) -> Path:
        """Return the expected TensorBoard logs folder."""
        return self.model_dir / f"{self.model_name}_logs"


@dataclass(frozen=True)
class PreviewDiagnosticMetric:
    """One preview diagnostic metric at a TensorBoard step."""

    name: str
    value: float
    current: float | None = None
    mean: float | None = None
    std: float | None = None
    count: float | None = None

    @property
    def display_name(self) -> str:
        """Return a compact display name."""
        return self.name.replace("_", " ")


@dataclass(frozen=True)
class PreviewDiagnosticSeries:
    """One preview diagnostic metric across a training run."""

    name: str
    values: tuple[float, ...]
    iterations: tuple[int, ...]

    @property
    def count(self) -> int:
        """Return number of points in the series."""
        return len(self.values)


@dataclass(frozen=True)
class PreviewDiagnosticsSnapshot:
    """A refreshed set of preview diagnostics metrics."""

    source: PreviewDiagnosticsSource | None
    session_id: int | None
    iteration: int | None
    metrics: tuple[PreviewDiagnosticMetric, ...]

    @property
    def is_empty(self) -> bool:
        """Return whether there are no preview diagnostics metrics."""
        return not self.metrics

    @property
    def metric_names(self) -> tuple[str, ...]:
        """Return metric names in display order."""
        return tuple(metric.name for metric in self.metrics)

    def metric(self, name: str) -> PreviewDiagnosticMetric | None:
        """Return a metric by name, if available."""
        return next((metric for metric in self.metrics if metric.name == name), None)


@dataclass
class _MetricParts:
    """Collected TensorBoard scalar parts for one logical diagnostic metric."""

    value: float | None = None
    current: float | None = None
    mean: float | None = None
    std: float | None = None
    count: float | None = None

    def to_metric(self, name: str) -> PreviewDiagnosticMetric | None:
        """Return a metric if the base EMA value was present."""
        if self.value is None:
            return None
        return PreviewDiagnosticMetric(
            name=name,
            value=self.value,
            current=self.current,
            mean=self.mean,
            std=self.std,
            count=self.count,
        )


class PreviewDiagnosticsService:
    """Read preview diagnostics scalars from TensorBoard logs."""

    PREFIX = "batch_preview_diagnostics/"
    STATE_SUFFIX = "_state.json"
    IMPORTANT_METRICS = (
        "reconstruction_mae_A",
        "reconstruction_mae_B",
        "reconstruction_imbalance_mae",
        "masked_reconstruction_mae_A",
        "masked_reconstruction_mae_B",
        "boundary_mae_A",
        "boundary_mae_B",
    )
    _PART_SUFFIXES = {
        "_current": "current",
        "_mean": "mean",
        "_std": "std",
        "_count": "count",
    }

    def __init__(self) -> None:
        self._source: PreviewDiagnosticsSource | None = None
        self._snapshot = PreviewDiagnosticsSnapshot(None, None, None, ())

    @property
    def source(self) -> PreviewDiagnosticsSource | None:
        """Return the currently configured source."""
        return self._source

    @property
    def snapshot(self) -> PreviewDiagnosticsSnapshot:
        """Return the latest diagnostics snapshot."""
        return self._snapshot

    def configure(
        self,
        *,
        model_folder: str | Path | None,
        model_name: str | None,
    ) -> PreviewDiagnosticsSource | None:
        """Configure a diagnostics source without requiring files to exist yet."""
        if model_folder is None or model_name is None:
            self.clear()
            return None
        self._source = PreviewDiagnosticsSource(Path(model_folder), model_name)
        self._snapshot = PreviewDiagnosticsSnapshot(self._source, None, None, ())
        return self._source

    def load_source(
        self,
        state_file_or_folder: str | Path,
        *,
        session_id: int | None = None,
    ) -> PreviewDiagnosticsSnapshot:
        """Load preview diagnostics from a model state file or model folder."""
        self._source = self.resolve_source(state_file_or_folder)
        return self.refresh(session_id=session_id)

    def refresh(self, *, session_id: int | None = None) -> PreviewDiagnosticsSnapshot:
        """Refresh diagnostics from the configured source."""
        if self._source is None:
            self._snapshot = PreviewDiagnosticsSnapshot(None, session_id, None, ())
            return self._snapshot
        session_ids = self.session_ids()
        selected_id = (
            session_id if session_id is not None else (session_ids[-1] if session_ids else None)
        )
        if selected_id is None:
            self._snapshot = PreviewDiagnosticsSnapshot(self._source, None, None, ())
            return self._snapshot
        event_file = self._event_file(selected_id)
        if event_file is None:
            self._snapshot = PreviewDiagnosticsSnapshot(self._source, selected_id, None, ())
            return self._snapshot
        self._snapshot = self._read_event_file(event_file, selected_id)
        return self._snapshot

    def session_ids(self) -> tuple[int, ...]:
        """Return session ids with TensorBoard event files."""
        source = self._source
        if source is None or not source.logs_dir.is_dir():
            return ()
        session_ids = []
        for path in source.logs_dir.glob("session_*"):
            session_id = self._session_id(path)
            if session_id is not None and self._event_file(session_id) is not None:
                session_ids.append(session_id)
        return tuple(sorted(session_ids))

    def get_series(self, session_id: int | None = None) -> tuple[PreviewDiagnosticSeries, ...]:
        """Return preview diagnostics series for the requested session or all sessions."""
        if self._source is None:
            return ()
        session_ids = self.session_ids() if session_id is None else (session_id,)
        collected: dict[str, list[tuple[int, float]]] = {}
        for idx in session_ids:
            event_file = self._event_file(idx)
            if event_file is None:
                continue
            for name, points in self._read_event_series(event_file).items():
                collected.setdefault(name, []).extend(points)

        series = []
        for name in self._sort_metric_names(collected):
            points = sorted(collected[name])
            if not points:
                continue
            iterations, values = zip(*points, strict=False)
            series.append(
                PreviewDiagnosticSeries(
                    name=name,
                    values=tuple(float(value) for value in values),
                    iterations=tuple(int(step) for step in iterations),
                )
            )
        return tuple(series)

    def clear(self) -> None:
        """Clear configured source and cached diagnostics."""
        self._source = None
        self._snapshot = PreviewDiagnosticsSnapshot(None, None, None, ())

    def resolve_source(self, state_file_or_folder: str | Path) -> PreviewDiagnosticsSource:
        """Resolve model folder/name from a state file or model folder."""
        path = Path(state_file_or_folder)
        if path.is_dir():
            state_file = self._state_file_from_folder(path)
        elif path.is_file():
            state_file = path
        else:
            raise PreviewDiagnosticsError(f"Preview diagnostics source does not exist: {path}")
        if not state_file.name.endswith(self.STATE_SUFFIX):
            raise PreviewDiagnosticsError(
                f"Preview diagnostics source is not a model state file: {state_file}"
            )
        model_name = state_file.name[: -len(self.STATE_SUFFIX)]
        return PreviewDiagnosticsSource(state_file.parent, model_name)

    def _state_file_from_folder(self, folder: Path) -> Path:
        """Return the first model state file from a folder."""
        state_files = sorted(
            path for path in folder.glob(f"*{self.STATE_SUFFIX}") if path.is_file()
        )
        if not state_files:
            raise PreviewDiagnosticsError(f"No model state file found in folder: {folder}")
        return state_files[0]

    def _event_file(self, session_id: int) -> Path | None:
        """Return the most recent TensorBoard event file for a session."""
        if self._source is None:
            return None
        train_dir = self._source.logs_dir / f"session_{session_id}" / "train"
        if not train_dir.is_dir():
            return None
        event_files = sorted(
            path for path in train_dir.iterdir() if path.name.startswith("events.out.tfevents")
        )
        return event_files[-1] if event_files else None

    @classmethod
    def _read_scalar(cls, event: event_pb2.Event) -> float | None:  # pyright:ignore[reportInvalidTypeForm]
        """Read a scalar value from a TensorBoard event."""
        summary = event.summary.value[0]
        value = summary.simple_value
        if value:
            return float(value)
        try:
            tensor_content = summary.tensor.tensor_content
        except AttributeError:
            return 0.0
        if len(tensor_content) < 4:
            return 0.0
        return float(struct.unpack("f", tensor_content[:4])[0])

    @classmethod
    def _metric_part(cls, tag: str) -> tuple[str, str]:
        """Return logical metric name and part from a TensorBoard tag suffix."""
        for suffix, part in cls._PART_SUFFIXES.items():
            if tag.endswith(suffix):
                return tag[: -len(suffix)], part
        return tag, "value"

    def _read_event_file(
        self,
        event_file: Path,
        session_id: int,
    ) -> PreviewDiagnosticsSnapshot:
        """Read the latest preview diagnostics step from a TensorBoard event file."""
        latest_step: int | None = None
        latest: dict[str, _MetricParts] = {}
        for record in RecordIterator(str(event_file)):
            event = event_pb2.Event.FromString(record)  # pyright:ignore[reportAttributeAccessIssue]
            if not event.summary.value:
                continue
            tag = event.summary.value[0].tag
            if not tag.startswith(self.PREFIX):
                continue
            if latest_step is None or event.step > latest_step:
                latest_step = event.step
                latest = {}
            if event.step != latest_step:
                continue
            metric_name, part = self._metric_part(tag[len(self.PREFIX) :])
            value = self._read_scalar(event)
            if value is None:
                continue
            setattr(latest.setdefault(metric_name, _MetricParts()), part, value)

        metrics = tuple(
            metric
            for name in self._sort_metric_names(latest)
            if (metric := latest[name].to_metric(name)) is not None
        )
        return PreviewDiagnosticsSnapshot(self._source, session_id, latest_step, metrics)

    def _read_event_series(self, event_file: Path) -> dict[str, list[tuple[int, float]]]:
        """Read all base preview diagnostic series from a TensorBoard event file."""
        series: dict[str, list[tuple[int, float]]] = {}
        for record in RecordIterator(str(event_file)):
            event = event_pb2.Event.FromString(record)  # pyright:ignore[reportAttributeAccessIssue]
            if not event.summary.value:
                continue
            tag = event.summary.value[0].tag
            if not tag.startswith(self.PREFIX):
                continue
            metric_name, part = self._metric_part(tag[len(self.PREFIX) :])
            if part != "value":
                continue
            value = self._read_scalar(event)
            if value is None:
                continue
            series.setdefault(metric_name, []).append((event.step, value))
        return series

    @classmethod
    def _sort_metric_names(cls, metrics: T.Mapping[str, object]) -> tuple[str, ...]:
        """Return metric names with headline metrics first."""
        ordered = [name for name in cls.IMPORTANT_METRICS if name in metrics]
        ordered.extend(sorted(name for name in metrics if name not in ordered))
        return tuple(ordered)

    @classmethod
    def compact_text(cls, snapshot: PreviewDiagnosticsSnapshot) -> str:
        """Return one-line preview diagnostics status text."""
        if snapshot.source is None:
            return "Preview diagnostics: no model source"
        if snapshot.is_empty:
            return "Preview diagnostics: no metrics logged"

        parts = []
        for name in cls.IMPORTANT_METRICS:
            metric = snapshot.metric(name)
            if metric is not None:
                parts.append(f"{metric.display_name}: {metric.value:.4f}")
            if len(parts) >= 4:
                break
        if not parts:
            parts = [
                f"{metric.display_name}: {metric.value:.4f}" for metric in snapshot.metrics[:4]
            ]

        sample_count = next(
            (metric.count for metric in snapshot.metrics if metric.count is not None),
            None,
        )
        suffix = f" | n={sample_count:.0f}" if sample_count is not None else ""
        step = f"iter {snapshot.iteration}" if snapshot.iteration is not None else "latest"
        return f"Preview diagnostics ({step}): " + " | ".join(parts) + suffix

    @staticmethod
    def _session_id(path: Path) -> int | None:
        """Return session id from a ``session_<id>`` folder."""
        suffix = path.name.rsplit("_", maxsplit=1)[-1]
        return int(suffix) if suffix.isdigit() else None


__all__ = get_module_objects(__name__)
