#!/usr/bin/env python3
"""Progress tracking and TRACE logging helpers for landmark pipeline tools."""

from __future__ import annotations

import json
import logging
import sys
import time
import typing as T
from dataclasses import dataclass
from pathlib import Path

TRACE_LEVEL = 5


def install_trace_level() -> None:
    """Register a TRACE logging level below DEBUG."""
    if logging.getLevelName(TRACE_LEVEL) != "TRACE":
        logging.addLevelName(TRACE_LEVEL, "TRACE")

    def trace(self: logging.Logger, message: str, *args: T.Any, **kwargs: T.Any) -> None:
        if self.isEnabledFor(TRACE_LEVEL):
            self._log(TRACE_LEVEL, message, args, **kwargs)

    if not hasattr(logging.Logger, "trace"):
        logging.Logger.trace = trace  # type: ignore[attr-defined]


def logging_level(value: str | int) -> int:
    """Return a logging level, including TRACE."""
    install_trace_level()
    if isinstance(value, int):
        return value
    normalized = str(value).strip().upper()
    if normalized == "TRACE":
        return TRACE_LEVEL
    level = getattr(logging, normalized, None)
    if isinstance(level, int):
        return level
    raise ValueError(f"unknown log level {value!r}; expected TRACE, DEBUG, INFO, WARNING, ERROR")


def configure_logging(level: str | int) -> None:
    """Configure root logging with TRACE support.

    Delegates to :func:`lib.logger.configure_tool_logging` so the landmark
    tools share the main Faceswap external-logger policy (matplotlib /
    libav INFO suppression, torch inductor warnings, blank-line elision)
    instead of installing an unmanaged root handler via
    ``logging.basicConfig()`` (issue #189).
    """
    install_trace_level()
    from lib.logger import configure_tool_logging  # local import: avoid bootstrapping cost

    configure_tool_logging(logging_level(level))


@dataclass(frozen=True)
class ProgressEvent:
    """One structured progress event."""

    event: str
    stage: str
    index: int
    total: int
    status: str
    elapsed_seconds: float
    message: str = ""

    def to_json(self) -> dict[str, T.Any]:
        return {
            "event": self.event,
            "stage": self.stage,
            "index": self.index,
            "total": self.total,
            "status": self.status,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "message": self.message,
        }


class PipelineProgress:
    """Stage-level progress reporter with optional terminal progress bar."""

    def __init__(
        self,
        *,
        total: int,
        output_path: Path | None = None,
        enabled: bool | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.total = max(total, 0)
        self.output_path = output_path
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = bool(sys.stderr.isatty()) if enabled is None else bool(enabled)
        self._started_at = time.time()
        self._current_stage = ""
        self._current_index = 0
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text("", encoding="utf-8")

    def start(self, stage: str, *, index: int, message: str = "") -> None:
        """Record the start of a stage."""
        self._current_stage = stage
        self._current_index = index
        self._emit("start", stage=stage, index=index, status="running", message=message)

    def finish(self, stage: str, *, index: int, status: str, message: str = "") -> None:
        """Record the end of a stage."""
        self._emit("finish", stage=stage, index=index, status=status, message=message)

    def event(self, stage: str, *, index: int, status: str, message: str) -> None:
        """Record an intermediate stage event."""
        self._emit("event", stage=stage, index=index, status=status, message=message)

    def _emit(self, event: str, *, stage: str, index: int, status: str, message: str = "") -> None:
        elapsed = time.time() - self._started_at
        progress_event = ProgressEvent(
            event=event,
            stage=stage,
            index=index,
            total=self.total,
            status=status,
            elapsed_seconds=elapsed,
            message=message,
        )
        payload = progress_event.to_json()
        if self.output_path is not None:
            with self.output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        log_message = "pipeline_progress event=%s stage=%s index=%s/%s status=%s %s"
        if event == "start":
            self.logger.debug(log_message, event, stage, index, self.total, status, message)
        elif status == "failed":
            self.logger.error(log_message, event, stage, index, self.total, status, message)
        else:
            self.logger.info(log_message, event, stage, index, self.total, status, message)
        if self.logger.isEnabledFor(TRACE_LEVEL):
            self.logger.log(
                TRACE_LEVEL, "pipeline_progress_payload=%s", json.dumps(payload, sort_keys=True)
            )
        if self.enabled:
            self._render_bar(index=index, stage=stage, status=status)

    _BAR: T.ClassVar[T.Any] = None

    def _render_bar(self, *, index: int, stage: str, status: str) -> None:
        """Render landmark pipeline progress via :mod:`tqdm`.

        Previously this method wrote a hand-rolled bar directly to
        ``sys.stderr``, which clashed with GUI status parsing and could
        intermix with other tqdm bars (issue #189). The bar is now an
        owned ``tqdm`` instance:

        * Lazily constructed on first render so import-time tqdm
          isn't required when no progress is actually emitted.
        * Updated in absolute terms (``bar.n = completed``) so it tracks
          the externally-supplied stage index, not delta increments.
        * Closed on the final or failed event so the bar doesn't leak
          across pipeline runs.

        ``set_postfix_str`` carries the current stage + status text in
        the same tqdm description payload the GUI ``ProgressParser``
        already consumes for ordinary Faceswap stages.
        """
        if self.total <= 0:
            return
        completed = min(max(index, 0), self.total)
        if self._BAR is None:
            from tqdm import tqdm  # pylint:disable=import-outside-toplevel

            self._BAR = tqdm(  # type: ignore[misc]
                total=self.total,
                desc="landmark pipeline",
                unit="stage",
                leave=False,
            )
        self._BAR.n = completed
        self._BAR.set_postfix_str(f"{stage} {status}", refresh=True)
        if completed >= self.total or status == "failed":
            self._BAR.close()
            self._BAR = None  # type: ignore[misc]


__all__ = [
    "TRACE_LEVEL",
    "PipelineProgress",
    "ProgressEvent",
    "configure_logging",
    "install_trace_level",
    "logging_level",
]
