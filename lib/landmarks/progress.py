#!/usr/bin/env python3
"""Small stderr progress helper for long-running landmark tools."""

from __future__ import annotations

import sys
import time
import typing as T

TItem = T.TypeVar("TItem")


class ProgressBar:
    """Minimal dependency-free progress bar for CLI tools."""

    def __init__(
        self,
        total: int,
        *,
        label: str,
        enabled: bool = True,
        width: int = 28,
        min_interval: float = 0.2,
    ) -> None:
        self.total = max(int(total), 0)
        self.label = label
        self.enabled = enabled and self.total > 0
        self.width = max(int(width), 10)
        self.min_interval = max(float(min_interval), 0.0)
        self.count = 0
        self.started = time.time()
        self._last_render = 0.0
        self._closed = False

    def update(self, step: int = 1) -> None:
        """Advance the bar by ``step`` units and render when useful."""
        if not self.enabled:
            self.count = min(self.total, self.count + step)
            return
        self.count = min(self.total, self.count + step)
        now = time.time()
        if self.count >= self.total or now - self._last_render >= self.min_interval:
            self._render(now)

    def close(self) -> None:
        """Render completion and end the line."""
        if self._closed:
            return
        if self.enabled:
            self.count = self.total
            self._render(time.time())
            print(file=sys.stderr, flush=True)
        self._closed = True

    def _render(self, now: float) -> None:
        self._last_render = now
        elapsed = max(now - self.started, 0.0)
        ratio = 1.0 if self.total <= 0 else min(1.0, self.count / float(self.total))
        filled = int(round(self.width * ratio))
        bar = "#" * filled + "-" * (self.width - filled)
        rate = self.count / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.count) / rate if rate > 0 and self.count < self.total else 0.0
        print(
            f"\r{self.label}: [{bar}] {self.count}/{self.total} "
            f"{ratio * 100:5.1f}% elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta)}",
            end="",
            file=sys.stderr,
            flush=True,
        )


def _format_seconds(seconds: float) -> str:
    """Return compact seconds/minutes display."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{int(remainder):02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m"


def progress_iter(
    values: T.Iterable[TItem],
    *,
    total: int | None = None,
    label: str,
    enabled: bool = True,
) -> T.Iterator[TItem]:
    """Yield values while rendering a stderr progress bar."""
    if total is None:
        try:
            total = len(values)  # type: ignore[arg-type]
        except TypeError:
            total = 0
    bar = ProgressBar(total, label=label, enabled=enabled)
    try:
        for value in values:
            yield value
            bar.update()
    finally:
        bar.close()
