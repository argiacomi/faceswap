#!/usr/bin/env python3
"""Parse GUI process output into runtime events."""

from __future__ import annotations

import re
import typing as T
from time import time

from lib.utils import get_module_objects

from .runtime_events import ParsedRuntimeOutput, RuntimeEvent


class ProgressParser:
    """Parse process stdout/stderr into structured runtime events.

    The parser mirrors the legacy GUI parsing behavior but returns events instead of mutating UI
    widgets directly. ``consumed`` indicates that the line should keep the old behavior of
    not being printed to the console.
    """

    _loss_re = re.compile(r"[\W]+(\d+)?[\W]+([a-zA-Z\s]*)[\W]+?(\d+\.\d+)")
    _tqdm_re = re.compile(
        r"(?P<dsc>.*?)(?P<pct>\d+%).*?(?P<itm>\S+/\S+)\W\["
        r"(?P<tme>[\d+:]+<.*),\W(?P<rte>.*)[a-zA-Z/]*\]"
    )
    _ffmpeg_re = re.compile(r"([a-zA-Z]+)=\s*(-?[\d|N/A]\S+)")

    def __init__(self) -> None:
        self._first_loss_seen = False
        self._train_stats: dict[T.Literal["iterations", "timestamp"], int | float | None] = {
            "iterations": 0,
            "timestamp": None,
        }

    @property
    def first_loss_seen(self) -> bool:
        """``True`` after the first training loss line has been parsed."""
        return self._first_loss_seen

    def reset(self) -> None:
        """Reset parser state after a job exits."""
        self._first_loss_seen = False
        self._train_stats = {"iterations": 0, "timestamp": None}

    def parse_stdout(self, command: str | None, output: str) -> ParsedRuntimeOutput:
        """Parse one stdout line into structured runtime events."""
        if command == "train" and not self._first_loss_seen:
            parsed = self._parse_training_determinate(output, stream="stdout")
            if parsed.consumed:
                return parsed

        if command == "train":
            parsed = self._parse_loss(output)
            if parsed.consumed:
                return parsed

        if command == "train" and output.strip() == "\x1b[2K":
            return ParsedRuntimeOutput(consumed=True)

        if command == "effmpeg":
            parsed = self._parse_ffmpeg(output, stream="stdout")
            if parsed.consumed:
                return parsed

        if command not in ("train", "effmpeg"):
            parsed = self._parse_tqdm(output, stream="stdout")
            if parsed.consumed:
                return parsed

        return ParsedRuntimeOutput()

    def parse_stderr(self, command: str | None, output: str) -> ParsedRuntimeOutput:
        """Parse one stderr line into structured runtime events."""
        if command != "train":
            parsed = self._parse_tqdm(output, stream="stderr")
            if parsed.consumed:
                return parsed

        if command == "train" and not self._first_loss_seen:
            parsed = self._parse_training_determinate(output, stream="stderr")
            if parsed.consumed:
                return parsed

        return ParsedRuntimeOutput()

    def final_status(self, command: str | None, returncode: int) -> RuntimeEvent:
        """Create the final status event for a completed process."""
        self.reset()
        if returncode in (0, 3221225786):
            status = "Ready"
            state = "ready"
        elif returncode == -15:
            status = f"Terminated - {command}.py"
            state = "terminated"
        elif returncode == -9:
            status = f"Killed - {command}.py"
            state = "killed"
        elif returncode == -6:
            status = f"Aborted - {command}.py"
            state = "aborted"
        else:
            status = f"Failed - {command}.py. Return Code: {returncode}"
            state = "failed"
        return RuntimeEvent(
            kind="status",
            message=status,
            payload={"command": command, "returncode": returncode, "state": state},
        )

    def _parse_training_determinate(self, output: str, *, stream: str) -> ParsedRuntimeOutput:
        """Parse training tqdm setup before the first loss line."""
        parsed = self._parse_tqdm(output, stream=stream)
        if not parsed.consumed:
            return parsed
        mode_event = RuntimeEvent(
            kind="status",
            message="Training progress is determinate",
            payload={"mode": "determinate", "source": stream},
        )
        return ParsedRuntimeOutput(events=(*parsed.events, mode_event), consumed=True)

    def _parse_loss(self, output: str) -> ParsedRuntimeOutput:
        """Parse a training loss line."""
        if not output.startswith("["):
            return ParsedRuntimeOutput()

        loss = self._loss_re.findall(output)
        if len(loss) != 2 or not all(len(item) == 3 for item in loss):
            return ParsedRuntimeOutput()

        total_iterations = int(loss[0][0])
        losses = {item[1].strip(): float(item[2]) for item in loss}
        message = f"Total Iterations: {total_iterations} | "
        message += "  ".join([f"{item[1]}: {item[2]}" for item in loss])
        if not message:
            return ParsedRuntimeOutput()

        iterations = self._train_stats["iterations"]
        assert isinstance(iterations, int)
        if iterations == 0:
            self._train_stats["timestamp"] = time()
        iterations += 1
        self._train_stats["iterations"] = iterations

        elapsed = self._calculate_elapsed()
        message = f"Elapsed: {elapsed} | Session Iterations: {iterations}  {message}"

        events: tuple[RuntimeEvent, ...] = ()
        if not self._first_loss_seen:
            events = (
                RuntimeEvent(
                    kind="status",
                    message="Training progress is indeterminate",
                    payload={"mode": "indeterminate", "source": "stdout"},
                ),
            )
            self._first_loss_seen = True

        return ParsedRuntimeOutput(
            events=(
                *events,
                RuntimeEvent(
                    kind="status",
                    message=message,
                    payload={
                        "parser": "loss",
                        "iterations": iterations,
                        "total_iterations": total_iterations,
                        "losses": losses,
                        "source": "stdout",
                    },
                ),
            ),
            consumed=True,
        )

    def _calculate_elapsed(self) -> str:
        """Calculate and format time since training started."""
        now = time()
        timestamp = self._train_stats["timestamp"]
        assert isinstance(timestamp, float)
        elapsed_time = now - timestamp
        try:
            hours = int(elapsed_time // 3600)
            fmt_hours = f"{hours:02d}" if hours < 10 else str(hours)
            minutes = f"{(int(elapsed_time % 3600) // 60):02d}"
            seconds = f"{(int(elapsed_time % 3600) % 60):02d}"
        except ZeroDivisionError:
            fmt_hours = minutes = seconds = "00"
        return f"{fmt_hours}:{minutes}:{seconds}"

    def _parse_tqdm(self, output: str, *, stream: str) -> ParsedRuntimeOutput:
        """Parse tqdm output."""
        match = self._tqdm_re.match(output)
        if not match:
            return ParsedRuntimeOutput()

        tqdm = match.groupdict()
        if any("?" in value for value in tqdm.values()):
            return ParsedRuntimeOutput(consumed=True)

        description = tqdm["dsc"].strip()
        description = description if description == "" else f"{description[:-1]}  |  "
        process_time = (
            f"Elapsed: {tqdm['tme'].split('<')[0]}  Remaining: {tqdm['tme'].split('<')[1]}"
        )
        message = (
            f"{description}{process_time}  |  {tqdm['rte']}  |  {tqdm['itm']}  |  {tqdm['pct']}"
        )

        percent = tqdm["pct"].replace("%", "")
        progress = float(percent) if percent.isdigit() else 0.0
        return ParsedRuntimeOutput(
            events=(
                RuntimeEvent(
                    kind="progress",
                    message=message,
                    progress=progress,
                    payload={
                        "parser": "tqdm",
                        "description": description.strip(),
                        "items": tqdm["itm"],
                        "rate": tqdm["rte"],
                        "source": stream,
                    },
                ),
            ),
            consumed=True,
        )

    def _parse_ffmpeg(self, output: str, *, stream: str) -> ParsedRuntimeOutput:
        """Parse ffmpeg key-value progress output."""
        ffmpeg = self._ffmpeg_re.findall(output)
        if len(ffmpeg) < 7:
            return ParsedRuntimeOutput()

        message = ""
        for item in ffmpeg:
            message += f"{item[0]}: {item[1]}  "
        if not message:
            return ParsedRuntimeOutput()

        return ParsedRuntimeOutput(
            events=(
                RuntimeEvent(
                    kind="progress",
                    message=message,
                    payload={
                        "parser": "ffmpeg",
                        "values": dict(ffmpeg),
                        "source": stream,
                    },
                ),
            ),
            consumed=True,
        )


__all__ = get_module_objects(__name__)
