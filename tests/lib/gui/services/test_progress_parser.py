#!/usr/bin/env python3
"""Tests for GUI process progress parsing."""

from __future__ import annotations

from lib.gui.services.progress_parser import ProgressParser


def test_plain_stdout_is_not_consumed() -> None:
    """Plain process output should pass through unchanged for console printing."""
    parsed = ProgressParser().parse_stdout("extract", "ordinary console line\n")

    assert not parsed.consumed
    assert parsed.events == ()


def test_tqdm_stdout_emits_progress_event() -> None:
    """A fake tqdm stdout line should become a structured progress event."""
    parsed = ProgressParser().parse_stdout(
        "extract",
        "Extracting:  25%|##5       | 5/20 [00:02<00:06,  2.50it/s]\n",
    )

    assert parsed.consumed
    event = parsed.events[0]
    assert event.kind == "progress"
    assert event.progress == 25.0
    assert "5/20" in event.message
    assert event.payload == {
        "parser": "tqdm",
        "description": "Extracting  |",
        "items": "5/20",
        "rate": " 2.50it/s",
        "source": "stdout",
    }


def test_training_tqdm_stderr_sets_determinate_mode() -> None:
    """Training tqdm output before first loss should emit progress and mode events."""
    parsed = ProgressParser().parse_stderr(
        "train",
        "Training:  50%|#####     | 10/20 [00:04<00:04,  2.50it/s]\n",
    )

    assert parsed.consumed
    assert [event.kind for event in parsed.events] == ["progress", "status"]
    assert parsed.events[0].progress == 50.0
    assert parsed.events[1].message == "Training progress is determinate"
    assert parsed.events[1].payload == {"mode": "determinate", "source": "stderr"}


def test_training_loss_stdout_emits_indeterminate_status_and_loss_payload() -> None:
    """A fake training loss line should emit status events with parsed loss values."""
    parsed = ProgressParser().parse_stdout("train", "[000001] loss A: 0.1234 - loss B: 0.5678\n")

    assert parsed.consumed
    assert [event.kind for event in parsed.events] == ["status", "status"]
    assert parsed.events[0].message == "Training progress is indeterminate"
    assert parsed.events[0].payload == {"mode": "indeterminate", "source": "stdout"}
    assert parsed.events[1].payload == {
        "parser": "loss",
        "iterations": 1,
        "total_iterations": 1,
        "losses": {"loss A": 0.1234, "loss B": 0.5678},
        "source": "stdout",
    }
    assert "Session Iterations: 1" in parsed.events[1].message


def test_effmpeg_stdout_emits_progress_payload_without_percentage() -> None:
    """A fake ffmpeg progress line should emit key-value progress details."""
    parsed = ProgressParser().parse_stdout(
        "effmpeg",
        "frame=   12 fps=24.0 q=-1.0 size=256kB "
        "time=00:00:02.00 bitrate=1000.0kbits/s speed=1.2x\n",
    )

    assert parsed.consumed
    event = parsed.events[0]
    assert event.kind == "progress"
    assert event.progress is None
    assert event.payload == {
        "parser": "ffmpeg",
        "values": {
            "frame": "12",
            "fps": "24.0",
            "q": "-1.0",
            "size": "256kB",
            "time": "00:00:02.00",
            "bitrate": "1000.0kbits/s",
            "speed": "1.2x",
        },
        "source": "stdout",
    }


def test_final_status_resets_training_state() -> None:
    """Final status should reset parser state and map process return codes."""
    parser = ProgressParser()
    parser.parse_stdout("train", "[000001] loss A: 0.1234 - loss B: 0.5678\n")

    event = parser.final_status("train", -9)

    assert event.kind == "status"
    assert event.message == "Killed - train.py"
    assert event.payload == {"command": "train", "returncode": -9, "state": "killed"}
    assert parser.first_loss_seen is False
