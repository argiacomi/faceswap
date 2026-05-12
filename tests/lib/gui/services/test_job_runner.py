#!/usr/bin/env python3
"""Tests for the GUI process runtime service."""

from __future__ import annotations

from lib.gui.services.job_runner import JobRunner, ProcessRuntimeService
from lib.gui.services.runtime_events import RuntimeEvent


def _collect_events(runner: ProcessRuntimeService) -> list[RuntimeEvent]:
    """Attach a test collector to a job runner."""
    events: list[RuntimeEvent] = []
    runner.add_event_callback(events.append)
    return events


def test_job_runner_alias_points_to_process_runtime_service() -> None:
    """The old JobRunner name should remain available during migration."""
    assert JobRunner is ProcessRuntimeService


def test_process_stdout_emits_parsed_events_to_callbacks() -> None:
    """Parsed stdout events should be forwarded to registered callbacks."""
    runner = ProcessRuntimeService(command="extract")
    events = _collect_events(runner)

    parsed = runner.process_stdout("Extracting:  25%|##5       | 5/20 [00:02<00:06,  2.50it/s]\n")

    assert parsed.consumed
    assert events[0].kind == "progress"
    assert events[0].progress == 25.0
    assert "5/20" in events[0].message


def test_process_stderr_emits_parsed_events_to_callbacks() -> None:
    """Parsed stderr events should be forwarded to registered callbacks."""
    runner = ProcessRuntimeService(command="train")
    events = _collect_events(runner)

    parsed = runner.process_stderr("Training:  50%|#####     | 10/20 [00:04<00:04,  2.50it/s]\n")

    assert parsed.consumed
    assert [event.kind for event in events] == ["progress", "status"]
    assert events[0].progress == 50.0
    assert events[1].message == "Training progress is determinate"


def test_plain_stdout_is_not_emitted_to_callbacks() -> None:
    """Plain process output should pass through unchanged for console printing."""
    runner = ProcessRuntimeService(command="extract")
    events = _collect_events(runner)

    parsed = runner.process_stdout("ordinary console line\n")

    assert not parsed.consumed
    assert parsed.events == ()
    assert events == []


def test_remove_event_callback_stops_future_notifications() -> None:
    """Removed callbacks should not receive later runtime events."""
    runner = ProcessRuntimeService(command="extract")
    events = _collect_events(runner)
    runner.remove_event_callback(events.append)

    runner.process_stdout("Extracting:  25%|##5       | 5/20 [00:02<00:06,  2.50it/s]\n")

    assert events == []


def test_finish_emits_final_status_event() -> None:
    """Finishing a fake process should emit a final status event."""
    runner = ProcessRuntimeService(command="train")
    events = _collect_events(runner)

    event = runner.finish(-9)

    assert event.kind == "status"
    assert event.message == "Killed - train.py"
    assert event.payload == {"command": "train", "returncode": -9, "state": "killed"}
    assert events == [event]
