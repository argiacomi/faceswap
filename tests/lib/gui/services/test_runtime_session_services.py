#!/usr/bin/env python3
"""Tests for shared GUI runtime session services."""

from __future__ import annotations

from lib.gui.services.command_context import CommandExecutionContext
from lib.gui.services.runtime_service import ProcessRuntimeService
from lib.gui.services.runtime_session_services import (
    PreviewOutputRuntimeService,
    RuntimeTerminationPolicy,
    TrainingSessionRuntimeService,
)


class _TrainingSessionDouble:
    """Small training Analysis session test double."""

    def __init__(self, *, is_training: bool = False) -> None:
        self.is_training = is_training
        self.initialized: list[tuple[str, str, bool]] = []
        self.stop_count = 0

    def initialize_session(
        self,
        model_folder: str,
        model_name: str,
        is_training: bool = False,
    ) -> None:
        """Capture initialization calls."""
        self.is_training = is_training
        self.initialized.append((model_folder, model_name, is_training))

    def stop_training(self) -> None:
        """Capture stop calls."""
        self.is_training = False
        self.stop_count += 1


class _PreviewExtractDouble:
    """Preview extract test double."""

    def __init__(self) -> None:
        self.output_paths: list[tuple[str, bool]] = []

    def set_faceswap_output_path(self, output_path: str, *, batch_mode: bool = False) -> None:
        """Capture preview output path setup."""
        self.output_paths.append((output_path, batch_mode))


class _ImagesDouble:
    """Image manager test double."""

    def __init__(self) -> None:
        self.preview_extract = _PreviewExtractDouble()
        self.delete_count = 0

    def delete_preview(self) -> None:
        """Capture preview cleanup."""
        self.delete_count += 1


class _PreviewTriggerDouble:
    """Preview trigger test double."""

    def __init__(self) -> None:
        self.clear_calls: list[object] = []

    def clear(self, trigger_type=None) -> None:  # type:ignore[no-untyped-def]
        """Capture clear calls."""
        self.clear_calls.append(trigger_type)


def test_runtime_termination_policy_uses_train_interrupt_plan() -> None:
    """Training termination should request graceful interrupt before escalation."""
    plan = RuntimeTerminationPolicy(train_timeout_seconds=3).plan("train.py")

    assert plan.command == "train"
    assert plan.interrupt_first is True
    assert plan.grace_ms == 3000
    assert plan.tree_timeout_seconds == 3
    assert plan.message == "Sending Exit Signal"


def test_runtime_termination_policy_uses_process_plan_for_other_commands() -> None:
    """Non-training commands should use standard process termination."""
    plan = RuntimeTerminationPolicy(default_timeout_seconds=7).plan("extract")

    assert plan.command == "extract"
    assert plan.interrupt_first is False
    assert plan.grace_ms == 7000
    assert plan.tree_timeout_seconds == 7
    assert plan.message == "Terminating Process..."


def test_training_session_service_configures_location_from_command_context() -> None:
    """Training context values should populate model folder and normalized model name."""
    service = TrainingSessionRuntimeService(_TrainingSessionDouble())
    context = CommandExecutionContext.from_values(
        "train",
        {"-m": "/models", "-t": "Original-Model"},
    )

    location = service.configure(context)

    assert location.model_folder == "/models"
    assert location.model_name == "original_model"
    assert location.is_complete is True


def test_training_session_service_initializes_and_refreshes_on_saved_model() -> None:
    """A standard saved-model train line should initialize Analysis and refresh graphs."""
    session = _TrainingSessionDouble()
    refreshes = []
    service = TrainingSessionRuntimeService(
        session, graph_refresh_callback=lambda: refreshes.append(True)
    )
    service.configure(model_folder="/models", model_name="model_a")

    events = service.events_for_stdout("train", "1000 [Saved Model] checkpoint written\n")

    assert session.initialized == [("/models", "model_a", True)]
    assert refreshes == [True]
    assert len(events) == 1
    assert events[0].kind == "training_session"
    assert events[0].payload == {
        "command": "train",
        "state": "saved_model",
        "saved_model": True,
        "initialized": True,
        "graph_refresh": True,
        "model_folder": "/models",
        "model_name": "model_a",
    }


def test_training_session_service_does_not_reinitialize_existing_training_session() -> None:
    """Saved-model refreshes should not reinitialize a session already marked training."""
    session = _TrainingSessionDouble(is_training=True)
    refreshes = []
    service = TrainingSessionRuntimeService(
        session, graph_refresh_callback=lambda: refreshes.append(True)
    )
    service.configure(model_folder="/models", model_name="model_a")

    events = service.events_for_stdout("train", "1000 [Saved Model] checkpoint written\n")

    assert session.initialized == []
    assert refreshes == [True]
    assert events[0].payload["initialized"] is False


def test_training_session_service_ignores_non_saved_model_output() -> None:
    """Only standard train saved-model lines should trigger session events."""
    service = TrainingSessionRuntimeService(_TrainingSessionDouble())

    assert service.events_for_stdout("extract", "1000 [Saved Model] checkpoint written\n") == ()
    assert service.events_for_stdout("train", "Saving for exit [Saved Model]\n") == ()
    assert service.events_for_stdout("train", "ordinary output\n") == ()


def test_preview_output_runtime_service_sets_output_path_and_cleans_up() -> None:
    """Preview setup and cleanup should be available without Tk globals in tests."""
    images = _ImagesDouble()
    trigger = _PreviewTriggerDouble()
    service = PreviewOutputRuntimeService(
        images_provider=lambda: images,
        preview_trigger_provider=lambda: trigger,
    )
    context = CommandExecutionContext.from_values(
        "extract",
        {"-o": "/preview", "-b": True},
    )

    applied = service.apply_context(context)
    service.cleanup()

    assert applied is True
    assert images.preview_extract.output_paths == [("/preview", True)]
    assert images.delete_count == 1
    assert trigger.clear_calls == [None]
    assert service.preview_output_path is None
    assert service.batch_mode is False


def test_preview_output_runtime_service_skips_side_effects_for_generate() -> None:
    """Generate-only command contexts should capture data but skip GUI side effects."""
    images = _ImagesDouble()
    service = PreviewOutputRuntimeService(images_provider=lambda: images)
    context = CommandExecutionContext.from_values("convert", {"-o": "/preview"})

    applied = service.apply_context(context, generate=True)

    assert applied is False
    assert service.preview_output_path == "/preview"
    assert images.preview_extract.output_paths == []


def test_process_runtime_service_emits_training_session_events_from_stdout() -> None:
    """ProcessRuntimeService should fan saved-model lines into runtime events."""
    session = _TrainingSessionDouble()
    training = TrainingSessionRuntimeService(session)
    training.configure(model_folder="/models", model_name="model_a")
    runtime = ProcessRuntimeService("train", training_session=training)
    events = []
    runtime.add_event_callback(events.append)

    parsed = runtime.process_stdout("1000 [Saved Model] checkpoint written\n")

    assert parsed.consumed is False
    assert [event.kind for event in parsed.events] == ["training_session"]
    assert events == list(parsed.events)
    assert session.initialized == [("/models", "model_a", True)]


def test_process_runtime_service_stops_training_session_only_for_train() -> None:
    """Final runtime cleanup should stop training only for train commands."""
    session = _TrainingSessionDouble(is_training=True)
    training = TrainingSessionRuntimeService(session)
    runtime = ProcessRuntimeService("extract", training_session=training)

    runtime.finish(0)
    assert session.stop_count == 0

    runtime.command = "train"
    runtime.finish(0)
    assert session.stop_count == 1
