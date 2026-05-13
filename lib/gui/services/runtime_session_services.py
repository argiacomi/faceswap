#!/usr/bin/env python3
"""Shared runtime session, preview and termination services for GUI shells."""

from __future__ import annotations

import typing as T
from dataclasses import dataclass

from lib.utils import get_module_objects

from .command_context import CommandExecutionContext
from .runtime_events import RuntimeEvent


class TrainingSessionProtocol(T.Protocol):
    """Small protocol for the legacy ``lib.gui.analysis.Session`` singleton."""

    is_training: bool

    def initialize_session(
        self, model_folder: str, model_name: str, is_training: bool = False
    ) -> None:
        """Initialize a training session."""

    def stop_training(self) -> None:
        """Stop the current training session."""


RuntimeEventCallback = T.Callable[[RuntimeEvent], None]
GraphRefreshCallback = T.Callable[[], None]
ImagesProvider = T.Callable[[], T.Any]
PreviewTriggerProvider = T.Callable[[], T.Any]


@dataclass(frozen=True)
class TrainingSessionLocation:
    """Model folder/name details needed to initialize a training Analysis session."""

    model_folder: str | None = None
    model_name: str | None = None

    @property
    def is_complete(self) -> bool:
        """Return whether both model details are available."""
        return bool(self.model_folder and self.model_name)


@dataclass(frozen=True)
class RuntimeTerminationPlan:
    """Shell-neutral process termination policy for a Faceswap command."""

    command: str | None
    interrupt_first: bool
    grace_ms: int
    tree_timeout_seconds: float
    message: str


class RuntimeTerminationPolicy:
    """Return command-aware process termination plans shared by GUI shells."""

    def __init__(
        self,
        *,
        train_timeout_seconds: float = 5.0,
        default_timeout_seconds: float = 5.0,
    ) -> None:
        self._train_timeout_seconds = train_timeout_seconds
        self._default_timeout_seconds = default_timeout_seconds

    def plan(self, command: str | None) -> RuntimeTerminationPlan:
        """Return a termination plan for ``command``."""
        normalized = self.normalize_command(command)
        if normalized == "train":
            return RuntimeTerminationPlan(
                command=normalized,
                interrupt_first=True,
                grace_ms=max(0, int(self._train_timeout_seconds * 1000)),
                tree_timeout_seconds=self._train_timeout_seconds,
                message="Sending Exit Signal",
            )
        return RuntimeTerminationPlan(
            command=normalized,
            interrupt_first=False,
            grace_ms=max(0, int(self._default_timeout_seconds * 1000)),
            tree_timeout_seconds=self._default_timeout_seconds,
            message="Terminating Process...",
        )

    @staticmethod
    def normalize_command(command: str | None) -> str | None:
        """Normalize command names passed from command builders or process events."""
        if command is None:
            return None
        return command.lower().replace(".py", "")


class TrainingSessionRuntimeService:
    """Shared training-session lifecycle helper for runtime process output."""

    def __init__(
        self,
        session: TrainingSessionProtocol | None = None,
        *,
        graph_refresh_callback: GraphRefreshCallback | None = None,
    ) -> None:
        self._session = session
        self._graph_refresh_callback = graph_refresh_callback
        self._location = TrainingSessionLocation()

    @property
    def location(self) -> TrainingSessionLocation:
        """Return the current training session location."""
        return self._location

    def configure(
        self,
        context: CommandExecutionContext | None = None,
        *,
        model_folder: str | None = None,
        model_name: str | None = None,
    ) -> TrainingSessionLocation:
        """Update model details from a command context or explicit values."""
        if context is not None:
            model_folder = (
                context.model_folder if context.model_folder is not None else model_folder
            )
            model_name = context.model_name if context.model_name is not None else model_name
        self._location = TrainingSessionLocation(model_folder, model_name)
        return self._location

    def clear(self) -> None:
        """Clear model details for the current runtime."""
        self._location = TrainingSessionLocation()

    def stop(self) -> None:
        """Stop the loaded training Analysis session when the adapter supports it."""
        session = self._session_or_default(import_default=False)
        if session is None:
            return
        stop_training = getattr(session, "stop_training", None)
        if callable(stop_training):
            stop_training()

    def events_for_stdout(
        self,
        command: str | None,
        output: str,
        *,
        is_training: bool = True,
    ) -> tuple[RuntimeEvent, ...]:
        """Return runtime events triggered by training stdout."""
        if not self.is_saved_model_line(command, output):
            return ()
        initialized = self.initialize_from_saved_model(is_training=is_training)
        payload: dict[str, object] = {
            "command": "train",
            "state": "saved_model",
            "saved_model": True,
            "initialized": initialized,
            "graph_refresh": True,
        }
        if self._location.model_folder is not None:
            payload["model_folder"] = self._location.model_folder
        if self._location.model_name is not None:
            payload["model_name"] = self._location.model_name
        return (
            RuntimeEvent(
                kind="training_session",
                message="Saved model detected",
                payload=payload,
            ),
        )

    def initialize_from_saved_model(self, *, is_training: bool = True) -> bool:
        """Initialize the Analysis session and trigger graph refresh after a model save."""
        initialized = False
        if self._location.is_complete:
            session = self._session_or_default(import_default=True)
            if session is not None and not bool(getattr(session, "is_training", False)):
                assert self._location.model_folder is not None
                assert self._location.model_name is not None
                session.initialize_session(
                    self._location.model_folder,
                    self._location.model_name,
                    is_training=is_training,
                )
                initialized = True
        if self._graph_refresh_callback is not None:
            self._graph_refresh_callback()
        return initialized

    @classmethod
    def is_saved_model_line(cls, command: str | None, output: str) -> bool:
        """Return whether stdout announces a standard train saved-model event."""
        if RuntimeTerminationPolicy.normalize_command(command) != "train":
            return False
        text = output.strip().lower()
        return "[saved model]" in text and not text.endswith("[saved model]")

    def _session_or_default(self, *, import_default: bool) -> TrainingSessionProtocol | None:
        """Return the injected session or lazy import the legacy singleton."""
        if self._session is not None or not import_default:
            return self._session
        from lib.gui.analysis import Session

        self._session = Session
        return self._session


class PreviewOutputRuntimeService:
    """Shared helper for preview output setup and cleanup side effects."""

    def __init__(
        self,
        *,
        images_provider: ImagesProvider | None = None,
        preview_trigger_provider: PreviewTriggerProvider | None = None,
    ) -> None:
        self._images_provider = images_provider
        self._preview_trigger_provider = preview_trigger_provider
        self.preview_output_path: str | None = None
        self.batch_mode = False

    def apply_context(
        self,
        context: CommandExecutionContext,
        *,
        generate: bool = False,
    ) -> bool:
        """Apply preview output side effects from a command execution context."""
        self.preview_output_path = context.preview_output_path
        self.batch_mode = context.batch_mode
        if generate or context.preview_output_path is None:
            return False
        images = self._images()
        images.preview_extract.set_faceswap_output_path(
            context.preview_output_path,
            batch_mode=context.batch_mode,
        )
        return True

    def cleanup(self) -> None:
        """Clear preview images and pending preview triggers."""
        images = self._images(optional=True)
        if images is not None:
            images.delete_preview()
        trigger = self._preview_trigger(optional=True)
        if trigger is not None:
            trigger.clear(trigger_type=None)
        self.preview_output_path = None
        self.batch_mode = False

    def _images(self, *, optional: bool = False) -> T.Any:
        """Return the configured image manager."""
        if self._images_provider is None:
            if optional:
                try:
                    from lib.gui.utils import get_images
                except ImportError:
                    return None
            else:
                from lib.gui.utils import get_images
            self._images_provider = get_images
        return self._images_provider()

    def _preview_trigger(self, *, optional: bool = False) -> T.Any:
        """Return the configured preview trigger helper."""
        if self._preview_trigger_provider is None:
            if optional:
                try:
                    from lib.gui.utils import preview_trigger
                except ImportError:
                    return None
            else:
                from lib.gui.utils import preview_trigger
            self._preview_trigger_provider = preview_trigger
        return self._preview_trigger_provider()


__all__ = get_module_objects(__name__)
