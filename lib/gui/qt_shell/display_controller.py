#!/usr/bin/env python3
"""Runtime-event driven controller for Qt shell display tabs."""

from __future__ import annotations

import re
import typing as T
from collections.abc import Mapping

from PySide6.QtCore import QObject, Qt
from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

if T.TYPE_CHECKING:
    from lib.gui.services.runtime_events import RuntimeEvent

PanelFactory = T.Callable[[], QWidget]


class DisplayController(QObject):
    """Create and remove display tabs from process/runtime state events."""

    ANALYSIS = "Analysis"
    PREVIEW = "Preview"
    GRAPH = "Graph"

    _ORDER = (ANALYSIS, PREVIEW, GRAPH)
    _PREVIEW_COMMANDS = frozenset(("extract", "convert"))
    _GRAPH_COMMANDS = frozenset(("train",))
    _TERMINAL_STATES = frozenset(
        (
            "aborted",
            "cancelled",
            "canceled",
            "complete",
            "completed",
            "failed",
            "finish",
            "finished",
            "idle",
            "killed",
            "ready",
            "stopped",
            "success",
            "terminated",
        )
    )
    _ACTIVE_STATES = frozenset(("active", "run", "running", "start", "started", "starting"))
    _ACTIVE_KINDS = frozenset(
        ("display", "graph", "preview", "process", "progress", "runtime", "state", "status")
    )
    _COMMAND_RE = re.compile(r"\b(extract|convert|train)\b", re.IGNORECASE)

    def __init__(
        self,
        tabs: QTabWidget,
        *,
        analysis_factory: PanelFactory | None = None,
        preview_factory: PanelFactory | None = None,
        graph_factory: PanelFactory | None = None,
        preserve_existing_tabs: bool = False,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._tabs = tabs
        self._preserve_existing_tabs = preserve_existing_tabs
        self._factories = {
            self.ANALYSIS: analysis_factory or (lambda: self._placeholder_panel(self.ANALYSIS)),
            self.PREVIEW: preview_factory or (lambda: self._placeholder_panel(self.PREVIEW)),
            self.GRAPH: graph_factory or (lambda: self._placeholder_panel(self.GRAPH)),
        }
        self._widgets: dict[str, QWidget] = {}
        self._command: str | None = None
        self._adopt_existing_tabs()
        self._ensure_tab(self.ANALYSIS)
        self._hide_tab(self.PREVIEW)
        self._hide_tab(self.GRAPH)

    @property
    def command(self) -> str | None:
        """Return the command currently driving display tab state."""
        return self._command

    def consume_event(self, event: RuntimeEvent) -> bool:
        """Consume a runtime event and update display tabs if it carries state."""
        kind = self._event_text(getattr(event, "kind", None))
        message = self._event_text(getattr(event, "message", None))
        payload = self._event_payload(event)
        state = self._state_from_event(kind, payload, message)
        command = self._command_from_event(payload, message)

        if state in self._TERMINAL_STATES:
            if command is None or command == self._command:
                self.set_runtime_state(None, running=False)
                return True
            return False

        if command is None and kind in ("progress", "status") and self._command is not None:
            command = self._command

        if command is None:
            return False

        has_progress = getattr(event, "progress", None) is not None
        if kind in self._ACTIVE_KINDS or state is not None or has_progress:
            self.set_runtime_state(command, running=True)
            return True

        return False

    def set_runtime_state(self, command: str | None, *, running: bool) -> None:
        """Set display tabs from explicit process state."""
        normalized = self._normalize_command(command)
        self._command = normalized if running else None
        self._ensure_tab(self.ANALYSIS)

        if running and normalized in self._PREVIEW_COMMANDS:
            self._show_tab(self.PREVIEW)
        else:
            self._hide_tab(self.PREVIEW)

        if running and normalized in self._GRAPH_COMMANDS:
            self._show_tab(self.GRAPH)
        else:
            self._hide_tab(self.GRAPH)

    def tab_names(self) -> tuple[str, ...]:
        """Return all tabs currently owned by the QTabWidget."""
        return tuple(self._tabs.tabText(index) for index in range(self._tabs.count()))

    def visible_tab_names(self) -> tuple[str, ...]:
        """Return visible display tab names in widget order."""
        return tuple(
            self._tabs.tabText(index)
            for index in range(self._tabs.count())
            if self._tabs.isTabVisible(index)
        )

    def _adopt_existing_tabs(self) -> None:
        """Track placeholder tabs that were created before the controller attached."""
        for index in range(self._tabs.count()):
            label = self._tabs.tabText(index)
            if label in self._ORDER:
                self._widgets[label] = self._tabs.widget(index)  # type: ignore[assignment]

    def _ensure_tab(self, label: str) -> None:
        """Ensure a tab exists and is visible."""
        if self._index_of(label) == -1:
            widget = self._widgets.get(label)
            if widget is None:
                widget = self._factories[label]()
                self._widgets[label] = widget
            self._tabs.insertTab(self._insert_index(label), widget, label)
        self._set_tab_visible(label, True)

    def _show_tab(self, label: str) -> None:
        """Show a dynamic tab, creating it if needed."""
        self._ensure_tab(label)

    def _hide_tab(self, label: str) -> None:
        """Hide or remove a dynamic tab while preserving the Analysis tab."""
        if label == self.ANALYSIS:
            self._ensure_tab(label)
            return
        index = self._index_of(label)
        if index == -1:
            return
        if self._preserve_existing_tabs:
            self._tabs.setTabVisible(index, False)
            return
        widget = self._tabs.widget(index)
        self._tabs.removeTab(index)
        widget.setParent(None)  # type: ignore[union-attr]

    def _set_tab_visible(self, label: str, visible: bool) -> None:
        """Set tab visibility when the tab exists."""
        index = self._index_of(label)
        if index != -1:
            self._tabs.setTabVisible(index, visible)

    def _insert_index(self, label: str) -> int:
        """Return the ordered insertion index for a display tab."""
        label_order = self._ORDER.index(label)
        insert_index = self._tabs.count()
        for later_label in self._ORDER[label_order + 1 :]:
            later_index = self._index_of(later_label)
            if later_index != -1:
                insert_index = min(insert_index, later_index)
        return insert_index

    def _index_of(self, label: str) -> int:
        """Return the tab index for a label, or -1 when absent."""
        for index in range(self._tabs.count()):
            if self._tabs.tabText(index) == label:
                return index
        return -1

    @classmethod
    def _command_from_event(cls, payload: T.Mapping[str, object], message: str) -> str | None:
        """Extract a Faceswap command name from event payload or message."""
        for key in ("command", "display", "process", "task", "job", "script"):
            command = cls._normalize_command(payload.get(key))
            if command is not None:
                return command
        match = cls._COMMAND_RE.search(message)
        return cls._normalize_command(match.group(1)) if match is not None else None

    @classmethod
    def _normalize_command(cls, value: object) -> str | None:
        """Normalize event command values to extract/convert/train where possible."""
        if not isinstance(value, str):
            return None
        command = value.lower().replace(".py", "")
        match = cls._COMMAND_RE.search(command)
        return match.group(1).lower() if match is not None else command or None

    @classmethod
    def _state_from_event(
        cls,
        kind: str,
        payload: T.Mapping[str, object],
        message: str,
    ) -> str | None:
        """Extract a normalized runtime state from event fields."""
        running_task = payload.get("running_task")
        if isinstance(running_task, bool):
            return "running" if running_task else "finished"
        for key in ("state", "status", "phase", "event", "action"):
            state = cls._event_text(payload.get(key))
            if state:
                return state
        if kind in cls._TERMINAL_STATES:
            return kind
        searchable_text = f"{kind} {message}".replace("_", " ")
        for state in cls._TERMINAL_STATES | cls._ACTIVE_STATES:
            if re.search(rf"\b{re.escape(state)}\b", searchable_text):
                return state
        return None

    @staticmethod
    def _event_payload(event: RuntimeEvent) -> T.Mapping[str, object]:
        """Return the event payload as a mapping."""
        payload = getattr(event, "payload", None)
        return payload if isinstance(payload, Mapping) else {}

    @staticmethod
    def _event_text(value: object) -> str:
        """Return a normalized event text value."""
        return value.lower().strip() if isinstance(value, str) else ""

    @staticmethod
    def _placeholder_panel(name: str) -> QWidget:
        """Create a simple display placeholder panel."""
        panel = QWidget()
        panel.setObjectName(f"qt-shell-{name.lower()}-panel")
        panel.setMinimumWidth(0)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 8)
        label = QLabel(f"{name} display placeholder")
        label.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        label.setWordWrap(True)
        label.setObjectName(f"qt-shell-{name.lower()}-placeholder")
        layout.addWidget(label, 1)
        return panel
