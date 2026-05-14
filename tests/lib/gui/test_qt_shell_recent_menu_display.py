#!/usr/bin/env python3
"""Tests for Qt recent-file menu display adapter."""

from __future__ import annotations

from lib.gui.services.recent_files_store import RecentFile, RecentFileDisplay
from lib.gui.qt_shell.recent_menu import install_recent_menu_display


class _SignalDouble:
    """Small signal stand-in."""

    def __init__(self) -> None:
        self.callbacks = []

    def connect(self, callback) -> None:  # type:ignore[no-untyped-def]
        """Capture a callback connection."""
        self.callbacks.append(callback)


class _ActionDouble:
    """Small QAction stand-in."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.tooltip = ""
        self.enabled = True
        self.triggered = _SignalDouble()

    def setToolTip(self, tooltip: str) -> None:  # noqa:N802
        """Capture tooltip text."""
        self.tooltip = tooltip

    def setEnabled(self, enabled: bool) -> None:  # noqa:N802
        """Capture enabled state."""
        self.enabled = enabled


class _MenuDouble:
    """Small QMenu stand-in."""

    def __init__(self) -> None:
        self.actions = []
        self.separator_count = 0

    def clear(self) -> None:
        """Clear menu actions."""
        self.actions.clear()
        self.separator_count = 0

    def addAction(self, text, callback=None):  # type:ignore[no-untyped-def] # noqa:N802
        """Add an action."""
        action = _ActionDouble(text)
        if callback is not None:
            action.triggered.connect(callback)
        self.actions.append(action)
        return action

    def addSeparator(self) -> None:  # noqa:N802
        """Capture separator creation."""
        self.separator_count += 1


class _RecentFilesDouble:
    """Small RecentFilesStore stand-in."""

    def __init__(self, entries: list[RecentFileDisplay]) -> None:
        self.entries = entries
        self.prune_count = 0
        self.display_args = []

    def prune_missing(self) -> list[RecentFile]:
        """Return existing recent files."""
        self.prune_count += 1
        return [entry.file for entry in self.entries]

    def display_items(self, entries: list[RecentFile]) -> list[RecentFileDisplay]:
        """Return display entries."""
        self.display_args.append(entries)
        return self.entries


class _WindowDouble:
    """Small MainWindow stand-in."""

    def __init__(self, entries: list[RecentFileDisplay]) -> None:
        self._recent_menu = _MenuDouble()
        self._recent_files = _RecentFilesDouble(entries)
        self.opened = []
        self.clear_count = 0

    def _refresh_recent_menu(self) -> None:
        """Original placeholder refresh."""
        raise AssertionError("original refresh should be replaced")

    def _open_session_file(self, filename: str, kind: str) -> None:
        """Capture opened recent file."""
        self.opened.append((filename, kind))

    def _clear_recent_files(self) -> None:
        """Capture clear action."""
        self.clear_count += 1


def _entry(label: str, filename: str, kind: str = "project") -> RecentFileDisplay:
    """Return one recent display entry."""
    return RecentFileDisplay(RecentFile(filename, kind), label, filename)


def test_recent_menu_uses_display_items_for_labels_and_tooltips() -> None:
    """Recent menu adapter should use duplicate-aware display labels from the store."""
    Window = type("Window", (_WindowDouble,), {})
    install_recent_menu_display(Window)
    entries = [
        _entry("Project: project.fsw (/one)", "/one/project.fsw"),
        _entry("Project: project.fsw (/two)", "/two/project.fsw"),
    ]
    window = Window(entries)

    window._refresh_recent_menu()

    assert window._recent_files.prune_count == 1
    assert [action.text for action in window._recent_menu.actions] == [
        "Project: project.fsw (/one)",
        "Project: project.fsw (/two)",
        "Clear Recent Files",
    ]
    assert [action.tooltip for action in window._recent_menu.actions[:2]] == [
        "/one/project.fsw",
        "/two/project.fsw",
    ]
    assert window._recent_menu.separator_count == 1


def test_recent_menu_actions_route_to_open_session_file() -> None:
    """Recent file actions should route filename and kind to the window open handler."""
    Window = type("Window", (_WindowDouble,), {})
    install_recent_menu_display(Window)
    window = Window([_entry("Task: task.fst", "/tmp/task.fst", "task")])
    window._refresh_recent_menu()

    window._recent_menu.actions[0].triggered.callbacks[0]()

    assert window.opened == [("/tmp/task.fst", "task")]


def test_recent_menu_empty_state_disables_actions() -> None:
    """Empty recent menus should show a disabled empty-state action and disabled clear."""
    Window = type("Window", (_WindowDouble,), {})
    install_recent_menu_display(Window)
    window = Window([])

    window._refresh_recent_menu()

    assert [action.text for action in window._recent_menu.actions] == [
        "No recent files",
        "Clear Recent Files",
    ]
    assert [action.enabled for action in window._recent_menu.actions] == [False, False]


def test_recent_menu_installation_is_idempotent() -> None:
    """Installing the adapter multiple times should not replace wrapped method again."""
    Window = type("Window", (_WindowDouble,), {})
    install_recent_menu_display(Window)
    refresh = Window._refresh_recent_menu

    install_recent_menu_display(Window)

    assert Window._refresh_recent_menu is refresh
