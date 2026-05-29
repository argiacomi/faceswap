#!/usr/bin/env python3
"""Regression tests for GUI display page teardown and subnotebook lifecycle."""

from __future__ import annotations

from unittest.mock import MagicMock

from lib.gui.display import DisplayNotebook
from lib.gui.display_page import DisplayOptionalPage, DisplayPage


class _FakeFrame:
    def __init__(self, master) -> None:
        self.master = master
        self.pack_calls: list[dict[str, object]] = []

    def pack(self, **kwargs) -> None:
        self.pack_calls.append(kwargs)


class _FakeNotebook:
    def __init__(self, *, exists: bool = True) -> None:
        self._exists = exists
        self.children: dict[str, object] = {}
        self.add_calls: list[tuple[object, str]] = []
        self.configure_calls: list[dict[str, object]] = []

    def winfo_exists(self) -> bool:
        return self._exists

    def add(self, frame, text: str) -> None:
        self.add_calls.append((frame, text))
        self.children[f"tab_{len(self.children)}"] = frame

    def configure(self, **kwargs) -> None:
        self.configure_calls.append(kwargs)


def test_display_page_recreates_invalid_subnotebook_before_adding_page(monkeypatch) -> None:
    """Adding a subnotebook page should recreate a destroyed notebook handle first."""
    import lib.gui.display_page as display_page_mod

    page = DisplayPage.__new__(DisplayPage)
    page.tabname = "preview"
    page.subnotebook = _FakeNotebook(exists=False)

    recreated = _FakeNotebook()
    page.subnotebook_show = lambda: setattr(page, "subnotebook", recreated)

    monkeypatch.setattr(display_page_mod.ttk, "Frame", _FakeFrame)

    frame = DisplayPage.subnotebook_add_page(page, "Preview")

    assert page.subnotebook is recreated
    assert isinstance(frame, _FakeFrame)
    assert frame.master is recreated
    assert recreated.add_calls == [(frame, "Preview")]
    assert recreated.configure_calls == [{"style": "single.TNotebook"}]


def test_display_optional_page_close_cancels_pending_update() -> None:
    """Closing an optional page should cancel any scheduled refresh and destroy children."""
    page = DisplayOptionalPage.__new__(DisplayOptionalPage)
    page._update_after_id = "after#1"
    page._after_ids = []
    page.after_cancel = MagicMock()
    child = MagicMock()
    page.winfo_children = lambda: [child]

    DisplayOptionalPage.close(page)

    page.after_cancel.assert_called_once_with("after#1")
    child.destroy.assert_called_once_with()


def test_display_notebook_remove_tabs_destroys_optional_page_widgets() -> None:
    """Removing optional tabs should destroy the forgotten page widget."""
    notebook = DisplayNotebook.__new__(DisplayNotebook)
    notebook._static_tabs = [".analysis"]
    notebook.tabs = lambda: [".analysis", ".preview1"]
    notebook.children = {"preview1": MagicMock()}
    notebook.forget = MagicMock()

    DisplayNotebook._remove_tabs(notebook)

    child_object = notebook.children["preview1"]
    child_object.close.assert_called_once_with()
    notebook.forget.assert_called_once_with(".preview1")
    child_object.destroy.assert_called_once_with()


def test_display_notebook_skips_rebuild_when_command_unchanged() -> None:
    """Setting the display var to the already-displayed command must be a no-op (#188).

    Re-entrant trace events were unconditionally tearing down + rebuilding the
    preview tab on every fire, which is how the duplicate-window-name crash
    appeared on extract/convert.
    """
    notebook = DisplayNotebook.__new__(DisplayNotebook)
    notebook._displayed_command = "extract"
    notebook._updating_displaybook = False
    notebook._wrapper_var = MagicMock()
    notebook._wrapper_var.get = MagicMock(return_value="extract")
    notebook._remove_tabs = MagicMock()
    notebook._command_display = MagicMock()

    DisplayNotebook._update_displaybook(notebook)

    notebook._remove_tabs.assert_not_called()
    notebook._command_display.assert_not_called()
    assert notebook._displayed_command == "extract"


def test_display_notebook_ignores_reentrant_update() -> None:
    """A re-entrant trace must not invoke teardown + rebuild a second time (#188)."""
    notebook = DisplayNotebook.__new__(DisplayNotebook)
    notebook._displayed_command = None
    notebook._updating_displaybook = True  # simulate an in-flight rebuild
    notebook._wrapper_var = MagicMock()
    notebook._wrapper_var.get = MagicMock(return_value="extract")
    notebook._remove_tabs = MagicMock()
    notebook._command_display = MagicMock()

    DisplayNotebook._update_displaybook(notebook)

    notebook._remove_tabs.assert_not_called()
    notebook._command_display.assert_not_called()
    # The guard must NOT mutate displayed_command on the re-entrant path —
    # the in-flight rebuild owns that field.
    assert notebook._displayed_command is None


def test_display_notebook_rebuild_updates_displayed_command_on_change() -> None:
    """A genuine command transition tears down, rebuilds, and records the new state."""
    notebook = DisplayNotebook.__new__(DisplayNotebook)
    notebook._displayed_command = "train"
    notebook._updating_displaybook = False
    notebook._wrapper_var = MagicMock()
    notebook._wrapper_var.get = MagicMock(return_value="extract")
    notebook._remove_tabs = MagicMock()
    notebook._command_display = MagicMock()

    DisplayNotebook._update_displaybook(notebook)

    notebook._remove_tabs.assert_called_once_with()
    notebook._command_display.assert_called_once_with("extract")
    assert notebook._displayed_command == "extract"
    assert notebook._updating_displaybook is False


def test_display_notebook_rebuild_clears_state_for_unknown_command() -> None:
    """An unknown / empty command tears down optional tabs and clears the
    displayed-command marker so a subsequent extract/train/convert request
    runs a full rebuild rather than being skipped by the unchanged-command
    guard."""
    notebook = DisplayNotebook.__new__(DisplayNotebook)
    notebook._displayed_command = "extract"
    notebook._updating_displaybook = False
    notebook._wrapper_var = MagicMock()
    notebook._wrapper_var.get = MagicMock(return_value="")
    notebook._remove_tabs = MagicMock()
    notebook._command_display = MagicMock()

    DisplayNotebook._update_displaybook(notebook)

    notebook._remove_tabs.assert_called_once_with()
    notebook._command_display.assert_not_called()
    assert notebook._displayed_command is None


def test_display_notebook_remove_tabs_tcl_destroy_fallback() -> None:
    """If the Python child lookup misses, fall back to a Tcl ``destroy`` so a
    stale widget cannot survive into the next rebuild (#188)."""
    notebook = DisplayNotebook.__new__(DisplayNotebook)
    notebook._static_tabs = [".analysis"]
    notebook.tabs = lambda: [".analysis", ".previewextract2"]
    # Empty children dict → Python lookup misses the optional tab.
    notebook.children = {}
    notebook.forget = MagicMock()
    fake_tk = MagicMock()
    notebook.tk = fake_tk

    DisplayNotebook._remove_tabs(notebook)

    notebook.forget.assert_called_once_with(".previewextract2")
    fake_tk.call.assert_called_once_with("destroy", ".previewextract2")


def test_display_notebook_remove_tabs_swallows_tcl_errors() -> None:
    """``forget``/``destroy`` raising TclError during teardown must not propagate
    — once the widget is gone, the rebuild path must still proceed."""
    import tkinter as tk

    notebook = DisplayNotebook.__new__(DisplayNotebook)
    notebook._static_tabs = []
    notebook.tabs = lambda: [".previewextract2"]
    notebook.children = {}
    notebook.forget = MagicMock(side_effect=tk.TclError("already destroyed"))
    fake_tk = MagicMock()
    fake_tk.call = MagicMock(side_effect=tk.TclError("already destroyed"))
    notebook.tk = fake_tk

    # Should not raise — both fallbacks tolerate the widget being already gone.
    DisplayNotebook._remove_tabs(notebook)

    notebook.forget.assert_called_once()
    fake_tk.call.assert_called_once_with("destroy", ".previewextract2")
