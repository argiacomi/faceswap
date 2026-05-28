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
