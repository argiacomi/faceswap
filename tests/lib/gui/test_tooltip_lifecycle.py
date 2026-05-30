#!/usr/bin/env python3
"""Lifecycle-safety tests for the Tk :class:`~lib.gui.custom_widgets.Tooltip` (issue #200).

These exercise the defensive show/hide/cleanup behaviour so that a tool-tip cannot be
left visible indefinitely after its source widget is no longer hovered, mapped, or alive.

A real Tk root is required.  The whole module is skipped when no display is available
(e.g. headless CI without an X server) rather than failing.
"""

from __future__ import annotations

import tkinter as tk
from types import SimpleNamespace

import pytest

# Skip the entire module if a Tk display cannot be created (headless CI).
_root: tk.Tk | None = None
try:
    _root = tk.Tk()
    _root.withdraw()
except tk.TclError:  # pragma: no cover - depends on the test host
    _root = None
    pytest.skip("No Tk display available", allow_module_level=True)


_THEME = {
    "tooltip": {
        "background_color": "#000000",
        "border_color": "#ffffff",
        "font_color": "#ffffff",
    }
}


@pytest.fixture(name="tk_root")
def fixture_tk_root() -> tk.Tk:
    """Return a shared, mapped Tk root for the module.

    The root must be visible (not withdrawn) so that child widgets report as mapped, which
    the defensive ``Tooltip._show`` guard requires before creating the tool-tip window.
    """
    assert _root is not None
    _root.deiconify()
    _root.update()
    return _root


@pytest.fixture(name="tooltip_cls")
def fixture_tooltip_cls(monkeypatch: pytest.MonkeyPatch):
    """Return the :class:`Tooltip` class with ``get_config`` stubbed to a fake theme.

    Patching ``get_config`` avoids needing the full GUI configuration stack just to read
    the tool-tip colour theme.
    """
    from lib.gui import custom_widgets

    monkeypatch.setattr(
        custom_widgets,
        "get_config",
        lambda: SimpleNamespace(user_theme=_THEME),
    )
    return custom_widgets.Tooltip


def test_hide_is_idempotent(tk_root: tk.Tk, tooltip_cls) -> None:
    """Forcing a show then hiding twice must not raise and must clear the window."""
    widget = tk.Frame(tk_root)
    widget.pack()
    tk_root.update()
    tip = tooltip_cls(widget, text="hello")

    tip._show()  # pylint:disable=protected-access
    assert tip._topwidget is not None  # pylint:disable=protected-access

    tip._hide()  # pylint:disable=protected-access
    tip._hide()  # second hide must be a safe no-op  # pylint:disable=protected-access
    assert tip._topwidget is None  # pylint:disable=protected-access

    widget.destroy()


def test_show_twice_tracks_single_window(tk_root: tk.Tk, tooltip_cls) -> None:
    """Calling ``_show`` repeatedly must not orphan the previous Toplevel."""
    widget = tk.Frame(tk_root)
    widget.pack()
    tk_root.update()
    tip = tooltip_cls(widget, text="hello")

    tip._show()  # pylint:disable=protected-access
    first = tip._topwidget  # pylint:disable=protected-access
    tip._show()  # pylint:disable=protected-access
    second = tip._topwidget  # pylint:disable=protected-access

    assert second is not None
    assert first is not second
    # The first window must have been destroyed rather than orphaned.
    assert not first.winfo_exists()

    tip._hide()  # pylint:disable=protected-access
    widget.destroy()


def test_destroying_source_widget_cleans_up(tk_root: tk.Tk, tooltip_cls) -> None:
    """Destroying the source widget while a tool-tip is visible must tear it down."""
    widget = tk.Frame(tk_root)
    widget.pack()
    tk_root.update()
    tip = tooltip_cls(widget, text="hello")

    tip._show()  # pylint:disable=protected-access
    topwidget = tip._topwidget  # pylint:disable=protected-access
    assert topwidget is not None

    widget.destroy()  # fires <Destroy>, routed to _on_destroy
    tk_root.update()

    assert tip._topwidget is None  # pylint:disable=protected-access
    assert not topwidget.winfo_exists()


def test_child_destroy_event_is_ignored(tk_root: tk.Tk, tooltip_cls) -> None:
    """A child widget being destroyed must not tear down the parent's tool-tip."""
    widget = tk.Frame(tk_root)
    widget.pack()
    child = tk.Frame(widget)
    child.pack()
    tk_root.update()
    tip = tooltip_cls(widget, text="hello")

    tip._show()  # pylint:disable=protected-access
    assert tip._topwidget is not None  # pylint:disable=protected-access

    # Destroying the child should be ignored by _on_destroy (event.widget is child).
    child.destroy()
    tk_root.update()
    assert tip._topwidget is not None  # pylint:disable=protected-access

    tip._hide()  # pylint:disable=protected-access
    widget.destroy()


def test_unschedule_survives_stale_callback(tk_root: tk.Tk, tooltip_cls) -> None:
    """Cancelling a stale/destroyed callback id must be suppressed, not raised."""
    widget = tk.Frame(tk_root)
    widget.pack()
    tk_root.update()
    tip = tooltip_cls(widget, text="hello")

    # Inject a bogus callback id; _unschedule must swallow the resulting TclError.
    tip._ident = "after#nonexistent"  # pylint:disable=protected-access
    tip._lifetime_ident = "after#alsogone"  # pylint:disable=protected-access
    tip._unschedule()  # pylint:disable=protected-access

    assert tip._ident is None  # pylint:disable=protected-access
    assert tip._lifetime_ident is None  # pylint:disable=protected-access

    widget.destroy()
