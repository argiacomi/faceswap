#!/usr/bin/env python3
"""Lifecycle-safety tests for the Tk :class:`~lib.gui.custom_widgets.Tooltip` (issue #200).

These exercise the defensive show/hide/cleanup behaviour so that a tool-tip cannot be
left visible indefinitely after its source widget is no longer hovered, mapped, or alive.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

_THEME = {
    "tooltip": {
        "background_color": "#000000",
        "border_color": "#ffffff",
        "font_color": "#ffffff",
    }
}


class _FakeWidget:
    """Display-free stand-in for the Tk source widget."""

    def __init__(self) -> None:
        self.exists = True
        self.mapped = True
        self.bindings: dict[str, object] = {}
        self.cancelled_after_ids: list[str] = []
        self._after_count = 0

    def bind(self, sequence: str, callback: object, *, add: str | None = None) -> None:
        """Record additive bindings without touching a display server."""
        assert add == "+"
        self.bindings[sequence] = callback

    def after(self, _wait_time: int, _callback: object) -> str:
        """Return a stable fake callback id."""
        self._after_count += 1
        return f"after#{self._after_count}"

    def after_cancel(self, callback_id: str) -> None:
        """Cancel known ids and raise for intentionally bogus stale ids."""
        if callback_id.startswith("after#nonexistent") or callback_id.startswith("after#alsogone"):
            from lib.gui import custom_widgets

            raise custom_widgets.TclError(callback_id)
        self.cancelled_after_ids.append(callback_id)

    def destroy(self) -> None:
        """Mark destroyed and emit the widget's own destroy callback."""
        self.exists = False
        event = SimpleNamespace(widget=self)
        callback = self.bindings.get("<Destroy>")
        if callback is not None:
            callback(event)  # type: ignore[operator]

    def winfo_exists(self) -> bool:
        """Return whether this fake widget still exists."""
        return self.exists

    def winfo_ismapped(self) -> bool:
        """Return whether this fake widget is mapped."""
        return self.mapped

    def winfo_screenwidth(self) -> int:
        """Return a fake screen width for tooltip placement."""
        return 1920

    def winfo_screenheight(self) -> int:
        """Return a fake screen height for tooltip placement."""
        return 1080

    def winfo_pointerxy(self) -> tuple[int, int]:
        """Return a fake pointer position for tooltip placement."""
        return (100, 100)


class _FakeTopLevel:
    """Display-free stand-in for ``tk.Toplevel``."""

    instances: list[_FakeTopLevel] = []

    def __init__(self, _widget: _FakeWidget) -> None:
        self.exists = True
        self.geometry = ""
        self.bindings: dict[str, object] = {}
        self.tk = SimpleNamespace(call=lambda *_args: None)
        self._w = ".tooltip"
        self.instances.append(self)

    def wm_overrideredirect(self, _value: bool) -> None:
        """Accept Tk window decoration changes."""

    def wm_geometry(self, geometry: str) -> None:
        """Record requested window geometry."""
        self.geometry = geometry

    def bind(self, sequence: str, callback: object, *, add: str | None = None) -> None:
        """Record tooltip-local bindings."""
        assert add == "+"
        self.bindings[sequence] = callback

    def winfo_exists(self) -> bool:
        """Return whether this fake toplevel still exists."""
        return self.exists

    def destroy(self) -> None:
        """Mark the fake toplevel destroyed."""
        self.exists = False


class _FakeFrame:
    """Display-free stand-in for ``tk.Frame``."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.gridded = False

    def grid(self, *_args: object, **_kwargs: object) -> None:
        """Record that the frame was laid out."""
        self.gridded = True


class _FakeLabel:
    """Display-free stand-in for ``tk.Label``."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.gridded = False

    def grid(self, *_args: object, **_kwargs: object) -> None:
        """Record that the label was laid out."""
        self.gridded = True

    def winfo_reqwidth(self) -> int:
        """Return a fake requested width."""
        return 120

    def winfo_reqheight(self) -> int:
        """Return a fake requested height."""
        return 30


@pytest.fixture(name="tooltip_cls")
def fixture_tooltip_cls(monkeypatch: pytest.MonkeyPatch):
    """Return the :class:`Tooltip` class with ``get_config`` stubbed to a fake theme.

    Patching ``get_config`` avoids needing the full GUI configuration stack just to read
    the tool-tip colour theme.
    """
    from lib.gui import custom_widgets

    monkeypatch.setattr(custom_widgets, "get_config", lambda: SimpleNamespace(user_theme=_THEME))
    monkeypatch.setattr(custom_widgets.tk, "Toplevel", _FakeTopLevel)
    monkeypatch.setattr(custom_widgets.tk, "Frame", _FakeFrame)
    monkeypatch.setattr(custom_widgets.tk, "Label", _FakeLabel)
    _FakeTopLevel.instances.clear()
    return custom_widgets.Tooltip


@pytest.fixture(name="source_widget")
def fixture_source_widget() -> _FakeWidget:
    """Return a mapped fake source widget."""
    return _FakeWidget()


def test_hide_is_idempotent(source_widget: _FakeWidget, tooltip_cls) -> None:
    """Forcing a show then hiding twice must not raise and must clear the window."""
    tip = tooltip_cls(source_widget, text="hello")

    tip._show()  # pylint:disable=protected-access
    assert tip._topwidget is not None  # pylint:disable=protected-access

    tip._hide()  # pylint:disable=protected-access
    tip._hide()  # second hide must be a safe no-op  # pylint:disable=protected-access
    assert tip._topwidget is None  # pylint:disable=protected-access


def test_show_twice_tracks_single_window(source_widget: _FakeWidget, tooltip_cls) -> None:
    """Calling ``_show`` repeatedly must not orphan the previous Toplevel."""
    tip = tooltip_cls(source_widget, text="hello")

    tip._show()  # pylint:disable=protected-access
    first = tip._topwidget  # pylint:disable=protected-access
    tip._show()  # pylint:disable=protected-access
    second = tip._topwidget  # pylint:disable=protected-access

    assert second is not None
    assert first is not second
    # The first window must have been destroyed rather than orphaned.
    assert not first.winfo_exists()

    tip._hide()  # pylint:disable=protected-access


def test_destroying_source_widget_cleans_up(source_widget: _FakeWidget, tooltip_cls) -> None:
    """Destroying the source widget while a tool-tip is visible must tear it down."""
    tip = tooltip_cls(source_widget, text="hello")

    tip._show()  # pylint:disable=protected-access
    topwidget = tip._topwidget  # pylint:disable=protected-access
    assert topwidget is not None

    source_widget.destroy()  # fires <Destroy>, routed to _on_destroy

    assert tip._topwidget is None  # pylint:disable=protected-access
    assert not topwidget.winfo_exists()


def test_child_destroy_event_is_ignored(source_widget: _FakeWidget, tooltip_cls) -> None:
    """A child widget being destroyed must not tear down the parent's tool-tip."""
    child = _FakeWidget()
    tip = tooltip_cls(source_widget, text="hello")

    tip._show()  # pylint:disable=protected-access
    assert tip._topwidget is not None  # pylint:disable=protected-access

    # Destroying the child should be ignored by _on_destroy (event.widget is child).
    tip._on_destroy(SimpleNamespace(widget=child))  # pylint:disable=protected-access
    assert tip._topwidget is not None  # pylint:disable=protected-access

    tip._hide()  # pylint:disable=protected-access


def test_unschedule_survives_stale_callback(source_widget: _FakeWidget, tooltip_cls) -> None:
    """Cancelling a stale/destroyed callback id must be suppressed, not raised."""
    tip = tooltip_cls(source_widget, text="hello")

    # Inject a bogus callback id; _unschedule must swallow the resulting TclError.
    tip._ident = "after#nonexistent"  # pylint:disable=protected-access
    tip._lifetime_ident = "after#alsogone"  # pylint:disable=protected-access
    tip._unschedule()  # pylint:disable=protected-access

    assert tip._ident is None  # pylint:disable=protected-access
    assert tip._lifetime_ident is None  # pylint:disable=protected-access
