#!/usr/bin/env python3
"""Targeted regression tests for the #193 Tk GUI cleanup commit.

Covers:

* Module-level regex constants in ``custom_widgets`` are reused across
  ``_SysOutRouter`` instances (not re-compiled per init).
* ``Stats._sessions`` returns ``{}`` on empty / missing state without
  raising; ``logging_disabled`` is safe on partial state too.
* ``GraphDisplay._add_trace_variables`` uses the module-level trace var
  names tuple instead of re-computing ``T.get_args(T.Literal[...])``.

These tests intentionally exercise the helpers WITHOUT spinning up a
full Tk root — the GUI suite already has display-dependent integration
coverage via ``display_lifecycle_test.py``.
"""

from __future__ import annotations

import re

from lib.gui import custom_widgets

# ---------------------------------------------------------------------------
# custom_widgets — module-level regex constants
# ---------------------------------------------------------------------------


def test_recolor_and_ansi_constants_are_compiled_once() -> None:
    """Both module constants exist, are ``re.Pattern`` instances, and
    every ``_SysOutRouter`` shares the SAME object (no per-instance
    ``re.compile`` cost)."""
    assert isinstance(custom_widgets._RECOLOR_RE, re.Pattern)
    assert isinstance(custom_widgets._ANSI_ESCAPE_RE, re.Pattern)

    class _FakeConsole:
        def insert(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            pass

        def see(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            pass

    a = custom_widgets._SysOutRouter(_FakeConsole(), "stdout")
    b = custom_widgets._SysOutRouter(_FakeConsole(), "stdout")

    assert a._recolor is custom_widgets._RECOLOR_RE
    assert a._ansi_escape is custom_widgets._ANSI_ESCAPE_RE
    assert b._recolor is a._recolor
    assert b._ansi_escape is a._ansi_escape


def test_recolor_pattern_still_matches_log_line() -> None:
    """The hoisted pattern must still pick out the ``lvl`` group from a
    real-shaped log line — regression against an accidental escape
    change during the refactor."""
    line = "12345 12:34:56 INFO     Test message"
    m = custom_widgets._RECOLOR_RE.match(line)
    assert m is not None
    assert m["lvl"] == "INFO"


def test_ansi_escape_strips_control_codes() -> None:
    """Sanity: hoisted ANSI pattern still removes the standard
    ``\\x1B[31m`` color codes."""
    coloured = "\x1b[31mERROR\x1b[0m: boom"
    assert custom_widgets._ANSI_ESCAPE_RE.sub("", coloured) == "ERROR: boom"


# ---------------------------------------------------------------------------
# analysis/stats — _sessions property + safe logging_disabled
# ---------------------------------------------------------------------------


def test_sessions_returns_empty_dict_on_missing_state() -> None:
    """``_sessions`` must NOT raise when the state file is missing or
    has no ``sessions`` key — the #193 P3 guard requirement."""
    from lib.gui.analysis.stats import GlobalSession as Session

    instance = Session.__new__(Session)

    instance._state = None
    assert instance._sessions == {}

    instance._state = {}
    assert instance._sessions == {}

    instance._state = {"sessions": {}}
    assert instance._sessions == {}

    instance._state = {"sessions": {"0": {"batchsize": 4, "no_logs": False}}}
    assert "0" in instance._sessions


def test_logging_disabled_returns_true_when_no_sessions() -> None:
    """``logging_disabled`` short-circuits via ``_sessions`` instead of
    indexing ``self._state["sessions"]`` directly — issue #193."""
    from lib.gui.analysis.stats import GlobalSession as Session

    instance = Session.__new__(Session)
    instance._state = None
    instance._is_training = False
    assert instance.logging_disabled is True

    instance._state = {"sessions": {}}
    assert instance.logging_disabled is True


def test_logging_disabled_returns_session_no_logs_flag() -> None:
    """On a state with two sessions, ``logging_disabled`` returns the
    ``no_logs`` flag from the highest-numbered session (the active one)."""
    from lib.gui.analysis.stats import GlobalSession as Session

    instance = Session.__new__(Session)
    instance._state = {
        "sessions": {
            "0": {"no_logs": False, "batchsize": 8},
            "1": {"no_logs": True, "batchsize": 16},
        }
    }
    instance._is_training = False
    assert instance.logging_disabled is True

    instance._state = {
        "sessions": {
            "0": {"no_logs": True, "batchsize": 8},
            "1": {"no_logs": False, "batchsize": 16},
        }
    }
    assert instance.logging_disabled is False


# ---------------------------------------------------------------------------
# display_command — module-level trace var names
# ---------------------------------------------------------------------------


def test_graph_trace_var_names_tuple_matches_literal() -> None:
    """The hoisted ``_GRAPH_TRACE_VAR_NAMES`` tuple matches the original
    ``T.get_args(T.Literal[...])`` it replaced — pins the trace-var
    surface so ``_add_trace_variables`` / ``_clear_trace_variables``
    keep agreeing on which Tk vars they're tracking."""
    import typing as T

    from lib.gui.display_command import _GRAPH_TRACE_VAR_NAMES

    assert T.get_args(T.Literal["smoothgraph", "display_iterations"]) == _GRAPH_TRACE_VAR_NAMES
    # Tuple, not list — frozen at import time.
    assert isinstance(_GRAPH_TRACE_VAR_NAMES, tuple)
