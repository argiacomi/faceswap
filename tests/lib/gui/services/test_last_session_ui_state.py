#!/usr/bin/env python3
"""Tests for last-session UI state persistence."""

from __future__ import annotations

from pathlib import Path

from lib.gui.services.project_session_service import PROJECT_KIND, LastSessionStore
from lib.serializer import get_serializer


def test_last_session_store_round_trips_ui_state(tmp_path: Path) -> None:
    """LastSessionStore should persist optional UI/session state."""
    session_file = tmp_path / "last.json"
    project_file = tmp_path / "example.fsw"
    project_file.write_text("{}", encoding="utf-8")
    store = LastSessionStore(get_serializer("json"), str(session_file))
    ui_state = {
        "display_tab": "Preview",
        "window_size": [1200, 800],
        "main_splitter": [400, 800],
        "vertical_splitter": [600, 200],
        "preview_source": str(tmp_path / "preview"),
    }

    store.save(str(project_file), PROJECT_KIND, ui_state)
    loaded = store.load()

    assert loaded is not None
    assert loaded.filename == str(project_file)
    assert loaded.kind == PROJECT_KIND
    assert loaded.ui_state == ui_state


def test_last_session_store_defaults_missing_ui_state(tmp_path: Path) -> None:
    """Older last-session payloads without UI state should still load."""
    session_file = tmp_path / "last.json"
    project_file = tmp_path / "example.fsw"
    project_file.write_text("{}", encoding="utf-8")
    serializer = get_serializer("json")
    serializer.save(
        str(session_file),
        {"filename": str(project_file), "kind": PROJECT_KIND},
    )
    store = LastSessionStore(serializer, str(session_file))

    loaded = store.load()

    assert loaded is not None
    assert loaded.ui_state == {}


def test_last_session_store_ignores_invalid_ui_state(tmp_path: Path) -> None:
    """Invalid UI-state payloads should fall back to an empty state."""
    session_file = tmp_path / "last.json"
    project_file = tmp_path / "example.fsw"
    project_file.write_text("{}", encoding="utf-8")
    serializer = get_serializer("json")
    serializer.save(
        str(session_file),
        {"filename": str(project_file), "kind": PROJECT_KIND, "ui_state": []},
    )
    store = LastSessionStore(serializer, str(session_file))

    loaded = store.load()

    assert loaded is not None
    assert loaded.ui_state == {}
