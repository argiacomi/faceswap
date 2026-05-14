#!/usr/bin/env python3
"""Recent-files menu adapter for the Qt shell."""

from __future__ import annotations

import functools

from lib.utils import get_module_objects


def install_recent_menu_display(main_window_class: type) -> None:
    """Install duplicate-aware recent-file labels on the Qt MainWindow class."""
    if getattr(main_window_class, "_recent_menu_display_installed", False):
        return

    original_refresh = main_window_class._refresh_recent_menu

    @functools.wraps(original_refresh)
    def refresh_recent_menu(self, *args, **kwargs):  # type:ignore[no-untyped-def]
        recent_menu = getattr(self, "_recent_menu", None)
        recent_files = getattr(self, "_recent_files", None)
        if recent_menu is None or recent_files is None:
            return original_refresh(self, *args, **kwargs)
        recent_menu.clear()
        entries = recent_files.prune_missing()
        display_items = recent_files.display_items(entries)
        if not display_items:
            action = recent_menu.addAction("No recent files")
            action.setEnabled(False)
        else:
            for display_item in display_items:
                action = recent_menu.addAction(display_item.label)
                action.setToolTip(display_item.tooltip)
                action.triggered.connect(
                    functools.partial(
                        self._open_session_file,
                        display_item.file.filename,
                        display_item.file.kind,
                    )
                )
        recent_menu.addSeparator()
        clear_action = recent_menu.addAction("Clear Recent Files", self._clear_recent_files)
        clear_action.setEnabled(bool(display_items))
        return None

    main_window_class._refresh_recent_menu = refresh_recent_menu
    main_window_class._recent_menu_display_installed = True


__all__ = get_module_objects(__name__)
