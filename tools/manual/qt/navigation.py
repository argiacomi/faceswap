#!/usr/bin/env python3
"""Qt Manual Tool frame filtering, transport and playback helpers."""

from __future__ import annotations

from lib.gui.qt_shell.theme import QtTheme, icon_for_action


class NavigationMixin:
    """Own filtered frame navigation, transport callbacks and playback state."""

    def _all_frame_indices(self) -> tuple[int, ...]:
        """Return every known source-frame index."""
        return tuple(range(self._thumbnail_panel.count()))

    def _refresh_filter_results(self, *, preserve_current: bool = True) -> None:
        """Recompute ``_filtered_frame_indices`` from the active filter."""
        from tools.manual.frame_filter import (
            DEFAULT_FILTER_MODE,
            filtered_frame_indices,
            misaligned_predicate_for_model,
        )

        mode = self._editor_state.filter_mode or DEFAULT_FILTER_MODE
        threshold = int(self._editor_state.filter_distance)
        predicate = misaligned_predicate_for_model(self._editable, threshold)
        self._filtered_frame_indices = filtered_frame_indices(
            self._all_frame_indices(),
            self._editable.face_count,
            mode,
            misaligned_predicate=predicate,
        )
        total = len(self._filtered_frame_indices)
        self._transport_bar.set_total(total)
        if total == 0:
            self._sync_actions()
            self._refresh_filter_controls()
            self._refresh_face_grid()
            return
        current_row = self._thumbnail_panel.currentRow()
        if preserve_current and current_row in self._filtered_frame_indices:
            position = self._filtered_frame_indices.index(current_row)
            self._transport_bar.set_position(position)
        else:
            new_row = self._filtered_frame_indices[0]
            self._thumbnail_panel.setCurrentRow(new_row)
        self._sync_actions()
        self._refresh_filter_controls()
        self._refresh_face_grid()

    def filtered_frame_indices(self) -> tuple[int, ...]:
        """Return the current filtered frame index list."""
        return self._filtered_frame_indices

    def _filtered_position(self) -> int:
        """Return the current frame's index in ``_filtered_frame_indices``."""
        row = self._thumbnail_panel.currentRow()
        if row < 0:
            return -1
        try:
            return self._filtered_frame_indices.index(row)
        except ValueError:
            return -1

    def goto_first_frame(self) -> None:
        """Select the first frame in the active filter."""
        if not self._filtered_frame_indices:
            self.statusBar().showMessage(self._no_filter_match_message(), 3000)
            return
        self._stop_playback()
        self._thumbnail_panel.setCurrentRow(self._filtered_frame_indices[0])

    def goto_last_frame(self) -> None:
        """Select the last frame in the active filter."""
        if not self._filtered_frame_indices:
            self.statusBar().showMessage(self._no_filter_match_message(), 3000)
            return
        self._stop_playback()
        self._thumbnail_panel.setCurrentRow(self._filtered_frame_indices[-1])

    def _previous_frame(self) -> None:
        """Select previous frame from the active filter."""
        self._stop_playback()
        if not self._filtered_frame_indices:
            return
        position = self._filtered_position()
        if position <= 0:
            return
        self._thumbnail_panel.setCurrentRow(self._filtered_frame_indices[position - 1])

    def _next_frame(self) -> None:
        """Select next frame from the active filter."""
        self._stop_playback()
        if not self._filtered_frame_indices:
            return
        position = self._filtered_position()
        if position < 0 or position >= len(self._filtered_frame_indices) - 1:
            return
        self._thumbnail_panel.setCurrentRow(self._filtered_frame_indices[position + 1])

    def toggle_play(self) -> None:
        """Toggle playback through the filtered frames."""
        if self._editor_state.is_playing:
            self._stop_playback()
            return
        if not self._filtered_frame_indices:
            self.statusBar().showMessage(self._no_filter_match_message(), 3000)
            return
        position = self._filtered_position()
        if position < 0 or position >= len(self._filtered_frame_indices) - 1:
            self._thumbnail_panel.setCurrentRow(self._filtered_frame_indices[0])
        self._editor_state.set("is_playing", True)
        self._play_timer.start()
        self._sync_play_action_icon()

    def _stop_playback(self) -> None:
        """Halt the auto-advance timer and reset the play-action icon."""
        if self._play_timer.isActive():
            self._play_timer.stop()
        if self._editor_state.is_playing:
            self._editor_state.set("is_playing", False)
        self._sync_play_action_icon()

    def _advance_during_playback(self) -> None:
        """Step forward through the filtered list while playing."""
        if not self._filtered_frame_indices:
            self._stop_playback()
            return
        position = self._filtered_position()
        if position < 0 or position >= len(self._filtered_frame_indices) - 1:
            self._stop_playback()
            return
        self._thumbnail_panel.setCurrentRow(self._filtered_frame_indices[position + 1])

    def _no_filter_match_message(self) -> str:
        """Compose the empty-filter user message."""
        mode = self._editor_state.filter_mode or "All Frames"
        return f"No frames match filter: {mode}"

    def _sync_play_action_icon(self) -> None:
        """Switch the Play/Pause toolbar icon to match playback state."""
        action = self._actions.get("play_pause")
        if action is None:
            return
        theme = QtTheme.default()
        icon_key = "pause" if self._editor_state.is_playing else "play"
        icon = icon_for_action(theme, icon_key)
        if not icon.isNull():
            action.setIcon(icon)
        text = "Pause" if self._editor_state.is_playing else "Play"
        tooltip = (
            "Pause playback (Space)" if self._editor_state.is_playing else "Play playback (Space)"
        )
        action.setText(text)
        action.setToolTip(tooltip)
        action.setStatusTip(tooltip)

    def _on_thumbnail_row_changed(self, _row: int) -> None:
        """Refresh action availability and keep transport position in sync."""
        self._sync_actions()
        if not self._filtered_frame_indices:
            return
        row = self._thumbnail_panel.currentRow()
        try:
            position = self._filtered_frame_indices.index(row)
        except ValueError:
            return
        self._transport_bar.set_position(position)

    def _on_transport_position_changed(self, position: int) -> None:
        """Apply a user-driven slider or jump-entry change."""
        if not self._filtered_frame_indices:
            return
        if not 0 <= position < len(self._filtered_frame_indices):
            return
        target_row = self._filtered_frame_indices[position]
        if target_row == self._thumbnail_panel.currentRow():
            return
        self._stop_playback()
        self._thumbnail_panel.setCurrentRow(target_row)


__all__ = ["NavigationMixin"]
