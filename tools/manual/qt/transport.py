#!/usr/bin/env python3
"""Qt Manual Tool implementation module."""

from __future__ import annotations

import logging

from PySide6.QtCore import (
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QIntValidator,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSlider,
    QWidget,
)

logger = logging.getLogger(__name__)


class ManualTransportBar(QWidget):
    """Transport controls below the frame view (slider + jump entry + counter).

    The widget is deliberately driven by ``set_total`` / ``set_position`` calls
    and emits ``position_changed`` for any user-driven motion (slider drag,
    jump-entry submit).  The host window owns the actual frame index — this is
    purely UI glue.

    Layout::

        [ slider ─────────────────────────────── ] [Frame: 12 / 250] [Go: 12]

    """

    position_changed = Signal(int)
    """Emitted when the user moves the slider or submits a jump-to value."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("qt-manual-transport-bar")
        self._total = 0
        self._suppress_signal = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(8)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setObjectName("qt-manual-transport-slider")
        self._slider.setRange(0, 0)
        self._slider.setEnabled(False)
        self._slider.setTracking(True)
        self._slider.valueChanged.connect(self._on_slider_value_changed)
        layout.addWidget(self._slider, 1)

        self._counter = QLabel("Frame: – / 0")
        self._counter.setObjectName("qt-manual-transport-counter")
        layout.addWidget(self._counter)

        self._jump_label = QLabel("Go:")
        layout.addWidget(self._jump_label)
        self._jump_entry = QLineEdit()
        self._jump_entry.setObjectName("qt-manual-transport-jump")
        self._jump_entry.setFixedWidth(60)
        # Jump entry is 1-based to match the visible ``Frame: X / N`` counter.
        # Validator range follows: ``[1, total]`` when total > 0, else ``[0, 0]``
        # so the entry stays uneditable when no frames are loaded.
        self._jump_entry.setValidator(QIntValidator(0, 0, self))
        self._jump_entry.setPlaceholderText("#")
        # ``editingFinished`` already fires on Return *and* on focus-out, so
        # there is no need to also connect ``returnPressed`` — doing both
        # makes a single Return keystroke emit ``position_changed`` twice
        # (issue #119 task 3).  ``editingFinished`` covers both paths.
        self._jump_entry.editingFinished.connect(self._on_jump_submit)
        layout.addWidget(self._jump_entry)

    @property
    def slider(self) -> QSlider:
        """Return the underlying transport slider widget (for tests/styling)."""
        return self._slider

    @property
    def jump_entry(self) -> QLineEdit:
        """Return the jump-to-frame line edit (for tests)."""
        return self._jump_entry

    @property
    def counter_label(self) -> QLabel:
        """Return the ``Frame: X / N`` counter label (for tests/styling)."""
        return self._counter

    def set_total(self, total: int) -> None:
        """Resize the transport range to ``total`` frames (0 disables the bar)."""
        self._total = max(0, int(total))
        last = max(0, self._total - 1)
        self._slider.setEnabled(self._total > 0)
        self._slider.setRange(0, last)
        self._jump_entry.setEnabled(self._total > 0)
        validator = self._jump_entry.validator()
        if isinstance(validator, QIntValidator):
            # 1-based validator: when no frames are loaded keep the entry pinned
            # to (0, 0) so it rejects all input; otherwise accept ``1..total``.
            if self._total > 0:
                validator.setRange(1, self._total)
            else:
                validator.setRange(0, 0)
        self._update_counter(self._slider.value())

    def set_position(self, position: int) -> None:
        """Snap the slider + counter to ``position`` without emitting signals."""
        if self._total <= 0:
            return
        clamped = max(0, min(int(position), self._total - 1))
        self._suppress_signal = True
        try:
            self._slider.setValue(clamped)
        finally:
            self._suppress_signal = False
        self._update_counter(clamped)

    def _on_slider_value_changed(self, value: int) -> None:
        """User dragged the slider — emit unless we're applying a programmatic snap."""
        self._update_counter(value)
        if not self._suppress_signal:
            self.position_changed.emit(int(value))

    def _on_jump_submit(self) -> None:
        """Parse a 1-based jump entry, clamp to ``[1, total]``, emit on change.

        The visible counter is ``Frame: X / N`` (1-based), so the jump entry
        accepts the same numbering: typing ``1`` selects the first frame,
        ``N`` selects the last.  The emitted ``position_changed`` signal
        carries the underlying 0-based transport position.
        """
        text = self._jump_entry.text().strip()
        current_one_based = self._slider.value() + 1 if self._total > 0 else 0
        if not text or self._total <= 0:
            self._jump_entry.setText(str(current_one_based) if self._total > 0 else "")
            return
        try:
            value = int(text)
        except ValueError:
            self._jump_entry.setText(str(current_one_based))
            return
        clamped_one_based = max(1, min(value, self._total))
        position = clamped_one_based - 1
        if position != self._slider.value():
            self.position_changed.emit(position)
        self._jump_entry.setText(str(clamped_one_based))

    def _update_counter(self, position: int) -> None:
        """Refresh the ``Frame: X / N`` label and the 1-based jump-entry text.

        The jump entry only refreshes when it is *not* focused so the user
        can keep typing without their input being clobbered by an in-flight
        slider move triggered by playback.
        """
        if self._total <= 0:
            self._counter.setText("Frame: – / 0")
            if not self._jump_entry.hasFocus():
                self._jump_entry.setText("")
            return
        one_based = position + 1
        self._counter.setText(f"Frame: {one_based} / {self._total}")
        if not self._jump_entry.hasFocus():
            self._jump_entry.setText(str(one_based))
