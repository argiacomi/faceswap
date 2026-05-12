#!/usr/bin/env python3
"""QProcess-backed command runner for the Qt shell prototype."""

from __future__ import annotations

from PySide6.QtCore import QObject, QProcess, QTimer, Signal


class JobRunner(QObject):
    """Run Faceswap commands through Qt's process/event loop integration."""

    stdout = Signal(str)
    stderr = Signal(str)
    progress = Signal(object)
    finished = Signal(int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.process: QProcess | None = None

    def start(self, argv: list[str]) -> None:
        """Start a command using QProcess."""
        if not argv:
            raise ValueError("Cannot start an empty command")
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            raise RuntimeError("A job is already running")

        self.process = QProcess(self)
        self.process.setProgram(argv[0])
        self.process.setArguments(argv[1:])
        self.process.readyReadStandardOutput.connect(self._stdout)
        self.process.readyReadStandardError.connect(self._stderr)
        self.process.errorOccurred.connect(self._error)
        self.process.finished.connect(self._finished)
        self.process.start()

    def stop(self) -> None:
        """Terminate the running process, escalating to kill if it does not exit."""
        if self.process is None or self.process.state() == QProcess.NotRunning:
            return
        self.process.terminate()
        QTimer.singleShot(5000, self._kill_if_running)

    def _kill_if_running(self) -> None:
        """Kill the process if terminate did not stop it."""
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            self.process.kill()

    def _stdout(self) -> None:
        """Emit decoded stdout."""
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        if data:
            self.stdout.emit(data)

    def _stderr(self) -> None:
        """Emit decoded stderr."""
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardError()).decode(errors="replace")
        if data:
            self.stderr.emit(data)

    def _error(self, error: QProcess.ProcessError) -> None:
        """Forward QProcess errors to the console."""
        self.stderr.emit(f"QProcess error: {error}")

    def _finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        """Drain final output, clear process state and emit finished."""
        self._stdout()
        self._stderr()
        self.process = None
        self.finished.emit(exit_code)
