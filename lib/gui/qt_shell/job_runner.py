#!/usr/bin/env python3
"""QProcess-backed command runner for the Qt shell prototype."""

from __future__ import annotations

import os
import signal
import typing as T

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

from lib.gui.services.command_context import CommandExecutionContext
from lib.gui.services.runtime_service import ProcessRuntimeService
from lib.gui.services.runtime_session_services import (
    ProcessTreeTerminator,
    RuntimeTerminationPlan,
    RuntimeTerminationPolicy,
)

StreamName = T.Literal["stdout", "stderr"]
IS_WINDOWS = os.name == "nt"


class JobRunner(QObject):
    """Run Faceswap commands through Qt's process/event loop integration."""

    stdout = Signal(str)
    stderr = Signal(str)
    progress = Signal(object)
    finished = Signal(int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.process: QProcess | None = None
        self._runtime = ProcessRuntimeService()
        self._runtime.add_event_callback(self.progress.emit)
        self._termination_policy = RuntimeTerminationPolicy()
        self._tree_terminator = ProcessTreeTerminator()
        self._pending_termination_plan: RuntimeTerminationPlan | None = None
        self._stdout_buffer = ""
        self._stderr_buffer = ""

    def configure_runtime_context(self, context: CommandExecutionContext) -> None:
        """Configure shared runtime services with command-derived context."""
        self._runtime.configure_context(context)

    def start(self, argv: list[str], *, command: str | None = None) -> None:
        """Start a command using QProcess."""
        if not argv:
            raise ValueError("Cannot start an empty command")
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            raise RuntimeError("A job is already running")

        self._runtime.command = command or self._command_from_argv(argv)
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self.process = QProcess(self)
        self.process.setProgram(argv[0])
        self.process.setArguments(argv[1:])
        self.process.readyReadStandardOutput.connect(self._stdout)
        self.process.readyReadStandardError.connect(self._stderr)
        self.process.errorOccurred.connect(self._error)
        self.process.finished.connect(self._finished)
        self.process.start()

    def stop(self) -> None:
        """Terminate the running process according to the command-aware policy."""
        if self.process is None or self.process.state() == QProcess.NotRunning:
            return
        plan = self._termination_policy.plan(self._runtime.command)
        self._pending_termination_plan = plan
        self.stdout.emit(f"{plan.message}\n")
        if not (plan.interrupt_first and self._interrupt_process()):
            self.process.terminate()
        QTimer.singleShot(plan.grace_ms, self._kill_if_running)

    def _interrupt_process(self) -> bool:
        """Request graceful interruption for train jobs where the platform supports it."""
        if self.process is None or IS_WINDOWS:
            return False
        try:
            pid = int(self.process.processId())
        except (AttributeError, TypeError, ValueError):
            return False
        if pid <= 0:
            return False
        try:
            os.kill(pid, signal.SIGINT)
        except OSError:
            return False
        return True

    def _kill_if_running(self) -> None:
        """Kill the process if terminate did not stop it."""
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            plan = self._pending_termination_plan
            pid = self._process_id()
            if plan is not None and pid is not None:
                self._tree_terminator.terminate(
                    pid,
                    timeout_seconds=plan.tree_timeout_seconds,
                )
            self.process.kill()

    def _process_id(self) -> int | None:
        """Return the current QProcess id when available."""
        if self.process is None:
            return None
        try:
            pid = int(self.process.processId())
        except (AttributeError, TypeError, ValueError):
            return None
        return pid if pid > 0 else None

    def _stdout(self) -> None:
        """Emit decoded stdout or structured runtime events."""
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        if data:
            self._process_output("stdout", data)

    def _stderr(self) -> None:
        """Emit decoded stderr or structured runtime events."""
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardError()).decode(errors="replace")
        if data:
            self._process_output("stderr", data)

    def _error(self, error: QProcess.ProcessError) -> None:
        """Forward QProcess errors to the console."""
        self.stderr.emit(f"QProcess error: {error}")

    def _finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        """Drain final output, emit final runtime status, and clear process state."""
        self._stdout()
        self._stderr()
        self._flush_output_buffers()
        self._runtime.finish(exit_code)
        self._pending_termination_plan = None
        self.process = None
        self.finished.emit(exit_code)

    def _process_output(self, stream: StreamName, data: str, *, flush: bool = False) -> None:
        """Buffer decoded process output and parse complete output units."""
        buffer = self._stdout_buffer if stream == "stdout" else self._stderr_buffer
        units, remainder = self._split_output_units(f"{buffer}{data}", flush=flush)
        if stream == "stdout":
            self._stdout_buffer = remainder
        else:
            self._stderr_buffer = remainder
        for unit in units:
            self._emit_output_unit(stream, unit)

    def _flush_output_buffers(self) -> None:
        """Flush partial stdout/stderr output when a process exits."""
        self._process_output("stdout", "", flush=True)
        self._process_output("stderr", "", flush=True)

    def _emit_output_unit(self, stream: StreamName, output: str) -> None:
        """Emit parsed runtime events or pass through plain process output."""
        parsed = (
            self._runtime.process_stdout(output)
            if stream == "stdout"
            else self._runtime.process_stderr(output)
        )
        if parsed.consumed:
            return
        if stream == "stdout":
            self.stdout.emit(output)
        else:
            self.stderr.emit(output)

    @staticmethod
    def _split_output_units(output: str, *, flush: bool = False) -> tuple[list[str], str]:
        """Split output into complete newline/carriage-return terminated units."""
        units: list[str] = []
        start = 0
        index = 0
        while index < len(output):
            if output[index] not in ("\n", "\r"):
                index += 1
                continue
            end = index + 1
            if output[index] == "\r" and end < len(output) and output[end] == "\n":
                end += 1
            units.append(output[start:end])
            start = end
            index = end
        if flush and start < len(output):
            units.append(output[start:])
            return units, ""
        return units, output[start:]

    @staticmethod
    def _command_from_argv(argv: list[str]) -> str | None:
        """Infer the Faceswap command name from command-builder argv when available."""
        for index, arg in enumerate(argv[:-1]):
            if os.path.basename(arg) in {"faceswap.py", "tools.py"}:
                return argv[index + 1]
        return None
