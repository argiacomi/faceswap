#!/usr/bin python3
"""Process wrapper for underlying faceswap commands for the GUI"""

from __future__ import annotations

import logging
import os
import re
import signal
import sys
import typing as T
from queue import Empty, Queue
from subprocess import PIPE, Popen
from threading import Thread
from time import time

import psutil

from lib.gui import gui_config as cfg
from lib.utils import get_module_objects

from .analysis import Session
from .services.command_builder import CommandBuilder
from .services.command_context import CommandExecutionContext
from .services.runtime_events import RuntimeEvent
from .services.runtime_service import ProcessRuntimeService
from .services.runtime_session_services import (
    PreviewOutputRuntimeService,
    TrainingSessionRuntimeService,
)
from .utils import LongRunningTask, get_config, get_images, preview_trigger

if os.name == "nt":
    import win32console  # pylint:disable=import-error

logger = logging.getLogger(__name__)

_ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_CONTROL_RE = re.compile(r"[\x00-\x1F\x7F]+")
_UNCONSUMED_TQDM_RE = re.compile(
    r"^.+?:\s+\d+(?:\.\d+)?\w+\s+\[\d+:\d+(?::\d+)?,\s+[^\]]+/s\]\s*$"
)


def _clean_console_control(output: str) -> str:
    """Remove terminal control characters before classifying process output."""
    cleaned = _ANSI_ESCAPE_RE.sub("", output)
    return _CONTROL_RE.sub("", cleaned)


def _has_console_content(output: str) -> bool:
    """Return whether subprocess output contains printable console content."""
    return bool(_clean_console_control(output).strip())


def _is_unconsumed_tqdm_line(output: str) -> bool:
    """Return whether output is a tqdm status line not consumed by the parser."""
    cleaned = _ANSI_ESCAPE_RE.sub("", output)
    cleaned = cleaned.replace("\r", "").replace("\n", "").strip()
    return bool(_UNCONSUMED_TQDM_RE.match(cleaned))


class ProcessWrapper:
    """Builds command, launches and terminates the underlying
    faceswap process. Updates GUI display depending on state"""

    def __init__(self) -> None:
        logger.debug("Initializing %s", self.__class__.__name__)
        self._tk_vars = get_config().tk_vars
        self._set_callbacks()
        self._command: str | None = None
        """ str | None: The currently executing command, when process running or ``None`` """

        self._statusbar = get_config().statusbar
        self._training_session_location: dict[T.Literal["model_name", "model_folder"], str] = {}
        self._training_session_service = TrainingSessionRuntimeService(
            Session, graph_refresh_callback=self._refresh_graph
        )
        self._preview_output_service = PreviewOutputRuntimeService(
            images_provider=get_images,
            preview_trigger_provider=preview_trigger,
        )
        self._task = FaceswapControl(self)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def task(self) -> FaceswapControl:
        """:class:`FaceswapControl`: The object that controls the underlying faceswap process"""
        return self._task

    def _set_callbacks(self) -> None:
        """Set the tkinter variable callbacks for performing an action or generating a command"""
        logger.debug("Setting tk variable traces")
        self._tk_vars.action_command.trace_add("write", self._action_command)
        self._tk_vars.generate_command.trace_add("write", self._generate_command)

    def _refresh_graph(self) -> None:
        """Trigger a graph/analysis refresh from a runtime service event."""
        self._tk_vars.refresh_graph.set(True)

    def _action_command(self, *args: tuple[str, str, str]):  # pylint:disable=unused-argument
        """Callback for when the Action button is pressed. Process command line options and
        launches the action

        Parameters
        ----------
        args:
            tuple[str, str, str]
                Tkinter variable callback args. Required but unused
        """
        if not self._tk_vars.action_command.get():
            return
        category, command = self._tk_vars.action_command.get().split(",")

        try:
            if self._tk_vars.running_task.get():
                self._task.terminate()
            else:
                self._command = command
                fs_args = self._build_args(category)
                self._prepare_after_args_built()
                self._task.execute_script(command, fs_args)
        except ValueError as err:
            self._handle_invalid_action_options(err)
        finally:
            self._tk_vars.action_command.set("")

    def _generate_command(
        self,  # pylint:disable=unused-argument
        *args: tuple[str, str, str],
    ) -> None:
        """Callback for when the Generate button is pressed. Process command line options and
        output the cli command

        Parameters
        ----------
        args:
            tuple[str, str, str]
                Tkinter variable callback args. Required but unused
        """
        if not self._tk_vars.generate_command.get():
            return
        category, command = self._tk_vars.generate_command.get().split(",")
        try:
            fs_args = self._build_args(category, command=command, generate=True)
            self._tk_vars.console_clear.set(True)
            logger.debug(" ".join(fs_args))
            print(" ".join(fs_args))
        except ValueError as err:
            self._handle_invalid_generate_options(err)
        finally:
            self._tk_vars.generate_command.set("")

    def _handle_invalid_action_options(self, err: ValueError) -> None:
        """Reset GUI state and display invalid action option errors."""
        logger.error("Invalid command options: %s", err)
        self._tk_vars.running_task.set(False)
        self._tk_vars.is_training.set(False)
        self._statusbar.stop()
        self._statusbar.message.set("Invalid command options")
        self._tk_vars.display.set("")
        self._command = None
        print(str(err), file=sys.stderr)

    def _handle_invalid_generate_options(self, err: ValueError) -> None:
        """Display invalid generate option errors without changing running task state."""
        logger.error("Invalid command options: %s", err)
        self._tk_vars.console_clear.set(True)
        print(str(err), file=sys.stderr)

    def _prepare_after_args_built(self) -> None:
        """Prepare the GUI state after command line arguments have been validated.

        Sets the 'running task' and 'console clear' global tkinter variables. If training, sets the
        'is training' variable.
        """
        logger.debug("Preparing for execution")
        assert self._command is not None
        self._tk_vars.running_task.set(True)
        self._tk_vars.console_clear.set(True)
        if self._command == "train":
            self._tk_vars.is_training.set(True)
        print("Loading...")

        self._statusbar.message.set(f"Executing - {self._command}.py")
        mode: T.Literal["indeterminate", "determinate"] = (
            "indeterminate" if self._command in ("effmpeg", "train") else "determinate"
        )
        self._statusbar.start(mode)
        self._tk_vars.display.set(self._command)
        logger.debug("Prepared for execution")

    def _build_args(
        self, category: str, command: str | None = None, generate: bool = False
    ) -> list[str]:
        """Build the faceswap command and arguments list.

        If training, pass the model folder and name to the training
        :class:`lib.gui.analysis.Session` for the GUI.

        Parameters
        ----------
        category: str, ["faceswap", "tools"]
            The script that is executing the command
        command: str, optional
            The main faceswap command to execute, if provided. The currently running task if
            ``None``. Default: ``None``
        generate: bool, optional
            ``True`` if the command is just to be generated for display. ``False`` if the command
            is to be executed

        Returns
        -------
        list[str]
            The full faceswap command to be executed or displayed
        """
        logger.debug(
            "Build cli arguments: (category: %s, command: %s, generate: %s)",
            category,
            command,
            generate,
        )
        command = command if command else self._command
        assert command is not None

        values = get_config().cli_opts.get_cli_argument_values(command)
        args = CommandBuilder().build(
            category,
            command,
            values,
            gui_mode=True,
            generate=generate,
        )
        context = CommandExecutionContext.from_values(command, values)
        self._apply_execution_context(context, generate=generate)

        logger.debug("Built cli arguments: (%s)", args)
        return args

    def _apply_execution_context(
        self, context: CommandExecutionContext, *, generate: bool = False
    ) -> None:
        """Apply GUI side effects derived from command values."""
        if generate:
            return

        if hasattr(self, "_training_session_service"):
            self._training_session_service.configure(context)

        if context.model_name is not None:
            self._training_session_location["model_name"] = context.model_name
            logger.debug("model_name: '%s'", self._training_session_location["model_name"])
        if context.model_folder is not None:
            self._training_session_location["model_folder"] = context.model_folder
            logger.debug("model_folder: '%s'", self._training_session_location["model_folder"])

        if hasattr(self, "_preview_output_service"):
            self._preview_output_service.apply_context(context)
        elif context.preview_output_path is not None:
            get_images().preview_extract.set_faceswap_output_path(
                context.preview_output_path, batch_mode=context.batch_mode
            )

    def terminate(self, message: str) -> None:
        """Finalize wrapper when process has exited. Stops the progress bar, sets the status
        message. If the terminating task is 'train', then triggers the training close down actions

        Parameters
        ----------
        message: str
            The message to display in the status bar
        """
        logger.debug("Terminating Faceswap processes")
        self._tk_vars.running_task.set(False)
        if self._task.command == "train":
            self._tk_vars.is_training.set(False)
            if hasattr(self, "_training_session_service"):
                self._training_session_service.stop()
            else:
                Session.stop_training()
        self._statusbar.stop()
        self._statusbar.message.set(message)
        self._tk_vars.display.set("")
        if hasattr(self, "_preview_output_service"):
            self._preview_output_service.cleanup()
        else:
            get_images().delete_preview()
            preview_trigger().clear(trigger_type=None)
        self._command = None
        logger.debug("Terminated Faceswap processes")
        print("Process exited.")


class FaceswapControl:
    """Control the underlying Faceswap tasks.

    wrapper: :class:`ProcessWrapper`
        The object responsible for managing this faceswap task
    """

    def __init__(self, wrapper: ProcessWrapper) -> None:
        logger.debug("Initializing %s (wrapper: %s)", self.__class__.__name__, wrapper)
        self._wrapper = wrapper
        self._session_info = wrapper._training_session_location
        self._config = get_config()
        self._statusbar = self._config.statusbar
        self._last_stdout_was_progress = False
        self._last_stderr_was_progress = False
        self._command: str | None = None
        self._process: Popen | None = None
        self._thread: LongRunningTask | None = None
        self._ui_queue: Queue[tuple[str, T.Any]] = Queue()
        self._runtime = ProcessRuntimeService(training_session=wrapper._training_session_service)
        self._runtime.add_event_callback(self._queue_runtime_event)
        self._config.root.after(50, self._drain_ui_queue)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def command(self) -> str | None:
        """str | None: The currently executing command, when process running or ``None``"""
        return self._command

    def _queue_ui_update(self, action: str, *args: T.Any) -> None:
        """Queue an action for execution on the Tkinter main thread.

        Parameters
        ----------
        action: str
            The UI action to execute
        *args: Any
            Arguments for the UI action
        """
        self._ui_queue.put((action, *args))

    def _queue_runtime_event(self, event: RuntimeEvent) -> None:
        """Queue a runtime event for execution on the Tkinter main thread."""
        self._queue_ui_update("runtime_event", event)

    def _drain_ui_queue(self) -> None:
        """Run queued UI actions on the Tkinter main thread."""
        while True:
            try:
                event = self._ui_queue.get_nowait()
            except Empty:
                break
            self._handle_ui_update(event)
        self._config.root.after(50, self._drain_ui_queue)

    def _handle_ui_update(self, event: tuple[str, T.Any]) -> None:
        """Handle a queued UI action.

        Parameters
        ----------
        event: tuple[str, Any]
            The queued action and its arguments
        """
        action, *args = event
        if action == "stdout":
            # Preserve intentional blank stdout lines. ``print`` emits the
            # message and newline separately after Tk redirects sys.stdout, so
            # final-sink filtering would collapse console formatting.
            print(args[0])
        elif action == "stderr":
            # Stderr is reserved for real error/status text. Blank/control-only
            # stderr has already been filtered in the pipe reader, but keep a
            # final guard so queued cleanup artifacts cannot reach the console.
            if _has_console_content(args[0]):
                print(args[0], file=sys.stderr)
        elif action == "progress_update":
            self._statusbar.progress_update(args[0], args[1], args[2])
        elif action == "status_mode":
            self._statusbar.set_mode(args[0])
        elif action == "runtime_event":
            self._handle_runtime_event(args[0])
        elif action == "terminate":
            self._wrapper.terminate(args[0])
        else:
            logger.debug("Unknown queued UI action: %s", action)

    def _handle_runtime_event(self, event: RuntimeEvent) -> None:
        """Apply a structured runtime event to the legacy Tk widgets."""
        payload = event.payload or {}

        if event.kind == "progress":
            position = int(event.progress) if event.progress is not None else 0
            self._statusbar.progress_update(event.message, position, event.progress is not None)
            return

        if event.kind == "training_session":
            if payload.get("graph_refresh"):
                self._config.tk_vars.refresh_graph.set(True)
            return

        if event.kind != "status":
            return

        mode = payload.get("mode")
        if mode in ("determinate", "indeterminate"):
            self._statusbar.set_mode(mode)

        if payload.get("parser") == "loss":
            self._statusbar.progress_update(event.message, 0, False)
        elif event.message:
            self._statusbar.message.set(event.message)

    def execute_script(self, command: str, args: list[str]) -> None:
        """Execute the requested Faceswap Script

        Parameters
        ----------
        command: str
            The faceswap command that is to be run
        args: list[str]
            The full command line arguments to be executed
        """
        logger.debug("Executing Faceswap: (command: '%s', args: %s)", command, args)
        self._thread = None
        self._command = command
        self._runtime.command = command

        proc = Popen(
            args,  # pylint:disable=consider-using-with
            stdout=PIPE,
            stderr=PIPE,
            bufsize=1,
            text=True,
            stdin=PIPE,
            encoding="utf-8",
            errors="backslashreplace",
        )
        self._process = proc
        self._thread_stdout()
        self._thread_stderr()
        logger.debug("Executed Faceswap")

    def _process_progress_stdout(self, output: str) -> bool:
        """Process stdout for any faceswap processes that update the status/progress bar(s)

        Parameters
        ----------
        output: str
            The output line read from stdout

        Returns
        -------
        bool
            ``True`` if all actions have been completed on the output line otherwise ``False``
        """
        return self._runtime.process_stdout(output).consumed

    def _read_stdout(self) -> None:
        """Read stdout from the subprocess."""
        logger.debug("Opening stdout reader")
        assert self._process is not None
        while True:
            try:
                buff = self._process.stdout
                assert buff is not None
                output: str = buff.readline()
            except ValueError as err:
                if str(err).lower().startswith("i/o operation on closed file"):
                    break
                raise

            if output == "" and self._process.poll() is not None:
                break

            if not output:
                continue

            if self._process_progress_stdout(output):
                self._last_stdout_was_progress = True
                continue

            if _is_unconsumed_tqdm_line(output):
                self._last_stdout_was_progress = True
                continue

            if not _has_console_content(output):
                if self._last_stdout_was_progress:
                    continue
                # Preserve intentional blank stdout from child ``print()``.
                self._queue_ui_update("stdout", output.rstrip("\r\n"))
                continue

            self._last_stdout_was_progress = False
            self._queue_ui_update("stdout", output.rstrip("\r\n"))

        returncode = self._process.poll()
        assert returncode is not None
        event = self._runtime.finish(returncode)
        self._queue_ui_update("terminate", event.message)
        logger.debug("Terminated stdout reader. returncode: %s", returncode)

    def _read_stderr(self) -> None:
        """Read stderr from the subprocess."""
        logger.debug("Opening stderr reader")
        assert self._process is not None
        while True:
            try:
                buff = self._process.stderr
                assert buff is not None
                output: str = buff.readline()
            except ValueError as err:
                if str(err).lower().startswith("i/o operation on closed file"):
                    break
                raise

            if output == "" and self._process.poll() is not None:
                break

            if not output:
                continue

            if self._runtime.process_stderr(output).consumed:
                self._last_stderr_was_progress = True
                continue

            if _is_unconsumed_tqdm_line(output):
                self._last_stderr_was_progress = True
                continue

            if not _has_console_content(output):
                # Stderr blank/control-only output is almost always terminal
                # drawing noise (for example tqdm leave=False cleanup), so do
                # not surface it in the Tk console.
                continue

            self._last_stderr_was_progress = False
            self._queue_ui_update("stderr", output.strip())
        logger.debug("Terminated stderr reader")

    def _thread_stdout(self) -> None:
        """Put the subprocess stdout so that it can be read without blocking"""
        logger.debug("Threading stdout")
        thread = Thread(target=self._read_stdout)
        thread.daemon = True
        thread.start()
        logger.debug("Threaded stdout")

    def _thread_stderr(self) -> None:
        """Put the subprocess stderr so that it can be read without blocking"""
        logger.debug("Threading stderr")
        thread = Thread(target=self._read_stderr)
        thread.daemon = True
        thread.start()
        logger.debug("Threaded stderr")

    def terminate(self) -> None:
        """Terminate the running process in a LongRunningTask so console can still be updated
        console"""
        if self._thread is None:
            logger.debug("Terminating wrapper in LongRunningTask")
            self._thread = LongRunningTask(
                target=self._terminate_in_thread, args=(self._command, self._process)
            )
            if self._command == "train":
                get_config().tk_vars.is_training.set(False)
            self._thread.start()
            self._config.root.after(1000, self.terminate)
        elif not self._thread.complete.is_set():
            logger.debug("Not finished terminating")
            self._config.root.after(1000, self.terminate)
        else:
            logger.debug("Termination Complete. Cleaning up")
            _ = self._thread.get_result()  # Terminate the LongRunningTask object
            self._thread = None

    def _terminate_in_thread(self, command: str | None, process: Popen | None) -> bool:
        """Terminate the subprocess

        Parameters
        ----------
        command: str
            The command that is running

        process: :class:`subprocess.Popen`
            The running process

        Returns
        -------
        bool
            ``True`` when this function exits
        """
        logger.debug("Terminating wrapper")
        if process is None:
            logger.debug("No process exists to terminate")
            return True
        if command == "train":
            timeout = cfg.timeout()
            logger.debug("Sending Exit Signal")
            self._queue_ui_update("stdout", "Sending Exit Signal")
            now = time()
            if os.name == "nt":
                logger.debug("Sending carriage return to process")
                con_in = win32console.GetStdHandle(  # pylint:disable=c-extension-no-member
                    win32console.STD_INPUT_HANDLE
                )  # pylint:disable=c-extension-no-member
                keypress = self._generate_windows_keypress("\n")
                con_in.WriteConsoleInput([keypress])
            else:
                logger.debug("Sending SIGINT to process")
                process.send_signal(signal.SIGINT)
            while True:
                timeelapsed = time() - now
                if process.poll() is not None:
                    break
                if timeelapsed > timeout:
                    logger.error("Timeout reached sending Exit Signal")
                    self._terminate_process_tree(process)
        else:
            self._terminate_process_tree(process)
        return True

    @classmethod
    def _generate_windows_keypress(cls, character: str) -> bytes:
        """Generate a Windows keypress

        Parameters
        ----------
        character: str
            The caracter to generate the keypress for

        Returns
        -------
        bytes
            The generated Windows keypress
        """
        buf = win32console.PyINPUT_RECORDType(  # pylint:disable=c-extension-no-member
            win32console.KEY_EVENT
        )  # pylint:disable=c-extension-no-member
        buf.KeyDown = 1
        buf.RepeatCount = 1
        buf.Char = character
        return buf

    def _terminate_process_tree(self, process: Popen) -> None:
        """Terminate the launched process and its children."""
        logger.debug("Terminating Process...")
        self._queue_ui_update("stdout", "Terminating Process...")
        try:
            root = psutil.Process(process.pid)
        except psutil.NoSuchProcess:
            logger.debug("Process already terminated")
            self._queue_ui_update("stdout", "Terminated")
            return

        children = root.children(recursive=True)
        for child in children:
            child.terminate()
        root.terminate()

        _, alive = psutil.wait_procs([root, *children], timeout=10)
        if not alive:
            logger.debug("Terminated")
            self._queue_ui_update("stdout", "Terminated")
            return

        logger.debug("Termination timed out. Killing Process...")
        self._queue_ui_update("stdout", "Termination timed out. Killing Process...")
        for proc in alive:
            proc.kill()
        _, alive = psutil.wait_procs(alive, timeout=10)
        if not alive:
            logger.debug("Killed")
            self._queue_ui_update("stdout", "Killed")
        else:
            for proc in alive:
                msg = f"Process {proc} survived SIGKILL. Giving up"
                logger.debug(msg)
                self._queue_ui_update("stdout", msg)


__all__ = get_module_objects(__name__)
