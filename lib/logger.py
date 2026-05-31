#!/usr/bin/python
"""Logging Functions for Faceswap."""

# NOTE: Don't import non stdlib packages. This module is accessed by setup.py
from __future__ import annotations

import collections
import logging
import os
import platform
import re
import sys
import time
import traceback
import typing as T
from datetime import datetime
from logging.handlers import RotatingFileHandler

from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    import numpy as np


# Add our custom levels to logger
for new_level in (("VERBOSE", 15), ("TRACE", 5)):
    level_name, level_num = new_level
    level_map = logging.getLevelNamesMapping()  # type: ignore[attr-defined]
    if level_name in level_map:
        continue
    logging.addLevelName(level_num, level_name)
    level_map[level_name] = level_num

_FACESWAP_HANDLER_ATTR = "_faceswap_managed_handler"


# ---------------------------------------------------------------------------
# External logger policy (issue #189)
#
# Noisy upstream libraries (matplotlib font discovery, libav transcoding
# noise, Torch inductor warnings, PIL/numexpr boilerplate) leaked INFO and
# WARNING lines onto the Faceswap console. The previous approach mutated
# LogRecord.levelno inside FaceswapFormatter, but a single record flows
# through multiple handlers (file, stream, crash) — formatter-side mutation
# can leak across handlers depending on handler order.
#
# The policy below is applied via HANDLER-BOUND filters from log_setup() so:
#   * Stream output is suppressed / down-weighted as configured.
#   * File and crash handlers continue to receive the original records, so
#     post-mortem analysis still has the full noise stream available.
# ---------------------------------------------------------------------------

EXTERNAL_LOGGER_POLICY: dict[str, dict[str, T.Any]] = {
    # Matplotlib font discovery spams INFO during first-import; downgrade so
    # the console only sees its WARNING+. File log is untouched.
    "matplotlib": {"lower_info_to_debug": True, "max_stream_level": "WARNING"},
    "matplotlib.font_manager": {"lower_info_to_debug": True},
    # libav (via PyAV) emits per-frame INFO that drowns extract/convert output.
    "libav": {
        "lower_info_to_debug": True,
        "max_stream_level": "WARNING",
        "suppress_stream_patterns": (
            ("libav.swscaler", None, "No accelerated colorspace conversion found from"),
        ),
    },
    # Pillow / numexpr boilerplate has no console value. Both downgrades
    # (drop INFO) and the WARNING ceiling are needed: lowering alone leaves
    # noisy WARNING-tier debug intact, and capping alone still lets INFO
    # through the stream handler's INFO floor.
    "PIL": {"lower_info_to_debug": True, "max_stream_level": "WARNING"},
    "numexpr": {"lower_info_to_debug": True, "max_stream_level": "WARNING"},
    # Specific Torch inductor warnings that always appear on supported GPUs
    # and add nothing for users.
    "torch._inductor": {
        "suppress_stream_patterns": (("torch._inductor.utils", "is_big_gpu", None),),
    },
    # Captured py.warnings noise originating from torch._inductor.
    "py.warnings": {
        "suppress_stream_patterns": (("py.warnings", None, "/torch/_inductor"),),
    },
}


class _ExternalLevelDowngradeFilter(logging.Filter):
    """Stream-only filter that drops external INFO records configured to be
    demoted to DEBUG.

    Replaces ``FaceswapFormatter._lower_external``. As a filter, the demotion
    is bound to a single handler — the file and crash handlers still see the
    original INFO record. Drop-on-match (rather than mutate the levelno) is
    used here because the stream handler is configured at INFO+, so mutating
    levelno to DEBUG would still pass its level threshold.
    """

    def __init__(self, lower_info_prefixes: tuple[str, ...]) -> None:
        super().__init__()
        self._prefixes = lower_info_prefixes

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno != logging.INFO:
            return True
        return not record.name.startswith(self._prefixes)


class _StreamMaxLevelFilter(logging.Filter):
    """Drop records whose level exceeds the configured per-logger ceiling.

    Counterpart to ``handler.setLevel()`` which only sets a floor. Used to
    silence chatty WARNINGs from PIL/numexpr/matplotlib/libav on the
    console while keeping the file log complete.
    """

    def __init__(self, max_levels: dict[str, int]) -> None:
        super().__init__()
        # Sort longest-prefix-first so ``matplotlib.font_manager`` wins
        # over the more general ``matplotlib`` prefix.
        self._max_levels = sorted(max_levels.items(), key=lambda kv: -len(kv[0]))

    def filter(self, record: logging.LogRecord) -> bool:
        for prefix, ceiling in self._max_levels:
            if record.name == prefix or record.name.startswith(prefix + "."):
                return record.levelno <= ceiling
        return True


class _StreamPatternSuppressionFilter(logging.Filter):
    """Drop records that match a configured ``(name, funcname, substring)``.

    Each pattern is a 3-tuple where ``funcname`` / ``substring`` may be
    ``None`` to skip the corresponding check. Subsumes the legacy hard-coded
    ``TorchWarningsFilter`` so its behaviour is now declared in
    ``EXTERNAL_LOGGER_POLICY`` rather than embedded in a class.
    """

    def __init__(self, patterns: tuple[tuple, ...]) -> None:
        super().__init__()
        self._patterns = patterns

    def filter(self, record: logging.LogRecord) -> bool:
        for name, funcname, substring in self._patterns:
            if record.name != name and not record.name.startswith(name + "."):
                continue
            if funcname is not None and record.funcName != funcname:
                continue
            if substring is not None and substring not in record.getMessage():
                continue
            return False
        return True


class _BlankMessageFilter(logging.Filter):
    """Drop records whose FORMATTED output is empty / whitespace-only.

    Without this, captured warnings (where the warning text contains only a
    blank line) and external noise can produce empty stream/tqdm lines that
    pollute progress bars and GUI status parsing.

    Critically, the filter formats a COPY of the record. ``FaceswapFormatter``
    still mutates incoming ``LogRecord`` instances (it rewrites captured
    ``py.warnings`` text into ``record.msg`` and escapes newlines on
    ``record.message``). Because handlers downstream of the stream handler
    (the crash buffer + file logger in non-setup mode are installed earlier,
    but ad-hoc embedding apps may add their own handlers later) see the
    SAME record object, formatting the original here would leak those
    mutations across handlers — defeating the "file/crash handlers continue
    to receive original records" guarantee. ``logging.makeLogRecord`` rebuilds
    a fresh record from a shallow copy of ``record.__dict__`` so the
    formatter's mutations stay local to this filter.
    """

    def __init__(self, formatter: logging.Formatter) -> None:
        super().__init__()
        self._formatter = formatter

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            isolated = logging.makeLogRecord(record.__dict__.copy())
            formatted = self._formatter.format(isolated)
        except Exception:  # pylint:disable=broad-except
            # Never silently swallow a record because formatting blew up.
            return True
        return bool(formatted) and bool(formatted.strip())


def _apply_external_logger_policy(
    rootlogger: logging.Logger,
    stream_handler: logging.Handler,
) -> None:
    """Install the configured external-logger policy on the stream handler.

    Called from :func:`log_setup` after the stream handler is constructed
    and (for non-setup modes) before the file/crash handlers are attached
    so the stream filters do not affect persisted log capture.
    """
    lower_info_prefixes: list[str] = []
    max_levels: dict[str, int] = {}
    patterns: list[tuple] = []

    for logger_name, policy in EXTERNAL_LOGGER_POLICY.items():
        if policy.get("lower_info_to_debug"):
            # Anchor at the logger's full name plus the dotted-child prefix
            # so child loggers (e.g. ``matplotlib.font_manager``) also match.
            lower_info_prefixes.append(logger_name + ".")
            lower_info_prefixes.append(logger_name)
        max_level = policy.get("max_stream_level")
        if isinstance(max_level, str):
            max_levels[logger_name] = logging.getLevelNamesMapping()[max_level.upper()]  # type: ignore[attr-defined]
        for pattern in policy.get("suppress_stream_patterns", ()):
            # Normalise to a 3-tuple ``(name, funcname, substring)``.
            if len(pattern) == 2:
                name, funcname = pattern
                substring = None
            elif len(pattern) == 3:
                name, funcname, substring = pattern
            else:
                raise ValueError(
                    f"Invalid suppress_stream_patterns entry for {logger_name!r}: {pattern!r}"
                )
            patterns.append((name, funcname, substring))

    if lower_info_prefixes:
        stream_handler.addFilter(_ExternalLevelDowngradeFilter(tuple(lower_info_prefixes)))
    if max_levels:
        stream_handler.addFilter(_StreamMaxLevelFilter(max_levels))
    if patterns:
        stream_handler.addFilter(_StreamPatternSuppressionFilter(tuple(patterns)))

    # Trust the stream handler's formatter to define what "blank" means here
    # so the filter sees exactly the bytes that would otherwise be written.
    formatter = stream_handler.formatter
    if formatter is not None:
        stream_handler.addFilter(_BlankMessageFilter(formatter))


class FaceswapLogger(logging.Logger):
    """A standard :class:`logging.logger` with additional "verbose" and "trace" levels added."""

    def verbose(self, msg: str, *args, **kwargs) -> None:
        # pylint:disable=wrong-spelling-in-docstring
        """Create a log message at severity level 15.

        Parameters
        ----------
        msg
            The log message to be recorded at Verbose level
        args
            Standard logging arguments
        kwargs
            Standard logging key word arguments
        """
        if self.isEnabledFor(15):
            kwargs.setdefault("stacklevel", 2)
            self._log(15, msg, args, **kwargs)

    def trace(self, msg: str, *args, **kwargs) -> None:
        # pylint:disable=wrong-spelling-in-docstring
        """Create a log message at severity level 5.

        Parameters
        ----------
        msg
            The log message to be recorded at Trace level
        args
            Standard logging arguments
        kwargs
            Standard logging key word arguments
        """
        if self.isEnabledFor(5):
            kwargs.setdefault("stacklevel", 2)
            self._log(5, msg, args, **kwargs)


class ColoredFormatter(logging.Formatter):
    """Overrides the stand :class:`logging.Formatter` to enable colored labels for message level
    labels on supported platforms

    Parameters
    ----------
    fmt
        The format string for the message as a whole
    pad_newlines
        If ``True`` new lines will be padded to appear in line with the log message, if ``False``
        they will be left aligned

    kwargs
        Standard :class:`logging.Formatter` keyword arguments
    """

    def __init__(self, fmt: str, pad_newlines: bool = False, **kwargs) -> None:
        super().__init__(fmt, **kwargs)
        self._use_color = self._get_color_compatibility()
        self._level_colors = {
            "CRITICAL": "\033[31m",  # red
            "ERROR": "\033[31m",  # red
            "WARNING": "\033[33m",  # yellow
            "INFO": "\033[32m",  # green
            "VERBOSE": "\033[34m",
        }  # blue
        self._default_color = "\033[0m"
        self._newline_padding = self._get_newline_padding(pad_newlines, fmt)

    @classmethod
    def _get_color_compatibility(cls) -> bool:
        """Return whether the system supports color ansi codes. Most OSes do other than Windows
        below Windows 10 version 1511.

        Returns
        -------
        ``True`` if the system supports color ansi codes otherwise ``False``
        """
        if platform.system().lower() != "windows":
            return True
        try:
            win = sys.getwindowsversion()  # type:ignore # pylint:disable=no-member
            if win.major >= 10 and win.build >= 10586:
                return True
        except Exception:  # pylint:disable=broad-except
            return False
        return False

    def _get_newline_padding(self, pad_newlines: bool, fmt: str) -> int:
        """Parses the format string to obtain padding for newlines if requested

        Parameters
        ----------
        fmt
            The format string for the message as a whole
        pad_newlines
            If ``True`` new lines will be padded to appear in line with the log message, if
            ``False`` they will be left aligned

        Returns
        -------
        The amount of padding to apply to the front of newlines
        """
        if not pad_newlines:
            return 0
        msg_idx = fmt.find("%(message)") + 1
        filtered = fmt[: msg_idx - 1]
        spaces = filtered.count(" ")
        pads = [int(pad.replace("s", "")) for pad in re.findall(r"\ds", filtered)]
        if "asctime" in filtered:
            pads.append(self._get_sample_time_string())
        return sum(pads) + spaces

    def _get_sample_time_string(self) -> int:
        """Obtain a sample time string and calculate correct padding.

        This may be inaccurate when ticking over an integer from single to double digits, but that
        shouldn't be a huge issue.

        Returns
        -------
        The length of the formatted date-time string
        """
        sample_time = time.time()
        date_format = self.datefmt if self.datefmt else self.default_time_format
        date_string = time.strftime(date_format, logging.Formatter.converter(sample_time))
        if not self.datefmt and self.default_msec_format:
            m_secs = (sample_time - int(sample_time)) * 1000
            date_string = self.default_msec_format % (date_string, m_secs)
        return len(date_string)

    def format(self, record: logging.LogRecord) -> str:
        """Color the log message level if supported otherwise return the standard log message.

        Parameters
        ----------
        record
            The incoming log record to be formatted for entry into the logger.

        Returns
        -------
        The formatted log message
        """
        formatted = super().format(record)
        levelname = record.levelname
        if self._use_color and levelname in self._level_colors:
            formatted = re.sub(
                levelname,
                f"{self._level_colors[levelname]}{levelname}{self._default_color}",
                formatted,
                1,
            )
        if self._newline_padding:
            formatted = formatted.replace("\n", f"\n{' ' * self._newline_padding}")
        return formatted


class FaceswapFormatter(logging.Formatter):
    """Overrides the standard :class:`logging.Formatter`.

    Strip newlines from incoming log messages.

    Rewrites some upstream warning messages to debug level to avoid spamming the console.
    """

    @classmethod
    def _format_warnings(cls, record: logging.LogRecord) -> logging.LogRecord:
        """Warnings redirected from the warnings module will have new lines inserted. We do not
        want this for logging

        Parameters
        ----------
        record
            The log record to check for rewriting

        Returns
        ----------
        The log rewritten or untouched record
        """
        if record.levelno != logging.WARNING or record.name != "py.warnings":
            return record

        msg = record.getMessage()
        # Strip new lines and trailing superfluous information from captured warnings
        msg = msg.replace("\n", " ").strip().rstrip("warnings.warn(")
        record.msg = msg
        record.args = ()
        return record

    def format(self, record: logging.LogRecord) -> str:
        """Strip new lines from log records and rewrite certain warning messages to debug level.

        Parameters
        ----------
        record
            The incoming log record to be formatted for entry into the logger.

        Returns
        -------
        The formatted log message
        """
        record = self._format_warnings(record)
        record.message = record.getMessage()
        # Strip embedded newlines from the message body itself. Tracebacks
        # / stack frames are appended below as ``exc_text`` / ``stack_info``
        # so they survive untouched — only the body line is collapsed.
        # Previously only DEBUG/VERBOSE/INFO were scrubbed; WARNING/ERROR
        # are now scrubbed too so external warnings cannot emit blank
        # stream lines (issue #189).
        if "\n" in record.message or "\r" in record.message:
            record.message = record.message.replace("\n", "\\n").replace("\r", "\\r")

        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)
        msg = self.formatMessage(record)
        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if msg[-1:] != "\n":
                msg = msg + "\n"
            msg = msg + record.exc_text
        if record.stack_info:
            if msg[-1:] != "\n":
                msg = msg + "\n"
            msg = msg + self.formatStack(record.stack_info)
        return msg


class TorchWarningsFilter:
    """Legacy alias for the Torch console-suppression policy.

    Retained for backwards-compatibility with any embedding application
    that previously imported this class directly. New code should not
    install it manually — the same suppression is now declared in
    :data:`EXTERNAL_LOGGER_POLICY` and applied automatically by
    :func:`log_setup` via stream-only handler filters.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno != logging.WARNING:
            return True
        if record.name == "torch._inductor.utils" and record.funcName == "is_big_gpu":
            return False
        if record.name != "py.warnings":
            return True
        return "/torch/_inductor" not in record.getMessage()


class RollingBuffer(collections.deque):
    """File-like that keeps a certain number of lines of text in memory for writing out to the
    crash log."""

    def write(self, buffer: str) -> None:
        """Splits lines from the incoming buffer and writes them out to the rolling buffer.

        Parameters
        ----------
        buffer
            The log messages to write to the rolling buffer
        """
        for line in buffer.rstrip().splitlines():
            self.append(f"{line}\n")


class TqdmHandler(logging.StreamHandler):
    """Overrides :class:`logging.StreamHandler` to use :func:`tqdm.tqdm.write` rather than writing
    to :func:`sys.stderr` so that log messages do not mess up tqdm progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        """Format the incoming message and pass to :func:`tqdm.tqdm.write`.

        Empty / whitespace-only formatted messages are dropped before
        reaching ``tqdm.write`` so captured warnings / external noise
        cannot inject blank lines that pollute progress bars or GUI
        status parsing (issue #189). The stream-handler-bound
        ``_BlankMessageFilter`` provides the same guarantee, but the
        belt-and-braces check here keeps the contract intact if this
        handler is ever wired up outside :func:`log_setup`.
        """
        from tqdm import tqdm  # pylint:disable=import-outside-toplevel

        msg = self.format(record)
        if not msg or not msg.strip():
            return
        tqdm.write(msg)


def _set_root_logger(loglevel: int = logging.INFO) -> logging.Logger:
    """Setup the root logger.

    Parameters
    ----------
    loglevel
        The log level to set the root logger to. Default :attr:`logging.INFO`

    Returns
    -------
    The root logger for Faceswap
    """
    rootlogger = logging.getLogger()
    rootlogger.setLevel(loglevel)
    logging.captureWarnings(True)
    return rootlogger


def _mark_faceswap_handler(handler: logging.Handler) -> logging.Handler:
    """Tag a handler managed by :func:`log_setup` for safe replacement."""
    setattr(handler, _FACESWAP_HANDLER_ATTR, True)
    return handler


def _is_faceswap_managed_handler(handler: logging.Handler) -> bool:
    """Return whether ``handler`` was installed by :func:`log_setup`.

    New handlers are tagged explicitly.  Older test-process handlers from
    modules that called ``log_setup`` before this marker existed will not have
    the tag, so also recognize the concrete handler/formatter combinations
    that ``log_setup`` creates.  This keeps pytest capture handlers and other
    external handlers intact while still cleaning stale Faceswap console/file
    handlers that would otherwise duplicate later Alignments logs.
    """
    if getattr(handler, _FACESWAP_HANDLER_ATTR, False):
        return True
    formatter = getattr(handler, "formatter", None)
    if not isinstance(formatter, (FaceswapFormatter, ColoredFormatter)):
        return False
    return isinstance(handler, (RotatingFileHandler, TqdmHandler, logging.StreamHandler))


def _clear_faceswap_handlers(rootlogger: logging.Logger) -> None:
    """Remove handlers previously installed by :func:`log_setup`.

    Test runs can import modules that call ``log_setup`` more than once.  The
    previous implementation appended file/stream/crash handlers on every call,
    causing every later log record to be emitted repeatedly.  Only handlers
    installed by Faceswap logging setup are removed so pytest's capture handler
    and any embedding application's handlers remain intact.
    """
    for handler in tuple(rootlogger.handlers):
        if not _is_faceswap_managed_handler(handler):
            continue
        rootlogger.removeHandler(handler)
        handler.close()


def log_setup(loglevel, log_file: str, command: str, is_gui: bool = False) -> None:
    """Set up logging for Faceswap.

    Sets up the root logger, the formatting for the crash logger and the file logger, and sets up
    the crash, file and stream log handlers.

    Parameters
    ----------
    loglevel
        The requested log level that Faceswap should be run at.
    log_file
        The location of the log file to write Faceswap's log to
    command
        The Faceswap command that is being run. Used to dictate whether the log file should
        have "_gui" appended to the filename or not.
    is_gui
        Whether Faceswap is running in the GUI or not. Dictates where the stream handler should
        output messages to. Default: ``False``
    """
    numeric_loglevel = get_loglevel(loglevel)
    root_loglevel = min(logging.DEBUG, numeric_loglevel)
    rootlogger = _set_root_logger(loglevel=root_loglevel)
    _clear_faceswap_handlers(rootlogger)

    if command == "setup":
        log_format = FaceswapFormatter(
            "%(asctime)s %(module)-16s %(funcName)-30s %(levelname)-8s %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
        )
        s_handler = _stream_setup_handler(numeric_loglevel)
        f_handler = _file_handler(root_loglevel, log_file, log_format, command)
    else:
        log_format = FaceswapFormatter(
            "%(asctime)s %(processName)-15s %(threadName)-30s "
            "%(module)-15s %(funcName)-30s %(levelname)-8s %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
        )
        s_handler = _stream_handler(numeric_loglevel, is_gui)
        f_handler = _file_handler(numeric_loglevel, log_file, log_format, command)

    # Install handler-bound external-logger policy on the stream handler
    # so file/crash logs keep the full noise stream for post-mortem use.
    _apply_external_logger_policy(rootlogger, s_handler)

    rootlogger.addHandler(_mark_faceswap_handler(f_handler))
    rootlogger.addHandler(_mark_faceswap_handler(s_handler))

    if command == "setup":
        return

    c_handler = _crash_handler(log_format)
    rootlogger.addHandler(_mark_faceswap_handler(c_handler))
    logging.info("Log level set to: %s", loglevel.upper())

    try:
        import torch  # noqa: F401  # pylint:disable=unused-import,import-outside-toplevel
    except ImportError:
        return

    # Elevate torch loggers to use our loggers
    for name in rootlogger.manager.loggerDict:
        if name.startswith("torch"):
            logger = logging.getLogger(name)
            logger.handlers.clear()
            logger.propagate = True


def _file_handler(
    loglevel, log_file: str, log_format: FaceswapFormatter, command: str
) -> RotatingFileHandler:
    """Add a rotating file handler for the current Faceswap session. 1 backup is always kept.

    Parameters
    ----------
    loglevel
        The requested log level that messages should be logged at.
    log_file
        The location of the log file to write Faceswap's log to
    log_format
        The formatting to store log messages as
    command
        The Faceswap command that is being run. Used to dictate whether the log file should
        have "_gui" appended to the filename or not.

    Returns
    -------
    The logging file handler
    """
    if log_file:
        filename = log_file
    else:
        filename = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), "faceswap")
        # Windows has issues sharing the log file with sub-processes, so log GUI separately
        filename += "_gui.log" if command == "gui" else ".log"

    should_rotate = os.path.isfile(filename)
    handler = RotatingFileHandler(filename, backupCount=1, encoding="utf-8")
    if should_rotate:
        handler.doRollover()
    handler.setFormatter(log_format)
    handler.setLevel(loglevel)
    return handler


def _stream_handler(loglevel: int, is_gui: bool) -> logging.StreamHandler | TqdmHandler:
    """Add a stream handler for the current Faceswap session. The stream handler will only ever
    output at a maximum of VERBOSE level to avoid spamming the console.

    Parameters
    ----------
    loglevel
        The requested log level that messages should be logged at.
    is_gui
        Whether Faceswap is running in the GUI or not. Dictates where the stream handler should
        output messages to.

    Returns
    -------
    The stream handler to use
    """
    # Don't set stdout to lower than verbose
    loglevel = max(loglevel, 15)
    log_format = FaceswapFormatter(
        "%(asctime)s %(levelname)-8s %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
    )

    if is_gui:
        # tqdm.write inserts extra lines in the GUI, so use standard output as
        # it is not needed there.
        log_console = logging.StreamHandler(sys.stdout)
    else:
        log_console = TqdmHandler(sys.stdout)
    log_console.setFormatter(log_format)
    log_console.setLevel(loglevel)
    return log_console


def _stream_setup_handler(loglevel: int) -> logging.StreamHandler:
    """Add a stream handler for faceswap's setup.py script
    This stream handler outputs a limited set of easy to use information using colored labels
    if available. It will only ever output at a minimum of INFO level

    Parameters
    ----------
    loglevel
        The requested log level that messages should be logged at.

    Returns
    -------
    The stream handler to use
    """
    loglevel = max(loglevel, 15)
    log_format = ColoredFormatter("%(levelname)-8s %(message)s", pad_newlines=True)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(log_format)
    handler.setLevel(loglevel)
    return handler


def _crash_handler(log_format: FaceswapFormatter) -> logging.StreamHandler:
    """Add a handler that stores the last 100 debug lines to :attr:'_DEBUG_BUFFER' for use in
    crash reports.

    Parameters
    ----------
    log_format
        The formatting to store log messages as

    Returns
    -------
    The crash log handler
    """
    log_crash = logging.StreamHandler(_DEBUG_BUFFER)
    log_crash.setFormatter(log_format)
    log_crash.setLevel(logging.DEBUG)
    return log_crash


def configure_tool_logging(
    loglevel: str | int = "INFO",
    *,
    stream: T.TextIO | None = None,
) -> None:
    """Configure stream-only logging for ad-hoc Faceswap CLI tools.

    Replaces standalone ``logging.basicConfig()`` calls in
    ``tools/landmarks/*.py`` (issue #189). The helper:

    * Installs a single :class:`TqdmHandler` (stream goes to stderr by
      default so stdout stays clean for piped tool output) tagged as a
      managed Faceswap handler, so a later :func:`log_setup` call replaces
      it without leaving a duplicate behind.
    * Applies :data:`EXTERNAL_LOGGER_POLICY` and the blank-message guard
      so tool output behaves the same way the main app's stream does.
    * Idempotent — repeated calls do not stack handlers; the previous
      managed handlers are cleared first, matching :func:`log_setup`.

    Parameters
    ----------
    loglevel
        A standard logging level name (``"TRACE"``, ``"DEBUG"``, ``"INFO"``,
        ``"VERBOSE"``, ``"WARNING"``, ``"ERROR"``) or numeric level.
    stream
        Where the handler writes. Defaults to :data:`sys.stderr` so this
        helper composes with tool stdout (which often carries the
        machine-readable output the tool was invoked to produce).
    """
    numeric = int(loglevel) if isinstance(loglevel, int) else get_loglevel(str(loglevel))
    rootlogger = _set_root_logger(loglevel=min(logging.DEBUG, numeric))
    _clear_faceswap_handlers(rootlogger)
    log_format = FaceswapFormatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler: logging.Handler = TqdmHandler(stream or sys.stderr)
    handler.setFormatter(log_format)
    handler.setLevel(max(numeric, 15))
    _apply_external_logger_policy(rootlogger, handler)
    rootlogger.addHandler(_mark_faceswap_handler(handler))


def get_loglevel(loglevel: str) -> int:
    """Check whether a valid log level has been supplied, and return the numeric log level that
    corresponds to the given string level.

    Parameters
    ----------
    loglevel
        The loglevel that has been requested

    Returns
    -------
    The numeric representation of the given loglevel
    """
    numeric_level = logging.getLevelNamesMapping()[loglevel.upper()]  # type: ignore[attr-defined]
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {loglevel}")
    return numeric_level


def crash_log() -> str:
    """On a crash, write out the contents of :func:`_DEBUG_BUFFER` containing the last 100 lines
    of debug messages to a crash report in the root Faceswap folder.

    Returns
    -------
    The filename of the file that contains the crash report
    """
    original_traceback = traceback.format_exc().encode("utf-8")
    path = os.path.dirname(os.path.realpath(sys.argv[0]))
    filename = os.path.join(path, datetime.now().strftime("crash_report.%Y.%m.%d.%H%M%S%f.log"))
    freeze_log = [line.encode("utf-8") for line in _DEBUG_BUFFER]
    try:
        from lib.system.sysinfo import sysinfo  # pylint:disable=import-outside-toplevel
    except Exception:  # pylint:disable=broad-except
        sysinfo = (
            "\n\nThere was an error importing System Information from lib.sysinfo. This is "
            f"probably a bug which should be fixed:\n{traceback.format_exc()}"
        )
    with open(filename, "wb") as outfile:
        outfile.writelines(freeze_log)
        outfile.write(original_traceback)
        outfile.write(sysinfo.encode("utf-8"))
    return filename


def format_array(array: np.ndarray) -> str:
    """Format arrays to be suitable for logging

    Parameters
    ----------
    array
        The array to be formatted for logging

    Returns
    -------
    String representation of an array for logging
    """
    try:
        import numpy as np  # pylint:disable=import-outside-toplevel
    except ImportError:
        return repr(array)

    if array.dtype == "object":
        retval = "np.array("
        for sub in array:
            retval += f"{format_array(sub)}, "
        if array.size:
            retval = retval[:-2]
        return f"{retval}, dtype='{array.dtype}')"

    if np.prod(array.shape) <= 10:
        return f"np.array({str(array.tolist())}, dtype='{array.dtype}')"
    return f"<array(shape{array.shape}, '{array.dtype}')>"


def _process_value(value: T.Any) -> T.Any:
    """Process the values from a local dict and return in a format suitable for logging

    Parameters
    ----------
    value
        The dictionary value

    Returns
    -------
    The original or amended value
    """
    if isinstance(value, list | tuple | set) and len(value) > 10:
        return f'[type: "{type(value).__name__}" len: {len(value)}]'

    try:
        import numpy as np  # pylint:disable=import-outside-toplevel
    except ImportError:
        return repr(value)

    if isinstance(value, np.ndarray):
        return format_array(value)

    return repr(value)


def parse_class_init(locals_dict: dict[str, T.Any]) -> str:
    """Parse a locals dict from a class and return in a format suitable for logging
    Parameters
    ----------
    locals_dict
        A locals() dictionary from a newly initialized class

    Returns
    -------
    The locals information suitable for logging
    """
    delimit = {
        k: _process_value(v) for k, v in locals_dict.items() if k not in ("self", "__class__")
    }
    dsp = ", ".join(f"{k}={v}" for k, v in delimit.items())
    dsp = f"({dsp})" if dsp else ""
    return f"Initializing {locals_dict['self'].__class__.__name__}{dsp}"


_OLD_FACTORY = logging.getLogRecordFactory()


def _faceswap_logrecord(*args, **kwargs) -> logging.LogRecord:
    """Add a flag to :class:`logging.LogRecord` to not strip formatting from particular
    records."""
    record = _OLD_FACTORY(*args, **kwargs)
    record.strip_spaces = True
    return record


logging.setLogRecordFactory(_faceswap_logrecord)

# Set logger class to custom logger
logging.setLoggerClass(FaceswapLogger)

# Stores the last 100 debug messages
_DEBUG_BUFFER = RollingBuffer(maxlen=100)


__all__ = get_module_objects(__name__)
