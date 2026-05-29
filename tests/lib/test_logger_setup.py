#!/usr/bin/env python3
"""Regression coverage for Faceswap logging setup."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from lib.logger import FaceswapFormatter, log_setup


def _faceswap_handlers() -> list[logging.Handler]:
    """Return root handlers installed by ``log_setup``."""
    return [
        handler
        for handler in logging.getLogger().handlers
        if getattr(handler, "_faceswap_managed_handler", False)
    ]


def test_log_setup_replaces_faceswap_handlers(tmp_path: Path) -> None:
    """Calling ``log_setup`` repeatedly must not duplicate console output.

    Several tests import modules that call ``log_setup`` during collection.  If
    every call appends another stream/file/crash handler, later GUI tests print
    each Alignments read/write log once per stale handler.  ``log_setup`` should
    replace only its own handlers and leave pytest/external handlers intact.
    """
    root_logger = logging.getLogger()
    original_external_handlers = tuple(
        handler
        for handler in root_logger.handlers
        if not getattr(handler, "_faceswap_managed_handler", False)
    )

    try:
        log_setup("INFO", str(tmp_path / "first.log"), "tools")
        first_handlers = tuple(_faceswap_handlers())
        assert len(first_handlers) == 3

        log_setup("VERBOSE", str(tmp_path / "second.log"), "tools")
        second_handlers = tuple(_faceswap_handlers())
        assert len(second_handlers) == 3
        assert not set(first_handlers).intersection(second_handlers)

        for handler in original_external_handlers:
            assert handler in root_logger.handlers

    finally:
        for handler in tuple(root_logger.handlers):
            if getattr(handler, "_faceswap_managed_handler", False):
                root_logger.removeHandler(handler)
                handler.close()
        for handler in original_external_handlers:
            if handler not in root_logger.handlers:
                root_logger.addHandler(handler)


def test_log_setup_replaces_legacy_untagged_faceswap_handlers(tmp_path: Path) -> None:
    """Untagged pre-marker Faceswap handlers should be treated as stale.

    This mirrors full-suite collection order where a legacy test module may
    have called ``log_setup`` before the explicit handler marker existed.  The
    cleanup must recognize Faceswap's handler/formatter shape too, otherwise a
    stale console handler survives and Alignments read/write logs are still
    printed twice.
    """
    root_logger = logging.getLogger()
    legacy_stream = logging.StreamHandler(sys.stdout)
    legacy_stream.setFormatter(FaceswapFormatter("%(asctime)s %(levelname)-8s %(message)s"))
    external_stream = logging.StreamHandler(sys.stdout)
    external_stream.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(legacy_stream)
    root_logger.addHandler(external_stream)

    try:
        log_setup("INFO", str(tmp_path / "fresh.log"), "tools")

        assert legacy_stream not in root_logger.handlers
        assert external_stream in root_logger.handlers
        assert len(tuple(_faceswap_handlers())) == 3
    finally:
        root_logger.removeHandler(external_stream)
        external_stream.close()
        for handler in tuple(_faceswap_handlers()):
            root_logger.removeHandler(handler)
            handler.close()


# ---------------------------------------------------------------------------
# External logger policy + blank-line suppression (issue #189)
# ---------------------------------------------------------------------------


from io import StringIO  # noqa: E402

from lib.logger import (  # noqa: E402
    EXTERNAL_LOGGER_POLICY,
    TqdmHandler,
    _apply_external_logger_policy,
)


def _make_stream_handler() -> tuple[logging.Handler, StringIO]:
    """Build a plain StreamHandler with the canonical Faceswap formatter."""
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(FaceswapFormatter("%(levelname)-8s %(name)s %(message)s"))
    handler.setLevel(logging.DEBUG)
    return handler, buf


def _make_record(name: str, level: int, msg: str, *, funcname: str = "fn") -> logging.LogRecord:
    return logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
        func=funcname,
    )


def test_external_policy_lowers_matplotlib_info_records_on_stream() -> None:
    """matplotlib INFO records should be dropped on the stream handler (issue #189)."""
    handler, buf = _make_stream_handler()
    _apply_external_logger_policy(logging.getLogger(), handler)

    handler.handle(_make_record("matplotlib.font_manager", logging.INFO, "noisy font discovery"))
    handler.handle(_make_record("matplotlib", logging.INFO, "another INFO"))

    assert "noisy font discovery" not in buf.getvalue()
    assert "another INFO" not in buf.getvalue()


def test_external_policy_keeps_matplotlib_warning_on_stream() -> None:
    """Lowering INFO must not silence WARNING records."""
    handler, buf = _make_stream_handler()
    _apply_external_logger_policy(logging.getLogger(), handler)

    handler.handle(_make_record("matplotlib", logging.WARNING, "real problem"))

    assert "real problem" in buf.getvalue()


def test_external_policy_caps_pil_error_on_stream() -> None:
    """``max_stream_level=WARNING`` drops ERROR records on the stream handler."""
    handler, buf = _make_stream_handler()
    _apply_external_logger_policy(logging.getLogger(), handler)

    handler.handle(_make_record("PIL.Image", logging.ERROR, "should not surface"))

    assert "should not surface" not in buf.getvalue()


def test_external_policy_suppresses_torch_inductor_is_big_gpu() -> None:
    """Specific Torch inductor noise should be dropped on the stream handler."""
    handler, buf = _make_stream_handler()
    _apply_external_logger_policy(logging.getLogger(), handler)

    handler.handle(
        _make_record(
            "torch._inductor.utils",
            logging.WARNING,
            "Not enough SMs to use max_autotune_gemm mode",
            funcname="is_big_gpu",
        )
    )

    assert "Not enough SMs" not in buf.getvalue()


def test_external_policy_does_not_modify_file_handler_records(tmp_path: Path) -> None:
    """Filters live on the stream handler only — file/crash handlers see everything (issue #189).

    Regression: the legacy ``FaceswapFormatter._lower_external`` mutated
    ``LogRecord.levelno`` during formatting, which could leak across
    handlers depending on handler order. The new policy uses handler-bound
    filters so this test asserts the file handler still receives the
    original INFO.
    """
    log_path = tmp_path / "external_policy.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(FaceswapFormatter("%(levelname)-8s %(name)s %(message)s"))
    file_handler.setLevel(logging.DEBUG)

    stream_handler, _stream_buf = _make_stream_handler()
    _apply_external_logger_policy(logging.getLogger(), stream_handler)

    rec = _make_record("matplotlib.font_manager", logging.INFO, "fontconfig boot")
    stream_handler.handle(rec)
    file_handler.handle(rec)
    file_handler.flush()

    contents = log_path.read_text(encoding="utf-8")
    assert "fontconfig boot" in contents
    # The level marker in the file should still be INFO (not DEBUG) — proof
    # the filter did not mutate the LogRecord shared across handlers.
    assert "INFO" in contents


def test_blank_message_filter_skips_whitespace_only_records() -> None:
    """Stream output must not contain blank lines from empty captured warnings (issue #189)."""
    handler, buf = _make_stream_handler()
    _apply_external_logger_policy(logging.getLogger(), handler)

    handler.handle(_make_record("py.warnings", logging.WARNING, "   \n\t  "))
    # An empty msg evaluates to a non-empty formatted line because of the
    # ``%(levelname)-8s %(name)s`` prefix — so this asserts the filter is
    # active only for genuinely whitespace-only POST-format output by
    # using an entirely empty formatter for clarity:
    bare_handler = logging.StreamHandler(StringIO())
    bare_handler.setFormatter(FaceswapFormatter("%(message)s"))
    _apply_external_logger_policy(logging.getLogger(), bare_handler)
    bare_handler.handle(_make_record("py.warnings", logging.WARNING, "   \n\t  "))

    assert bare_handler.stream.getvalue() == ""


def test_tqdm_handler_skips_blank_messages(monkeypatch) -> None:
    """TqdmHandler.emit must not call ``tqdm.write`` for empty messages (issue #189)."""
    handler = TqdmHandler(StringIO())
    handler.setFormatter(FaceswapFormatter("%(message)s"))
    written: list[str] = []

    class _FakeTqdm:
        @staticmethod
        def write(msg: str) -> None:
            written.append(msg)

    monkeypatch.setitem(sys.modules, "tqdm", type("M", (), {"tqdm": _FakeTqdm}))

    handler.emit(_make_record("py.warnings", logging.WARNING, "   "))
    handler.emit(_make_record("py.warnings", logging.WARNING, ""))
    handler.emit(_make_record("any", logging.INFO, "real content"))

    assert written == ["real content"]


def test_warning_records_have_internal_newlines_collapsed() -> None:
    """WARNING/ERROR records must not introduce blank stream lines (issue #189)."""
    handler, buf = _make_stream_handler()

    handler.handle(_make_record("test", logging.WARNING, "line one\nline two\nline three"))

    lines = buf.getvalue().splitlines()
    # The multiline body should collapse into a single line; the only
    # newlines should be the trailing record terminator.
    assert len(lines) == 1
    assert "line one\\nline two\\nline three" in lines[0]


def test_warning_records_preserve_exception_traceback() -> None:
    """Tracebacks remain multi-line even after newline collapsing."""
    handler, buf = _make_stream_handler()

    try:
        raise ValueError("boom")
    except ValueError:
        import sys as _sys

        exc_info = _sys.exc_info()

    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="explanation",
        args=(),
        exc_info=exc_info,
        func="fn",
    )
    handler.handle(record)

    out = buf.getvalue()
    assert "explanation" in out
    # The traceback block is preserved verbatim (multiline).
    assert "Traceback (most recent call last):" in out
    assert "ValueError: boom" in out


def test_external_logger_policy_is_centralized_in_lib_logger() -> None:
    """The policy dict must own every external-logger configuration knob (issue #189)."""
    # The policy declares each known noisy logger explicitly so changes
    # land in one place and are testable.
    assert "matplotlib" in EXTERNAL_LOGGER_POLICY
    assert "libav" in EXTERNAL_LOGGER_POLICY
    assert "PIL" in EXTERNAL_LOGGER_POLICY
    assert "torch._inductor" in EXTERNAL_LOGGER_POLICY


def test_configure_tool_logging_is_idempotent(tmp_path: Path) -> None:
    """Repeated calls must not stack handlers (issue #189)."""
    import lib.logger as logger_mod

    rootlogger = logging.getLogger()
    external_handlers_before = tuple(
        h for h in rootlogger.handlers if not getattr(h, "_faceswap_managed_handler", False)
    )

    try:
        logger_mod.configure_tool_logging("INFO")
        first = _faceswap_handlers()
        logger_mod.configure_tool_logging("DEBUG")
        second = _faceswap_handlers()

        assert len(first) == 1
        assert len(second) == 1
        # Different instance after the second call — the prior handler was
        # cleaned up rather than left around.
        assert first[0] is not second[0]
        for handler in external_handlers_before:
            assert handler in rootlogger.handlers
    finally:
        for handler in tuple(rootlogger.handlers):
            if getattr(handler, "_faceswap_managed_handler", False):
                rootlogger.removeHandler(handler)
                handler.close()
