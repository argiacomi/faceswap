#!/usr/bin/env python3
"""Regression coverage for Faceswap logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

from lib.logger import log_setup


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
