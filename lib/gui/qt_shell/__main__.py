#!/usr/bin/env python3
"""Run the Qt shell prototype with ``python -m lib.gui.qt_shell``."""

from __future__ import annotations

import sys

from lib.gui.qt_shell.main import main

if __name__ == "__main__":
    sys.exit(main())
