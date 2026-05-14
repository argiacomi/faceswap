#!/usr/bin/env python3
"""Command line interface helpers."""

from lib.cli.gui_launch_hardening import install_gui_launch_hardening as _install_gui_launch_hardening
from lib.cli.launcher import ScriptExecutor

_install_gui_launch_hardening(ScriptExecutor)
