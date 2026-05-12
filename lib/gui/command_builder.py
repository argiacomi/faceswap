#!/usr/bin/env python3
"""Compatibility import for GUI command builder service."""

from lib.gui.services.command_builder import CommandBuilder as CommandBuilder
from lib.utils import get_module_objects

__all__ = get_module_objects(__name__)
