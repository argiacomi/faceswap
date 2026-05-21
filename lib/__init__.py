#!/usr/bin/env python3
"""Initialization for faceswap's lib section"""

from __future__ import annotations

import os

os.environ["OBJC_DISABLE_CLASS_WARNINGS"] = "YES"
os.environ["OBJC_DEBUG_DUPLICATE_CLASSES"] = "NO"

# Import logger here so our custom loglevels are set for when executing code outside of FS
from . import logger
