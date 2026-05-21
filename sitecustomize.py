#!/usr/bin/env python3
"""Process-wide Python startup tweaks for local tools and tests."""

from __future__ import annotations

import os

os.environ["OBJC_DISABLE_CLASS_WARNINGS"] = "YES"
