#!/usr/bin/env python3
"""Compatibility shim — :mod:`lib.landmarks.search.promotion_gates` is the new home.

Promotion gating is part of the search/promotion subsystem, not a metric.
Migrating call sites incrementally; this shim keeps existing imports
working in the meantime.
"""

from __future__ import annotations

from lib.landmarks.search.promotion_gates import *  # noqa: F401, F403
from lib.landmarks.search.promotion_gates import __all__  # noqa: F401
