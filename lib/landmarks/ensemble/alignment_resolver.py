#!/usr/bin/env python3
"""Compatibility shim — :mod:`lib.landmarks.alignment.resolver` is the new home.

The resolver routes faces through low-risk / high-risk / invalid paths
using geometry-risk evidence. That's alignment decisioning, not ensemble
weighting, so the module moved under :mod:`lib.landmarks.alignment`.
Existing imports keep working via this shim.
"""

from __future__ import annotations

from lib.landmarks.alignment.resolver import *  # noqa: F401, F403
from lib.landmarks.alignment.resolver import __all__  # noqa: F401
