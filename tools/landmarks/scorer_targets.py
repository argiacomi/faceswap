#!/usr/bin/env python3
"""Compatibility shim for moved scorer target helpers.

The implementation lives in :mod:`lib.landmarks.ensemble.scorer_targets`.
Keep this module until downstream imports have migrated.
"""

from lib.landmarks.ensemble.scorer_targets import *  # noqa: F401,F403
