#!/usr/bin/env python3
"""Compatibility shim — :mod:`lib.landmarks.search.candidate_search` is the new home.

The candidate-search subsystem moved out of ``eval/`` because it isn't just
metric evaluation — it owns Candidate / CandidateResult dataclasses, the
search-space enumeration, fitted-weight caching, scoring, and the report
payloads. Migrating call sites incrementally; this shim keeps existing
imports working in the meantime.
"""

from __future__ import annotations

from lib.landmarks.search.candidate_search import *  # noqa: F401, F403
from lib.landmarks.search.candidate_search import __all__  # noqa: F401
