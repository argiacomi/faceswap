#!/usr/bin/env python3
"""Hermetic defaults for the landmark-ensemble plugin test folder.

When the local Faceswap config has ``use_alignment_resolver=True`` and/or
``resolver_policy="learned_quality_v2"`` (e.g. an installed production
bundle), ``Ensemble(...)`` calls that don't explicitly opt out fall through
to those defaults and raise:

* ``ProductionBundleMissing`` when no bundle/setup/scorer kwargs are
  supplied — the test environment intentionally doesn't ship one.
* ``ValueError: learned_quality_v2 requires resolver_scorer_path`` when the
  policy demands a scorer path but the test doesn't provide one.

This conftest pins the implicit defaults to:

* ``use_alignment_resolver = False`` — tests that aren't *about* the
  resolver stay on the legacy non-resolver fusion path.
* ``resolver_policy = "roll_aware_veto"`` — for tests that *do* opt into
  the resolver but don't supply a scorer, ``roll_aware_veto`` runs
  without one (Tk-parity safety check); tests that intentionally exercise
  ``learned_quality_v1`` / ``v1_1`` / ``v2`` still pass their own
  ``resolver_policy=...`` kwarg, which the test fixture leaves alone.

Tests that explicitly want the resolver on remain unchanged because they
already pass ``use_alignment_resolver=True`` to the constructor — the
explicit kwarg wins.

The fixture is ``autouse`` so every test in this folder picks it up; it
does *not* touch production behavior or environment variables.
"""

from __future__ import annotations

import pytest

from plugins.extract.align import ensemble_defaults as cfg


@pytest.fixture(autouse=True)
def hermetic_landmark_ensemble_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the implicit landmark-ensemble defaults for every test in this folder.

    See module docstring for the rationale.  The two pins are:

    * ``use_alignment_resolver()`` → ``False`` so non-resolver tests don't
      accidentally route through the production resolver path when the local
      machine's config has it enabled.
    * ``resolver_policy()`` → ``"roll_aware_veto"`` so tests that opt into
      the resolver but don't provide a scorer don't fail the
      ``learned_quality_*`` policy → ``resolver_scorer_path`` check.
    """
    monkeypatch.setattr(cfg, "use_alignment_resolver", lambda: False)
    monkeypatch.setattr(cfg, "resolver_policy", lambda: "roll_aware_veto")
