#!/usr/bin/env python3
"""Pytest process setup."""

from __future__ import annotations

import gc
import os
import shutil
import tempfile
import tracemalloc

os.environ["OBJC_DISABLE_CLASS_WARNINGS"] = "YES"
os.environ["OBJC_DEBUG_DUPLICATE_CLASSES"] = "NO"
# Keep Qt widget tests headless by default.  Use setdefault so developers can
# still override the platform plugin explicitly when debugging locally.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Isolate tests from any landmark-ensemble production bundle that happens to be
# installed at the project's default ``.fs_cache/landmark_ensemble/current/``
# location. Without this, the plugin would pick up a real bundle and try to
# load its promoted setup file in unit tests that have no business touching
# disk artifacts. We force the env var to a per-process tmp dir that we wipe
# at startup so leftover state from earlier sessions (including any pipeline
# tests that wrote a real bundle here) does not leak between runs.
_FACESWAP_TEST_BUNDLE_DIR = os.path.join(tempfile.gettempdir(), "__faceswap_test_no_bundle__")
shutil.rmtree(_FACESWAP_TEST_BUNDLE_DIR, ignore_errors=True)
os.environ["FACESWAP_LANDMARK_ENSEMBLE_ARTIFACTS"] = _FACESWAP_TEST_BUNDLE_DIR


import pytest  # noqa: E402


def pytest_configure(config: pytest.Config) -> None:
    """Enable allocation traces for unraisable cleanup warnings.

    A flaky ``tempfile._TemporaryFileCloser`` warning is currently surfacing
    wherever cyclic GC happens to run, not where the bad wrapper was created.
    Pytest's warning text explicitly says to enable tracemalloc for the
    allocation traceback; doing it here keeps the next failure actionable
    without suppressing or filtering the warning.
    """
    if not tracemalloc.is_tracing():
        tracemalloc.start(25)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> object:
    """Force unraisable cleanup warnings to surface at the originating test.

    Without an explicit collection point, a leaked/broken temporary-file
    wrapper may not be finalized until a later, unrelated test.  Collecting
    after each test's teardown keeps the warning associated with the test
    that created the object while preserving the warning itself.
    """
    outcome = yield
    gc.collect()
    return outcome


@pytest.fixture(autouse=True)
def _isolate_landmark_ensemble_bundle(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Give every test its own empty bundle directory.

    Pipeline tests that exercise ``--write-config`` install a bundle into
    ``FACESWAP_LANDMARK_ENSEMBLE_ARTIFACTS``; without per-test isolation
    that bundle leaks into later unit tests that read from the same path.
    Each test gets a unique tmp directory, wiped between tests automatically
    by pytest's tmp_path machinery. Tests that want a specific bundle can
    still monkeypatch the env var to a path of their choice.
    """
    per_test_dir = tmp_path_factory.mktemp("landmark_ensemble_bundle")
    monkeypatch.setenv("FACESWAP_LANDMARK_ENSEMBLE_ARTIFACTS", str(per_test_dir))
