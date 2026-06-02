#!/usr/bin/env python3
"""Pytest process setup."""

from __future__ import annotations

import contextlib
import gc
import os
import shutil
import tempfile

# import tracemalloc

os.environ["OBJC_DISABLE_CLASS_WARNINGS"] = "YES"
os.environ["OBJC_DEBUG_DUPLICATE_CLASSES"] = "NO"

# PyTorch and LightGBM each ship their own libomp. Loading both into one process
# (which happens when a single ``pytest`` run touches the torch-based plugins and
# the lightgbm-based learned_quality_v2 scorer training) triggers an "OMP: System
# error #22" / segfault on macOS. ``faceswap.py`` guards against this before any
# torch/lightgbm import at runtime, but pytest never imports ``faceswap.py``, so
# mirror the same guard here before any heavy module is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LIBOMP_NUM_THREADS", "1")

# Import lightgbm eagerly (before any torch-importing test module) so its libomp
# initializes first; mixing the load order with torch in one process otherwise
# segfaults during lightgbm training on macOS.
with contextlib.suppress(Exception):
    import lightgbm  # noqa: F401

# Keep Qt widget tests headless under pytest, even if a developer shell has a
# display-backed Qt platform configured.
os.environ["QT_QPA_PLATFORM"] = "offscreen"

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
    # if not tracemalloc.is_tracing():
    #     tracemalloc.start(25)


# Path fragments that auto-classify collected tests into tier markers. Keeping
# the routing here avoids sprinkling ``pytestmark`` boilerplate across hundreds
# of files and gives one obvious place to retune the fast/slow split.
_GUI_PATH_FRAGMENTS = (
    f"tests{os.sep}lib{os.sep}gui{os.sep}",
    f"tests{os.sep}scripts{os.sep}test_gui_qt_launch.py",
    f"tests{os.sep}tools{os.sep}manual{os.sep}qt{os.sep}",
    f"tests{os.sep}tools{os.sep}preview{os.sep}",
)
_SLOW_PATH_FRAGMENTS = (
    # Heavy phaze-a parametrizations (bottleneck x latent_norm matrix, decoder
    # contract variants, etc.) belong to the slow tier; the per-commit run
    # still exercises representative model construction / state coverage via
    # the non-matrix tests in test_phaze_a.py.
    f"tests{os.sep}plugins{os.sep}train{os.sep}test_phaze_a_decoder_contracts.py",
    f"tests{os.sep}plugins{os.sep}train{os.sep}test_phaze_a_refinement_tail.py",
    f"tests{os.sep}plugins{os.sep}train{os.sep}test_phaze_a_decoder_norm.py",
    f"tests{os.sep}plugins{os.sep}train{os.sep}test_phaze_a_residual_scale.py",
    f"tests{os.sep}plugins{os.sep}train{os.sep}test_phaze_a_activation_config.py",
    f"tests{os.sep}plugins{os.sep}train{os.sep}test_phaze_a_antialias.py",
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply the ``gui`` and ``slow`` tier markers based on file location."""
    for item in items:
        path = str(item.fspath)
        if any(fragment in path for fragment in _GUI_PATH_FRAGMENTS):
            item.add_marker(pytest.mark.gui)
        if any(fragment in path for fragment in _SLOW_PATH_FRAGMENTS):
            item.add_marker(pytest.mark.slow)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> object:
    """Force unraisable cleanup warnings to surface at the originating test.

    Only Qt/Tk GUI tests need an explicit ``gc.collect()`` after teardown:
    those suites create cyclic widget graphs whose temporary-file finalisers
    otherwise fire in unrelated later tests.  Pure-Python unit tests do not
    pay the cost of a full collection after every test.
    """
    outcome = yield
    if item.get_closest_marker("gui") is not None:
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
