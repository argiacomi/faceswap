#!/usr/bin/env python3
"""GUI-neutral aligner service for the Manual Tool (#104).

The Tk Manual Tool's :class:`tools.manual.manual.Aligner` couples plugin
loading, the alignment pipeline, and a Tk-shaped ``DetectedFaces`` cache into
a single class.  The Qt path needs an aligner surface that *doesn't* import
tkinter, can be loaded lazily off the UI thread, and is straightforward to
stub from a unit test.

This module exposes:

* :class:`ManualAlignerService` — caches loaded aligners by
  ``(plugin, normalization)``, exposes ``available_aligners`` /
  ``available_normalizations``, and runs ``align(image, bbox)`` to return a
  68-point landmark array.
* :class:`AlignerStatus` — a small dataclass the service emits while loading
  / running so the Qt host can surface progress and errors uniformly.
* :func:`make_default_aligner_factory` — the production factory that wraps
  the real :class:`lib.infer.align.Align` pipeline.  Tests inject a
  custom callable to bypass plugin loading.
"""

from __future__ import annotations

import logging
import threading
import typing as T
from dataclasses import dataclass

import numpy as np

from lib.utils import get_module_objects

logger = logging.getLogger(__name__)

NORMALIZATION_CHOICES: tuple[str, ...] = ("none", "clahe", "hist", "mean")
"""Normalization methods exposed in the Bounding Box editor's Normalization
radio set.  Matches the Tk Manual Tool's
:class:`tools.manual.frame_viewer.editor.bounding_box.BoundingBox` controls.
"""

DEFAULT_NORMALIZATION = "hist"
"""Default normalization (Tk parity)."""


@dataclass(frozen=True)
class AlignerStatus:
    """Status update emitted by the aligner service while loading or running."""

    kind: str
    """``"loading"`` / ``"ready"`` / ``"failed"`` / ``"aligning"`` / ``"aligned"``."""
    aligner: str
    """The aligner plugin display name this status applies to."""
    message: str = ""
    """User-facing message for status bar + console."""


StatusCallback = T.Callable[[AlignerStatus], None]
"""Type alias for status hooks consumed by the Qt host."""


class _AlignerBackend(T.Protocol):
    """Protocol that the cached aligner backend must satisfy.

    The production implementation wraps :class:`lib.infer.align.Align`
    through ``Align(plugin, normalization=...)()`` and pumps detected faces
    through the pipeline. Tests inject any callable that satisfies the same
    shape — typically a small class that emits canned landmarks.
    """

    def align(
        self,
        image: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> np.ndarray:
        """Return ``(N, 2)`` landmark array for the face at ``bbox`` in ``image``."""
        ...

    def set_normalization(self, method: str) -> None:
        """Switch the normalization method without rebuilding the pipeline."""
        ...


AlignerFactory = T.Callable[[str, str], _AlignerBackend]
"""``factory(aligner_name, normalization) -> _AlignerBackend``."""


def make_default_aligner_factory() -> AlignerFactory:
    """Return the production aligner factory.

    Lazily imports the alignment pipeline to keep ``import`` cheap for
    unit tests that never actually align.  The factory raises if model
    loading fails so the host can surface a clear status/error message
    without corrupting the editable model.
    """

    def _factory(aligner_name: str, normalization: str) -> _AlignerBackend:
        from lib.infer.align import Align  # local import — keeps tests light

        return _PluginAlignerBackend(aligner_name, normalization, Align)

    return _factory


class _PluginAlignerBackend:
    """Production backend wrapping the Faceswap alignment pipeline."""

    def __init__(
        self,
        aligner_name: str,
        normalization: str,
        align_cls: T.Any,
    ) -> None:
        self._aligner_name = aligner_name
        self._align_cls = align_cls
        self._normalization = normalization
        # Defer the pipeline build until align() so the host can ``__init__``
        # the backend in a worker thread without blocking the GUI.
        self._pipeline: T.Any = None

    def _ensure_pipeline(self) -> T.Any:
        if self._pipeline is None:
            self._pipeline = self._align_cls(
                self._aligner_name, normalization=self._normalization
            )()
        return self._pipeline

    def align(
        self,
        image: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> np.ndarray:
        """Push ``image`` + ``bbox`` through the extractor pipeline."""
        from lib.align import DetectedFace

        pipeline = self._ensure_pipeline()
        face = DetectedFace(
            x=int(round(bbox[0])),
            y=int(round(bbox[1])),
            w=int(round(bbox[2])),
            h=int(round(bbox[3])),
        )
        # filename can be a placeholder — the pipeline keys results by
        # filename but we only consume the single returned face below.
        result = pipeline.put(
            "manual",
            image,
            detected_faces=[face],
            passthrough=True,
        ).detected_faces[0]
        return np.asarray(result.landmarks_xy, dtype=np.float32)

    def set_normalization(self, method: str) -> None:
        """Propagate a normalization change to the underlying pipeline."""
        self._normalization = method
        if self._pipeline is None:
            return  # take effect next time _ensure_pipeline runs
        handler = getattr(self._pipeline, "handler", None)
        if handler is None:
            return
        set_normalize = getattr(handler, "set_normalize_method", None)
        if callable(set_normalize):
            set_normalize(method)


class ManualAlignerService:
    """Cache + facade over one or more aligner backends.

    The host calls :meth:`align` with the active aligner + normalization and
    a source image / bbox.  Backends are constructed on first request through
    ``factory`` (overridable for tests).  Status callbacks fire for loading
    progress, alignment kicks, and failures so the Qt status bar / console
    can mirror the Tk console's logging.

    The service is itself thread-safe: ``_backends`` is guarded so two
    background threads can request the same aligner without producing two
    half-loaded pipelines.
    """

    def __init__(
        self,
        *,
        available: T.Callable[[], T.Sequence[str]] | None = None,
        default: T.Callable[[], str] | None = None,
        factory: AlignerFactory | None = None,
        status_callback: StatusCallback | None = None,
    ) -> None:
        if available is None:
            from tools.manual.aligners import available_aligners as _default_available

            available = _default_available
        if default is None:
            from tools.manual.aligners import default_aligner as _default_default

            default = _default_default
        self._available = available
        self._default = default
        self._factory: AlignerFactory = factory or make_default_aligner_factory()
        self._status_callback = status_callback
        self._backends: dict[tuple[str, str], _AlignerBackend] = {}
        self._lock = threading.Lock()
        # Cache the discovery result so the host can render selection
        # dropdowns without hitting the plugin loader repeatedly.
        self._available_cache: tuple[str, ...] | None = None

    @property
    def available_aligners(self) -> tuple[str, ...]:
        """Return the available aligner plugin display names.

        Cached after the first call — plugin discovery requires walking
        :mod:`plugins.plugin_loader` and we don't want to repeat the
        filesystem scan every time the host paints a dropdown.
        """
        if self._available_cache is None:
            try:
                self._available_cache = tuple(self._available())
            except Exception:  # noqa: BLE001 - tolerate plugin loader failure
                logger.exception("Manual Tool: failed to enumerate aligners")
                self._available_cache = ()
        return self._available_cache

    def available_normalizations(self) -> tuple[str, ...]:
        """Return the supported normalization methods."""
        return NORMALIZATION_CHOICES

    def default_aligner(self) -> str:
        """Return the preferred aligner name (falls back to the first available)."""
        try:
            return self._default()
        except Exception:  # noqa: BLE001 - tolerate plugin loader failure
            logger.exception("Manual Tool: default aligner lookup failed")
            aligners = self.available_aligners
            return aligners[0] if aligners else ""

    def preload(self, aligner: str, normalization: str = DEFAULT_NORMALIZATION) -> bool:
        """Eagerly construct the backend so a subsequent :meth:`align` is fast.

        Returns ``True`` on success.  On failure the backend is *not*
        cached so the next ``align`` call can retry instead of replaying
        the same exception.  Status callbacks fire ``"loading"`` then
        either ``"ready"`` or ``"failed"`` so the host can update a
        determinate progress bar without polling.
        """
        self._emit_status("loading", aligner, f"Loading aligner '{aligner}'…")
        try:
            self._get_backend(aligner, normalization)
        except Exception as err:  # noqa: BLE001 - surface to caller
            logger.exception("Manual Tool: aligner preload failed (%s)", aligner)
            self._emit_status("failed", aligner, f"Aligner '{aligner}' failed to load: {err}")
            return False
        self._emit_status("ready", aligner, f"Aligner '{aligner}' ready")
        return True

    def align(
        self,
        image: np.ndarray,
        bbox: tuple[float, float, float, float],
        *,
        aligner: str | None = None,
        normalization: str = DEFAULT_NORMALIZATION,
    ) -> np.ndarray:
        """Push ``image``+``bbox`` through the selected aligner.

        Returns the ``(N, 2)`` float32 landmark array.  Raises whatever
        the backend raises if the alignment fails — callers are expected
        to wrap the call so a failed run does not corrupt the editable
        model.  Emits ``"aligning"`` and ``"aligned"`` status updates
        so progress is visible.
        """
        aligner = aligner or self.default_aligner()
        if not aligner:
            raise RuntimeError("No aligner plugins are available")
        self._emit_status("aligning", aligner, f"Running aligner '{aligner}'…")
        backend = self._get_backend(aligner, normalization)
        landmarks = backend.align(image, bbox)
        landmarks_array = np.asarray(landmarks, dtype=np.float32)
        self._emit_status(
            "aligned",
            aligner,
            f"Aligner '{aligner}' produced {len(landmarks_array)} landmark(s)",
        )
        return landmarks_array

    def set_normalization(self, normalization: str, aligner: str | None = None) -> None:
        """Propagate a normalization change to one or all loaded backends.

        Mirrors :meth:`tools.manual.manual.Aligner.set_normalization_method`:
        Tk propagates the change to every loaded plugin so the user's
        choice persists across aligner switches.
        """
        with self._lock:
            for (name, _norm), backend in list(self._backends.items()):
                if aligner is not None and name != aligner:
                    continue
                try:
                    backend.set_normalization(normalization)
                except Exception:  # noqa: BLE001 - non-fatal
                    logger.exception("Manual Tool: aligner '%s' refused normalization", name)

    def _get_backend(self, aligner: str, normalization: str) -> _AlignerBackend:
        """Return a cached backend for ``(aligner, normalization)``."""
        key = (aligner, normalization)
        with self._lock:
            backend = self._backends.get(key)
            if backend is not None:
                return backend
        # Construct outside the lock so concurrent requests for *different*
        # aligners don't queue behind a slow plugin load.
        backend = self._factory(aligner, normalization)
        with self._lock:
            self._backends.setdefault(key, backend)
            return self._backends[key]

    def _emit_status(self, kind: str, aligner: str, message: str) -> None:
        """Forward a status update if a callback is installed."""
        if self._status_callback is None:
            return
        try:
            self._status_callback(AlignerStatus(kind=kind, aligner=aligner, message=message))
        except Exception:  # noqa: BLE001 - status hooks must never raise
            logger.exception("Manual Tool: aligner status callback failed")


__all__ = get_module_objects(__name__)
