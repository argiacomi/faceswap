#!/usr/env/bin/python3
"""General utility functions for Faceswap inference"""

from __future__ import annotations

import logging
import typing as T
from collections.abc import Iterable, Mapping
from threading import Event, Lock
from time import perf_counter, sleep

import cv2
import numpy as np
import torch

from lib.torch_utils import accelerator_empty_cache
from lib.utils import get_module_objects

from .compile_modes import (
    CompilePolicy,
    CompileResult,
    compile_module,
    is_compiled_module,
    record_compile_outcome,
    reset_compiled_module,
    resolve_compile_policy,
)

if T.TYPE_CHECKING:
    from plugins.extract.base import ExtractPlugin


logger = logging.getLogger(__name__)


def random_input_from_plugin(
    plugin: ExtractPlugin, batch_size: int, channels_last: bool
) -> np.ndarray:
    """Obtain a random input array from a plugin's information for the given batch size

    Parameters
    ----------
    plugin
        The plugin to obtain the input array for
    batch_size : int
        The batch size for the input array
    channels_last : bool
        ``True`` if the data should be formatted channels last

    Returns
    -------
    A random input array in the correct format for the given plugin at the given batch size
    """
    size = plugin.input_size
    low, high = plugin.scale
    im_range = high - low
    retval = np.random.random((batch_size, 3, size, size)).astype(plugin.dtype) * im_range
    retval += low
    if channels_last:
        retval = retval.transpose(0, 2, 3, 1)
    return retval  # type: ignore[no-any-return]


def get_torch_modules(
    obj: T.Any,  # noqa: C901  # pylint:disable=too-many-branches,too-many-return-statements
    mod: str | None = None,
    seen: set[int] | None = None,
    results: list[torch.nn.Module] | None = None,
) -> list[torch.nn.Module]:
    """Recursively search a plugin's model attribute to find any parent :class:`torch.nn.Module`s

    Parameters
    ----------
    obj
        The object to check if it is a torch Module. This should be a plugin's `model` attribute
    mod
        The module that the parent model class belongs to. Default: ``None`` (Collected from the
        first object entered into the recursive function)
    seen
        A set of seen object IDs to prevent self-recursion. Default: ``None`` (Created when the
        first object enters the recursive function)
    results
        List of discovered torch modules. Default: ``None`` (Created when the first object enters
        the recursive function)

    Returns
    -------
    The list of discovered torch Modules
    """
    seen = set() if seen is None else seen
    retval: list[torch.nn.Module] = [] if results is None else results
    mod = obj.__class__.__module__ if mod is None else mod

    obj_id = id(obj)
    if obj_id in seen:
        return retval
    seen.add(obj_id)

    if isinstance(obj, torch.nn.Module):
        logger.debug("Torch module found in %s(%s)", obj.__class__.__name__, type(obj))
        retval.append(obj)
        return retval

    if isinstance(obj, (str, bytes, int, float, bool, type(None))):
        # Fast exit on primitive
        return retval

    if hasattr(obj, "__class__") and obj.__class__.__module__ not in (mod, "builtins"):
        # Never leave the plugin module
        return retval

    if isinstance(obj, Mapping):
        # Mapping before iterable as a mapping is also an iterable
        for val in obj.values():
            retval = get_torch_modules(val, mod, seen=seen, results=retval)

    if isinstance(obj, Iterable):
        for val in obj:
            retval = get_torch_modules(val, mod, seen=seen, results=retval)

    if hasattr(obj, "__dict__"):
        for val in obj.__dict__.values():
            retval = get_torch_modules(val, mod, seen=seen, results=retval)
    return retval


def warmup_plugin(
    plugin: ExtractPlugin,  # noqa: C901
    batch_size: int,
    channels_last: bool | None = None,
) -> bool | None:
    """Warm up a plugin that contains torch modules. If channels_last is ``None`` then attempt to
    send a channels first batch through. If it fails, send a channels last batch through

    Parameters
    ----------
    plugin
        The plugin to warmup
    batch_size
        The batch size to put through the model
    channels_last
        The expected channel order of the plugin or ``None`` to detect

    Returns
    -------
    bool
        ``True`` if the plugin is detected as channels last, ``False`` for channels first, ``None``
        for could not be detected
    """
    cv2_loglevel = None
    cv2_setlevel = None
    if channels_last is None:
        # cv2 outputs scary warnings when we are testing channels first/last with cv2-dnn plugins
        # so disable logging
        try:  # cv2 arbitrarily moves this based on build options :/
            cv2_loglevel = cv2.getLogLevel()  # type:ignore[attr-defined]
            cv2_setlevel = cv2.setLogLevel  # type: ignore[attr-defined]
        except AttributeError:
            try:
                cv2_loglevel = cv2.utils.logging.getLogLevel()
                cv2_setlevel = cv2.utils.logging.setLogLevel
            except AttributeError:
                pass

    chan_list = [False, True] if channels_last is None else [channels_last]
    is_chan_last = None

    if cv2_setlevel is not None:
        cv2_setlevel(0)

    for chan_last in chan_list:
        try:
            inp = random_input_from_plugin(plugin, batch_size, chan_last)
            plugin.process(inp)
            is_chan_last = chan_last
            break
        except Exception as err:  # pylint:disable=broad-except
            logger.debug("Exception with channels_last=%s: %s", chan_last, str(err).strip())

    if cv2_setlevel is not None:
        cv2_setlevel(cv2_loglevel)
    logger.debug("[%s] Warmed up. channels_last: %s", plugin.name, is_chan_last)
    return is_chan_last


_COMPILE_LOCK = Lock()
_COMPILE_LOGGED = Event()
_MPS_COMPILE_SKIP_PLUGINS = {
    "ORFormer": (
        "Skipping torch.compile on Apple Silicon for ORFormer due to known "
        "TorchInductor/MPS codegen instability during first compiled execution."
    ),
    "SPIGA": (
        "Skipping torch.compile on Apple Silicon for SPIGA due to known MPS shape-guard "
        "instability during first compiled execution."
    ),
}


def _elapsed_ms(start: float) -> float:
    """Return elapsed wall-clock milliseconds since ``start``."""
    return (perf_counter() - start) * 1000.0


def _unique_modules(modules: list[torch.nn.Module]) -> tuple[list[torch.nn.Module], int]:
    """Return modules with duplicate references removed while preserving order."""
    retval: list[torch.nn.Module] = []
    seen: set[int] = set()
    duplicates = 0
    for module in modules:
        module_id = id(module)
        if module_id in seen:
            duplicates += 1
            continue
        seen.add(module_id)
        retval.append(module)
    return retval, duplicates


def compile_models(
    plugin: ExtractPlugin,
    modules: list[torch.nn.Module],
    compile_mode: CompilePolicy | str | bool,
) -> None:
    """Compile any Torch modules in the plugin's `model` attribute

    Parameters
    ----------
    plugin
        The plugin containing Torch modules to be compiled
    modules
        The list of Torch modules contained within the plugin's `model` attribute
    compile_mode
        The requested or resolved compile mode for the current backend
    """
    policy = resolve_compile_policy(compile_mode)
    modules, duplicate_refs = _unique_modules(modules)
    module_names = ", ".join(mod.__class__.__name__ for mod in modules)
    with _COMPILE_LOCK:
        if not _COMPILE_LOGGED.is_set():
            _COMPILE_LOGGED.set()
            sleep(0.5)  # Let other plugins log their output first
            logger.info("Compiling PyTorch models...")
        logger.info(
            "[%s] Compile start: modules=%s backend=%s requested_mode=%s "
            "effective_mode=%s duplicate_refs=%s",
            plugin.name,
            module_names or "none",
            policy.backend,
            policy.requested,
            policy.effective,
            duplicate_refs,
        )
        skip_reason = _MPS_COMPILE_SKIP_PLUGINS.get(plugin.name)
        if policy.backend == "apple_silicon" and skip_reason and policy.enabled:
            logger.warning("[%s] %s", plugin.name, skip_reason)
            record_compile_outcome(
                policy,
                CompileResult(
                    compiled=False,
                    fallback_status="backend_skip",
                    fullgraph=None,
                    compile_time_ms=0.0,
                    final_execution_mode="eager",
                    error_summary=skip_reason,
                ),
            )
            eager_start = perf_counter()
            warmup_plugin(plugin, plugin.batch_size)
            logger.info(
                "[%s] Compile skipped: eager_warmup_elapsed_ms=%.2f batch_size=%s backend=%s",
                plugin.name,
                _elapsed_ms(eager_start),
                plugin.batch_size,
                policy.backend,
            )
            return
        if policy.requested != policy.effective:
            logger.info(
                "[%s] Compile mode '%s' downgraded to '%s' on backend '%s'. %s",
                plugin.name,
                policy.requested,
                policy.effective,
                policy.backend,
                policy.reason,
            )
        elif policy.experimental:
            logger.debug(
                "[%s] Compile mode '%s' on backend '%s' is experimental. %s",
                plugin.name,
                policy.effective,
                policy.backend,
                policy.reason,
            )

        total_start = perf_counter()
        channel_warmup_start = perf_counter()
        channels_last = warmup_plugin(plugin, 1)
        channel_warmup_ms = _elapsed_ms(channel_warmup_start)
        logger.info(
            "[%s] Channel-order warmup complete: channels_last=%s batch_size=%s elapsed_ms=%.2f",
            plugin.name,
            channels_last,
            plugin.batch_size,
            channel_warmup_ms,
        )
        for mod in modules:
            if is_compiled_module(mod):
                logger.warning(
                    "[%s/%s] Skipping compile for already-compiled module",
                    plugin.name,
                    mod.__class__.__name__,
                )
                continue
            logger.verbose(  # type:ignore[attr-defined]
                "Compiling %s (%s) with mode '%s'...",
                plugin.name,
                mod.__class__.__name__,
                policy.effective,
            )
            result = compile_module(mod, policy)
            if not result.compiled:
                logger.warning(
                    "[%s] backend=%s mode=%s fullgraph=%s dynamic=%s fallback=%s "
                    "compile_wrap_time_ms=%.2f final_execution_mode=%s error=%s",
                    plugin.name,
                    policy.backend,
                    policy.effective,
                    "n/a",
                    policy.dynamic,
                    result.fallback_status,
                    result.compile_time_ms,
                    result.final_execution_mode,
                    result.error_summary or "n/a",
                )
                continue
            logger.info(
                "[%s] backend=%s mode=%s fullgraph=%s dynamic=%s fallback=%s "
                "compile_wrap_time_ms=%.2f final_execution_mode=%s",
                plugin.name,
                policy.backend,
                policy.effective,
                result.fullgraph,
                policy.dynamic,
                result.fallback_status,
                result.compile_time_ms,
                result.final_execution_mode,
            )
        # Send the warmup batch here as we need to keep the lock when tracing
        compiled_warmup_start = perf_counter()
        compiled_ready = warmup_plugin(plugin, plugin.batch_size, channels_last=channels_last)
        compiled_warmup_ms = _elapsed_ms(compiled_warmup_start)
        if compiled_ready is None:
            error_summary = (
                "First compiled execution warmup failed; reverting compiled "
                "modules to eager execution"
            )
            logger.warning(
                "[%s] First compiled execution warmup failed: elapsed_ms=%.2f batch_size=%s "
                "backend=%s. Reverting to eager execution.",
                plugin.name,
                compiled_warmup_ms,
                plugin.batch_size,
                policy.backend,
            )
            for mod in modules:
                reset_compiled_module(mod)
            record_compile_outcome(
                policy,
                CompileResult(
                    compiled=False,
                    fallback_status="runtime_eager_fallback",
                    fullgraph=None,
                    compile_time_ms=compiled_warmup_ms,
                    final_execution_mode="eager",
                    error_summary=error_summary,
                ),
            )
            eager_warmup_start = perf_counter()
            eager_ready = warmup_plugin(
                plugin,
                plugin.batch_size,
                channels_last=channels_last,
            )
            eager_warmup_ms = _elapsed_ms(eager_warmup_start)
            if eager_ready is None:
                eager_error = "Compiled execution failed and eager fallback warmup also failed"
                logger.error(
                    "[%s] Compile fallback failed: eager warmup also failed. "
                    "elapsed_ms=%.2f batch_size=%s backend=%s",
                    plugin.name,
                    eager_warmup_ms,
                    plugin.batch_size,
                    policy.backend,
                )
                record_compile_outcome(
                    policy,
                    CompileResult(
                        compiled=False,
                        fallback_status="runtime_eager_fallback_failed",
                        fullgraph=None,
                        compile_time_ms=eager_warmup_ms,
                        final_execution_mode="failed",
                        error_summary=eager_error,
                    ),
                )
                return
            logger.info(
                "[%s] Compile fallback complete: eager_warmup_elapsed_ms=%.2f "
                "total_compile_ready_ms=%.2f backend=%s",
                plugin.name,
                eager_warmup_ms,
                _elapsed_ms(total_start),
                policy.backend,
            )
            return
        logger.info(
            "[%s] First compiled execution warmup complete: elapsed_ms=%.2f batch_size=%s "
            "backend=%s",
            plugin.name,
            compiled_warmup_ms,
            plugin.batch_size,
            policy.backend,
        )
        logger.info(
            "[%s] Compile ready: modules=%s final_batch_size=%s backend=%s "
            "total_compile_ready_ms=%.2f",
            plugin.name,
            len(modules),
            plugin.batch_size,
            policy.backend,
            _elapsed_ms(total_start),
        )
    if policy.backend in ("nvidia", "rocm"):
        accelerator_empty_cache()  # Need to clear cache or we may run out of VRAM


__all__ = get_module_objects(__name__)
