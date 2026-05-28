#!/usr/bin/env python3
"""Backend-aware Torch compile mode policy and fallback helpers."""

from __future__ import annotations

import argparse
import logging
import typing as T
from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter

import torch

from lib.utils import (
    ValidBackends,
    deprecation_warning,
    get_backend,
    get_module_objects,
)

logger = logging.getLogger(__name__)

CompileMode = T.Literal[
    "off",
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]

COMPILE_MODES: tuple[CompileMode, ...] = (
    "off",
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)
_COMPILE_ALIASES: dict[str, CompileMode] = {
    "true": "default",
    "false": "off",
}


@dataclass(frozen=True)
class CompilePolicy:
    """The backend-safe compile mode selected for an extract run."""

    requested: CompileMode
    effective: CompileMode
    backend: ValidBackends
    reason: str = ""
    experimental: bool = False
    dynamic: bool = False

    @property
    def enabled(self) -> bool:
        """``True`` when model compilation should run for this backend and mode."""
        return self.effective != "off"

    def __bool__(self) -> bool:
        return self.enabled

    def compile_kwargs(self, *, fullgraph: bool) -> dict[str, bool | str]:
        """Build the kwargs for a single compile attempt."""
        retval: dict[str, bool | str] = {
            "fullgraph": fullgraph,
            "dynamic": self.dynamic,
        }
        if self.effective != "default":
            retval["mode"] = self.effective
        return retval


@dataclass(frozen=True)
class CompileResult:
    """The outcome of compiling a single module."""

    compiled: bool
    fallback_status: str
    fullgraph: bool | None
    compile_time_ms: float
    final_execution_mode: str
    error_summary: str = ""


@dataclass(frozen=True)
class CompileBackendSummary:
    """Aggregate compile outcomes for a single backend during the current process."""

    backend: ValidBackends
    requested: CompileMode
    effective: CompileMode
    experimental: bool
    attempts: int
    compiled: int
    fallback_statuses: tuple[str, ...]
    final_execution_modes: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass
class _CompileBackendState:
    """Mutable compile tracking used to build :class:`CompileBackendSummary` snapshots."""

    requested: CompileMode
    effective: CompileMode
    experimental: bool
    attempts: int = 0
    compiled: int = 0
    fallback_statuses: set[str] = field(default_factory=set)
    final_execution_modes: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)


_COMPILE_SUMMARY_LOCK = Lock()
_COMPILE_SUMMARIES: dict[ValidBackends, _CompileBackendState] = {}


def compile_mode_choices() -> tuple[CompileMode, ...]:
    """Return the canonical compile mode choices exposed to the CLI."""
    return COMPILE_MODES


def parse_compile_mode(value: str) -> CompileMode:
    """Parse CLI input into a canonical compile mode."""
    normalized = value.strip().lower()
    if normalized in _COMPILE_ALIASES:
        return _COMPILE_ALIASES[normalized]
    if normalized in COMPILE_MODES:
        return T.cast(CompileMode, normalized)
    modes = ", ".join(COMPILE_MODES)
    raise argparse.ArgumentTypeError(
        f"Invalid compile mode '{value}'. Choose from: {modes}"
    )


def resolve_compile_policy(
    value: CompilePolicy | CompileMode | str | bool | None,
    *,
    backend: ValidBackends | None = None,
    warn_on_legacy: bool = False,
) -> CompilePolicy:
    """Resolve a requested compile mode to a backend-safe compile policy."""
    if isinstance(value, CompilePolicy):
        return value

    resolved_backend = get_backend() if backend is None else backend
    requested = _normalize_requested_mode(value, warn_on_legacy=warn_on_legacy)

    if resolved_backend == "cpu":
        reason = (
            "" if requested == "off" else "PyTorch compile is disabled on CPU backends"
        )
        return CompilePolicy(requested, "off", resolved_backend, reason=reason)

    if resolved_backend == "apple_silicon":
        if requested == "off":
            return CompilePolicy(requested, "off", resolved_backend)
        reason = "MPS only supports the default compile mode during extract"
        return CompilePolicy(
            requested, "default", resolved_backend, reason=reason, experimental=True
        )

    if resolved_backend == "rocm":
        if requested in ("off", "default", "reduce-overhead"):
            return CompilePolicy(requested, requested, resolved_backend)
        return CompilePolicy(
            requested,
            "reduce-overhead",
            resolved_backend,
            reason="ROCm compile support is currently limited to default and "
            "reduce-overhead",
        )

    return CompilePolicy(requested, requested, resolved_backend)


def compile_module(module: torch.nn.Module, policy: CompilePolicy) -> CompileResult:
    """Compile a single module with fullgraph retry and eager fallback."""
    start = perf_counter()
    if not policy.enabled:
        result = CompileResult(
            compiled=False,
            fallback_status="disabled",
            fullgraph=None,
            compile_time_ms=_elapsed_ms(start),
            final_execution_mode="eager",
        )
        _record_compile_summary(policy, result)
        return result

    if not hasattr(torch, "compile") or not hasattr(module, "compile"):
        result = CompileResult(
            compiled=False,
            fallback_status="unavailable",
            fullgraph=None,
            compile_time_ms=_elapsed_ms(start),
            final_execution_mode="eager",
            error_summary="torch.compile is unavailable",
        )
        _record_compile_summary(policy, result)
        return result

    last_error = ""
    for fullgraph in (True, False):
        try:
            module.compile(**policy.compile_kwargs(fullgraph=fullgraph))
            result = CompileResult(
                compiled=True,
                fallback_status=("none" if fullgraph else "fullgraph_retry"),
                fullgraph=fullgraph,
                compile_time_ms=_elapsed_ms(start),
                final_execution_mode=f"compiled:{policy.effective}",
            )
            _record_compile_summary(policy, result)
            return result
        except Exception as err:  # pylint:disable=broad-except
            last_error = _error_summary(err)
            logger.debug(
                "Torch compile failed for %s with fullgraph=%s: %s",
                module.__class__.__name__,
                fullgraph,
                last_error,
            )

    result = CompileResult(
        compiled=False,
        fallback_status="eager_fallback",
        fullgraph=None,
        compile_time_ms=_elapsed_ms(start),
        final_execution_mode="eager",
        error_summary=last_error,
    )
    _record_compile_summary(policy, result)
    return result


def is_compiled_module(module: torch.nn.Module) -> bool:
    """Return whether ``module`` currently has a compiled call implementation."""
    return getattr(module, "_compiled_call_impl", None) is not None


def reset_compiled_module(module: torch.nn.Module) -> None:
    """Restore ``module`` to eager execution if it was previously compiled."""
    if hasattr(module, "_compiled_call_impl"):
        module._compiled_call_impl = None  # type:ignore[attr-defined]


def record_compile_outcome(policy: CompilePolicy, result: CompileResult) -> None:
    """Record a compile-related outcome outside the wrapper call itself."""
    _record_compile_summary(policy, result)


def get_compile_backend_summary(backend: ValidBackends) -> CompileBackendSummary | None:
    """Return a snapshot of compile outcomes for *backend* if any exist."""
    with _COMPILE_SUMMARY_LOCK:
        state = _COMPILE_SUMMARIES.get(backend)
        if state is None:
            return None
        return CompileBackendSummary(
            backend=backend,
            requested=state.requested,
            effective=state.effective,
            experimental=state.experimental,
            attempts=state.attempts,
            compiled=state.compiled,
            fallback_statuses=tuple(sorted(state.fallback_statuses)),
            final_execution_modes=tuple(sorted(state.final_execution_modes)),
            errors=tuple(state.errors),
        )


def reset_compile_backend_summaries() -> None:
    """Clear any recorded compile summaries."""
    with _COMPILE_SUMMARY_LOCK:
        _COMPILE_SUMMARIES.clear()


def _normalize_requested_mode(
    value: CompileMode | str | bool | None, *, warn_on_legacy: bool
) -> CompileMode:
    if isinstance(value, bool):
        if warn_on_legacy:
            deprecation_warning(
                "Boolean compile mode",
                "Use '--compile default' or '--compile off' instead.",
            )
        return "default" if value else "off"

    if value is None:
        return "off"

    if isinstance(value, str):
        normalized = parse_compile_mode(value)
        if warn_on_legacy and value.strip().lower() in _COMPILE_ALIASES:
            deprecation_warning(
                "Boolean compile mode",
                "Use '--compile default' or '--compile off' instead.",
            )
        return normalized

    return T.cast(CompileMode, value)


def _elapsed_ms(start: float) -> float:
    return (perf_counter() - start) * 1000.0


def _error_summary(err: Exception) -> str:
    return " ".join(str(err).strip().split())


def _record_compile_summary(policy: CompilePolicy, result: CompileResult) -> None:
    """Track compile outcomes so profile reports can expose backend fallback status."""
    with _COMPILE_SUMMARY_LOCK:
        state = _COMPILE_SUMMARIES.setdefault(
            policy.backend,
            _CompileBackendState(
                requested=policy.requested,
                effective=policy.effective,
                experimental=policy.experimental,
            ),
        )
        state.requested = policy.requested
        state.effective = policy.effective
        state.experimental = policy.experimental
        state.attempts += 1
        state.compiled += int(result.compiled)
        state.fallback_statuses.add(result.fallback_status)
        state.final_execution_modes.add(result.final_execution_mode)
        if result.error_summary:
            state.errors.append(result.error_summary)


__all__ = get_module_objects(__name__)
