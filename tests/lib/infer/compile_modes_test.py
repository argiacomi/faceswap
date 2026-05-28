#!/usr/bin/env python3
"""Tests for backend-aware extract compile mode policy."""

from __future__ import annotations

import argparse
import logging
from typing import Any
from unittest.mock import patch

import torch

from lib.cli.args_extract_convert import ExtractArgs
from lib.infer.compile_modes import (
    CompilePolicy,
    compile_module,
    get_compile_backend_summary,
    reset_compile_backend_summaries,
    resolve_compile_policy,
)
from lib.infer.plugin_utils import _COMPILE_LOGGED, compile_models


class _CompileModule(torch.nn.Module):
    """A torch module that records compile attempts and returns programmed results."""

    def __init__(self, responses: list[Exception | None]) -> None:
        super().__init__()
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def forward(self, *args, **kwargs):  # pylint:disable=unused-argument
        raise NotImplementedError

    def compile(self, **kwargs) -> None:  # type:ignore[override]
        self.calls.append(kwargs)
        self._compiled_call_impl = object()
        response = self._responses.pop(0)
        if response is not None:
            raise response


class _Plugin:
    """Minimal extract plugin stub for compile logging tests."""

    name = "stub-plugin"
    batch_size = 4


class _OrformerPlugin(_Plugin):
    name = "ORFormer"


class _SpigaPlugin(_Plugin):
    name = "SPIGA"


def _compile_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    option = next(
        opt
        for opt in ExtractArgs.get_optional_arguments()
        if "--compile" in opt["opts"]
    )
    kwargs = {key: val for key, val in option.items() if key not in ("opts", "group")}
    parser.add_argument(*option["opts"], **kwargs)
    return parser


def test_compile_flag_maps_to_default() -> None:
    """`--compile` should retain legacy boolean behavior and select default mode."""
    args = _compile_parser().parse_args(["--compile"])
    assert args.compile == "default"


def test_compile_true_maps_to_default() -> None:
    """`--compile true` should map to the default compile mode."""
    args = _compile_parser().parse_args(["--compile", "true"])
    assert args.compile == "default"


def test_compile_false_maps_to_off() -> None:
    """`--compile false` should disable compilation."""
    args = _compile_parser().parse_args(["--compile", "false"])
    assert args.compile == "off"


def test_bool_style_compile_logs_deprecation(caplog) -> None:
    """Legacy bool compile settings should still map cleanly with a deprecation warning."""
    with caplog.at_level(logging.WARNING):
        policy = resolve_compile_policy(True, backend="nvidia", warn_on_legacy=True)
    assert policy.requested == "default"
    assert "deprecated" in caplog.text.lower()


def test_cpu_defaults_to_off() -> None:
    """CPU backends must always resolve to eager mode."""
    policy = resolve_compile_policy("default", backend="cpu")
    assert policy.requested == "default"
    assert policy.effective == "off"
    assert not policy


def test_mps_rejects_cuda_graph_modes() -> None:
    """MPS should never retain a CUDA-graph compile mode."""
    policy = resolve_compile_policy("max-autotune", backend="apple_silicon")
    assert policy.requested == "max-autotune"
    assert policy.effective == "default"
    assert policy.experimental is True


def test_cuda_accepts_max_autotune() -> None:
    """CUDA backends should accept the highest autotune mode unchanged."""
    policy = resolve_compile_policy("max-autotune", backend="nvidia")
    assert policy.effective == "max-autotune"


def test_fullgraph_failure_retries_without_fullgraph() -> None:
    """A fullgraph compile error should retry with fullgraph disabled."""
    module = _CompileModule([RuntimeError("fullgraph failed"), None])
    policy = resolve_compile_policy("default", backend="nvidia")

    result = compile_module(module, policy)

    assert result.compiled is True
    assert result.fallback_status == "fullgraph_retry"
    assert result.fullgraph is False
    assert module.calls == [
        {"fullgraph": True, "dynamic": False},
        {"fullgraph": False, "dynamic": False},
    ]


def test_compile_failure_falls_back_to_eager() -> None:
    """A second compile failure should fall back to eager execution."""
    module = _CompileModule(
        [RuntimeError("fullgraph failed"), RuntimeError("retry failed")]
    )
    policy = resolve_compile_policy("default", backend="nvidia")

    result = compile_module(module, policy)

    assert result.compiled is False
    assert result.fallback_status == "eager_fallback"
    assert result.final_execution_mode == "eager"
    assert result.error_summary == "retry failed"


def test_logs_include_backend_mode_and_fallback(caplog) -> None:
    """Compile logs should capture the selected backend, mode, and fallback outcome."""
    _COMPILE_LOGGED.clear()
    module = _CompileModule([RuntimeError("fullgraph failed"), None])
    policy = CompilePolicy("default", "default", "nvidia")

    with (
        patch("lib.infer.plugin_utils.warmup_plugin", return_value=False),
        caplog.at_level(logging.INFO),
    ):
        compile_models(_Plugin(), [module], policy)

    assert "backend=nvidia" in caplog.text
    assert "mode=default" in caplog.text
    assert "fallback=fullgraph_retry" in caplog.text
    assert "Compile start" in caplog.text
    assert "Channel-order warmup complete" in caplog.text
    assert "First compiled execution warmup complete" in caplog.text
    assert "Compile ready" in caplog.text


def test_compile_models_dedupes_duplicate_module_references() -> None:
    """The same module object should not be wrapped more than once."""
    _COMPILE_LOGGED.clear()
    module = _CompileModule([None])
    policy = CompilePolicy("default", "default", "nvidia")

    with patch("lib.infer.plugin_utils.warmup_plugin", return_value=False):
        compile_models(_Plugin(), [module, module], policy)

    assert len(module.calls) == 1


def test_compile_models_falls_back_to_eager_when_compiled_warmup_fails(caplog) -> None:
    """A failed first compiled execution should revert the module to eager mode."""
    _COMPILE_LOGGED.clear()
    module = _CompileModule([None])
    policy = CompilePolicy("default", "default", "apple_silicon", experimental=True)

    with (
        patch("lib.infer.plugin_utils.warmup_plugin", side_effect=[False, None, False]),
        caplog.at_level(logging.INFO),
    ):
        compile_models(_Plugin(), [module], policy)

    assert getattr(module, "_compiled_call_impl", None) is None
    assert "First compiled execution warmup failed" in caplog.text
    assert "Compile fallback complete" in caplog.text


def test_compile_models_logs_failed_eager_fallback_when_rewarmup_fails(caplog) -> None:
    """A failed eager re-warmup must be recorded as a failed fallback."""
    _COMPILE_LOGGED.clear()
    reset_compile_backend_summaries()
    module = _CompileModule([None])
    policy = CompilePolicy("default", "default", "apple_silicon", experimental=True)

    with (
        patch("lib.infer.plugin_utils.warmup_plugin", side_effect=[False, None, None]),
        caplog.at_level(logging.INFO),
    ):
        compile_models(_Plugin(), [module], policy)

    assert getattr(module, "_compiled_call_impl", None) is None
    assert "First compiled execution warmup failed" in caplog.text
    assert "Compile fallback failed: eager warmup also failed" in caplog.text
    assert "Compile fallback complete" not in caplog.text
    summary = get_compile_backend_summary("apple_silicon")
    assert summary is not None
    assert "runtime_eager_fallback_failed" in summary.fallback_statuses
    assert "failed" in summary.final_execution_modes
    assert "Compiled execution failed and eager fallback warmup also failed" in summary.errors


def test_compile_models_skips_orformer_on_mps(caplog) -> None:
    """ORFormer should bypass torch.compile on Apple Silicon until MPS codegen is stable."""
    _COMPILE_LOGGED.clear()
    module = _CompileModule([None])
    policy = CompilePolicy("default", "default", "apple_silicon", experimental=True)

    with (
        patch("lib.infer.plugin_utils.warmup_plugin", return_value=False),
        caplog.at_level(logging.INFO),
    ):
        compile_models(_OrformerPlugin(), [module], policy)

    assert module.calls == []
    assert "Skipping torch.compile on Apple Silicon for ORFormer" in caplog.text


def test_compile_models_skips_spiga_on_mps(caplog) -> None:
    """SPIGA should bypass torch.compile on Apple Silicon until MPS shape guards are stable."""
    _COMPILE_LOGGED.clear()
    module = _CompileModule([None])
    policy = CompilePolicy("default", "default", "apple_silicon", experimental=True)

    with (
        patch("lib.infer.plugin_utils.warmup_plugin", return_value=True),
        caplog.at_level(logging.INFO),
    ):
        compile_models(_SpigaPlugin(), [module], policy)

    assert module.calls == []
    assert "Skipping torch.compile on Apple Silicon for SPIGA" in caplog.text
