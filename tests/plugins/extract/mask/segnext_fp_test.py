#!/usr/bin/env python3
"""Unit tests for :mod:`plugins.extract.mask.segnext_fp`."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from lib.utils import FaceswapError
from plugins.extract.mask import segnext_fp_defaults as cfg
from plugins.plugin_loader import PluginLoader


def _import_segnext_module():
    """Import ``segnext_fp`` without executing ``lib.infer.__init__`` during tests."""
    if "lib.infer" not in sys.modules:
        infer_pkg = types.ModuleType("lib.infer")
        infer_pkg.__path__ = [str(Path(__file__).resolve().parents[4] / "lib" / "infer")]
        sys.modules["lib.infer"] = infer_pkg
    return importlib.import_module("plugins.extract.mask.segnext_fp")


def _minimal_checkpoint_state(num_classes: int, channels: int) -> dict[str, torch.Tensor]:
    """Return the minimal classifier tensors needed for checkpoint validation."""
    return {
        "decode_head.conv_seg.weight": torch.zeros((num_classes, channels, 1, 1)),
        "decode_head.conv_seg.bias": torch.zeros((num_classes,)),
    }


def _patch_plugin_config(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str = "small",
    *,
    include_hair: bool = False,
    include_ears: bool = False,
    include_glasses: bool = True,
) -> None:
    """Patch SegNeXt-FP config accessors for constructor-only tests."""
    monkeypatch.setattr("plugins.extract.mask.segnext_fp.cfg.batch_size", lambda: 1)
    monkeypatch.setattr("plugins.extract.mask.segnext_fp.cfg.cpu", lambda: True)
    monkeypatch.setattr("plugins.extract.mask.segnext_fp.cfg.model", lambda: model_name)
    monkeypatch.setattr("plugins.extract.mask.segnext_fp.cfg.include_ears", lambda: include_ears)
    monkeypatch.setattr("plugins.extract.mask.segnext_fp.cfg.include_hair", lambda: include_hair)
    monkeypatch.setattr(
        "plugins.extract.mask.segnext_fp.cfg.include_glasses", lambda: include_glasses
    )


def test_plugin_loader_discovers_segnext_fp() -> None:
    """SegNeXt-FP must be selectable wherever mask extractors are enumerated."""
    assert "segnext-fp" in PluginLoader.get_available_extractors("mask")


def test_plugin_loader_extends_segnext_fp_storage_names() -> None:
    """Face/head mask storage names must be exposed for convert/train mask choices."""
    maskers = PluginLoader.get_available_extractors("mask", extend_plugin=True)

    assert "segnext-fp_face" in maskers
    assert "segnext-fp_head" in maskers


def test_constructor_uses_face_centering_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default SegNeXt-FP masks should match BiSeNet-FP's face-centered storage."""
    segnext_fp = _import_segnext_module()
    _patch_plugin_config(monkeypatch)

    plugin = segnext_fp.SegNeXtFP()

    assert plugin.centering == "face"
    assert plugin.storage_name == "segnext-fp_face"


def test_hair_enabled_mode_uses_head_centering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hair-enabled SegNeXt-FP masks should store under head centering."""
    segnext_fp = _import_segnext_module()
    _patch_plugin_config(monkeypatch, include_hair=True)

    plugin = segnext_fp.SegNeXtFP()

    assert plugin.centering == "head"
    assert plugin.storage_name == "segnext-fp_head"


def test_process_and_post_process_produce_valid_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A SegNeXt-FP logits batch should reduce to a binary float32 face mask."""
    segnext_fp = _import_segnext_module()
    _patch_plugin_config(monkeypatch, include_glasses=False)
    plugin = segnext_fp.SegNeXtFP()
    logits: np.ndarray = np.zeros((1, 19, 4, 4), dtype=np.float32)
    logits[:, 1] = 10.0
    plugin.from_torch = lambda batch: logits

    processed = plugin.process(np.zeros((1, 3, 4, 4), dtype=np.float32))
    output = plugin.post_process(processed)

    assert processed.shape == (1, 4, 4, 19)
    assert output.shape == (1, 4, 4)
    assert output.dtype == np.float32
    assert np.all(output == 1.0)
    assert output.source_id == "segnext-fp_face"
    assert output.per_class_probs is not None
    assert output.per_class_probs.shape == (1, 4, 4, 19)


@pytest.mark.parametrize("model_name", ["small", "base"])
def test_registry_entries_validate_as_19_class_checkpoints(
    tmp_path,
    model_name: str,
) -> None:
    """The published small/base registry entries must both validate as 19-class heads."""
    segnext_fp = _import_segnext_module()
    checkpoint = segnext_fp._CHECKPOINTS[model_name]  # pylint:disable=protected-access
    path = tmp_path / f"{model_name}.pth"
    torch.save(
        {
            "state_dict": _minimal_checkpoint_state(
                checkpoint.num_classes, checkpoint.config.decoder_channels
            )
        },
        path,
    )

    cleaned = segnext_fp._sanitize_checkpoint(
        str(path),  # pylint:disable=protected-access
        "0" * 64,
        expected_num_classes=checkpoint.num_classes,
    )
    state_dict = torch.load(cleaned, map_location="cpu", weights_only=True)

    segnext_fp._validate_checkpoint_classes(
        state_dict,  # pylint:disable=protected-access
        checkpoint.num_classes,
    )


def test_sanitize_checkpoint_rejects_150_class_head(tmp_path) -> None:
    """A mismatched 150-class decode head must fail before load_state_dict is attempted."""
    segnext_fp = _import_segnext_module()
    path = tmp_path / "bad.pth"
    torch.save({"state_dict": _minimal_checkpoint_state(150, 512)}, path)

    with pytest.raises(
        FaceswapError,
        match=(
            "SegNeXt-FP checkpoint class-count mismatch: expected 19 CelebAMask-HQ "
            "classes, got 150\\. The selected model/checkpoint pair is incompatible\\."
        ),
    ):
        segnext_fp._sanitize_checkpoint(
            str(path),  # pylint:disable=protected-access
            "1" * 64,
            expected_num_classes=19,
        )


def test_base_model_uses_aiart_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exposed base option must point at the validated AiArt-Gao 19-class artifact."""
    segnext_fp = _import_segnext_module()
    _patch_plugin_config(monkeypatch, "base")
    plugin = segnext_fp.SegNeXtFP()

    assert plugin._checkpoint.filename == "iter_160000.pth"  # pylint:disable=protected-access
    assert plugin._checkpoint.url == segnext_fp._AIART_BASE_URL  # pylint:disable=protected-access
    assert (
        plugin._checkpoint.sha256
        == "6cfe7fa84f4f931dbed2f47fd04466f4b1d2acfd7fdc4e512e1f22796d3b5853"
    )  # pylint:disable=protected-access,line-too-long
    assert plugin._checkpoint.google_drive_file_id == segnext_fp._AIART_BASE_FILE_ID  # pylint:disable=protected-access


def test_base_checkpoint_path_uses_shared_google_drive_helper(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AiArt-Gao base checkpoints should resolve through the shared Drive helper."""
    segnext_fp = _import_segnext_module()
    checkpoint = segnext_fp._CHECKPOINTS["base"]  # pylint:disable=protected-access
    captured: dict[str, object] = {}
    expected_path = tmp_path / ".fs_cache" / checkpoint.filename

    def _fake_download(
        file_id: str,
        destination: Path,
        *,
        expected_sha256: str | None = None,
        quiet: bool = False,
    ) -> Path:
        captured["file_id"] = file_id
        captured["destination"] = destination
        captured["expected_sha256"] = expected_sha256
        captured["quiet"] = quiet
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"checkpoint")
        return destination

    monkeypatch.setattr(segnext_fp, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(segnext_fp, "download_google_drive_file", _fake_download)

    result = segnext_fp._resolve_checkpoint_path(checkpoint)  # pylint:disable=protected-access

    assert result == str(expected_path)
    assert captured == {
        "file_id": segnext_fp._AIART_BASE_FILE_ID,
        "destination": expected_path,
        "expected_sha256": checkpoint.sha256,
        "quiet": False,
    }


def test_config_choices_only_expose_valid_19_class_checkpoints() -> None:
    """The user-facing model choices must stay aligned with validated 19-class checkpoints."""
    segnext_fp = _import_segnext_module()
    assert list(cfg.model.choices) == ["small", "base"]
    assert set(cfg.model.choices) == set(segnext_fp._CHECKPOINTS)  # pylint:disable=protected-access
    assert all(
        segnext_fp._CHECKPOINTS[name].num_classes == 19  # pylint:disable=protected-access
        for name in cfg.model.choices
    )
