#!/usr/bin/env python3
"""Pure unit tests for the deterministic training phase scheduler."""

from __future__ import annotations

import pytest

from lib.training.phase_scheduler import (
    PhasePreset,
    ScheduledMultipliers,
    TrainingPhaseScheduler,
)


def test_off_mode_is_behavior_noop() -> None:
    state = TrainingPhaseScheduler.for_mode("off").at(75_000)

    assert state.phase == "off"
    assert state.apply is False
    assert state.dry_run is False
    assert state.scheduled_multipliers == ScheduledMultipliers.identity()
    assert state.effective_multipliers == ScheduledMultipliers.identity()
    assert state.transition_reason == "automation disabled"


def test_broad_phase_suppresses_optional_detail_multipliers() -> None:
    state = TrainingPhaseScheduler.for_mode("balanced").at(10_000)

    assert state.phase == "broad_reconstruction"
    assert state.apply is True
    assert state.effective_multipliers.primary_loss == 1.0
    assert state.effective_multipliers.secondary_loss == 0.0
    assert state.effective_multipliers.boundary_loss == 0.0
    assert state.effective_multipliers.region_perceptual_loss == 0.0
    assert state.effective_multipliers.identity_loss == 0.0


def test_detail_phase_ramps_linearly() -> None:
    scheduler = TrainingPhaseScheduler.for_mode("balanced")

    halfway = scheduler.at(30_000)
    complete = scheduler.at(40_000)

    assert halfway.phase == "detail_ramp"
    assert halfway.effective_multipliers.secondary_loss == pytest.approx(0.5)
    assert halfway.effective_multipliers.boundary_loss == pytest.approx(0.5)
    assert halfway.effective_multipliers.region_perceptual_loss == pytest.approx(0.5)
    assert halfway.effective_multipliers.identity_loss == 0.0

    assert complete.phase == "detail_ramp"
    assert complete.effective_multipliers.secondary_loss == pytest.approx(1.0)
    assert complete.effective_multipliers.boundary_loss == pytest.approx(1.0)
    assert complete.effective_multipliers.region_perceptual_loss == pytest.approx(1.0)


def test_final_phase_ramps_identity_loss() -> None:
    scheduler = TrainingPhaseScheduler.for_mode("balanced")

    halfway = scheduler.at(90_000)
    complete = scheduler.at(100_000)

    assert halfway.phase == "final_refinement"
    assert halfway.effective_multipliers.secondary_loss == pytest.approx(1.0)
    assert halfway.effective_multipliers.identity_loss == pytest.approx(0.5)

    assert complete.phase == "final_refinement"
    assert complete.effective_multipliers.identity_loss == pytest.approx(1.0)


def test_dry_run_computes_schedule_but_does_not_apply() -> None:
    dry_run = TrainingPhaseScheduler.for_mode("dry_run").at(90_000)
    balanced = TrainingPhaseScheduler.for_mode("balanced").at(90_000)

    assert dry_run.dry_run is True
    assert dry_run.apply is False
    assert dry_run.phase == balanced.phase
    assert dry_run.scheduled_multipliers == balanced.scheduled_multipliers
    assert dry_run.effective_multipliers == ScheduledMultipliers.identity()


def test_dry_run_flag_previews_selected_preset_without_applying() -> None:
    """The dry-run flag previews the selected preset (not balanced) and never applies."""
    applied = TrainingPhaseScheduler.for_mode("aggressive").at(20_000)
    preview = TrainingPhaseScheduler.for_mode("aggressive", dry_run=True).at(20_000)

    assert preview.dry_run is True
    assert preview.apply is False
    # Same selected-preset schedule as the applied aggressive mode, unlike the "dry_run" alias
    # which previews the balanced preset.
    assert preview.phase == applied.phase
    assert preview.scheduled_multipliers == applied.scheduled_multipliers
    assert preview.effective_multipliers == ScheduledMultipliers.identity()


def test_tensorboard_scalars_include_phase_index() -> None:
    """Phase transitions are exposed as a stable numeric scalar for TensorBoard charts."""
    broad = TrainingPhaseScheduler.for_mode("balanced").at(10_000)
    final = TrainingPhaseScheduler.for_mode("balanced").at(90_000)

    assert broad.tensorboard_scalars()["phase_scheduler/phase"] == pytest.approx(1.0)
    assert final.tensorboard_scalars()["phase_scheduler/phase"] == pytest.approx(3.0)


def test_modes_have_different_timing_and_strength() -> None:
    aggressive = TrainingPhaseScheduler.for_mode("aggressive").at(20_000)
    conservative = TrainingPhaseScheduler.for_mode("conservative").at(20_000)

    assert aggressive.phase == "detail_ramp"
    assert aggressive.effective_multipliers.secondary_loss == pytest.approx(1.0)

    assert conservative.phase == "broad_reconstruction"
    assert conservative.effective_multipliers.secondary_loss == 0.0


def test_conservative_mode_caps_detail_and_identity_targets() -> None:
    scheduler = TrainingPhaseScheduler.for_mode("conservative")

    detail_complete = scheduler.at(80_000)
    final_complete = scheduler.at(200_000)

    assert detail_complete.effective_multipliers.secondary_loss == pytest.approx(0.75)
    assert detail_complete.effective_multipliers.identity_loss == 0.0

    assert final_complete.effective_multipliers.secondary_loss == pytest.approx(0.75)
    assert final_complete.effective_multipliers.identity_loss == pytest.approx(0.50)


def test_scheduler_is_resume_deterministic() -> None:
    before_restart = TrainingPhaseScheduler.for_mode("balanced").at(45_000)
    after_restart = TrainingPhaseScheduler.for_mode("balanced").at(45_000)

    assert before_restart == after_restart


def test_custom_preset_supports_zero_length_ramps() -> None:
    preset = PhasePreset(
        mode="balanced",
        detail_start_iteration=10,
        final_start_iteration=20,
        detail_ramp_iterations=0,
        final_ramp_iterations=0,
    )
    scheduler = TrainingPhaseScheduler(mode="balanced", preset=preset)

    detail = scheduler.at(10)
    final = scheduler.at(20)

    assert detail.effective_multipliers.secondary_loss == pytest.approx(1.0)
    assert final.effective_multipliers.identity_loss == pytest.approx(1.0)


def test_preset_rejects_invalid_iterations() -> None:
    with pytest.raises(ValueError, match="detail_start_iteration"):
        PhasePreset(
            mode="balanced",
            detail_start_iteration=-1,
            final_start_iteration=20,
            detail_ramp_iterations=10,
            final_ramp_iterations=10,
        )

    with pytest.raises(ValueError, match="final_start_iteration"):
        PhasePreset(
            mode="balanced",
            detail_start_iteration=20,
            final_start_iteration=10,
            detail_ramp_iterations=10,
            final_ramp_iterations=10,
        )


def test_multipliers_reject_negative_values() -> None:
    with pytest.raises(ValueError, match="identity_loss"):
        ScheduledMultipliers(identity_loss=-0.1)


def test_tensorboard_scalars_are_stable() -> None:
    state = TrainingPhaseScheduler.for_mode("balanced").at(90_000)

    scalars = state.tensorboard_scalars()

    assert scalars["phase_scheduler/primary_loss"] == pytest.approx(1.0)
    assert scalars["phase_scheduler/identity_loss"] == pytest.approx(0.5)
    assert scalars["phase_scheduler/apply"] == pytest.approx(1.0)
    assert scalars["phase_scheduler/dry_run"] == pytest.approx(0.0)
