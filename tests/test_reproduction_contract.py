"""Integrity checks for the frozen Luo et al. 2022 R0 contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


CONTRACT_PATH = Path(__file__).resolve().parents[1] / "configs" / "luo2022_r0.json"


def load_contract() -> dict:
    with CONTRACT_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_luo2022_r0_contract_is_frozen_and_runtime_bound() -> None:
    contract = load_contract()

    assert contract["profile_id"] == "luo2022_thz_r0"
    assert contract["freeze_version"] == "2026-07-19.3"
    assert contract["status"]["contract"] == "frozen"
    assert contract["status"]["runtime_binding"] == "implemented"
    assert contract["status"]["validation"] == "small_run_passed_under_2026-07-19.3"
    assert contract["status"]["readiness"] == "ready_for_cuda_retraining_2026-07-19"
    assert contract["status"]["small_run_artifact"] == (
        "outputs/luo2022_r0_small/manifest.json"
    )
    assert contract["status"]["full_scale_diffuser_artifact"] == (
        "outputs/luo2022_r0_assessment/assessment.json"
    )
    assert contract["experiment_class"] == "E4"
    assert contract["comparison_level"] == "R0"
    assert "main_zotero_attachment_key" not in contract["paper"]
    assert "supplement_zotero_attachment_key" not in contract["paper"]


def test_luo2022_r0_paper_confirmed_core_values_do_not_drift() -> None:
    contract = load_contract()

    assert contract["illumination"]["wavelength_m"] == pytest.approx(0.00075)
    assert contract["grid"]["shape"] == [240, 240]
    assert contract["grid"]["pixel_pitch_m"] == pytest.approx(0.0003)
    assert contract["geometry"] == {
        "object_to_diffuser_m": 0.04,
        "diffuser_to_first_layer_m": 0.002,
        "layer_to_layer_m": 0.002,
        "last_layer_to_output_m": 0.007,
        "evidence": "paper_confirmed",
    }
    assert contract["d2nn"]["layers"] == 4
    assert contract["training"]["objects_per_batch"] == 4
    assert contract["training"]["diffusers_per_epoch"] == 20
    assert contract["training"]["epochs"] == 100
    assert contract["evaluation"]["diffuser_sets_for_n20"]["all_training_diffusers"] == 2000
    assert contract["evaluation"]["diffuser_sets_for_n20"]["new_diffusers"] == 20
    assert contract["diffuser"]["uniqueness"] == {
        "metric": "mean_pixelwise_absolute_difference_of_mean_centered_phase",
        "minimum_radians": pytest.approx(3.141592653589793 / 2),
        "phase_representation": "minus_pi_to_pi",
        "comparison_scope": (
            "all_previously_accepted_training_diffusers_and_current_epoch_candidates"
        ),
        "metric_and_threshold_evidence": "paper_confirmed",
        "phase_representation_evidence": "paper_inferred_project_choice",
        "comparison_scope_evidence": "paper_inferred",
    }


def test_luo2022_r0_inferred_learning_rate_matches_paper_end_value() -> None:
    learning_rate = load_contract()["training"]["learning_rate"]

    expected_epoch_100 = learning_rate["initial"] * learning_rate["gamma"] ** 99
    assert learning_rate["update_interval"] == "epoch"
    assert learning_rate["epoch_100_approx"] == pytest.approx(expected_epoch_100)
    assert 3e-4 < expected_epoch_100 < 4e-4


def test_luo2022_r0_unpublished_choices_are_explicit() -> None:
    contract = load_contract()

    assert contract["d2nn"]["phase_initialization"]["evidence"] == "project_choice"
    assert contract["diffuser"]["finite_kernel_choice"]["evidence"] == "project_choice"
    assert contract["diffuser"]["correlation_estimator"]["field"] == (
        "mean_centered_complex_transmittance"
    )
    assert contract["diffuser"]["correlation_estimator"]["field_evidence"] == (
        "paper_inferred_project_choice"
    )
    assert contract["propagation"]["reference_backend"]["evidence"] == "project_choice"
    assert contract["training"]["primary_seed"]["evidence"] == "project_choice"
    assert contract["training"]["diffuser_seed_schedule"] == {
        "epoch_base_seed_formula": "primary_seed_plus_epoch_times_stride",
        "epoch_stride": 100_000,
        "evidence": "project_choice",
    }
    assert contract["evaluation"]["diffuser_seed_schedule"] == {
        "base_seed_formula": "primary_seed_plus_offset",
        "offset": 1_000_000_000,
        "must_be_disjoint_from_all_training_epoch_base_seeds": True,
        "evidence": "project_choice",
    }
    assert all(item["reason"] for item in contract["deferred"])
    freeze_change = contract["change_control"]["current_freeze_change"]
    assert freeze_change["from"] == "2026-07-17.2"
    assert freeze_change["to"] == "2026-07-19.3"
    assert freeze_change["acceptance"]["seed_namespace_overlap"] is False
