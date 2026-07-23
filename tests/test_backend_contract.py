"""Contract checks for the fixed-four-layer digital-backend ablation."""

from __future__ import annotations

import json
from pathlib import Path


CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "luo2022_fixed4_backend.json"
)


def load_contract() -> dict:
    with CONTRACT_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_backend_contract_identity_and_frozen_r0_reference() -> None:
    contract = load_contract()

    assert contract["schema_version"] == 1
    assert contract["profile_id"] == "luo2022_fixed4_backend"
    assert contract["status"] == "exploratory fixed-depth backend ablation"
    assert contract["comparison_scope"] == "B0_R1_R2"
    assert contract["r0_reference"] == {
        "profile_id": "luo2022_thz_r0",
        "freeze_version": "2026-07-19.3",
        "public_tag": "luo2022-r0-v3",
        "optical_layers": 4,
        "frozen": True,
        "rule": (
            "The R0 optical model, geometry, diffuser protocol, phase parameters, "
            "and sealed metrics remain unchanged."
        ),
    }


def test_backend_contract_has_comparable_variant_schedules() -> None:
    contract = load_contract()
    variants = contract["variants"]

    assert set(variants) == {"B0", "R1", "R2"}
    assert variants["B0"]["definition"] == "direct_no_d2nn_plus_supervised_unet"
    assert variants["B0"]["epochs"] == {
        "supervised": 30,
        "adversarial": 0,
        "total": 30,
    }
    assert variants["R1"]["epochs"] == {
        "shared_supervised_warmup": 20,
        "supervised_continuation": 10,
        "adversarial": 0,
        "total": 30,
    }
    assert variants["R2"]["epochs"] == {
        "shared_supervised_warmup": 20,
        "adversarial_continuation": 10,
        "total": 30,
    }
    assert variants["B0"]["epochs"]["total"] == (
        variants["R1"]["epochs"]["shared_supervised_warmup"]
        + variants["R1"]["epochs"]["supervised_continuation"]
    )
    assert variants["B0"]["epochs"]["total"] == (
        variants["R2"]["epochs"]["shared_supervised_warmup"]
        + variants["R2"]["epochs"]["adversarial_continuation"]
    )
    assert "identical epoch-20" in variants["R2"]["branch_rule"]


def test_backend_contract_freezes_cache_scaling_and_assignment() -> None:
    contract = load_contract()

    assert contract["assignment"]["evidence"] == "project_choice"
    assert contract["assignment"]["schema"] == "luo2022_fixed4_assignment_v1"
    assert contract["assignment"]["train_object_ids"] == [0, 49_999]
    assert contract["assignment"]["validation_object_ids"] == [50_000, 59_999]
    assert contract["assignment"]["training_diffusers"] == 2_000
    assert contract["assignment"]["unit"] == "object_diffuser_pair"
    assert contract["cache"]["quantity"] == "raw_detector_intensity"
    assert contract["cache"]["dtype"] == "float32"
    assert contract["cache"]["normalization_at_write"] == "none"

    scaling = contract["input_scaling"]
    assert scaling["method"] == "per_operator_global_dataset_max"
    assert scaling["fit_split"] == "train_only"
    assert scaling["fit_separately_per_operator"] is True
    assert scaling["apply_frozen_training_statistic_to_validation_and_evaluation"] is True
    assert scaling["per_image_normalization"] is False
    assert scaling["clipping"] is False


def test_backend_contract_freezes_model_optimization_and_losses() -> None:
    contract = load_contract()

    assert contract["model"]["base_channels"] == 4
    assert contract["training"]["batch_size"] == 32
    assert contract["training"]["seed"] == 0
    assert contract["training"]["generator_optimizer"] == {
        "name": "Adam",
        "learning_rate": 0.002,
        "betas": [0.9, 0.999],
    }
    assert contract["training"]["discriminator_optimizer"] == {
        "name": "Adam",
        "learning_rate": 0.0002,
        "betas": [0.5, 0.999],
    }
    assert contract["training"]["loss"] == {
        "reconstruction": "L1",
        "reconstruction_weight": 1.0,
        "adversarial_weight": 0.01,
        "B0_adversarial_weight": 0.0,
        "R1_adversarial_weight": 0.0,
        "R2_adversarial_weight": 0.01,
    }


def test_backend_contract_freezes_evaluation_and_claim_boundary() -> None:
    contract = load_contract()
    evaluation = contract["evaluation"]

    assert evaluation["diffuser_conditions"] == [
        "final_epoch_known",
        "seed_disjoint_unseen",
        "no_diffuser",
    ]
    assert evaluation["report_separately_by_condition"] is True
    assert evaluation["fixed_example_object_ids"] == [3, 2, 1, 18, 4, 8, 11, 0, 61, 7]
    assert evaluation["fixed_example_labels"] == list(range(10))
    assert "full_canvas_pearson" in evaluation["metrics"]
    assert "worst_5_percent_target_support_pearson" in evaluation["metrics"]
    assert contract["claim_boundary"]["label"] == "exploratory fixed-depth backend ablation"
    assert "depth efficiency" in contract["claim_boundary"]["excluded"]


def test_backend_contract_contains_only_portable_public_metadata() -> None:
    contract = load_contract()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                assert not key.endswith("_hash")
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str):
            assert not value.startswith("/")

    visit(contract)
