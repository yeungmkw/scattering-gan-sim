import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import experiment
from coherent_data import CoherentD2NNDataset
from experiment import (
    compare_runs,
    evaluate_generator,
    evaluate_reconstructor,
    save_gan_grid,
    save_reconstruction_grid,
    train_gan_one_epoch,
    train_unet_one_epoch,
)
from losses import ReconstructionLossWeights
from patchgan import PatchDiscriminator
from runtime import write_json
from unet import UNetReconstructor


def _coherent_loader(seed: int) -> DataLoader:
    images = torch.rand(4, 1, 16, 16)
    labels = torch.arange(4)
    dataset = CoherentD2NNDataset(TensorDataset(images, labels), corruption="phase", seed=seed)
    return DataLoader(dataset, batch_size=2, shuffle=False)


def test_experiment_cli_help() -> None:
    try:
        experiment.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0

    for command in ("d2nn", "unet", "gan", "compare", "full"):
        try:
            experiment.main([command, "--help"])
        except SystemExit as exc:
            assert exc.code == 0


def test_training_cli_accepts_diffuser_ids() -> None:
    args = experiment.build_parser().parse_args(
        [
            "unet",
            "--train-diffuser-ids",
            "0",
            "1",
            "--eval-diffuser-ids",
            "2",
            "3",
        ]
    )

    assert args.train_diffuser_ids == [0, 1]
    assert args.eval_diffuser_ids == [2, 3]


def test_d2nn_cli_exposes_isolated_luo2022_profile() -> None:
    args = experiment.build_parser().parse_args(
        [
            "d2nn",
            "--profile",
            "luo2022_r0",
            "--action",
            "train",
            "--small-run",
            "--grid-size",
            "48",
            "--diffuser-chunk-size",
            "1",
            "--review-eval-batches",
            "3",
            "--resume",
        ]
    )

    assert args.profile == "luo2022_r0"
    assert args.action == "train"
    assert args.small_run is True
    assert args.grid_size == 48
    assert args.diffuser_chunk_size == 1
    assert args.review_eval_batches == 3
    assert args.resume is True


def test_d2nn_cli_exposes_luo2022_readiness_assessment() -> None:
    args = experiment.build_parser().parse_args(
        [
            "d2nn",
            "--profile",
            "luo2022_r0",
            "--action",
            "assess",
            "--output-dir",
            "outputs/assessment",
        ]
    )

    assert args.profile == "luo2022_r0"
    assert args.action == "assess"
    assert args.output_dir == "outputs/assessment"


def test_d2nn_cli_exposes_luo2022_scatter_correlation_audit() -> None:
    args = experiment.build_parser().parse_args(
        [
            "d2nn",
            "--profile",
            "luo2022_r0",
            "--action",
            "scatter-audit",
            "--scatter-audit-output-dir",
            "outputs/independent_scatter_evidence",
        ]
    )

    assert args.action == "scatter-audit"
    assert args.scatter_audit_output_dir == Path("outputs/independent_scatter_evidence")


def test_d2nn_cli_exposes_luo2022_posthoc_evaluation() -> None:
    args = experiment.build_parser().parse_args(
        [
            "d2nn",
            "--profile",
            "luo2022_r0",
            "--action",
            "evaluate",
            "--output-dir",
            "outputs/frozen_run",
            "--posthoc-output-dir",
            "outputs/evidence",
            "--posthoc-populations",
            "new",
            "no_diffuser",
            "--posthoc-training-epochs",
            "100",
            "--posthoc-roi-metrics",
            "--diffuser-chunk-size",
            "2",
        ]
    )

    assert args.action == "evaluate"
    assert args.output_dir == "outputs/frozen_run"
    assert args.posthoc_output_dir == Path("outputs/evidence")
    assert args.posthoc_populations == ["new", "no_diffuser"]
    assert args.posthoc_training_epochs == [100]
    assert args.posthoc_roi_metrics is True
    assert args.diffuser_chunk_size == 2


def test_d2nn_cli_exposes_luo2022_optical_control_ladder() -> None:
    args = experiment.build_parser().parse_args(
        [
            "d2nn",
            "--profile",
            "luo2022_r0",
            "--action",
            "control-ladder",
            "--run-dir",
            "outputs/frozen_run",
            "--control-output-dir",
            "outputs/independent_control_evidence",
            "--control-populations",
            "training",
            "new",
            "no_diffuser",
            "--control-training-epochs",
            "2",
            "--diffuser-chunk-size",
            "2",
        ]
    )

    assert args.action == "control-ladder"
    assert args.run_dir == Path("outputs/frozen_run")
    assert args.control_output_dir == Path("outputs/independent_control_evidence")
    assert args.control_populations == ["training", "new", "no_diffuser"]
    assert args.control_training_epochs == [2]
    assert args.diffuser_chunk_size == 2


def test_d2nn_cli_exposes_read_only_luo2022_diagnosis() -> None:
    args = experiment.build_parser().parse_args(
        [
            "d2nn",
            "--profile",
            "luo2022_r0",
            "--action",
            "diagnose",
            "--run-dir",
            "outputs/frozen_run",
            "--diagnostic-output-dir",
            "outputs/independent_evidence",
            "--diagnostic-batches",
            "2",
            "--diagnostic-diffusers",
            "4",
            "--diagnostic-pad-factors",
            "2",
            "4",
            "--diagnostic-cross-bank-audit",
        ]
    )

    assert args.action == "diagnose"
    assert args.run_dir == Path("outputs/frozen_run")
    assert args.diagnostic_output_dir == Path("outputs/independent_evidence")
    assert args.diagnostic_batches == 2
    assert args.diagnostic_diffusers == 4
    assert args.diagnostic_pad_factors == [2, 4]
    assert args.diagnostic_cross_bank_audit is True


def test_luo2022_diagnosis_rejects_output_inside_frozen_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "frozen"

    with pytest.raises(ValueError, match="outside the frozen run"):
        experiment.run_luo2022_diagnosis(
            run_dir=run_dir,
            diagnostic_output_dir=run_dir / "diagnosis",
        )


def test_luo2022_scatter_correlation_audit_writes_reproducible_read_only_json(
    tmp_path: Path,
) -> None:
    first = experiment.run_luo2022_scatter_correlation_convention_audit(
        output_dir=tmp_path / "first",
        sample_count=4,
        seed=112,
    )
    second = experiment.run_luo2022_scatter_correlation_convention_audit(
        output_dir=tmp_path / "second",
        sample_count=4,
        seed=112,
    )

    assert first["read_only"] is True
    assert first["generation"]["reduced_test_audit"] is True
    assert first["paper_constraints"]["published_gaussian_sigma_lambda"] == pytest.approx(4.0)
    assert first["paper_constraints"]["configured_gaussian_sigma_lambda"] == pytest.approx(4.0)
    assert [item["id"] for item in first["conventions"]] == [
        "unwrapped_phase_frozen_fit",
        "zero_to_2pi_phase_frozen_fit",
        "minus_pi_to_pi_phase_frozen_fit",
        "minus_pi_to_pi_phase_low_correlation_fit",
        "complex_transmittance_frozen_fit",
    ]
    assert list((tmp_path / "first").iterdir()) == [
        tmp_path / "first" / "scatter_correlation_convention_audit.json"
    ]
    assert [
        item["sample_mean_correlation_length_lambda"] for item in first["conventions"]
    ] == pytest.approx(
        [item["sample_mean_correlation_length_lambda"] for item in second["conventions"]]
    )


def test_luo2022_scatter_correlation_audit_rejects_nonfrozen_contract(
    tmp_path: Path,
) -> None:
    contract = deepcopy(experiment.load_config(experiment.DEFAULT_LUO2022_CONFIG))
    contract["diffuser"]["gaussian_sigma_lambda"] = 3.0
    altered_path = tmp_path / "altered_contract.json"
    write_json(altered_path, contract)

    with pytest.raises(ValueError, match="exact frozen R0 contract"):
        experiment.run_luo2022_scatter_correlation_convention_audit(
            output_dir=tmp_path / "evidence",
            config_path=altered_path,
            sample_count=4,
        )


def test_luo2022_roi_posthoc_rejects_output_inside_frozen_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "frozen"

    with pytest.raises(ValueError, match="outside the frozen run"):
        experiment.run_luo2022_roi_posthoc_evaluation(
            run_dir=run_dir,
            output_dir=run_dir / "roi_evidence",
        )


def test_luo2022_control_ladder_rejects_output_inside_frozen_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "frozen"

    with pytest.raises(ValueError, match="outside the frozen run"):
        experiment.run_luo2022_c0_optical_control_ladder(
            run_dir=run_dir,
            control_output_dir=run_dir / "control_ladder",
        )


def test_luo2022_control_ladder_rejects_truncated_frozen_evaluation(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "completed"
    _contract, runtime_config = _write_completed_luo2022_run_fixture(run_dir)
    runtime_config["runtime"]["max_eval_batches"] = 1
    write_json(run_dir / "config.json", runtime_config)
    checkpoint_path = run_dir / "checkpoints" / "luo2022_d2nn.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["runtime_config"] = runtime_config
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="requires full frozen evaluation"):
        experiment.run_luo2022_c0_optical_control_ladder(
            run_dir=run_dir,
            control_output_dir=tmp_path / "control_ladder",
        )


def test_luo2022_runtime_config_labels_and_controls_small_overrides() -> None:
    contract = experiment.load_config(experiment.DEFAULT_LUO2022_CONFIG)

    config = experiment.build_luo2022_runtime_config(
        contract,
        small_run=True,
        device=torch.device("cpu"),
        epochs=2,
        train_limit=8,
    )

    assert config["status_label"] == "small run"
    assert config["runtime"]["epochs"] == 2
    assert config["runtime"]["train_limit"] == 8
    assert config["runtime"]["diffuser_chunk_size"] == 2
    assert config["execution_controls"]["gradient_accumulation_preserves_fields_per_update"] is True
    assert config["overrides_from_frozen_contract"]["epochs"]["paper_value"] == 100
    assert "not a paper result" in config["claim_boundary"]

    with pytest.raises(ValueError, match="require --small-run"):
        experiment.build_luo2022_runtime_config(
            contract,
            small_run=False,
            device=torch.device("cpu"),
            epochs=2,
        )


def test_luo2022_full_runtime_allows_execution_only_chunking() -> None:
    contract = experiment.load_config(experiment.DEFAULT_LUO2022_CONFIG)

    config = experiment.build_luo2022_runtime_config(
        contract,
        small_run=False,
        device=torch.device("cuda"),
        diffuser_chunk_size=2,
        review_eval_batches=10,
    )

    assert config["runtime"]["diffusers_per_epoch"] == 20
    assert config["runtime"]["diffuser_chunk_size"] == 2
    assert config["runtime"]["review_eval_batches"] == 10
    assert config["overrides_from_frozen_contract"] == {}
    assert config["diffuser_seed_schedule"] == {
        "training_epoch_formula": "primary_seed_plus_epoch_times_stride",
        "training_stride": 100_000,
        "evaluation_formula": "primary_seed_plus_offset",
        "evaluation_offset": 1_000_000_000,
        "evaluation_base_seed": 1_000_000_000,
        "disjoint_for_configured_epochs": True,
    }


def test_luo2022_diffuser_seed_schedule_rejects_training_evaluation_overlap() -> None:
    with pytest.raises(ValueError, match="overlaps"):
        experiment.luo2022_diffuser_seed_schedule(
            seed=0,
            epochs=100,
            training_stride=100_000,
            evaluation_offset=10_000_000,
        )


def test_frozen_diffuser_seed_schedule_requires_or_validates_isolation_evidence() -> None:
    contract = experiment.load_config(experiment.DEFAULT_LUO2022_CONFIG)
    runtime_config = experiment.build_luo2022_runtime_config(
        contract,
        small_run=True,
        device=torch.device("cpu"),
        epochs=2,
        train_limit=8,
    )

    schedule, provenance = experiment._luo2022_frozen_diffuser_seed_schedule(
        runtime_config,
        contract,
    )
    assert provenance == "validated_runtime_copy"
    assert schedule["training_stride"] == 100_000
    assert schedule["evaluation_offset"] == 1_000_000_000

    without_runtime_copy = deepcopy(runtime_config)
    without_runtime_copy.pop("diffuser_seed_schedule")
    derived, derived_provenance = experiment._luo2022_frozen_diffuser_seed_schedule(
        without_runtime_copy,
        contract,
    )
    assert derived_provenance == "derived_from_frozen_source_config"
    assert derived == schedule

    inconsistent_runtime = deepcopy(runtime_config)
    inconsistent_runtime["diffuser_seed_schedule"]["evaluation_base_seed"] = 7
    with pytest.raises(ValueError, match="does not match"):
        experiment._luo2022_frozen_diffuser_seed_schedule(inconsistent_runtime, contract)

    pre_isolation_contract = deepcopy(contract)
    pre_isolation_contract["training"].pop("diffuser_seed_schedule")
    pre_isolation_contract["evaluation"].pop("diffuser_seed_schedule")
    with pytest.raises(ValueError, match="cannot be certified"):
        experiment._luo2022_frozen_diffuser_seed_schedule(without_runtime_copy, pre_isolation_contract)


def _write_completed_luo2022_run_fixture(run_dir: Path) -> tuple[dict, dict]:
    contract = experiment.load_config(experiment.DEFAULT_LUO2022_CONFIG)
    runtime_config = experiment.build_luo2022_runtime_config(
        contract,
        small_run=True,
        device=torch.device("cpu"),
        epochs=2,
        train_limit=8,
    )
    checkpoint_path = run_dir / "checkpoints" / "luo2022_d2nn.pt"
    checkpoint_path.parent.mkdir(parents=True)
    write_json(run_dir / "config.json", runtime_config)
    write_json(run_dir / "source_config.json", contract)
    source_config_sha256 = experiment._sha256_file(run_dir / "source_config.json")
    optics_config = experiment._luo2022_optics_config_from_frozen_run(runtime_config, contract)
    model = experiment.Luo2022FourLayerD2NN(optics_config)
    write_json(
        run_dir / "manifest.json",
        {
            "profile_id": contract["profile_id"],
            "source_freeze_version": contract["freeze_version"],
            "source_config_sha256": source_config_sha256,
        },
    )
    write_json(
        run_dir / "run_state.json",
        {
            "status": "completed",
            "target_epochs": 2,
            "completed_epoch": 2,
        },
    )
    torch.save(
        {
            "source_freeze_version": contract["freeze_version"],
            "runtime_config": runtime_config,
            "source_config_sha256": source_config_sha256,
            "model": model.state_dict(),
        },
        checkpoint_path,
    )
    return contract, runtime_config


def test_luo2022_diagnosis_loader_requires_run_local_completed_contract(tmp_path: Path) -> None:
    run_dir = tmp_path / "completed"
    contract, runtime_config = _write_completed_luo2022_run_fixture(run_dir)

    artifacts = experiment._load_luo2022_frozen_run_artifacts(
        run_dir=run_dir,
        config_path=experiment.DEFAULT_LUO2022_CONFIG,
        device=torch.device("cpu"),
    )

    assert artifacts.contract == contract
    assert artifacts.runtime_config == runtime_config
    assert artifacts.run_state_sha256
    assert artifacts.source_config_integrity == "sha256_bound_by_manifest_and_checkpoint"

    write_json(
        run_dir / "run_state.json",
        {
            "status": "completed",
            "target_epochs": 2,
            "completed_epoch": 1,
        },
    )
    with pytest.raises(ValueError, match="epoch does not match"):
        experiment._load_luo2022_frozen_run_artifacts(
            run_dir=run_dir,
            config_path=experiment.DEFAULT_LUO2022_CONFIG,
            device=torch.device("cpu"),
        )

    write_json(
        run_dir / "run_state.json",
        {
            "status": "completed",
            "target_epochs": 2,
            "completed_epoch": 2,
        },
    )
    changed_contract = deepcopy(contract)
    changed_contract["geometry"]["layer_to_layer_m"] = 0.003
    write_json(run_dir / "source_config.json", changed_contract)
    changed_hash = experiment._sha256_file(run_dir / "source_config.json")
    write_json(
        run_dir / "manifest.json",
        {
            "profile_id": contract["profile_id"],
            "source_freeze_version": contract["freeze_version"],
            "source_config_sha256": changed_hash,
        },
    )
    checkpoint_path = run_dir / "checkpoints" / "luo2022_d2nn.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["source_config_sha256"] = changed_hash
    torch.save(checkpoint, checkpoint_path)
    changed_config_path = tmp_path / "changed_source_config.json"
    write_json(changed_config_path, changed_contract)
    with pytest.raises(ValueError, match="immutable fields"):
        experiment._load_luo2022_frozen_run_artifacts(
            run_dir=run_dir,
            config_path=changed_config_path,
            device=torch.device("cpu"),
        )

    (run_dir / "source_config.json").unlink()
    with pytest.raises(FileNotFoundError, match="frozen source config"):
        experiment._load_luo2022_frozen_run_artifacts(
            run_dir=run_dir,
            config_path=experiment.DEFAULT_LUO2022_CONFIG,
            device=torch.device("cpu"),
        )


def test_luo2022_diagnosis_rejects_invalid_checkpoint_before_writing_state(tmp_path: Path) -> None:
    run_dir = tmp_path / "completed"
    _write_completed_luo2022_run_fixture(run_dir)
    checkpoint_path = run_dir / "checkpoints" / "luo2022_d2nn.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["model"] = {"not_a_phase_parameter": torch.zeros(1)}
    torch.save(checkpoint, checkpoint_path)
    diagnostic_output_dir = tmp_path / "independent_diagnosis"

    with pytest.raises(ValueError, match="model state"):
        experiment.run_luo2022_diagnosis(
            run_dir=run_dir,
            diagnostic_output_dir=diagnostic_output_dir,
        )

    assert not diagnostic_output_dir.exists()


def test_luo2022_roi_posthoc_isolated_resume_and_full_canvas_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "completed"
    contract, runtime_config = _write_completed_luo2022_run_fixture(run_dir)
    values = runtime_config["runtime"]
    generator = torch.Generator().manual_seed(37)
    dataset = TensorDataset(
        torch.rand(int(values["eval_limit"]), 1, 28, 28, generator=generator),
        torch.zeros(int(values["eval_limit"]), dtype=torch.long),
    )
    monkeypatch.setattr(experiment, "build_torchvision_dataset", lambda **_kwargs: dataset)

    optics_config = experiment._luo2022_optics_config_from_frozen_run(runtime_config, contract)
    checkpoint_path = run_dir / "checkpoints" / "luo2022_d2nn.pt"
    checkpoint_sha256_before = experiment._sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = experiment.Luo2022FourLayerD2NN(optics_config)
    model.load_state_dict(checkpoint["model"], strict=True)
    eval_loader = DataLoader(dataset, batch_size=int(values["batch_size"]), shuffle=False)
    diffuser_dir = run_dir / "diffusers"
    diffuser_dir.mkdir()
    training_banks = {
        epoch: torch.rand(
            int(values["diffusers_per_epoch"]),
            *optics_config.field_shape,
            generator=generator,
        )
        for epoch in range(1, int(values["epochs"]) + 1)
    }
    for epoch, phases in training_banks.items():
        torch.save(phases, diffuser_dir / f"training_epoch_{epoch:03d}.pt")

    diffuser_kwargs = experiment._luo2022_diffuser_kwargs(optics_config, contract)
    seed_schedule, _ = experiment._luo2022_frozen_diffuser_seed_schedule(
        runtime_config,
        contract,
    )
    uniqueness = contract["diffuser"]["uniqueness"]
    new_diffusers = experiment.make_unique_correlated_diffusers(
        int(values["eval_diffusers"]),
        field_shape=optics_config.field_shape,
        base_seed=int(seed_schedule["evaluation_base_seed"]),
        minimum_difference_radians=float(uniqueness["minimum_radians"]),
        phase_representation=str(uniqueness["phase_representation"]),
        **diffuser_kwargs,
    )
    checkpoint_sha256 = experiment._sha256_file(checkpoint_path)
    legacy_rows: list[dict] = []

    def append_legacy_rows(
        phases: torch.Tensor,
        metadata: list[dict],
    ) -> None:
        metrics = experiment.evaluate_luo2022_model_per_diffuser(
            model,
            eval_loader,
            phases,
            resized_shape=(int(values["input_size"]), int(values["input_size"])),
            canvas_shape=optics_config.field_shape,
            device=torch.device("cpu"),
            diffuser_chunk_size=1,
        )
        legacy_rows.extend(
            {
                **row_metadata,
                **metric,
                "checkpoint_sha256": checkpoint_sha256,
                "source_freeze_version": contract["freeze_version"],
            }
            for row_metadata, metric in zip(metadata, metrics, strict=True)
        )

    append_legacy_rows(
        training_banks[2],
        [
            {
                "diffuser_id": f"training:e002:i{index:02d}",
                "population": "training",
                "training_epoch": 2,
                "within_epoch_index": index,
            }
            for index in range(int(values["diffusers_per_epoch"]))
        ],
    )
    append_legacy_rows(
        new_diffusers,
        [
            {
                "diffuser_id": f"new:i{index:02d}",
                "population": "new",
                "training_epoch": None,
                "within_epoch_index": index,
            }
            for index in range(int(values["eval_diffusers"]))
        ],
    )
    append_legacy_rows(
        torch.zeros((1, *optics_config.field_shape)),
        [
            {
                "diffuser_id": "no_diffuser",
                "population": "no_diffuser",
                "training_epoch": None,
                "within_epoch_index": 0,
            }
        ],
    )
    experiment._append_luo2022_posthoc_rows(
        run_dir / "posthoc_evaluation" / "per_diffuser_metrics.jsonl",
        legacy_rows,
    )

    evidence_dir = tmp_path / "independent_roi_evidence"
    summary = experiment.run_luo2022_roi_posthoc_evaluation(
        run_dir=run_dir,
        output_dir=evidence_dir,
        device_name="cpu",
        populations=("training", "new", "no_diffuser"),
        training_epochs=(2,),
        diffuser_chunk_size=2,
    )

    assert summary["status"] == "completed"
    assert summary["read_only"] is True
    assert summary["completed_population_counts"] == {
        "training": 2,
        "new": 2,
        "no_diffuser": 1,
    }
    assert (
        summary["full_canvas_regression"][
            "max_abs_roi_full_canvas_minus_frozen_pearson"
        ]
        <= experiment.LUO2022_ROI_REGRESSION_TOLERANCE
    )
    assert (
        summary["full_canvas_regression"][
            "max_abs_roi_pearson_minus_legacy_pearson"
        ]
        <= experiment.LUO2022_ROI_REGRESSION_TOLERANCE
    )
    assert summary["artifact_integrity"]["model_phase_unchanged"] is True
    assert experiment._sha256_file(checkpoint_path) == checkpoint_sha256_before
    rows_path = evidence_dir / "roi_per_diffuser_metrics.jsonl"
    assert len(rows_path.read_text(encoding="utf-8").splitlines()) == 5

    resumed = experiment.run_luo2022_roi_posthoc_evaluation(
        run_dir=run_dir,
        output_dir=evidence_dir,
        device_name="cpu",
        populations=("training", "new", "no_diffuser"),
        training_epochs=(2,),
        diffuser_chunk_size=1,
    )

    assert resumed["status"] == "completed"
    assert len(rows_path.read_text(encoding="utf-8").splitlines()) == 5


def test_luo2022_control_ladder_isolated_resume_and_trained_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "completed"
    contract, runtime_config = _write_completed_luo2022_run_fixture(run_dir)
    values = runtime_config["runtime"]
    generator = torch.Generator().manual_seed(91)
    dataset = TensorDataset(
        torch.rand(int(values["eval_limit"]), 1, 28, 28, generator=generator),
        torch.zeros(int(values["eval_limit"]), dtype=torch.long),
    )
    monkeypatch.setattr(experiment, "build_torchvision_dataset", lambda **_kwargs: dataset)

    optics_config = experiment._luo2022_optics_config_from_frozen_run(runtime_config, contract)
    checkpoint_path = run_dir / "checkpoints" / "luo2022_d2nn.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    saved_phase = checkpoint["model"]["phase"]
    checkpoint["model"]["phase"] = torch.linspace(
        0.05,
        1.25,
        saved_phase.numel(),
        dtype=saved_phase.dtype,
    ).reshape_as(saved_phase)
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256_before = experiment._sha256_file(checkpoint_path)
    model = experiment.Luo2022FourLayerD2NN(optics_config)
    model.load_state_dict(checkpoint["model"], strict=True)
    eval_loader = DataLoader(dataset, batch_size=int(values["batch_size"]), shuffle=False)

    diffuser_dir = run_dir / "diffusers"
    diffuser_dir.mkdir()
    training_banks = {
        epoch: torch.rand(
            int(values["diffusers_per_epoch"]),
            *optics_config.field_shape,
            generator=generator,
        )
        for epoch in range(1, int(values["epochs"]) + 1)
    }
    for epoch, phases in training_banks.items():
        torch.save(phases, diffuser_dir / f"training_epoch_{epoch:03d}.pt")

    diffuser_kwargs = experiment._luo2022_diffuser_kwargs(optics_config, contract)
    seed_schedule, _ = experiment._luo2022_frozen_diffuser_seed_schedule(
        runtime_config,
        contract,
    )
    uniqueness = contract["diffuser"]["uniqueness"]
    new_diffusers = experiment.make_unique_correlated_diffusers(
        int(values["eval_diffusers"]),
        field_shape=optics_config.field_shape,
        base_seed=int(seed_schedule["evaluation_base_seed"]),
        minimum_difference_radians=float(uniqueness["minimum_radians"]),
        phase_representation=str(uniqueness["phase_representation"]),
        **diffuser_kwargs,
    )
    legacy_rows: list[dict] = []

    def append_legacy_rows(phases: torch.Tensor, metadata: list[dict]) -> None:
        metrics = experiment.evaluate_luo2022_model_per_diffuser(
            model,
            eval_loader,
            phases,
            resized_shape=(int(values["input_size"]), int(values["input_size"])),
            canvas_shape=optics_config.field_shape,
            device=torch.device("cpu"),
            diffuser_chunk_size=1,
        )
        legacy_rows.extend(
            {
                **row_metadata,
                **metric,
                "checkpoint_sha256": checkpoint_sha256_before,
                "source_freeze_version": contract["freeze_version"],
            }
            for row_metadata, metric in zip(metadata, metrics, strict=True)
        )

    append_legacy_rows(
        training_banks[2],
        [
            {
                "diffuser_id": f"training:e002:i{index:02d}",
                "population": "training",
                "training_epoch": 2,
                "within_epoch_index": index,
            }
            for index in range(int(values["diffusers_per_epoch"]))
        ],
    )
    append_legacy_rows(
        new_diffusers,
        [
            {
                "diffuser_id": f"new:i{index:02d}",
                "population": "new",
                "training_epoch": None,
                "within_epoch_index": index,
            }
            for index in range(int(values["eval_diffusers"]))
        ],
    )
    append_legacy_rows(
        torch.zeros((1, *optics_config.field_shape)),
        [
            {
                "diffuser_id": "no_diffuser",
                "population": "no_diffuser",
                "training_epoch": None,
                "within_epoch_index": 0,
            }
        ],
    )
    experiment._append_luo2022_posthoc_rows(
        run_dir / "posthoc_evaluation" / "per_diffuser_metrics.jsonl",
        legacy_rows,
    )

    evidence_dir = tmp_path / "independent_control_evidence"
    final_training_bank = diffuser_dir / "training_epoch_002.pt"
    final_training_bank_sha256_before = experiment._sha256_file(final_training_bank)
    final_training_bank_bytes = final_training_bank.read_bytes()
    checkpoint_bytes = checkpoint_path.read_bytes()
    legacy_rows_path = run_dir / "posthoc_evaluation" / "per_diffuser_metrics.jsonl"
    legacy_rows_bytes = legacy_rows_path.read_bytes()
    original_evaluator = experiment._evaluate_luo2022_forward_per_diffuser
    original_direct_operator = (
        experiment.Luo2022FourLayerD2NN.forward_without_diffractive_layers
    )
    original_four_layer_operator = experiment.Luo2022FourLayerD2NN.forward
    zero_phase_model = experiment.Luo2022FourLayerD2NN(optics_config)
    control_operators = {
        "direct_free_space_no_d2nn": model.forward_without_diffractive_layers,
        "zero_phase_four_layer": zero_phase_model.forward,
        "trained_four_layer": model.forward,
    }
    experiment._validate_luo2022_control_ladder_operator_bindings(
        control_operators,
        trained_model=model,
        zero_phase_model=zero_phase_model,
    )
    swapped_control_operators = dict(control_operators)
    swapped_control_operators["direct_free_space_no_d2nn"] = zero_phase_model.forward
    with pytest.raises(
        RuntimeError,
        match="direct_free_space_no_d2nn is not bound to the trained model's "
        "direct-propagation operator",
    ):
        experiment._validate_luo2022_control_ladder_operator_bindings(
            swapped_control_operators,
            trained_model=model,
            zero_phase_model=zero_phase_model,
    )
    swapped_control_operators = dict(control_operators)
    swapped_control_operators["zero_phase_four_layer"] = model.forward
    with pytest.raises(
        RuntimeError,
        match="zero_phase_four_layer is not bound to its expected four-layer optical operator",
    ):
        experiment._validate_luo2022_control_ladder_operator_bindings(
            swapped_control_operators,
            trained_model=model,
            zero_phase_model=zero_phase_model,
        )
    operator_calls: set[str] = set()
    routed_controls: list[str] = []

    def track_direct_operator(
        optical_model: experiment.Luo2022FourLayerD2NN,
        object_field: torch.Tensor,
        diffuser_phase: torch.Tensor,
    ) -> torch.Tensor:
        operator_calls.add("direct_free_space_no_d2nn")
        return original_direct_operator(optical_model, object_field, diffuser_phase)

    def track_four_layer_operator(
        optical_model: experiment.Luo2022FourLayerD2NN,
        object_field: torch.Tensor,
        diffuser_phase: torch.Tensor,
    ) -> torch.Tensor:
        operator_calls.add(
            (
                "zero_phase_four_layer"
                if torch.count_nonzero(optical_model.phase).item() == 0
                else "trained_four_layer"
            )
        )
        return original_four_layer_operator(optical_model, object_field, diffuser_phase)

    monkeypatch.setattr(
        experiment.Luo2022FourLayerD2NN,
        "forward_without_diffractive_layers",
        track_direct_operator,
    )
    monkeypatch.setattr(
        experiment.Luo2022FourLayerD2NN,
        "forward",
        track_four_layer_operator,
    )

    def classify_c0_forward(forward: object) -> str:
        if getattr(forward, "__func__", None) is track_direct_operator:
            return "direct_free_space_no_d2nn"
        phase = getattr(forward, "phase", None)
        if not isinstance(phase, torch.Tensor):
            phase = getattr(getattr(forward, "__self__", None), "phase", None)
        if not isinstance(phase, torch.Tensor):
            pytest.fail("C0 evaluator received an unrecognized forward operator")
        return (
            "zero_phase_four_layer"
            if torch.count_nonzero(phase).item() == 0
            else "trained_four_layer"
        )

    def record_c0_forward_route(args: tuple[object, ...]) -> None:
        assert args
        routed_controls.append(classify_c0_forward(args[0]))

    evaluation_calls = 0

    def interrupt_after_direct_training(*args: object, **kwargs: object) -> object:
        nonlocal evaluation_calls
        record_c0_forward_route(args)
        evaluation_calls += 1
        if evaluation_calls == 2:
            raise RuntimeError("simulated C0 interruption")
        return original_evaluator(*args, **kwargs)

    monkeypatch.setattr(
        experiment,
        "_evaluate_luo2022_forward_per_diffuser",
        interrupt_after_direct_training,
    )
    with pytest.raises(RuntimeError, match="simulated C0 interruption"):
        experiment.run_luo2022_c0_optical_control_ladder(
            run_dir=run_dir,
            control_output_dir=evidence_dir,
            device_name="cpu",
            populations=("training", "new", "no_diffuser"),
            training_epochs=(2,),
            diffuser_chunk_size=2,
        )
    rows_path = evidence_dir / "control_ladder_per_diffuser_metrics.jsonl"
    partial_rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(partial_rows) == 2
    assert {row["control_id"] for row in partial_rows} == {
        "direct_free_space_no_d2nn"
    }
    partial_state = experiment.load_config(evidence_dir / "control_ladder_state.json")
    assert partial_state["status"] == "running"
    assert experiment._sha256_file(checkpoint_path) == checkpoint_sha256_before
    assert experiment._sha256_file(final_training_bank) == final_training_bank_sha256_before

    tampered_checkpoint = deepcopy(checkpoint)
    tampered_checkpoint["model"]["phase"] = (
        tampered_checkpoint["model"]["phase"] + 0.001
    )
    torch.save(tampered_checkpoint, checkpoint_path)
    with pytest.raises(ValueError):
        experiment.run_luo2022_c0_optical_control_ladder(
            run_dir=run_dir,
            control_output_dir=evidence_dir,
            device_name="cpu",
            populations=("training", "new", "no_diffuser"),
            training_epochs=(2,),
            diffuser_chunk_size=2,
        )
    checkpoint_path.write_bytes(checkpoint_bytes)

    torch.save(training_banks[2] + 0.001, final_training_bank)
    with pytest.raises(ValueError, match="does not match the frozen inputs or request"):
        experiment.run_luo2022_c0_optical_control_ladder(
            run_dir=run_dir,
            control_output_dir=evidence_dir,
            device_name="cpu",
            populations=("training", "new", "no_diffuser"),
            training_epochs=(2,),
            diffuser_chunk_size=2,
        )
    final_training_bank.write_bytes(final_training_bank_bytes)

    tampered_legacy_rows = [
        json.loads(line)
        for line in legacy_rows_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    tampered_legacy_rows[0]["pearson"] = float(tampered_legacy_rows[0]["pearson"]) + 0.001
    legacy_rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in tampered_legacy_rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match the frozen inputs or request"):
        experiment.run_luo2022_c0_optical_control_ladder(
            run_dir=run_dir,
            control_output_dir=evidence_dir,
            device_name="cpu",
            populations=("training", "new", "no_diffuser"),
            training_epochs=(2,),
            diffuser_chunk_size=2,
        )
    legacy_rows_path.write_bytes(legacy_rows_bytes)

    legacy_rows_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in tampered_legacy_rows[1:]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing requested C0 diffuser rows"):
        experiment.run_luo2022_c0_optical_control_ladder(
            run_dir=run_dir,
            control_output_dir=evidence_dir,
            device_name="cpu",
            populations=("training", "new", "no_diffuser"),
            training_epochs=(2,),
            diffuser_chunk_size=2,
        )
    legacy_rows_path.write_bytes(legacy_rows_bytes)

    assert experiment._sha256_file(checkpoint_path) == checkpoint_sha256_before
    assert experiment._sha256_file(final_training_bank) == final_training_bank_sha256_before

    def record_c0_evaluator(*args: object, **kwargs: object) -> object:
        record_c0_forward_route(args)
        return original_evaluator(*args, **kwargs)

    monkeypatch.setattr(
        experiment,
        "_evaluate_luo2022_forward_per_diffuser",
        record_c0_evaluator,
    )

    summary = experiment.run_luo2022_c0_optical_control_ladder(
        run_dir=run_dir,
        control_output_dir=evidence_dir,
        device_name="cpu",
        populations=("training", "new", "no_diffuser"),
        training_epochs=(2,),
        diffuser_chunk_size=2,
    )

    assert summary["status"] == "completed"
    assert summary["read_only"] is True
    assert summary["expected_record_count"] == 15
    assert summary["completed_population_counts"] == {
        control_id: {"training": 2, "new": 2, "no_diffuser": 1}
        for control_id in experiment.LUO2022_CONTROL_LADDER_IDS
    }
    assert (
        summary["trained_four_layer_legacy_full_canvas_regression"]["status"]
        == "verified_against_run_local_posthoc"
    )
    assert (
        summary["trained_four_layer_legacy_full_canvas_regression"]["max_abs_error"]
        <= experiment.LUO2022_ROI_REGRESSION_TOLERANCE
    )
    assert summary["artifact_integrity"]["trained_phase_unchanged"] is True
    assert summary["artifact_integrity"]["zero_phase_unchanged"] is True
    assert summary["artifact_integrity"]["frozen_input_hashes_unchanged"] is True
    assert (
        summary["artifact_integrity"]["trained_phase_before_sha256"]
        != summary["artifact_integrity"]["zero_phase_before_sha256"]
    )
    assert experiment._sha256_file(checkpoint_path) == checkpoint_sha256_before
    assert operator_calls == set(experiment.LUO2022_CONTROL_LADDER_IDS)
    assert routed_controls == [
        "direct_free_space_no_d2nn",
        "direct_free_space_no_d2nn",
        "direct_free_space_no_d2nn",
        "direct_free_space_no_d2nn",
        "zero_phase_four_layer",
        "zero_phase_four_layer",
        "zero_phase_four_layer",
        "trained_four_layer",
        "trained_four_layer",
        "trained_four_layer",
    ]

    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 15
    assert len({row["record_id"] for row in rows}) == 15
    assert {
        row["control_id"] for row in rows
    } == set(experiment.LUO2022_CONTROL_LADDER_IDS)
    assert all(row["object_count"] == int(values["eval_limit"]) for row in rows)
    assert all(
        row["roi_full_canvas_metric_abs_error"]
        <= experiment.LUO2022_ROI_REGRESSION_TOLERANCE
        for row in rows
    )
    direct_rows = [
        row for row in rows if row["control_id"] == "direct_free_space_no_d2nn"
    ]
    assert all(row["operator"] == "single_direct_propagation" for row in direct_rows)
    assert all(row["phase_dependency"] == "none" for row in direct_rows)
    assert all(row["control_phase_sha256"] is None for row in direct_rows)
    assert all(row["post_diffuser_distance_m"] == pytest.approx(0.015) for row in direct_rows)
    assert all(row["explicit_physical_aperture"] == "none" for row in direct_rows)
    assert all(
        row["post_diffuser_window_applications"] == 1 for row in direct_rows
    )
    assert all(
        "one center-cropped finite" in row["finite_numerical_window"]
        for row in direct_rows
    )
    sampled_rows = [
        row
        for row in rows
        if row["control_id"]
        in {"zero_phase_four_layer", "trained_four_layer"}
    ]
    assert all(row["post_diffuser_window_applications"] == 5 for row in sampled_rows)

    def fail_if_completed_control_re_evaluates(*_args: object, **_kwargs: object) -> object:
        pytest.fail("completed C0 evidence should return without evaluating another forward pass")

    monkeypatch.setattr(
        experiment,
        "_evaluate_luo2022_forward_per_diffuser",
        fail_if_completed_control_re_evaluates,
    )
    resumed = experiment.run_luo2022_c0_optical_control_ladder(
        run_dir=run_dir,
        control_output_dir=evidence_dir,
        device_name="cpu",
        populations=("training", "new", "no_diffuser"),
        training_epochs=(2,),
        diffuser_chunk_size=1,
    )

    assert resumed["status"] == "completed"
    assert len(rows_path.read_text(encoding="utf-8").splitlines()) == 15

    finalizing_state = experiment.load_config(evidence_dir / "control_ladder_state.json")
    finalizing_state["status"] = "finalizing"
    finalizing_state["stage"] = "finalizing"
    experiment._write_luo2022_control_ladder_json(
        evidence_dir / "control_ladder_state.json",
        finalizing_state,
    )
    finalized_after_interruption = experiment.run_luo2022_c0_optical_control_ladder(
        run_dir=run_dir,
        control_output_dir=evidence_dir,
        device_name="cpu",
        populations=("training", "new", "no_diffuser"),
        training_epochs=(2,),
        diffuser_chunk_size=1,
    )
    assert finalized_after_interruption["status"] == "completed"
    assert (
        experiment.load_config(evidence_dir / "control_ladder_state.json")["status"]
        == "completed"
    )

    summary_path = evidence_dir / "control_ladder_summary.json"
    saved_summary = experiment.load_config(summary_path)
    tampered_summary = deepcopy(saved_summary)
    tampered_summary["groups"] = {}
    experiment._write_luo2022_control_ladder_json(summary_path, tampered_summary)
    with pytest.raises(ValueError, match="does not match its evidence rows"):
        experiment.run_luo2022_c0_optical_control_ladder(
            run_dir=run_dir,
            control_output_dir=evidence_dir,
            device_name="cpu",
            populations=("training", "new", "no_diffuser"),
            training_epochs=(2,),
            diffuser_chunk_size=1,
        )
    experiment._write_luo2022_control_ladder_json(summary_path, saved_summary)

    corrupt_rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    corrupt_rows[0]["pearson"] = float("nan")
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in corrupt_rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite 'pearson'"):
        experiment.run_luo2022_c0_optical_control_ladder(
            run_dir=run_dir,
            control_output_dir=evidence_dir,
            device_name="cpu",
            populations=("training", "new", "no_diffuser"),
            training_epochs=(2,),
            diffuser_chunk_size=1,
        )


def test_luo2022_diffuser_chunking_preserves_one_optimizer_update() -> None:
    generator = torch.Generator().manual_seed(19)
    images = torch.rand(2, 1, 28, 28, generator=generator)
    labels = torch.zeros(2, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)
    config = experiment.Luo2022OpticsConfig(field_shape=(48, 48))
    full_model = experiment.Luo2022FourLayerD2NN(config)
    chunked_model = experiment.Luo2022FourLayerD2NN(config)
    chunked_model.load_state_dict(full_model.state_dict())
    diffusers = torch.rand(4, 48, 48, generator=generator)
    full_optimizer = torch.optim.SGD(full_model.parameters(), lr=1e-3)
    chunked_optimizer = torch.optim.SGD(chunked_model.parameters(), lr=1e-3)

    full_metrics = experiment.train_luo2022_one_epoch(
        full_model,
        loader,
        diffusers,
        full_optimizer,
        resized_shape=(32, 32),
        canvas_shape=(48, 48),
        device=torch.device("cpu"),
        diffuser_chunk_size=4,
    )
    chunked_metrics = experiment.train_luo2022_one_epoch(
        chunked_model,
        loader,
        diffusers,
        chunked_optimizer,
        resized_shape=(32, 32),
        canvas_shape=(48, 48),
        device=torch.device("cpu"),
        diffuser_chunk_size=1,
    )

    assert chunked_metrics == pytest.approx(full_metrics, abs=1e-6)
    assert torch.allclose(chunked_model.phase, full_model.phase, atol=1e-8, rtol=1e-5)


def test_luo2022_per_diffuser_metrics_preserve_global_mean() -> None:
    generator = torch.Generator().manual_seed(29)
    images = torch.rand(4, 1, 28, 28, generator=generator)
    labels = torch.zeros(4, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)
    config = experiment.Luo2022OpticsConfig(field_shape=(48, 48))
    model = experiment.Luo2022FourLayerD2NN(config)
    diffusers = torch.rand(3, 48, 48, generator=generator)

    aggregate = experiment.evaluate_luo2022_model(
        model,
        loader,
        diffusers,
        resized_shape=(32, 32),
        canvas_shape=(48, 48),
        device=torch.device("cpu"),
        diffuser_chunk_size=2,
    )
    per_diffuser = experiment.evaluate_luo2022_model_per_diffuser(
        model,
        loader,
        diffusers,
        resized_shape=(32, 32),
        canvas_shape=(48, 48),
        device=torch.device("cpu"),
        diffuser_chunk_size=1,
    )
    per_diffuser_with_roi = experiment.evaluate_luo2022_model_per_diffuser(
        model,
        loader,
        diffusers,
        resized_shape=(32, 32),
        canvas_shape=(48, 48),
        device=torch.device("cpu"),
        diffuser_chunk_size=2,
        include_roi_metrics=True,
    )

    assert len(per_diffuser) == 3
    assert all(row["object_count"] == 4 for row in per_diffuser)
    for metric in ("total", "negative_pearson", "energy", "pearson"):
        assert sum(float(row[metric]) for row in per_diffuser) / 3 == pytest.approx(
            aggregate[metric],
            abs=1e-6,
        )
    assert len(per_diffuser_with_roi) == len(per_diffuser)
    for plain_row, roi_row in zip(per_diffuser, per_diffuser_with_roi, strict=True):
        for metric in ("total", "negative_pearson", "energy", "pearson"):
            assert roi_row[metric] == pytest.approx(plain_row[metric], abs=1e-6)
        assert roi_row["roi_full_canvas_pearson"] == pytest.approx(
            roi_row["pearson"],
            abs=1e-6,
        )
        for metric in experiment.LUO2022_ROI_METRIC_FIELDS:
            assert metric in roi_row



def test_luo2022_posthoc_summary_recreates_paper_populations() -> None:
    rows = [
        {
            "population": "training",
            "training_epoch": epoch,
            "object_count": 10,
            "pearson": float(epoch) / 10,
            "negative_pearson": -float(epoch) / 10,
            "energy": -0.1,
            "total": -float(epoch) / 10 - 0.1,
        }
        for epoch in range(1, 13)
    ]
    rows.extend(
        [
            {
                "population": "new",
                "training_epoch": None,
                "object_count": 10,
                "pearson": 0.5,
                "negative_pearson": -0.5,
                "energy": -0.1,
                "total": -0.6,
            },
            {
                "population": "no_diffuser",
                "training_epoch": None,
                "object_count": 10,
                "pearson": 0.7,
                "negative_pearson": -0.7,
                "energy": -0.1,
                "total": -0.8,
            },
        ]
    )

    summary = experiment.summarize_luo2022_posthoc_rows(rows, target_epochs=12)

    assert summary["all_training_diffusers"]["diffuser_count"] == 12
    assert summary["epochs_1_to_penultimate_training_diffusers"]["diffuser_count"] == 11
    assert summary["last_10_epoch_training_diffusers"]["diffuser_count"] == 10
    assert summary["final_epoch_known_diffusers"]["diffuser_count"] == 1
    assert summary["new_unseen_diffusers"]["diffuser_count"] == 1
    assert summary["no_diffuser_control"]["diffuser_count"] == 1
    assert summary["all_training_diffusers"]["metrics"]["pearson"]["sample_std"] is not None


def test_training_cli_builds_shared_reconstruction_weights() -> None:
    args = experiment.build_parser().parse_args(
        [
            "unet",
            "--l1-weight",
            "0.8",
            "--negative-pearson-weight",
            "0.1",
            "--ssim-weight",
            "0.2",
            "--fourier-weight",
            "0.05",
        ]
    )

    assert experiment.reconstruction_weights_from_args(args) == ReconstructionLossWeights(
        l1=0.8,
        negative_pearson=0.1,
        ssim=0.2,
        fourier=0.05,
    )


def test_experiment_classes_follow_the_roadmap() -> None:
    assert experiment.experiment_class_for_run(
        corruption="phase", train_diffuser_ids=(0,), eval_diffuser_ids=(0,), uses_gan=False
    ) == "E0"
    assert experiment.experiment_class_for_run(
        corruption="phase", train_diffuser_ids=(0, 1), eval_diffuser_ids=(2,), uses_gan=False
    ) == "E1"
    assert experiment.experiment_class_for_run(
        corruption="phase", train_diffuser_ids=(0,), eval_diffuser_ids=(0,), uses_gan=True
    ) == "E0+E2"
    assert experiment.experiment_class_for_run(
        corruption="particles", train_diffuser_ids=(0,), eval_diffuser_ids=(0,), uses_gan=True
    ) == "E3+E2"


def test_coherent_training_config_uses_canonical_schema() -> None:
    config = experiment.coherent_training_config(
        command="unet",
        experiment_class="E1",
        corruption="phase",
        seed=42,
        d2nn_seed=7961,
        train_diffuser_ids=(0, 1),
        eval_diffuser_ids=(2,),
        train_limit=32,
        eval_limit=8,
        epochs=2,
        batch_size=4,
        base_channels=8,
        lr=2e-3,
        device=torch.device("cpu"),
        materialize=True,
        sample_every=1,
        max_train_batches=None,
        max_eval_batches=None,
        num_workers=0,
        reconstruction_weights=ReconstructionLossWeights(l1=1.0, negative_pearson=0.1),
    )

    assert config["schema_version"] == experiment.CONFIG_SCHEMA_VERSION
    assert config["experiment_class"] == "E1"
    assert config["diffuser_split"]["evaluation"] == "unseen"
    assert config["optimization"]["reconstruction_loss_weights"]["negative_pearson"] == 0.1
    assert config["evaluation"]["metrics_protocol"] == experiment.metrics_protocol_metadata()


def test_coherent_unet_train_eval_and_grid(tmp_path: Path) -> None:
    loader = _coherent_loader(seed=6)
    model = UNetReconstructor(base_channels=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    train_metrics = train_unet_one_epoch(
        model,
        loader,
        optimizer,
        ReconstructionLossWeights(l1=1.0),
        device=device,
    )
    eval_metrics = evaluate_reconstructor(model, loader, device=device)
    output_path = tmp_path / "coherent_grid.png"
    save_reconstruction_grid(model, loader, output_path, device=device)

    assert train_metrics["total"] > 0
    assert {"l1", "mse", "psnr", "ssim", "pearson"}.issubset(eval_metrics)
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_coherent_patchgan_train_eval_and_grid(tmp_path: Path) -> None:
    loader = _coherent_loader(seed=9)
    generator = UNetReconstructor(base_channels=4)
    discriminator = PatchDiscriminator(base_channels=4)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
    device = torch.device("cpu")

    train_metrics = train_gan_one_epoch(
        generator,
        discriminator,
        loader,
        generator_optimizer,
        discriminator_optimizer,
        ReconstructionLossWeights(l1=1.0),
        adversarial_weight=0.01,
        device=device,
    )
    eval_metrics = evaluate_generator(generator, loader, device=device)
    output_path = tmp_path / "coherent_gan_grid.png"
    save_gan_grid(generator, loader, output_path, device=device)

    assert train_metrics["generator_total"] > 0
    assert train_metrics["discriminator_total"] > 0
    assert "reconstruction_l1" in train_metrics
    assert "adversarial" in train_metrics
    assert {"l1", "mse", "psnr", "ssim", "pearson"}.issubset(eval_metrics)
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_gan_training_reuses_fake_and_restores_discriminator_gradients(monkeypatch) -> None:
    loader = _coherent_loader(seed=12)
    generator = UNetReconstructor(base_channels=4)
    discriminator = PatchDiscriminator(base_channels=4)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
    original_forward = generator.forward
    calls = 0

    def counting_forward(source: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_forward(source)

    monkeypatch.setattr(generator, "forward", counting_forward)
    train_gan_one_epoch(
        generator,
        discriminator,
        loader,
        generator_optimizer,
        discriminator_optimizer,
        ReconstructionLossWeights(l1=1.0),
        adversarial_weight=0.01,
        device=torch.device("cpu"),
        max_batches=1,
    )

    assert calls == 1
    assert all(parameter.requires_grad for parameter in discriminator.parameters())


def test_experiment_compare_writes_metric_summary(tmp_path: Path) -> None:
    unet_dir = tmp_path / "unet"
    gan_dir = tmp_path / "gan"
    output_dir = tmp_path / "comparison"
    unet_dir.mkdir()
    gan_dir.mkdir()
    write_json(unet_dir / "metrics.json", {"l1": 0.4, "mse": 0.2, "psnr": 7.0, "ssim": 0.1, "pearson": 0.2})
    write_json(gan_dir / "metrics.json", {"l1": 0.3, "mse": 0.15, "psnr": 8.0, "ssim": 0.2, "pearson": 0.3})

    result = compare_runs(unet_dir, gan_dir, output_dir)

    assert result["metric_comparison"]["l1"]["gan_better"] is True
    assert result["metric_comparison"]["psnr"]["gan_better"] is True
    assert result["schema_version"] == experiment.MANIFEST_SCHEMA_VERSION
    assert result["metrics_protocol"]["psnr"] == "mean of per-image PSNR values"
    assert (output_dir / "comparison.json").is_file()
    assert (output_dir / "comparison_metrics.png").is_file()
