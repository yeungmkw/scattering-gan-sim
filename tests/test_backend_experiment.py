"""Execution tests for the fixed-four-layer digital-backend orchestration."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

import experiment
from coherent_data import verify_luo2022_intensity_cache
from runtime import write_json


class _TinyMNIST:
    def __init__(self, count: int = 6) -> None:
        self.items: list[tuple[torch.Tensor, int]] = []
        for index in range(count):
            image = torch.zeros(1, 28, 28)
            row = 2 + index % 12
            column = 3 + (index * 3) % 12
            image[:, row : row + 8, column : column + 7] = 0.35 + 0.05 * index
            image[:, row + 2 : row + 5, column + 1 : column + 5] = 1.0
            self.items.append((image, index % 10))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.items[index]


def _write_frozen_r0_fixture(run_dir: Path) -> None:
    contract = experiment.load_config(experiment.DEFAULT_LUO2022_CONFIG)
    runtime_config = experiment.build_luo2022_runtime_config(
        contract,
        small_run=True,
        device=torch.device("cpu"),
        grid_size=48,
        input_size=24,
        epochs=1,
        train_limit=4,
        eval_limit=2,
        diffusers_per_epoch=2,
        eval_diffusers=1,
    )
    checkpoint_path = run_dir / "checkpoints" / "luo2022_d2nn.pt"
    checkpoint_path.parent.mkdir(parents=True)
    (run_dir / "diffusers").mkdir()
    write_json(run_dir / "config.json", runtime_config)
    write_json(run_dir / "source_config.json", contract)
    source_config_sha256 = experiment._sha256_file(run_dir / "source_config.json")
    optics_config = experiment._luo2022_optics_config_from_frozen_run(
        runtime_config,
        contract,
    )
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
        {"status": "completed", "target_epochs": 1, "completed_epoch": 1},
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
    phase_generator = torch.Generator(device="cpu")
    phase_generator.manual_seed(123)
    torch.save(
        0.1 * torch.rand(2, 48, 48, generator=phase_generator),
        run_dir / "diffusers" / "training_epoch_001.pt",
    )


@pytest.fixture(scope="module")
def backend_runs(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("backend-e2e")
    run_dir = root / "r0"
    _write_frozen_r0_fixture(run_dir)
    tiny_dataset = _TinyMNIST()
    patcher = pytest.MonkeyPatch()
    patcher.setattr(
        experiment,
        "build_torchvision_dataset",
        lambda **_kwargs: tiny_dataset,
    )
    cache_dir = root / "cache"
    cache_result = experiment.run_luo2022_backend_cache(
        run_dir=run_dir,
        cache_dir=cache_dir,
        device_name="cpu",
        small_run=True,
        train_limit=4,
        validation_limit=2,
        shard_size=2,
    )
    common = {
        "run_dir": run_dir,
        "cache_dir": cache_dir,
        "device_name": "cpu",
        "small_run": True,
        "batch_size": 2,
        "max_updates_per_epoch": 1,
    }
    b0_dir = root / "b0"
    warmup_dir = root / "warmup"
    r1_dir = root / "r1"
    r2_dir = root / "r2"
    experiment.run_luo2022_backend_training(
        **common,
        output_dir=b0_dir,
        condition="b0",
        epochs=2,
    )
    experiment.run_luo2022_backend_training(
        **common,
        output_dir=warmup_dir,
        condition="warmup",
        epochs=1,
    )
    experiment.run_luo2022_backend_training(
        **common,
        output_dir=r1_dir,
        condition="r1",
        warmup_dir=warmup_dir,
        epochs=1,
    )
    experiment.run_luo2022_backend_training(
        **common,
        output_dir=r2_dir,
        condition="r2",
        warmup_dir=warmup_dir,
        epochs=1,
    )
    evaluation_dir = root / "evaluation"
    evaluation = experiment.run_luo2022_backend_evaluation(
        run_dir=run_dir,
        output_dir=evaluation_dir,
        cache_dir=cache_dir,
        warmup_dir=warmup_dir,
        b0_dir=b0_dir,
        r1_dir=r1_dir,
        r2_dir=r2_dir,
        device_name="cpu",
        max_eval_batches=1,
    )
    yield {
        "root": root,
        "run_dir": run_dir,
        "cache_dir": cache_dir,
        "cache_result": cache_result,
        "b0_dir": b0_dir,
        "warmup_dir": warmup_dir,
        "r1_dir": r1_dir,
        "r2_dir": r2_dir,
        "evaluation_dir": evaluation_dir,
        "evaluation": evaluation,
    }
    patcher.undo()


def test_backend_cli_exposes_cache_train_and_evaluate_actions() -> None:
    parser = experiment.build_parser()
    cache_args = parser.parse_args(
        [
            "d2nn",
            "--profile",
            "luo2022_r0",
            "--action",
            "backend-cache",
            "--run-dir",
            "r0",
            "--small-run",
        ]
    )
    train_args = parser.parse_args(
        [
            "d2nn",
            "--profile",
            "luo2022_r0",
            "--action",
            "backend-train",
            "--run-dir",
            "r0",
            "--backend-condition",
            "r2",
            "--backend-warmup-dir",
            "warmup",
        ]
    )
    evaluate_args = parser.parse_args(
        [
            "d2nn",
            "--profile",
            "luo2022_r0",
            "--action",
            "backend-evaluate",
            "--run-dir",
            "r0",
            "--backend-warmup-dir",
            "warmup",
            "--backend-b0-dir",
            "b0",
            "--backend-r1-dir",
            "r1",
            "--backend-r2-dir",
            "r2",
        ]
    )

    assert cache_args.backend_cache_operators == ("direct", "frozen_four_layer")
    assert train_args.backend_condition == "r2"
    assert train_args.backend_warmup_dir == Path("warmup")
    assert evaluate_args.backend_warmup_dir == Path("warmup")
    assert evaluate_args.backend_b0_dir == Path("b0")

    missing_warmup_args = parser.parse_args(
        [
            "d2nn",
            "--profile",
            "luo2022_r0",
            "--action",
            "backend-evaluate",
            "--run-dir",
            "r0",
            "--backend-b0-dir",
            "b0",
            "--backend-r1-dir",
            "r1",
            "--backend-r2-dir",
            "r2",
        ]
    )
    with pytest.raises(ValueError, match="--backend-warmup-dir"):
        experiment.dispatch(missing_warmup_args)


def test_b0_forward_and_supervised_backward_update_generator() -> None:
    generator = experiment.UNetReconstructor(base_channels=4)
    optimizer = torch.optim.Adam(generator.parameters(), lr=2e-3)
    source = torch.rand(2, 1, 20, 20)
    target = torch.rand(2, 1, 20, 20)
    before = experiment._backend_state_sha256(generator.state_dict())

    metrics = experiment.backend_supervised_train_step(
        generator,
        optimizer,
        source,
        target,
    )

    assert metrics["l1"] > 0
    assert experiment._backend_state_sha256(generator.state_dict()) != before
    assert any(parameter.grad is not None for parameter in generator.parameters())


def test_r2_step_produces_generator_and_discriminator_gradients() -> None:
    generator = experiment.UNetReconstructor(base_channels=4)
    discriminator = experiment.PatchDiscriminator(base_channels=4)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-3)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4)

    metrics = experiment.backend_gan_train_step(
        generator,
        discriminator,
        generator_optimizer,
        discriminator_optimizer,
        torch.rand(2, 1, 20, 20),
        torch.rand(2, 1, 20, 20),
        adversarial_weight=0.01,
    )

    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert any(parameter.grad is not None for parameter in generator.parameters())
    assert any(parameter.grad is not None for parameter in discriminator.parameters())
    assert all(parameter.requires_grad for parameter in discriminator.parameters())


def test_frozen_r0_loader_enforces_eval_no_grad_and_hash_stability(backend_runs) -> None:
    artifacts, model, before = experiment._load_luo2022_frozen_backend_model(
        run_dir=backend_runs["run_dir"],
        device=torch.device("cpu"),
    )
    phase_before = experiment._sha256_tensor(model.phase)
    target = torch.rand(1, 1, 48, 48)
    field = experiment.amplitude_to_complex_field(target)
    with torch.no_grad():
        output = model(field, torch.zeros(1, 48, 48))
    frontend = experiment.Luo2022FrozenBackendFrontend(artifacts, model, before)
    integrity = experiment._assert_luo2022_backend_r0_unchanged(frontend)

    assert output.shape == (1, 1, 48, 48)
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert model.phase.grad is None
    assert experiment._sha256_tensor(model.phase) == phase_before
    assert integrity["hashes_unchanged"] is True


def test_cache_reuses_shared_assignment_and_train_only_scale(backend_runs) -> None:
    cache_dir = backend_runs["cache_dir"]
    manifest = backend_runs["cache_result"]["manifest"]
    assert manifest["assignment"]["shared_across_operators"] is True
    assert manifest["r0"]["hashes_unchanged"] is True

    operator_records = {}
    for operator_id in ("direct_no_d2nn", "r0_four_layer"):
        train = verify_luo2022_intensity_cache(cache_dir / operator_id / "train")
        validation = verify_luo2022_intensity_cache(
            cache_dir / operator_id / "validation"
        )
        assert train["assignment_sha"] == manifest["assignment"]["train_sha256"]
        assert validation["assignment_sha"] == manifest["assignment"]["validation_sha256"]
        assert validation["scale"]["value"] == train["scale"]["value"]
        assert validation["scale"]["reuse_mode"] == "frozen_training_statistic"
        assert validation["scale"]["provenance"]["source_cache_root_fingerprint"] == train[
            "root_fingerprint"
        ]
        records_path = cache_dir / operator_id / "validation" / validation["shards"][0][
            "records_file"
        ]
        records = json.loads(records_path.read_text("utf-8"))
        operator_records[operator_id] = records
        assert all(record["object_id"] >= 50_000 for record in records)
        assert {record["diffuser_id"] for record in records}.issubset({0, 1})
        assert [
            (record["diffuser_id"], record["row_id"]) for record in records
        ] == sorted(
            (record["diffuser_id"], record["row_id"]) for record in records
        )
    assert operator_records["direct_no_d2nn"] == operator_records["r0_four_layer"]


def test_warmup_branches_have_identical_start_order_and_update_budget(backend_runs) -> None:
    warmup = torch.load(
        backend_runs["warmup_dir"] / "checkpoints" / "latest.pt",
        map_location="cpu",
        weights_only=True,
    )
    r1 = torch.load(
        backend_runs["r1_dir"] / "checkpoints" / "latest.pt",
        map_location="cpu",
        weights_only=True,
    )
    r2 = torch.load(
        backend_runs["r2_dir"] / "checkpoints" / "latest.pt",
        map_location="cpu",
        weights_only=True,
    )
    configs = [
        experiment.load_config(backend_runs[name] / "config.json")
        for name in ("b0_dir", "warmup_dir", "r1_dir", "r2_dir")
    ]

    assert len(
        {config["fresh_generator_initial_state_sha256"] for config in configs}
    ) == 1
    assert r1["branch_start"] == r2["branch_start"]
    assert r1["branch_start"]["generator_state_sha256"] == experiment._backend_state_sha256(
        warmup["generator"]
    )
    assert r1["branch_start"]["generator_optimizer_sha256"] == experiment._backend_state_sha256(
        warmup["generator_optimizer"]
    )
    assert [row["consumed_order_sha256"] for row in r1["epoch_orders"]] == [
        row["consumed_order_sha256"] for row in r2["epoch_orders"]
    ]
    assert r1["generator_update_count"] == r2["generator_update_count"] == 1
    assert r1["discriminator"] is None
    assert r2["discriminator"] is not None

    fairness = backend_runs["evaluation"]["metrics"]["model_and_budget"][
        "training_fairness"
    ]
    b0 = torch.load(
        backend_runs["b0_dir"] / "checkpoints" / "latest.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert b0["epoch_orders"] == [*warmup["epoch_orders"], *r1["epoch_orders"]]
    assert fairness["b0_order_sequence_sha256"] == fairness[
        "warmup_plus_r1_order_sequence_sha256"
    ]
    assert fairness["warmup_generator_updates"] == 1
    assert fairness["b0_total_generator_updates"] == 2
    assert fairness["r1_total_generator_updates"] == 2
    assert fairness["r2_total_generator_updates"] == 2


def test_backend_fairness_rejects_wrong_warmup_hash_or_b0_order(backend_runs) -> None:
    cache_manifest = backend_runs["cache_result"]["manifest"]
    configs = {
        name: experiment.load_config(backend_runs[f"{name}_dir"] / "config.json")
        for name in ("warmup", "b0", "r1", "r2")
    }
    checkpoints = {
        name: torch.load(
            backend_runs[f"{name}_dir"] / "checkpoints" / "latest.pt",
            map_location="cpu",
            weights_only=True,
        )
        for name in ("warmup", "b0", "r1", "r2")
    }
    warmup_sha256 = experiment._sha256_file(
        backend_runs["warmup_dir"] / "checkpoints" / "latest.pt"
    )
    common = {
        "cache_manifest": cache_manifest,
        "warmup_config": configs["warmup"],
        "warmup_checkpoint": checkpoints["warmup"],
        "b0_config": configs["b0"],
        "b0_checkpoint": checkpoints["b0"],
        "r1_config": configs["r1"],
        "r1_checkpoint": checkpoints["r1"],
        "r2_config": configs["r2"],
        "r2_checkpoint": checkpoints["r2"],
    }

    with pytest.raises(ValueError, match="supplied warm-up"):
        experiment._validate_backend_comparison_runs(
            **common,
            warmup_checkpoint_sha256="0" * 64,
        )

    tampered_b0 = copy.deepcopy(checkpoints["b0"])
    tampered_b0["epoch_orders"][1]["consumed_order_sha256"] = "tampered"
    with pytest.raises(ValueError, match="B0 order sequence"):
        experiment._validate_backend_comparison_runs(
            **{**common, "b0_checkpoint": tampered_b0},
            warmup_checkpoint_sha256=warmup_sha256,
        )


def test_backend_evaluator_rejects_non_warmup_source_dir(backend_runs) -> None:
    with pytest.raises(ValueError, match="required warmup condition"):
        experiment.run_luo2022_backend_evaluation(
            run_dir=backend_runs["run_dir"],
            output_dir=backend_runs["root"] / "wrong-warmup-evaluation",
            cache_dir=backend_runs["cache_dir"],
            warmup_dir=backend_runs["b0_dir"],
            b0_dir=backend_runs["b0_dir"],
            r1_dir=backend_runs["r1_dir"],
            r2_dir=backend_runs["r2_dir"],
            device_name="cpu",
            max_eval_batches=1,
        )


def test_resume_rejects_incompatible_batch_size(backend_runs) -> None:
    with pytest.raises(ValueError, match="resume config"):
        experiment.run_luo2022_backend_training(
            run_dir=backend_runs["run_dir"],
            output_dir=backend_runs["b0_dir"],
            cache_dir=backend_runs["cache_dir"],
            condition="b0",
            device_name="cpu",
            small_run=True,
            epochs=2,
            batch_size=1,
            max_updates_per_epoch=1,
            resume=True,
        )


def test_completed_epoch_resume_matches_uninterrupted_training(
    backend_runs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resumed_dir = backend_runs["root"] / "b0-resumed"
    original_save = experiment._save_backend_epoch_checkpoint
    interrupted = False

    def save_then_interrupt(*, output_dir: Path, checkpoint: dict):
        nonlocal interrupted
        saved = original_save(output_dir=output_dir, checkpoint=checkpoint)
        if int(checkpoint["completed_epoch"]) == 1 and not interrupted:
            interrupted = True
            raise RuntimeError("simulated completed-epoch interruption")
        return saved

    monkeypatch.setattr(experiment, "_save_backend_epoch_checkpoint", save_then_interrupt)
    with pytest.raises(RuntimeError, match="simulated completed-epoch interruption"):
        experiment.run_luo2022_backend_training(
            run_dir=backend_runs["run_dir"],
            output_dir=resumed_dir,
            cache_dir=backend_runs["cache_dir"],
            condition="b0",
            device_name="cpu",
            small_run=True,
            epochs=2,
            batch_size=2,
            max_updates_per_epoch=1,
        )
    monkeypatch.setattr(experiment, "_save_backend_epoch_checkpoint", original_save)

    experiment.run_luo2022_backend_training(
        run_dir=backend_runs["run_dir"],
        output_dir=resumed_dir,
        cache_dir=backend_runs["cache_dir"],
        condition="b0",
        device_name="cpu",
        small_run=True,
        epochs=2,
        batch_size=2,
        max_updates_per_epoch=1,
        resume=True,
    )
    uninterrupted = torch.load(
        backend_runs["b0_dir"] / "checkpoints" / "latest.pt",
        map_location="cpu",
        weights_only=True,
    )
    resumed = torch.load(
        resumed_dir / "checkpoints" / "latest.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert resumed["completed_epoch"] == uninterrupted["completed_epoch"] == 2
    assert resumed["generator_update_count"] == uninterrupted["generator_update_count"] == 2
    assert resumed["history"] == uninterrupted["history"]
    assert resumed["epoch_orders"] == uninterrupted["epoch_orders"]
    assert experiment._backend_state_sha256(resumed["generator"]) == experiment._backend_state_sha256(
        uninterrupted["generator"]
    )
    assert experiment._backend_state_sha256(
        resumed["generator_optimizer"]
    ) == experiment._backend_state_sha256(uninterrupted["generator_optimizer"])


def test_small_backend_e2e_writes_complete_metrics_and_fixed_outputs(backend_runs) -> None:
    evaluation_dir = backend_runs["evaluation_dir"]
    metrics = backend_runs["evaluation"]["metrics"]

    assert set(metrics["conditions"]) == {
        "final_epoch_known",
        "seed_disjoint_unseen",
        "no_diffuser",
    }
    assert set(metrics["conditions"]["seed_disjoint_unseen"]) == {
        "R0",
        "B0",
        "R1",
        "R2",
    }
    assert metrics["model_and_budget"]["training_fairness"]["matched"] is True
    models = metrics["model_and_budget"]["models"]
    assert models["B0"]["inference_digital_parameter_count"] == 7_557
    assert models["R1"]["inference_digital_parameter_count"] == 7_557
    assert models["R2"]["inference_digital_parameter_count"] == 7_557
    assert models["R2"]["digital_parameter_count"] == 10_442
    assert models["R2"]["training_only_digital_parameter_count"] == 2_885
    assert models["R2"]["training_total_digital_parameter_count"] == 10_442
    assert models["R2"]["generator_parameter_count"] == 7_557
    assert models["R2"]["discriminator_parameter_count"] == 2_885
    assert models["B0"]["training_only_digital_parameter_count"] == 0
    assert models["R1"]["training_only_digital_parameter_count"] == 0
    assert set(metrics["causal_deltas"]) == {
        "digital_reconstruction_total_gain_R1_minus_R0",
        "optical_frontend_net_contribution_R1_minus_B0",
        "gan_marginal_contribution_R2_minus_R1",
    }
    paired = metrics["causal_delta_inference"][
        "gan_marginal_contribution_R2_minus_R1"
    ]
    assert paired["seed_disjoint_unseen"]["pearson_target_support"][
        "aggregation_unit"
    ] == "matched_diffuser"
    assert paired["no_diffuser"]["pearson_target_support"][
        "aggregation_unit"
    ] == "object"
    target_summary = metrics["conditions"]["seed_disjoint_unseen"]["R2"]["metrics"][
        "pearson_target_support"
    ]
    assert target_summary["distribution"]["aggregation_unit"] == "diffuser"
    assert target_summary["worst_5_percent"]["pair_count"] == 2
    assert metrics["conditions"]["no_diffuser"]["R0"]["metrics"]["psnr"][
        "distribution"
    ]["aggregation_unit"] == "object"
    assert metrics["input_scaling"]["fit_split"] == "train_only"
    assert (evaluation_dir / "per_object_metrics.csv").is_file()
    assert (evaluation_dir / "metrics_comparison.png").is_file()
    assert (evaluation_dir / "cost.png").is_file()
    assert (evaluation_dir / "training_curves.png").is_file()
    assert (evaluation_dir / "model_metadata.json").is_file()
    assert (evaluation_dir / "cost_plot_series.json").is_file()
    assert (evaluation_dir / "training_curve_series.json").is_file()
    assert (evaluation_dir / "samples" / "seed_disjoint_unseen.png").is_file()

    assert experiment.load_config(evaluation_dir / "model_metadata.json") == models
    cost_series = experiment.load_config(evaluation_dir / "cost_plot_series.json")
    assert cost_series["R2"]["inference_digital_parameter_count"] == 7_557
    assert cost_series["R2"]["training_only_digital_parameter_count"] == 2_885
    assert cost_series["R2"]["inference_digital_parameter_count"] != models["R2"][
        "digital_parameter_count"
    ]
    training_series = experiment.load_config(
        evaluation_dir / "training_curve_series.json"
    )
    assert training_series["B0"]["global_epoch"] == [1, 2]
    assert training_series["R1"]["global_epoch"] == [1, 2]
    assert training_series["R2"]["global_epoch"] == [1, 2]
    assert training_series["R1"]["training_l1"][0] == training_series["R2"][
        "training_l1"
    ][0]
    assert training_series["R2"]["gan_continuation"]["global_epoch"] == [2]
    assert set(training_series["R2"]["gan_continuation"]) == {
        "global_epoch",
        "generator_adversarial",
        "generator_total",
        "discriminator_real",
        "discriminator_fake",
        "discriminator_total",
    }
