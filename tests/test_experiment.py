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
            "--diffuser-chunk-size",
            "2",
        ]
    )

    assert args.action == "evaluate"
    assert args.output_dir == "outputs/frozen_run"
    assert args.posthoc_output_dir == Path("outputs/evidence")
    assert args.posthoc_populations == ["new", "no_diffuser"]
    assert args.diffuser_chunk_size == 2


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

    assert len(per_diffuser) == 3
    assert all(row["object_count"] == 4 for row in per_diffuser)
    for metric in ("total", "negative_pearson", "energy", "pearson"):
        assert sum(float(row[metric]) for row in per_diffuser) / 3 == pytest.approx(
            aggregate[metric],
            abs=1e-6,
        )


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
