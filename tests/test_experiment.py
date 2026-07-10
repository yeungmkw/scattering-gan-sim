from pathlib import Path

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
    assert (output_dir / "comparison.json").is_file()
    assert (output_dir / "comparison_metrics.png").is_file()
