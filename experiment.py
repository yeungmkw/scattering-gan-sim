"""Main experiment entrypoint for coherent scattering reconstruction.

This file intentionally replaces the earlier pile of one-off scripts. It keeps
one CLI surface for the reusable prototype system:

``d2nn``: inspect the coherent optical path,
``unet``: train the coherent U-Net reconstructor,
``gan``: refine a trained U-Net with conditional PatchGAN,
``compare``: compare U-Net and U-Net+GAN runs,
``full``: run U-Net, GAN, and comparison in sequence.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from coherent_data import build_coherent_mnist_datasets, materialize_coherent_dataset
from d2nn import (
    AngularSpectrumPropagator,
    CoherentOpticsConfig,
    SingleLayerD2NN,
    apply_amplitude_particles,
    apply_phase_screen,
    field_intensity,
    field_phase,
    image_to_complex_field,
    make_amplitude_particles,
    make_random_phase_screen,
)
from data import build_torchvision_dataset
from losses import ReconstructionLossWeights, reconstruction_loss
from metrics import reconstruction_metrics
from patchgan import PatchDiscriminator
from runtime import prepare_output_dir, run_metadata, seed_everything, select_device, write_json
from unet import UNetReconstructor


DEFAULT_SEED = 42
DEFAULT_DATA_ROOT = Path("data")
DEFAULT_IMAGE_INDEX = 0
DEFAULT_IMAGE_SIZE = 64
DEFAULT_BATCH_SIZE = 4
DEFAULT_BASE_CHANNELS = 8
DEFAULT_UNET_LR = 2e-3
DEFAULT_GAN_LR = 2e-4
DEFAULT_TRAIN_LIMIT = 8
DEFAULT_EVAL_LIMIT = 4
DEFAULT_ADVERSARIAL_WEIGHT = 0.01
DEFAULT_SAMPLE_EVERY = 1
LOWER_IS_BETTER = {"l1", "mse"}
HIGHER_IS_BETTER = {"psnr", "ssim", "pearson"}
ORDERED_METRICS = ("l1", "mse", "psnr", "ssim", "pearson")


def coherent_forward_model_metadata(
    corruption: str,
    *,
    optics_config: CoherentOpticsConfig | None = None,
) -> dict[str, Any]:
    """Describe the fixed coherent forward model used by the current CLI."""

    config = optics_config or CoherentOpticsConfig(field_shape=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))
    phase_range = float(2 * torch.pi)
    scattering: dict[str, Any] = {
        "type": corruption,
        "seeded_per_diffuser": True,
    }
    if corruption == "phase":
        scattering["phase_range_radians"] = phase_range
    else:
        scattering.update(
            {
                "num_particles": 12,
                "radius_range_pixels": [2, 6],
                "amplitude_attenuation": 0.15,
            }
        )
    return {
        "optics": {**asdict(config), "field_shape": list(config.field_shape)},
        "scattering": scattering,
        "d2nn": {
            "layers": 1,
            "phase_only": True,
            "trainable": False,
            "phase_range_radians": phase_range,
        },
        "observation_preprocessing": {
            "clean": "input intensity clamped to [0, 1], encoded as zero-phase complex amplitude sqrt(intensity)",
            "dirty_intensity": "per-sample spatial min-max normalization to [0, 1]",
            "d2nn_intensity": "per-sample spatial min-max normalization to [0, 1]",
            "dirty_phase": "wrapped phase mapped from [-pi, pi] to [0, 1] for visualization only",
        },
    }


def experiment_class_for_run(
    *,
    corruption: str,
    train_diffuser_ids: tuple[int, ...] | list[int],
    eval_diffuser_ids: tuple[int, ...] | list[int],
    uses_gan: bool,
) -> str:
    """Map a run to the E0--E3 roadmap labels recorded in its manifest."""

    if corruption == "particles":
        base_class = "E3"
    elif len(set(train_diffuser_ids)) > 1 or diffuser_evaluation_split(train_diffuser_ids, eval_diffuser_ids) != "seen":
        base_class = "E1"
    else:
        base_class = "E0"
    return f"{base_class}+E2" if uses_gan else base_class


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = dispatch(args)
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    d2nn_parser = subparsers.add_parser("d2nn", help="Save one-image coherent path inspection images.")
    d2nn_parser.add_argument("--output-dir", default="outputs/d2nn_inspection")
    d2nn_parser.add_argument("--download", action="store_true")
    d2nn_parser.add_argument("--corruption", choices=("phase", "particles"), default="phase")

    unet_parser = subparsers.add_parser("unet", help="Train the coherent U-Net reconstructor.")
    add_training_args(unet_parser, default_output="outputs/coherent_unet")
    unet_parser.add_argument("--lr", type=float, default=DEFAULT_UNET_LR)

    gan_parser = subparsers.add_parser("gan", help="Train U-Net+PatchGAN coherent refinement.")
    add_training_args(gan_parser, default_output="outputs/coherent_gan")
    gan_parser.add_argument("--lr", type=float, default=DEFAULT_GAN_LR)
    gan_parser.add_argument("--adversarial-weight", type=float, default=DEFAULT_ADVERSARIAL_WEIGHT)
    gan_parser.add_argument("--generator-init", type=Path, default=None)

    compare_parser = subparsers.add_parser("compare", help="Compare U-Net and U-Net+GAN run directories.")
    compare_parser.add_argument("--unet-dir", type=Path, required=True)
    compare_parser.add_argument("--gan-dir", type=Path, required=True)
    compare_parser.add_argument("--output-dir", type=Path, required=True)

    full_parser = subparsers.add_parser("full", help="Run U-Net, GAN, then comparison.")
    add_training_args(full_parser, default_output="outputs/coherent_full")
    full_parser.add_argument("--unet-epochs", type=int, default=20)
    full_parser.add_argument("--gan-epochs", type=int, default=10)
    full_parser.add_argument("--unet-lr", type=float, default=DEFAULT_UNET_LR)
    full_parser.add_argument("--gan-lr", type=float, default=DEFAULT_GAN_LR)
    full_parser.add_argument("--adversarial-weight", type=float, default=DEFAULT_ADVERSARIAL_WEIGHT)

    return parser


def add_training_args(parser: argparse.ArgumentParser, *, default_output: str) -> None:
    parser.add_argument("--output-dir", default=default_output)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--corruption", choices=("phase", "particles"), default="phase")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--base-channels", type=int, default=DEFAULT_BASE_CHANNELS)
    parser.add_argument("--train-limit", type=int, default=DEFAULT_TRAIN_LIMIT)
    parser.add_argument("--eval-limit", type=int, default=DEFAULT_EVAL_LIMIT)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-materialize", action="store_true")
    parser.add_argument("--sample-every", type=int, default=DEFAULT_SAMPLE_EVERY)
    parser.add_argument(
        "--train-diffuser-ids",
        type=int,
        nargs="+",
        metavar="ID",
        default=(0,),
        help="Diffuser IDs used during training. Use multiple IDs for E1.",
    )
    parser.add_argument(
        "--eval-diffuser-ids",
        type=int,
        nargs="+",
        metavar="ID",
        default=(0,),
        help="Diffuser IDs used for evaluation; use IDs disjoint from training for unseen-diffuser E1.",
    )


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "d2nn":
        image, label = load_mnist_image(
            root=DEFAULT_DATA_ROOT,
            image_index=DEFAULT_IMAGE_INDEX,
            image_size=DEFAULT_IMAGE_SIZE,
            download=args.download,
        )
        return run_d2nn_inspection(
            image,
            output_dir=Path(args.output_dir),
            label=label,
            image_index=DEFAULT_IMAGE_INDEX,
            seed=DEFAULT_SEED,
            corruption=args.corruption,
        )
    if args.command == "unet":
        return run_unet_training(
            output_dir=Path(args.output_dir),
            download=args.download,
            corruption=args.corruption,
            device_name=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            base_channels=args.base_channels,
            lr=args.lr,
            train_limit=args.train_limit,
            eval_limit=args.eval_limit,
            max_train_batches=args.max_train_batches,
            max_eval_batches=args.max_eval_batches,
            num_workers=args.num_workers,
            materialize=not args.no_materialize,
            sample_every=args.sample_every,
            train_diffuser_ids=args.train_diffuser_ids,
            eval_diffuser_ids=args.eval_diffuser_ids,
        )
    if args.command == "gan":
        return run_gan_training(
            output_dir=Path(args.output_dir),
            download=args.download,
            corruption=args.corruption,
            device_name=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            base_channels=args.base_channels,
            lr=args.lr,
            train_limit=args.train_limit,
            eval_limit=args.eval_limit,
            adversarial_weight=args.adversarial_weight,
            generator_init=args.generator_init,
            max_train_batches=args.max_train_batches,
            max_eval_batches=args.max_eval_batches,
            num_workers=args.num_workers,
            materialize=not args.no_materialize,
            sample_every=args.sample_every,
            train_diffuser_ids=args.train_diffuser_ids,
            eval_diffuser_ids=args.eval_diffuser_ids,
        )
    if args.command == "compare":
        return compare_runs(args.unet_dir, args.gan_dir, args.output_dir)
    if args.command == "full":
        return run_full_pipeline(args)
    raise ValueError(f"unknown command {args.command}")


def run_full_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    unet_dir = output_dir / "unet"
    gan_dir = output_dir / "gan"
    comparison_dir = output_dir / "comparison"
    unet_result = run_unet_training(
        output_dir=unet_dir,
        download=args.download,
        corruption=args.corruption,
        device_name=args.device,
        epochs=args.unet_epochs,
        batch_size=args.batch_size,
        base_channels=args.base_channels,
        lr=args.unet_lr,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        num_workers=args.num_workers,
        materialize=not args.no_materialize,
        sample_every=args.sample_every,
        train_diffuser_ids=args.train_diffuser_ids,
        eval_diffuser_ids=args.eval_diffuser_ids,
    )
    gan_result = run_gan_training(
        output_dir=gan_dir,
        download=args.download,
        corruption=args.corruption,
        device_name=args.device,
        epochs=args.gan_epochs,
        batch_size=args.batch_size,
        base_channels=args.base_channels,
        lr=args.gan_lr,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        adversarial_weight=args.adversarial_weight,
        generator_init=unet_dir / "checkpoints" / "coherent_unet.pt",
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        num_workers=args.num_workers,
        materialize=not args.no_materialize,
        sample_every=args.sample_every,
        train_diffuser_ids=args.train_diffuser_ids,
        eval_diffuser_ids=args.eval_diffuser_ids,
    )
    comparison = compare_runs(unet_dir, gan_dir, comparison_dir)
    return {
        "unet_metrics": unet_result["metrics"],
        "gan_metrics": gan_result["metrics"],
        "comparison": comparison["metric_comparison"],
        "output_dir": str(output_dir),
    }


def load_mnist_image(
    *,
    root: Path,
    image_index: int,
    image_size: int,
    download: bool,
) -> tuple[torch.Tensor, int]:
    if image_index < 0:
        raise ValueError("image_index must be non-negative")
    dataset = build_torchvision_dataset(
        name="MNIST",
        root=root,
        train=False,
        image_size=image_size,
        download=download,
    )
    image, label = dataset[image_index]
    return image, int(label)


def run_d2nn_inspection(
    image: torch.Tensor,
    *,
    output_dir: Path,
    label: int,
    image_index: int,
    seed: int,
    corruption: str = "phase",
) -> dict[str, Any]:
    """Run the coherent optical inspection path and save images."""

    if corruption not in {"phase", "particles"}:
        raise ValueError("corruption must be 'phase' or 'particles'")
    seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("input.png", "dirty_intensity.png", "dirty_phase.png", "output_intensity.png", "manifest.json"):
        path = output_dir / filename
        if path.exists():
            path.unlink()

    field = image_to_complex_field(image)
    config = CoherentOpticsConfig(field_shape=tuple(field.shape[-2:]))
    if corruption == "phase":
        phase_screen = make_random_phase_screen(config.field_shape, seed=seed + 1)
        scattered_field = apply_phase_screen(field, phase_screen)
    else:
        amplitude_mask = make_amplitude_particles(config.field_shape, seed=seed + 1)
        scattered_field = apply_amplitude_particles(field, amplitude_mask)

    dirty_field = AngularSpectrumPropagator(config).propagate(scattered_field)
    output_field = SingleLayerD2NN(config, seed=seed + 2, trainable=False)(dirty_field)

    save_image(output_dir / "input.png", field_intensity(field)[0])
    save_image(output_dir / "dirty_intensity.png", field_intensity(dirty_field)[0])
    save_phase(output_dir / "dirty_phase.png", field_phase(dirty_field)[0])
    save_image(output_dir / "output_intensity.png", field_intensity(output_field)[0])

    manifest = {
        "status_label": "inspection",
        "experiment_class": "E3-inspection",
        "dataset": "MNIST",
        "image_index": int(image_index),
        "label": int(label),
        "seed": int(seed),
        "corruption": corruption,
        "field_shape": list(config.field_shape),
        "wavelength": config.wavelength,
        "pixel_size": config.pixel_size,
        "propagation_distance": config.propagation_distance,
        "forward_model": coherent_forward_model_metadata(corruption, optics_config=config),
        "runtime": run_metadata(),
        "artifacts": {
            "input": "input.png",
            "dirty_intensity": "dirty_intensity.png",
            "dirty_phase": "dirty_phase.png",
            "output_intensity": "output_intensity.png",
        },
        "physical_effects_included": [
            "zero-phase complex field encoding",
            "random phase screen" if corruption == "phase" else "amplitude particle mask",
            "free-space propagation after corruption",
            "single phase-only D2NN layer",
            "angular-spectrum propagation",
            "output intensity readout",
        ],
        "physical_effects_omitted": coherent_omitted_effects(include_gan=False, inspection=True),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def run_unet_training(
    *,
    output_dir: Path,
    download: bool = False,
    corruption: str = "phase",
    seed: int = DEFAULT_SEED,
    train_limit: int = DEFAULT_TRAIN_LIMIT,
    eval_limit: int = DEFAULT_EVAL_LIMIT,
    device_name: str = "auto",
    epochs: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
    base_channels: int = DEFAULT_BASE_CHANNELS,
    lr: float = DEFAULT_UNET_LR,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
    num_workers: int = 0,
    materialize: bool = True,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    train_diffuser_ids: tuple[int, ...] | list[int] = (0,),
    eval_diffuser_ids: tuple[int, ...] | list[int] = (0,),
) -> dict[str, Any]:
    """Train a coherent U-Net reconstructor from D2NN intensity to clean image."""

    validate_training_inputs(
        corruption,
        epochs,
        batch_size,
        base_channels,
        train_limit,
        eval_limit,
        sample_every,
    )
    seed_everything(seed)
    prepare_output_dir(output_dir)
    train_dataset, eval_dataset = build_coherent_mnist_datasets(
        corruption=corruption,
        seed=seed,
        download=download,
        limit_train=train_limit,
        limit_eval=eval_limit,
        train_diffuser_ids=train_diffuser_ids,
        eval_diffuser_ids=eval_diffuser_ids,
    )
    train_diffuser_ids = list(train_dataset.diffuser_ids)
    eval_diffuser_ids = list(eval_dataset.diffuser_ids)
    d2nn_seed = int(train_dataset.d2nn_seed)
    if materialize:
        train_dataset = materialize_coherent_dataset(train_dataset)
        eval_dataset = materialize_coherent_dataset(eval_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    visualization_batch = cache_visualization_batch(eval_loader)
    device = select_device(device_name)
    model = UNetReconstructor(base_channels=base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    weights = ReconstructionLossWeights(l1=1.0)

    history: list[dict[str, Any]] = []
    eval_metrics: dict[str, float] = {}
    for epoch in range(1, epochs + 1):
        train_metrics = train_unet_one_epoch(
            model,
            train_loader,
            optimizer,
            weights,
            device=device,
            max_batches=max_train_batches,
        )
        eval_metrics = evaluate_reconstructor(model, eval_loader, device=device, max_batches=max_eval_batches)
        if should_save_epoch_sample(epoch, epochs, sample_every):
            save_reconstruction_grid(
                model,
                eval_loader,
                output_dir / "samples" / f"epoch_{epoch:03d}.png",
                device=device,
                batch=visualization_batch,
            )
        history.append({"epoch": epoch, "train": train_metrics, "eval": eval_metrics})
        write_json(output_dir / "history.json", history)

    save_reconstruction_grid(
        model,
        eval_loader,
        output_dir / "samples" / "coherent_reconstruction.png",
        device=device,
        batch=visualization_batch,
    )
    torch.save(model.state_dict(), output_dir / "checkpoints" / "coherent_unet.pt")
    small_sized = epochs == 1 and train_limit <= DEFAULT_TRAIN_LIMIT and eval_limit <= DEFAULT_EVAL_LIMIT
    manifest = {
        "status_label": "small run" if small_sized else "exploratory result",
        "experiment_class": experiment_class_for_run(
            corruption=corruption,
            train_diffuser_ids=train_diffuser_ids,
            eval_diffuser_ids=eval_diffuser_ids,
            uses_gan=False,
        ),
        "dataset": "MNIST",
        "corruption": corruption,
        "model_input": "d2nn_intensity",
        "target": "clean",
        "seed": int(seed),
        "d2nn_seed": d2nn_seed,
        "train_diffuser_ids": train_diffuser_ids,
        "eval_diffuser_ids": eval_diffuser_ids,
        "eval_diffuser_split": diffuser_evaluation_split(train_diffuser_ids, eval_diffuser_ids),
        "train_limit": int(train_limit),
        "eval_limit": int(eval_limit),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "base_channels": int(base_channels),
        "lr": float(lr),
        "device": str(device),
        "materialized_dataset": bool(materialize),
        "sample_every": int(sample_every),
        "loss_weights": asdict(weights),
        "forward_model": coherent_forward_model_metadata(corruption),
        "runtime": run_metadata(),
        "artifacts": {
            "sample_grid": "samples/coherent_reconstruction.png",
            "checkpoint": "checkpoints/coherent_unet.pt",
            "metrics": "metrics.json",
        },
        "physical_effects_included": [
            "zero-phase complex field encoding",
            "random phase screen" if corruption == "phase" else "amplitude particle mask",
            "free-space propagation after corruption",
            "single phase-only D2NN layer",
            "DNN reconstruction from D2NN output intensity",
        ],
        "physical_effects_omitted": ["GAN refinement", *coherent_omitted_effects(include_gan=False)],
    }
    write_json(output_dir / "metrics.json", eval_metrics)
    write_json(output_dir / "manifest.json", manifest)
    return {"history": history, "metrics": eval_metrics, "manifest": manifest}


def run_gan_training(
    *,
    output_dir: Path,
    download: bool = False,
    corruption: str = "phase",
    seed: int = DEFAULT_SEED,
    train_limit: int = DEFAULT_TRAIN_LIMIT,
    eval_limit: int = DEFAULT_EVAL_LIMIT,
    device_name: str = "auto",
    epochs: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
    base_channels: int = DEFAULT_BASE_CHANNELS,
    lr: float = DEFAULT_GAN_LR,
    adversarial_weight: float = DEFAULT_ADVERSARIAL_WEIGHT,
    generator_init: Path | None = None,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
    num_workers: int = 0,
    materialize: bool = True,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    train_diffuser_ids: tuple[int, ...] | list[int] = (0,),
    eval_diffuser_ids: tuple[int, ...] | list[int] = (0,),
) -> dict[str, Any]:
    """Train a conditional PatchGAN refinement stage on coherent observations."""

    validate_training_inputs(
        corruption,
        epochs,
        batch_size,
        base_channels,
        train_limit,
        eval_limit,
        sample_every,
    )
    if adversarial_weight < 0:
        raise ValueError("adversarial_weight must be non-negative")
    seed_everything(seed)
    prepare_output_dir(output_dir)
    train_dataset, eval_dataset = build_coherent_mnist_datasets(
        corruption=corruption,
        seed=seed,
        download=download,
        limit_train=train_limit,
        limit_eval=eval_limit,
        train_diffuser_ids=train_diffuser_ids,
        eval_diffuser_ids=eval_diffuser_ids,
    )
    train_diffuser_ids = list(train_dataset.diffuser_ids)
    eval_diffuser_ids = list(eval_dataset.diffuser_ids)
    d2nn_seed = int(train_dataset.d2nn_seed)
    if materialize:
        train_dataset = materialize_coherent_dataset(train_dataset)
        eval_dataset = materialize_coherent_dataset(eval_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    visualization_batch = cache_visualization_batch(eval_loader)
    device = select_device(device_name)
    generator = UNetReconstructor(base_channels=base_channels).to(device)
    if generator_init is not None:
        generator.load_state_dict(torch.load(generator_init, map_location=device))
    discriminator = PatchDiscriminator(base_channels=base_channels).to(device)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    weights = ReconstructionLossWeights(l1=1.0)

    history: list[dict[str, Any]] = []
    eval_metrics: dict[str, float] = {}
    for epoch in range(1, epochs + 1):
        train_metrics = train_gan_one_epoch(
            generator,
            discriminator,
            train_loader,
            generator_optimizer,
            discriminator_optimizer,
            weights,
            adversarial_weight=adversarial_weight,
            device=device,
            max_batches=max_train_batches,
        )
        eval_metrics = evaluate_generator(generator, eval_loader, device=device, max_batches=max_eval_batches)
        if should_save_epoch_sample(epoch, epochs, sample_every):
            save_gan_grid(
                generator,
                eval_loader,
                output_dir / "samples" / f"epoch_{epoch:03d}.png",
                device=device,
                batch=visualization_batch,
            )
        history.append({"epoch": epoch, "train": train_metrics, "eval": eval_metrics})
        write_json(output_dir / "history.json", history)

    save_gan_grid(
        generator,
        eval_loader,
        output_dir / "samples" / "coherent_gan_reconstruction.png",
        device=device,
        batch=visualization_batch,
    )
    torch.save(generator.state_dict(), output_dir / "checkpoints" / "coherent_gan_generator.pt")
    torch.save(discriminator.state_dict(), output_dir / "checkpoints" / "coherent_gan_discriminator.pt")
    small_sized = epochs == 1 and train_limit <= DEFAULT_TRAIN_LIMIT and eval_limit <= DEFAULT_EVAL_LIMIT
    manifest = {
        "status_label": "small run" if small_sized else "exploratory result",
        "experiment_class": experiment_class_for_run(
            corruption=corruption,
            train_diffuser_ids=train_diffuser_ids,
            eval_diffuser_ids=eval_diffuser_ids,
            uses_gan=True,
        ),
        "dataset": "MNIST",
        "corruption": corruption,
        "model_input": "d2nn_intensity",
        "target": "clean",
        "seed": int(seed),
        "d2nn_seed": d2nn_seed,
        "train_diffuser_ids": train_diffuser_ids,
        "eval_diffuser_ids": eval_diffuser_ids,
        "eval_diffuser_split": diffuser_evaluation_split(train_diffuser_ids, eval_diffuser_ids),
        "train_limit": int(train_limit),
        "eval_limit": int(eval_limit),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "base_channels": int(base_channels),
        "lr": float(lr),
        "device": str(device),
        "materialized_dataset": bool(materialize),
        "sample_every": int(sample_every),
        "forward_model": coherent_forward_model_metadata(corruption),
        "runtime": run_metadata(),
        "generator": "unet_reconstructor",
        "discriminator": "conditional_patchgan",
        "generator_init": str(generator_init) if generator_init is not None else None,
        "loss_weights": {**asdict(weights), "adversarial": adversarial_weight},
        "artifacts": {
            "sample_grid": "samples/coherent_gan_reconstruction.png",
            "generator_checkpoint": "checkpoints/coherent_gan_generator.pt",
            "discriminator_checkpoint": "checkpoints/coherent_gan_discriminator.pt",
            "metrics": "metrics.json",
        },
        "paper_rationale": {
            "coherent_forward_model": "Move beyond random PSF corruption by using a phase screen or particle mask with free-space propagation before the D2NN intensity readout.",
            "gan_refinement": "Use conditional adversarial refinement only after the coherent U-Net path has a measurable supervised baseline.",
        },
        "physical_effects_included": [
            "zero-phase complex field encoding",
            "random phase screen" if corruption == "phase" else "amplitude particle mask",
            "free-space propagation after corruption",
            "single phase-only D2NN layer",
            "DNN reconstruction from D2NN output intensity",
            "conditional adversarial reconstruction refinement",
        ],
        "physical_effects_omitted": coherent_omitted_effects(include_gan=True),
    }
    write_json(output_dir / "metrics.json", eval_metrics)
    write_json(output_dir / "manifest.json", manifest)
    return {"history": history, "metrics": eval_metrics, "manifest": manifest}


def train_unet_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    weights: ReconstructionLossWeights,
    *,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = defaultdict(float)
    count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        source = batch["d2nn_intensity"].to(device)
        target = batch["clean"].to(device)
        prediction = model(source)
        loss, components = reconstruction_loss(prediction, target, weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch_size = int(target.shape[0])
        for name, value in components.items():
            totals[name] += float(value.detach().item()) * batch_size
        count += batch_size
    if count == 0:
        raise ValueError("training loader yielded no batches")
    return {name: value / count for name, value in totals.items()}


def train_gan_one_epoch(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    loader: DataLoader,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    weights: ReconstructionLossWeights,
    *,
    adversarial_weight: float,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    generator.train()
    discriminator.train()
    totals: dict[str, float] = defaultdict(float)
    count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        source = batch["d2nn_intensity"].to(device)
        target = batch["clean"].to(device)

        fake = generator(source)
        real_logits = discriminator(source, target)
        fake_logits = discriminator(source, fake.detach())
        discriminator_loss_real = adversarial_loss(real_logits, target_is_real=True)
        discriminator_loss_fake = adversarial_loss(fake_logits, target_is_real=False)
        discriminator_loss = 0.5 * (discriminator_loss_real + discriminator_loss_fake)
        discriminator_optimizer.zero_grad(set_to_none=True)
        discriminator_loss.backward()
        discriminator_optimizer.step()

        reconstruction_total, reconstruction_components = reconstruction_loss(fake, target, weights)
        set_requires_grad(discriminator, False)
        try:
            generator_adversarial = adversarial_loss(discriminator(source, fake), target_is_real=True)
            generator_loss = reconstruction_total + adversarial_weight * generator_adversarial
            generator_optimizer.zero_grad(set_to_none=True)
            generator_loss.backward()
            generator_optimizer.step()
        finally:
            set_requires_grad(discriminator, True)

        batch_size = int(target.shape[0])
        totals["generator_total"] += float(generator_loss.detach().item()) * batch_size
        totals["discriminator_total"] += float(discriminator_loss.detach().item()) * batch_size
        totals["adversarial"] += float(generator_adversarial.detach().item()) * batch_size
        totals["discriminator_real"] += float(discriminator_loss_real.detach().item()) * batch_size
        totals["discriminator_fake"] += float(discriminator_loss_fake.detach().item()) * batch_size
        for name, value in reconstruction_components.items():
            totals[f"reconstruction_{name}"] += float(value.detach().item()) * batch_size
        count += batch_size
    if count == 0:
        raise ValueError("training loader yielded no batches")
    return {name: value / count for name, value in totals.items()}


def evaluate_reconstructor(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            source = batch["d2nn_intensity"].to(device)
            target = batch["clean"].to(device)
            prediction = model(source)
            metrics = reconstruction_metrics(prediction, target)
            batch_size = int(target.shape[0])
            for name, value in metrics.items():
                totals[name] += value * batch_size
            count += batch_size
    if count == 0:
        raise ValueError("evaluation loader yielded no batches")
    return {name: value / count for name, value in totals.items()}


def evaluate_generator(
    generator: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    return evaluate_reconstructor(generator, loader, device=device, max_batches=max_batches)


def adversarial_loss(logits: torch.Tensor, *, target_is_real: bool) -> torch.Tensor:
    targets = torch.ones_like(logits) if target_is_real else torch.zeros_like(logits)
    return F.binary_cross_entropy_with_logits(logits, targets)


def set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    """Enable or freeze parameter gradients without changing module mode."""

    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def save_reconstruction_grid(
    model: torch.nn.Module,
    loader: DataLoader,
    output_path: Path,
    *,
    device: torch.device,
    max_items: int = 4,
    batch: dict[str, torch.Tensor] | None = None,
) -> None:
    model.eval()
    batch = next(iter(loader)) if batch is None else batch
    clean = batch["clean"][:max_items].to(device)
    dirty_intensity = batch["dirty_intensity"][:max_items].to(device)
    dirty_phase = batch["dirty_phase"][:max_items].to(device)
    d2nn_intensity = batch["d2nn_intensity"][:max_items].to(device)
    with torch.no_grad():
        reconstruction = model(d2nn_intensity).clamp(0.0, 1.0)
    save_coherent_grid(
        [
            ("clean", clean),
            ("dirty intensity", dirty_intensity),
            ("dirty phase", dirty_phase),
            ("D2NN intensity", d2nn_intensity),
            ("reconstruction", reconstruction),
            ("error", (reconstruction - clean).abs()),
        ],
        output_path,
    )


def save_gan_grid(
    generator: torch.nn.Module,
    loader: DataLoader,
    output_path: Path,
    *,
    device: torch.device,
    max_items: int = 4,
    batch: dict[str, torch.Tensor] | None = None,
) -> None:
    generator.eval()
    batch = next(iter(loader)) if batch is None else batch
    clean = batch["clean"][:max_items].to(device)
    dirty_intensity = batch["dirty_intensity"][:max_items].to(device)
    dirty_phase = batch["dirty_phase"][:max_items].to(device)
    d2nn_intensity = batch["d2nn_intensity"][:max_items].to(device)
    with torch.no_grad():
        reconstruction = generator(d2nn_intensity).clamp(0.0, 1.0)
    save_coherent_grid(
        [
            ("clean", clean),
            ("dirty intensity", dirty_intensity),
            ("dirty phase", dirty_phase),
            ("D2NN intensity", d2nn_intensity),
            ("GAN reconstruction", reconstruction),
            ("error", (reconstruction - clean).abs()),
        ],
        output_path,
    )


def cache_visualization_batch(loader: DataLoader) -> dict[str, torch.Tensor]:
    batch = next(iter(loader))
    return {
        name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }


def should_save_epoch_sample(epoch: int, total_epochs: int, sample_every: int) -> bool:
    return epoch == total_epochs or epoch % sample_every == 0


def save_coherent_grid(panels: list[tuple[str, torch.Tensor]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    item_count = panels[0][1].shape[0]
    fig, axes = plt.subplots(len(panels), item_count, figsize=(2.0 * item_count, 10.0))
    if item_count == 1:
        axes = np.expand_dims(axes, axis=1)
    for row, (title, images) in enumerate(panels):
        for col in range(item_count):
            axis = axes[row, col]
            axis.imshow(images[col, 0].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            axis.set_xticks([])
            axis.set_yticks([])
            if col == 0:
                axis.set_ylabel(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def compare_runs(unet_dir: Path, gan_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Write JSON and image artifacts comparing final reconstruction metrics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    unet_metrics = read_json(unet_dir / "metrics.json")
    gan_metrics = read_json(gan_dir / "metrics.json")
    unet_manifest = read_optional_json(unet_dir / "manifest.json")
    gan_manifest = read_optional_json(gan_dir / "manifest.json")
    metric_comparison = compare_metrics(unet_metrics, gan_metrics)
    result = {
        "unet_dir": str(unet_dir),
        "gan_dir": str(gan_dir),
        "unet_manifest": unet_manifest,
        "gan_manifest": gan_manifest,
        "unet_metrics": unet_metrics,
        "gan_metrics": gan_metrics,
        "metric_comparison": metric_comparison,
        "claim_boundary": "Exploratory comparison only; GAN is useful only if visual refinement does not hide worse fidelity metrics.",
        "artifacts": {
            "metrics_plot": "comparison_metrics.png",
            "sample_grid": "comparison_samples.png",
            "comparison_json": "comparison.json",
        },
    }
    write_json(output_dir / "comparison.json", result)
    save_metric_plot(unet_metrics, gan_metrics, output_dir / "comparison_metrics.png")
    save_sample_comparison(unet_dir, gan_dir, output_dir / "comparison_samples.png")
    return result


def compare_metrics(unet_metrics: dict[str, float], gan_metrics: dict[str, float]) -> dict[str, dict[str, Any]]:
    comparison: dict[str, dict[str, Any]] = {}
    for name in ORDERED_METRICS:
        if name not in unet_metrics or name not in gan_metrics:
            continue
        unet_value = float(unet_metrics[name])
        gan_value = float(gan_metrics[name])
        delta = gan_value - unet_value
        if name in LOWER_IS_BETTER:
            gan_better = delta < 0
            direction = "lower_is_better"
        elif name in HIGHER_IS_BETTER:
            gan_better = delta > 0
            direction = "higher_is_better"
        else:
            gan_better = None
            direction = "unknown"
        comparison[name] = {
            "unet": unet_value,
            "gan": gan_value,
            "gan_minus_unet": delta,
            "direction": direction,
            "gan_better": gan_better,
        }
    return comparison


def save_metric_plot(unet_metrics: dict[str, float], gan_metrics: dict[str, float], output_path: Path) -> None:
    metrics = [name for name in ORDERED_METRICS if name in unet_metrics and name in gan_metrics]
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.4 * len(metrics), 3.0))
    if len(metrics) == 1:
        axes = [axes]
    for axis, name in zip(axes, metrics, strict=True):
        axis.bar(["U-Net", "GAN"], [unet_metrics[name], gan_metrics[name]], color=["#4c78a8", "#f58518"])
        axis.set_title(name)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_sample_comparison(unet_dir: Path, gan_dir: Path, output_path: Path) -> None:
    unet_sample = unet_dir / "samples" / "coherent_reconstruction.png"
    gan_sample = gan_dir / "samples" / "coherent_gan_reconstruction.png"
    if not unet_sample.exists() or not gan_sample.exists():
        return
    images = [plt.imread(unet_sample), plt.imread(gan_sample)]
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 16.0))
    for axis, title, image in zip(axes, ("U-Net", "U-Net + PatchGAN"), images, strict=True):
        axis.imshow(image)
        axis.set_title(title)
        axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_image(path: Path, image: torch.Tensor) -> None:
    normalized = normalize_for_display(image)
    plt.imsave(path, normalized.detach().cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)


def save_phase(path: Path, phase: torch.Tensor) -> None:
    normalized = ((phase + torch.pi) / (2 * torch.pi)).clamp(0.0, 1.0)
    plt.imsave(path, normalized.detach().cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)


def normalize_for_display(image: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    image = image.detach().to(dtype=torch.float32)
    low = image.amin()
    high = image.amax()
    return (image - low) / (high - low).clamp_min(eps)


def validate_training_inputs(
    corruption: str,
    epochs: int,
    batch_size: int,
    base_channels: int,
    train_limit: int,
    eval_limit: int,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
) -> None:
    if corruption not in {"phase", "particles"}:
        raise ValueError("corruption must be 'phase' or 'particles'")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if base_channels <= 0:
        raise ValueError("base_channels must be positive")
    if train_limit <= 0 or eval_limit <= 0:
        raise ValueError("train_limit and eval_limit must be positive")
    if sample_every <= 0:
        raise ValueError("sample_every must be positive")


def diffuser_evaluation_split(
    train_diffuser_ids: tuple[int, ...] | list[int],
    eval_diffuser_ids: tuple[int, ...] | list[int],
) -> str:
    """Classify whether evaluation uses seen, unseen, or mixed diffusers."""

    train_ids = set(train_diffuser_ids)
    eval_ids = set(eval_diffuser_ids)
    if eval_ids.issubset(train_ids):
        return "seen"
    if train_ids.isdisjoint(eval_ids):
        return "unseen"
    return "mixed"


def coherent_omitted_effects(*, include_gan: bool, inspection: bool = False) -> list[str]:
    """List omitted effects consistently across coherent manifests."""

    omitted = [
        "PSF calibration",
        "phase-screen material parameters and spatial correlation calibration",
        "sensor noise",
        "detector geometry and calibration",
        "hardware alignment",
        "fabrication constraints",
    ]
    if inspection:
        omitted.insert(0, "training")
    if include_gan:
        omitted.append("optical GAN implementation")
    return omitted


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_optional_json(path: Path) -> Any:
    return read_json(path) if path.exists() else None


if __name__ == "__main__":
    main()
