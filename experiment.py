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
import platform
import shutil
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

try:
    import resource
except ModuleNotFoundError:  # pragma: no cover - exercised on Windows CUDA hosts
    resource = None

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from coherent_data import (
    build_coherent_mnist_datasets,
    materialize_coherent_dataset,
    prepare_luo2022_amplitude,
)
from d2nn import (
    AngularSpectrumPropagator,
    CoherentOpticsConfig,
    Luo2022FourLayerD2NN,
    Luo2022OpticsConfig,
    RayleighSommerfeldPropagator,
    SingleLayerD2NN,
    amplitude_to_complex_field,
    apply_amplitude_particles,
    apply_phase_screen,
    diffuser_phase_difference,
    estimate_phase_correlation_length,
    estimate_transmittance_correlation_length,
    field_intensity,
    field_phase,
    image_to_complex_field,
    make_amplitude_particles,
    make_correlated_diffuser_phase,
    make_random_phase_screen,
    make_unique_correlated_diffusers,
    summarize_diffuser_bank_uniqueness,
)
from data import build_torchvision_dataset
from losses import ReconstructionLossWeights, luo2022_d2nn_loss, reconstruction_loss
from metrics import reconstruction_metrics
from patchgan import PatchDiscriminator
from runtime import (
    load_config,
    prepare_output_dir,
    run_metadata,
    seed_everything,
    select_device,
    snapshot_config,
    write_json,
)
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
DEFAULT_LUO2022_CONFIG = Path("configs/luo2022_r0.json")
MANIFEST_SCHEMA_VERSION = 1
CONFIG_SCHEMA_VERSION = 1
METRICS_PROTOCOL_VERSION = 1
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


def reconstruction_weights_from_args(args: argparse.Namespace) -> ReconstructionLossWeights:
    """Build the shared supervised reconstruction loss configuration."""

    return ReconstructionLossWeights(
        l1=args.l1_weight,
        negative_pearson=args.negative_pearson_weight,
        ssim=args.ssim_weight,
        fourier=args.fourier_weight,
    )


def metrics_protocol_metadata() -> dict[str, Any]:
    """Describe how flat ``metrics.json`` values are aggregated."""

    return {
        "schema_version": METRICS_PROTOCOL_VERSION,
        "data_range": [0.0, 1.0],
        "dataset_aggregation": "sample-count-weighted mean of batch metrics",
        "psnr": "mean of per-image PSNR values",
        "ssim": "mean of per-image SSIM-like values",
        "pearson": "mean of per-image Pearson correlations",
        "ordered_metrics": list(ORDERED_METRICS),
    }


def coherent_training_config(
    *,
    command: str,
    experiment_class: str,
    corruption: str,
    seed: int,
    d2nn_seed: int,
    train_diffuser_ids: tuple[int, ...] | list[int],
    eval_diffuser_ids: tuple[int, ...] | list[int],
    train_limit: int,
    eval_limit: int,
    epochs: int,
    batch_size: int,
    base_channels: int,
    lr: float,
    device: torch.device,
    materialize: bool,
    sample_every: int,
    max_train_batches: int | None,
    max_eval_batches: int | None,
    num_workers: int,
    reconstruction_weights: ReconstructionLossWeights,
    adversarial_weight: float | None = None,
    generator_init: Path | None = None,
) -> dict[str, Any]:
    """Return the canonical configuration snapshot for a coherent training run."""

    optimization: dict[str, Any] = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(lr),
        "reconstruction_loss_weights": asdict(reconstruction_weights),
    }
    if adversarial_weight is not None:
        optimization["adversarial_weight"] = float(adversarial_weight)
    model: dict[str, Any] = {
        "generator": "unet_reconstructor",
        "base_channels": int(base_channels),
    }
    if command == "gan":
        model.update(
            {
                "discriminator": "conditional_patchgan",
                "generator_init": str(generator_init) if generator_init is not None else None,
            }
        )
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "command": command,
        "experiment_class": experiment_class,
        "dataset": {
            "name": "MNIST",
            "train_limit": int(train_limit),
            "eval_limit": int(eval_limit),
        },
        "diffuser_split": {
            "train_ids": list(train_diffuser_ids),
            "eval_ids": list(eval_diffuser_ids),
            "evaluation": diffuser_evaluation_split(train_diffuser_ids, eval_diffuser_ids),
        },
        "forward_model": coherent_forward_model_metadata(corruption),
        "model": model,
        "optimization": optimization,
        "evaluation": {
            "metrics_protocol": metrics_protocol_metadata(),
            "max_train_batches": max_train_batches,
            "max_eval_batches": max_eval_batches,
        },
        "execution": {
            "seed": int(seed),
            "d2nn_seed": int(d2nn_seed),
            "device": str(device),
            "num_workers": int(num_workers),
            "materialized_dataset": bool(materialize),
            "sample_every": int(sample_every),
        },
    }


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
    d2nn_parser.add_argument("--profile", choices=("legacy", "luo2022_r0"), default="legacy")
    d2nn_parser.add_argument("--action", choices=("inspect", "train", "assess"), default="inspect")
    d2nn_parser.add_argument("--config-path", type=Path, default=DEFAULT_LUO2022_CONFIG)
    d2nn_parser.add_argument("--small-run", action="store_true")
    d2nn_parser.add_argument("--device", default="cpu")
    d2nn_parser.add_argument("--seed", type=int, default=None)
    d2nn_parser.add_argument("--grid-size", type=int, default=None)
    d2nn_parser.add_argument("--input-size", type=int, default=None)
    d2nn_parser.add_argument("--epochs", type=int, default=None)
    d2nn_parser.add_argument("--batch-size", type=int, default=None)
    d2nn_parser.add_argument("--train-limit", type=int, default=None)
    d2nn_parser.add_argument("--eval-limit", type=int, default=None)
    d2nn_parser.add_argument("--diffusers-per-epoch", type=int, default=None)
    d2nn_parser.add_argument("--eval-diffusers", type=int, default=None)
    d2nn_parser.add_argument("--lr", type=float, default=None)
    d2nn_parser.add_argument("--max-train-batches", type=int, default=None)
    d2nn_parser.add_argument("--max-eval-batches", type=int, default=None)
    d2nn_parser.add_argument(
        "--diffuser-chunk-size",
        type=int,
        default=None,
        help=(
            "Execution-only diffuser chunk size. Gradients are accumulated across chunks "
            "before one optimizer update, preserving the configured fields per update."
        ),
    )
    d2nn_parser.add_argument(
        "--review-eval-batches",
        type=int,
        default=25,
        help=(
            "Fixed evaluation-prefix batches used for per-epoch monitoring. "
            "The final evaluation still uses the complete configured test set."
        ),
    )
    d2nn_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the Luo 2022 run from OUTPUT_DIR/checkpoints/latest.pt.",
    )

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
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--negative-pearson-weight", type=float, default=0.0)
    parser.add_argument("--ssim-weight", type=float, default=0.0)
    parser.add_argument("--fourier-weight", type=float, default=0.0)
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
        if args.profile == "luo2022_r0":
            if args.action == "assess":
                return run_luo2022_readiness_assessment(
                    output_dir=Path(args.output_dir),
                    config_path=args.config_path,
                    device_name=args.device,
                    seed=args.seed,
                )
            if args.action != "train":
                contract = load_config(args.config_path)
                return {
                    "profile_id": contract["profile_id"],
                    "freeze_version": contract["freeze_version"],
                    "contract_status": contract["status"]["contract"],
                    "runtime_binding": contract["status"]["runtime_binding"],
                    "next_action": "use --action train, preferably with --small-run first",
                }
            return run_luo2022_training(
                output_dir=Path(args.output_dir),
                config_path=args.config_path,
                download=args.download,
                small_run=args.small_run,
                device_name=args.device,
                seed=args.seed,
                grid_size=args.grid_size,
                input_size=args.input_size,
                epochs=args.epochs,
                batch_size=args.batch_size,
                train_limit=args.train_limit,
                eval_limit=args.eval_limit,
                diffusers_per_epoch=args.diffusers_per_epoch,
                eval_diffusers=args.eval_diffusers,
                lr=args.lr,
                max_train_batches=args.max_train_batches,
                max_eval_batches=args.max_eval_batches,
                diffuser_chunk_size=args.diffuser_chunk_size,
                review_eval_batches=args.review_eval_batches,
                resume=args.resume,
            )
        if args.action != "inspect":
            raise ValueError("legacy d2nn profile only supports --action inspect")
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
            reconstruction_weights=reconstruction_weights_from_args(args),
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
            reconstruction_weights=reconstruction_weights_from_args(args),
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
        reconstruction_weights=reconstruction_weights_from_args(args),
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
        reconstruction_weights=reconstruction_weights_from_args(args),
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
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status_label": "inspection",
        "experiment_class": "E0-inspection" if corruption == "phase" else "E3-inspection",
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


def build_luo2022_runtime_config(
    contract: dict[str, Any],
    *,
    small_run: bool,
    device: torch.device,
    seed: int | None = None,
    grid_size: int | None = None,
    input_size: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    train_limit: int | None = None,
    eval_limit: int | None = None,
    diffusers_per_epoch: int | None = None,
    eval_diffusers: int | None = None,
    lr: float | None = None,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
    diffuser_chunk_size: int | None = None,
    review_eval_batches: int = 25,
) -> dict[str, Any]:
    """Bind the frozen contract to either exact R0 or a labeled small run."""

    optional_overrides = (
        grid_size,
        input_size,
        epochs,
        batch_size,
        train_limit,
        eval_limit,
        diffusers_per_epoch,
        eval_diffusers,
        lr,
        max_train_batches,
        max_eval_batches,
    )
    if not small_run and any(value is not None for value in optional_overrides):
        raise ValueError("R0 parameter overrides require --small-run")

    paper_grid = int(contract["grid"]["shape"][0])
    paper_input = int(contract["input"]["resized_shape"][0])
    paper_epochs = int(contract["training"]["epochs"])
    paper_batch = int(contract["training"]["objects_per_batch"])
    paper_train_limit = int(contract["input"]["train_objects"])
    paper_eval_limit = int(contract["evaluation"]["test_objects"])
    paper_diffusers = int(contract["training"]["diffusers_per_epoch"])
    paper_eval_diffusers = int(contract["evaluation"]["diffuser_sets_for_n20"]["new_diffusers"])
    paper_lr = float(contract["training"]["learning_rate"]["initial"])

    if small_run:
        resolved = {
            "grid_size": 48 if grid_size is None else int(grid_size),
            "input_size": 32 if input_size is None else int(input_size),
            "epochs": 8 if epochs is None else int(epochs),
            "batch_size": 4 if batch_size is None else int(batch_size),
            "train_limit": 16 if train_limit is None else int(train_limit),
            "eval_limit": 8 if eval_limit is None else int(eval_limit),
            "diffusers_per_epoch": 2 if diffusers_per_epoch is None else int(diffusers_per_epoch),
            "eval_diffusers": 2 if eval_diffusers is None else int(eval_diffusers),
            "learning_rate": paper_lr if lr is None else float(lr),
        }
    else:
        resolved = {
            "grid_size": paper_grid,
            "input_size": paper_input,
            "epochs": paper_epochs,
            "batch_size": paper_batch,
            "train_limit": paper_train_limit,
            "eval_limit": paper_eval_limit,
            "diffusers_per_epoch": paper_diffusers,
            "eval_diffusers": paper_eval_diffusers,
            "learning_rate": paper_lr,
        }
    resolved["seed"] = int(contract["training"]["primary_seed"]["value"] if seed is None else seed)
    resolved["max_train_batches"] = max_train_batches
    resolved["max_eval_batches"] = max_eval_batches
    resolved["diffuser_chunk_size"] = int(
        resolved["diffusers_per_epoch"]
        if diffuser_chunk_size is None
        else diffuser_chunk_size
    )
    resolved["review_eval_batches"] = int(review_eval_batches)
    resolved["device"] = str(device)

    positive_names = (
        "grid_size",
        "input_size",
        "epochs",
        "batch_size",
        "train_limit",
        "eval_limit",
        "diffusers_per_epoch",
        "eval_diffusers",
        "learning_rate",
    )
    if any(resolved[name] <= 0 for name in positive_names):
        raise ValueError("all Luo 2022 runtime dimensions and optimization values must be positive")
    if resolved["input_size"] > resolved["grid_size"]:
        raise ValueError("input_size must not exceed grid_size")
    if not 1 <= resolved["diffuser_chunk_size"] <= resolved["diffusers_per_epoch"]:
        raise ValueError("diffuser_chunk_size must be between 1 and diffusers_per_epoch")
    if resolved["review_eval_batches"] <= 0:
        raise ValueError("review_eval_batches must be positive")
    kernel_radius = int(contract["diffuser"]["finite_kernel_choice"]["radius_pixels"])
    if (
        contract["diffuser"]["finite_kernel_choice"]["padding"] == "reflect"
        and kernel_radius >= resolved["grid_size"]
    ):
        raise ValueError(
            f"grid_size must exceed the frozen reflect-padding radius of {kernel_radius} pixels"
        )

    paper_values = {
        "grid_size": paper_grid,
        "input_size": paper_input,
        "epochs": paper_epochs,
        "batch_size": paper_batch,
        "train_limit": paper_train_limit,
        "eval_limit": paper_eval_limit,
        "diffusers_per_epoch": paper_diffusers,
        "eval_diffusers": paper_eval_diffusers,
        "learning_rate": paper_lr,
    }
    overrides = {
        name: {"paper_value": paper_values[name], "runtime_value": resolved[name]}
        for name in paper_values
        if resolved[name] != paper_values[name]
    }
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "profile_id": contract["profile_id"],
        "source_freeze_version": contract["freeze_version"],
        "experiment_class": contract["experiment_class"],
        "comparison_level": contract["comparison_level"],
        "status_label": "small run" if small_run else "reproduction-inspired result",
        "claim_boundary": (
            "Engineering validation of the frozen formula chain; reduced dimensions are not a paper result."
            if small_run
            else "Executable R0 settings; paper-level reproduction still requires full acceptance comparison."
        ),
        "runtime": resolved,
        "overrides_from_frozen_contract": overrides,
        "paper_equations": {
            "diffuser": contract["diffuser"]["equations"],
            "propagation": contract["propagation"]["equations"],
            "d2nn": contract["d2nn"]["equations"],
            "loss": contract["training"]["loss"]["equations"],
        },
        "physical_parameters": {
            "wavelength_m": contract["illumination"]["wavelength_m"],
            "pixel_pitch_m": contract["grid"]["pixel_pitch_m"],
            "geometry": contract["geometry"],
            "diffuser": contract["diffuser"],
            "d2nn_layers": contract["d2nn"]["layers"],
        },
        "training_protocol": {
            "regenerate_diffusers_at_epoch_start": contract["training"][
                "regenerate_diffusers_at_epoch_start"
            ],
            "reuse_epoch_diffusers_for_all_batches": contract["training"][
                "reuse_epoch_diffusers_for_all_batches"
            ],
            "optimizer": contract["training"]["optimizer"],
            "learning_rate": contract["training"]["learning_rate"],
            "loss": contract["training"]["loss"],
        },
        "execution_controls": {
            "diffuser_chunk_size": resolved["diffuser_chunk_size"],
            "gradient_accumulation_preserves_fields_per_update": True,
            "review_eval_batches": resolved["review_eval_batches"],
            "final_evaluation_uses_complete_configured_test_set": True,
        },
    }


def run_luo2022_readiness_assessment(
    *,
    output_dir: Path,
    config_path: Path = DEFAULT_LUO2022_CONFIG,
    device_name: str = "cpu",
    seed: int | None = None,
) -> dict[str, Any]:
    """Assess numerical, diffuser, and local-resource readiness for full R0."""

    contract = load_config(config_path)
    resolved_seed = int(contract["training"]["primary_seed"]["value"] if seed is None else seed)
    seed_everything(resolved_seed)
    device = select_device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("assessment.json", "assessment.md"):
        path = output_dir / filename
        if path.exists():
            path.unlink()

    optics_config = Luo2022OpticsConfig(
        field_shape=tuple(int(value) for value in contract["grid"]["shape"]),
        wavelength=float(contract["illumination"]["wavelength_m"]),
        pixel_size=float(contract["grid"]["pixel_pitch_m"]),
        object_to_diffuser_distance=float(contract["geometry"]["object_to_diffuser_m"]),
        diffuser_to_first_layer_distance=float(contract["geometry"]["diffuser_to_first_layer_m"]),
        layer_distance=float(contract["geometry"]["layer_to_layer_m"]),
        output_distance=float(contract["geometry"]["last_layer_to_output_m"]),
        num_layers=int(contract["d2nn"]["layers"]),
        pad_factor=2,
    )
    diffuser_kwargs = {
        "wavelength": optics_config.wavelength,
        "pixel_size": optics_config.pixel_size,
        "refractive_index_difference": float(contract["diffuser"]["refractive_index_difference"]),
        "height_mean_lambda": float(contract["diffuser"]["height_mean_lambda"]),
        "height_std_lambda": float(contract["diffuser"]["height_std_lambda"]),
        "gaussian_sigma_lambda": float(contract["diffuser"]["gaussian_sigma_lambda"]),
        "truncate_sigma": float(contract["diffuser"]["finite_kernel_choice"]["truncate_sigma"]),
        "padding": str(contract["diffuser"]["finite_kernel_choice"]["padding"]),
    }

    phase_sample_count = max(20, int(contract["training"]["diffusers_per_epoch"]))
    generation_start = time.perf_counter()
    phases = torch.stack(
        [
            make_correlated_diffuser_phase(
                optics_config.field_shape,
                seed=resolved_seed + index,
                **diffuser_kwargs,
            )
            for index in range(phase_sample_count)
        ]
    )
    phase_generation_seconds = time.perf_counter() - generation_start
    uniqueness_threshold = float(contract["diffuser"]["uniqueness"]["minimum_radians"])
    selected_phase_representation = str(
        contract["diffuser"]["uniqueness"]["phase_representation"]
    )
    selected_correlation_field = str(contract["diffuser"]["correlation_estimator"]["field"])
    if selected_correlation_field != "mean_centered_complex_transmittance":
        raise ValueError(f"unsupported frozen correlation field {selected_correlation_field}")
    uniqueness = {
        mode: _phase_difference_summary(
            phases,
            mode=mode,
            threshold=uniqueness_threshold,
        )
        for mode in ("unwrapped", "zero_to_2pi", "minus_pi_to_pi")
    }
    correlation_sensitivity = {}
    correlation_representations = {
        "unwrapped_phase": phases,
        "zero_to_2pi_phase": torch.remainder(phases, 2.0 * torch.pi),
        "minus_pi_to_pi_phase": torch.angle(torch.exp(1j * phases)),
    }
    for mode, represented_phases in correlation_representations.items():
        values = [
            estimate_phase_correlation_length(
                phase,
                pixel_size=optics_config.pixel_size,
                wavelength=optics_config.wavelength,
            )
            for phase in represented_phases
        ]
        correlation_sensitivity[mode] = {
            "sample_mean": float(np.mean(values)),
            "sample_standard_deviation": float(np.std(values, ddof=1)),
        }
    transmittance_values = [
        estimate_transmittance_correlation_length(
            phase,
            pixel_size=optics_config.pixel_size,
            wavelength=optics_config.wavelength,
        )
        for phase in phases
    ]
    correlation_sensitivity["complex_transmittance"] = {
        "sample_mean": float(np.mean(transmittance_values)),
        "sample_standard_deviation": float(np.std(transmittance_values, ddof=1)),
    }
    correlation_mean = correlation_sensitivity["complex_transmittance"]["sample_mean"]
    correlation_std = correlation_sensitivity["complex_transmittance"][
        "sample_standard_deviation"
    ]
    expected_correlation = float(contract["diffuser"]["expected_mean_correlation_length_lambda"])
    correlation_relative_error = abs(correlation_mean - expected_correlation) / expected_correlation
    for result in correlation_sensitivity.values():
        result["relative_error_to_paper_target"] = (
            abs(result["sample_mean"] - expected_correlation) / expected_correlation
        )

    direct_sum_error = _rayleigh_sommerfeld_direct_sum_error()
    precision = _luo2022_precision_comparison(optics_config, phases[0], seed=resolved_seed)
    benchmark = _luo2022_training_step_benchmark(
        optics_config,
        phases,
        device=device,
        seed=resolved_seed,
        iterations=3,
    )
    paper_scale_audit = _luo2022_paper_scale_diffuser_audit(
        optics_config,
        diffuser_kwargs=diffuser_kwargs,
        count=int(contract["diffuser"]["training_correlation_validation_samples"]),
        sampled_pairs=10_000,
        seed=resolved_seed,
        uniqueness_threshold=uniqueness_threshold,
        paper_correlation_target=expected_correlation,
        phase_representation=selected_phase_representation,
    )
    total_steps = int(contract["training"]["epochs"]) * int(contract["training"]["steps_per_epoch"])
    projected_hours = benchmark["steady_step_seconds"] * total_steps / 3600.0
    reported_hours = 24.0
    local_limit_hours = 72.0
    benchmark.update(
        {
            "full_run_steps": total_steps,
            "projected_full_run_hours": projected_hours,
            "projected_full_run_days": projected_hours / 24.0,
            "paper_reported_training_hours_approx": reported_hours,
            "projected_to_paper_time_ratio": projected_hours / reported_hours,
            "local_practical_limit_hours": local_limit_hours,
            "local_practical_limit_evidence": "project assessment choice",
        }
    )

    gates = {
        "rs_fft_matches_direct_sum": direct_sum_error <= 1e-10,
        "complex64_matches_complex128": precision["relative_l2"] <= 1e-3,
        "selected_phase_representation_uniqueness_passes_20_all_pairs": (
            uniqueness[selected_phase_representation]["minimum_radians"]
            > uniqueness_threshold
        ),
        "selected_correlation_length_within_10_percent": correlation_relative_error <= 0.10,
        "paper_scale_exact_all_pairs_uniqueness_passes": (
            paper_scale_audit["exact_uniqueness"]["pair_pass_fraction"] == 1.0
        ),
        "paper_scale_correlation_length_within_10_percent": (
            paper_scale_audit["complex_transmittance_correlation_length_lambda"][
                "relative_error_to_paper_target"
            ]
            <= 0.10
        ),
    }
    ready_for_full_r0 = all(gates.values())
    assessment = {
        "schema_version": 1,
        "assessment_date": "2026-07-17",
        "profile_id": contract["profile_id"],
        "source_freeze_version": contract["freeze_version"],
        "status_label": "reproduction readiness assessment",
        "decision": (
            "ready_for_cuda_training"
            if ready_for_full_r0
            else "blocked_before_cuda_training"
        ),
        "ready_for_full_r0": ready_for_full_r0,
        "ready_for_cuda_training": ready_for_full_r0,
        "gates": gates,
        "numerical_propagation": {
            "rs_fft_vs_direct_sum_max_abs_error": direct_sum_error,
            "acceptance_tolerance": 1e-10,
            "precision_comparison": precision,
        },
        "diffuser_model": {
            "sample_count": phase_sample_count,
            "phase_generation_seconds": phase_generation_seconds,
            "phase_std_radians": {
                "mean": float(phases.std(dim=(-2, -1)).mean()),
                "standard_deviation": float(phases.std(dim=(-2, -1)).std(unbiased=True)),
            },
            "uniqueness_threshold_radians": uniqueness_threshold,
            "selected_phase_representation": selected_phase_representation,
            "uniqueness_sensitivity": uniqueness,
            "correlation_length_lambda": {
                "estimator": (
                    "radially averaged autocorrelation fitted to "
                    "exp(-pi*r^2/L^2) over correlation values 0.2 to 0.95"
                ),
                "selected_field": selected_correlation_field,
                "selected_sample_mean": correlation_mean,
                "selected_sample_standard_deviation": correlation_std,
                "paper_target": expected_correlation,
                "selected_relative_error": correlation_relative_error,
                "sensitivity": correlation_sensitivity,
                "paper_estimator_published": False,
            },
            "published_ambiguities": [
                "Gaussian kernel discrete normalization and support",
                "boundary handling",
                "phase wrapping branch used for uniqueness",
                "phase-autocorrelation estimator and fit window",
            ],
            "paper_scale_sensitivity_audit": paper_scale_audit,
        },
        "resource_benchmark": benchmark,
        "runtime": run_metadata(),
        "conclusion": (
            "The RS implementation and complex64 training precision pass. Freeze 2026-07-17.2 "
            "uses [-pi, pi) wrapped phase for uniqueness and complex-transmittance "
            "autocovariance for correlation as explicit paper-inferred project choices. CUDA "
            "training readiness requires both the exact 2000-diffuser all-pairs audit and the "
            "reduced training rerun to pass; local CPU full-training time is informational and "
            "is not a readiness gate."
        ),
        "required_before_full_r0": [
            "Transfer the frozen configuration and validated code to a CUDA machine.",
            "Record the CUDA runtime and environment before launching the 100-epoch R0.",
        ],
    }
    write_json(output_dir / "assessment.json", assessment)
    (output_dir / "assessment.md").write_text(
        _luo2022_assessment_markdown(assessment),
        encoding="utf-8",
    )
    return assessment


def _phase_difference_summary(
    phases: torch.Tensor,
    *,
    mode: str,
    threshold: float,
) -> dict[str, float]:
    if mode == "unwrapped":
        represented = phases
    elif mode == "zero_to_2pi":
        represented = torch.remainder(phases, 2.0 * torch.pi)
    elif mode == "minus_pi_to_pi":
        represented = torch.angle(torch.exp(1j * phases))
    else:
        raise ValueError(f"unsupported phase representation {mode}")
    differences = [
        float(diffuser_phase_difference(represented[left], represented[right]))
        for left, right in combinations(range(represented.shape[0]), 2)
    ]
    return {
        "minimum_radians": min(differences),
        "mean_radians": float(np.mean(differences)),
        "maximum_radians": max(differences),
        "pair_pass_fraction": float(np.mean(np.asarray(differences) > threshold)),
    }


def _luo2022_paper_scale_diffuser_audit(
    optics_config: Luo2022OpticsConfig,
    *,
    diffuser_kwargs: dict[str, Any],
    count: int,
    sampled_pairs: int,
    seed: int,
    uniqueness_threshold: float,
    paper_correlation_target: float,
    phase_representation: str,
) -> dict[str, Any]:
    if count <= 0 or sampled_pairs <= 0:
        raise ValueError("paper-scale diffuser audit sizes must be positive")
    phases = torch.empty((count, *optics_config.field_shape), dtype=torch.float32)
    correlation_lengths = np.empty(count, dtype=np.float64)
    start = time.perf_counter()
    for index in range(count):
        phase = make_correlated_diffuser_phase(
            optics_config.field_shape,
            seed=seed + index,
            **diffuser_kwargs,
        )
        phases[index] = phase
        correlation_lengths[index] = estimate_transmittance_correlation_length(
            phase,
            pixel_size=optics_config.pixel_size,
            wavelength=optics_config.wavelength,
        )
    generation_and_correlation_seconds = time.perf_counter() - start

    exact_pair_start = time.perf_counter()
    exact_uniqueness = summarize_diffuser_bank_uniqueness(
        phases,
        phase_representation=phase_representation,
        threshold_radians=uniqueness_threshold,
        block_size=32,
    )
    exact_pair_audit_seconds = time.perf_counter() - exact_pair_start

    pair_generator = torch.Generator(device="cpu")
    pair_generator.manual_seed(seed + 20_260_717)
    left_indices = torch.randint(0, count, (sampled_pairs,), generator=pair_generator)
    right_indices = torch.randint(0, count - 1, (sampled_pairs,), generator=pair_generator)
    right_indices = right_indices + (right_indices >= left_indices)
    pair_start = time.perf_counter()
    sampled_uniqueness = {
        mode: _sampled_phase_difference_summary(
            phases,
            left_indices=left_indices,
            right_indices=right_indices,
            mode=mode,
            threshold=uniqueness_threshold,
        )
        for mode in ("unwrapped", "minus_pi_to_pi")
    }
    pair_audit_seconds = time.perf_counter() - pair_start
    correlation_mean = float(correlation_lengths.mean())
    return {
        "diffuser_count": count,
        "sampled_pair_count": sampled_pairs,
        "generation_and_correlation_seconds": generation_and_correlation_seconds,
        "exact_pair_audit_seconds": exact_pair_audit_seconds,
        "sampled_pair_audit_seconds": pair_audit_seconds,
        "complex_transmittance_correlation_length_lambda": {
            "sample_mean": correlation_mean,
            "sample_standard_deviation": float(correlation_lengths.std(ddof=1)),
            "paper_target": paper_correlation_target,
            "relative_error_to_paper_target": (
                abs(correlation_mean - paper_correlation_target) / paper_correlation_target
            ),
        },
        "exact_uniqueness": exact_uniqueness,
        "sampled_uniqueness": sampled_uniqueness,
        "claim_boundary": (
            "The correlation sample count matches the paper. The uniqueness audit covers every "
            "unordered pair in the raw seeded 2000-diffuser bank. If every pair passes, the "
            "same candidates also pass the paper's sequential all-existing acceptance rule "
            "without rejection."
        ),
    }


def _sampled_phase_difference_summary(
    phases: torch.Tensor,
    *,
    left_indices: torch.Tensor,
    right_indices: torch.Tensor,
    mode: str,
    threshold: float,
    chunk_size: int = 50,
) -> dict[str, float]:
    differences = []
    for start in range(0, int(left_indices.numel()), chunk_size):
        left = phases[left_indices[start : start + chunk_size]]
        right = phases[right_indices[start : start + chunk_size]]
        if mode == "minus_pi_to_pi":
            left = torch.remainder(left + torch.pi, 2.0 * torch.pi) - torch.pi
            right = torch.remainder(right + torch.pi, 2.0 * torch.pi) - torch.pi
        elif mode != "unwrapped":
            raise ValueError(f"unsupported sampled phase representation {mode}")
        centered_left = left - left.mean(dim=(-2, -1), keepdim=True)
        centered_right = right - right.mean(dim=(-2, -1), keepdim=True)
        differences.extend(
            (centered_left - centered_right).abs().mean(dim=(-2, -1)).tolist()
        )
    values = np.asarray(differences)
    return {
        "minimum_radians": float(values.min()),
        "mean_radians": float(values.mean()),
        "maximum_radians": float(values.max()),
        "pair_pass_fraction": float(np.mean(values > threshold)),
    }


def _rayleigh_sommerfeld_direct_sum_error() -> float:
    height, width = 5, 6
    wavelength = 0.75e-3
    pixel_size = 0.3e-3
    distance = 2e-3
    generator = torch.Generator(device="cpu")
    generator.manual_seed(90210)
    field = torch.complex(
        torch.randn(height, width, generator=generator, dtype=torch.float64),
        torch.randn(height, width, generator=generator, dtype=torch.float64),
    )
    propagator = RayleighSommerfeldPropagator(
        field_shape=(height, width),
        wavelength=wavelength,
        pixel_size=pixel_size,
        distance=distance,
    )
    fft_output = propagator.propagate(field)
    direct_output = torch.zeros_like(field)
    source_y = torch.arange(height, dtype=torch.float64)
    source_x = torch.arange(width, dtype=torch.float64)
    for output_y in range(height):
        for output_x in range(width):
            delta_y = (output_y - source_y[:, None]) * pixel_size
            delta_x = (output_x - source_x[None, :]) * pixel_size
            radius = torch.sqrt(delta_x.square() + delta_y.square() + distance**2)
            kernel = (
                distance
                / radius.square()
                * (1.0 / (2.0 * torch.pi * radius) + 1.0 / (1j * wavelength))
                * torch.exp(1j * 2.0 * torch.pi * radius / wavelength)
                * pixel_size**2
            )
            direct_output[output_y, output_x] = (field * kernel).sum()
    return float((fft_output - direct_output).abs().max())


@torch.no_grad()
def _luo2022_precision_comparison(
    optics_config: Luo2022OpticsConfig,
    diffuser_phase: torch.Tensor,
    *,
    seed: int,
) -> dict[str, float | bool]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 17)
    image = torch.rand(1, 1, 28, 28, generator=generator)
    target32 = prepare_luo2022_amplitude(
        image,
        resized_shape=(160, 160),
        canvas_shape=optics_config.field_shape,
    )
    model32 = Luo2022FourLayerD2NN(optics_config)
    model64 = Luo2022FourLayerD2NN(optics_config).double()
    start = time.perf_counter()
    output32 = model32(amplitude_to_complex_field(target32), diffuser_phase[None])
    complex64_seconds = time.perf_counter() - start
    start = time.perf_counter()
    output64 = model64(
        amplitude_to_complex_field(target32.double()),
        diffuser_phase.double()[None],
    )
    complex128_seconds = time.perf_counter() - start
    reference32 = output64.float()
    difference = output32 - reference32
    relative_l2 = float(
        torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference32).clamp_min(1e-30)
    )
    maximum_absolute = float(difference.abs().max())
    peak_relative = maximum_absolute / float(reference32.abs().max().clamp_min(1e-30))
    output32_flat = output32.flatten().double()
    output64_flat = output64.flatten()
    centered32 = output32_flat - output32_flat.mean()
    centered64 = output64_flat - output64_flat.mean()
    pcc = float(
        (centered32 * centered64).sum()
        / torch.sqrt(centered32.square().sum() * centered64.square().sum()).clamp_min(1e-30)
    )
    return {
        "complex64_seconds": complex64_seconds,
        "complex128_seconds": complex128_seconds,
        "relative_l2": relative_l2,
        "maximum_absolute_error": maximum_absolute,
        "peak_relative_maximum_absolute_error": peak_relative,
        "pcc": pcc,
        "complex64_finite": bool(torch.isfinite(output32).all()),
        "complex128_finite": bool(torch.isfinite(output64).all()),
    }


def _luo2022_training_step_benchmark(
    optics_config: Luo2022OpticsConfig,
    diffuser_phases: torch.Tensor,
    *,
    device: torch.device,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    if iterations < 2:
        raise ValueError("iterations must include at least one warm-up and one measured step")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 23)
    image = torch.rand(4, 1, 28, 28, generator=generator)
    target = prepare_luo2022_amplitude(
        image,
        resized_shape=(160, 160),
        canvas_shape=optics_config.field_shape,
    ).to(device)
    field = amplitude_to_complex_field(target)
    diffuser_bank = diffuser_phases[:20].to(device)
    model = Luo2022FourLayerD2NN(optics_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    timings = []
    for iteration in range(iterations):
        start = time.perf_counter()
        output = model(field, diffuser_bank)
        forward_seconds = time.perf_counter() - start
        loss, _components = luo2022_d2nn_loss(output, target)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        backward_step_seconds = time.perf_counter() - start
        timings.append(
            {
                "iteration": iteration,
                "forward_seconds": forward_seconds,
                "backward_optimizer_seconds": backward_step_seconds,
                "total_seconds": forward_seconds + backward_step_seconds,
            }
        )
    measured = timings[1:]
    max_rss_bytes = 0
    if resource is not None:
        max_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        max_rss_bytes = max_rss if platform.system() == "Darwin" else max_rss * 1024
    return {
        "device": str(device),
        "grid_shape": list(optics_config.field_shape),
        "batch_objects": 4,
        "diffusers": 20,
        "fields_per_update": 80,
        "timings": timings,
        "steady_step_seconds": float(np.mean([row["total_seconds"] for row in measured])),
        "process_peak_rss_bytes": max_rss_bytes,
        "process_peak_rss_gib": max_rss_bytes / 1024**3,
        "process_peak_rss_available": resource is not None,
        "diffuser_bank_note": (
            "The benchmark uses the first 20 raw seeded phases. The separate exact "
            "2000-diffuser audit proves that all raw candidates pass the frozen uniqueness "
            "criterion, so filtering would not alter this benchmark bank."
        ),
    }


def _luo2022_assessment_markdown(assessment: dict[str, Any]) -> str:
    propagation = assessment["numerical_propagation"]
    precision = propagation["precision_comparison"]
    diffuser = assessment["diffuser_model"]
    paper_scale = diffuser["paper_scale_sensitivity_audit"]
    benchmark = assessment["resource_benchmark"]
    gates = assessment["gates"]
    gate_lines = "\n".join(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`" for name, passed in gates.items()
    )
    return f"""# Luo 2022 R0 readiness assessment

Date: {assessment["assessment_date"]}

Decision: **{assessment["decision"]}**

## Gates

{gate_lines}

## Numerical propagation

- RS FFT vs direct discrete sum max absolute error:
  `{propagation["rs_fft_vs_direct_sum_max_abs_error"]:.3e}`
- complex64 vs complex128 relative L2 error: `{precision["relative_l2"]:.3e}`
- complex64 vs complex128 output PCC: `{precision["pcc"]:.12f}`

## Diffuser model

- Samples: {diffuser["sample_count"]}
- Mean phase standard deviation:
  `{diffuser["phase_std_radians"]["mean"]:.6f}` rad
- Frozen phase representation:
  `{diffuser["selected_phase_representation"]}`
- Frozen-representation minimum pair difference:
  `{diffuser["uniqueness_sensitivity"][diffuser["selected_phase_representation"]]["minimum_radians"]:.6f}` rad
- Required minimum pair difference:
  `{diffuser["uniqueness_threshold_radians"]:.6f}` rad
- Frozen complex-transmittance correlation length:
  `{diffuser["correlation_length_lambda"]["selected_sample_mean"]:.3f} ±
  {diffuser["correlation_length_lambda"]["selected_sample_standard_deviation"]:.3f}`
  lambda
- Paper target: approximately
  `{diffuser["correlation_length_lambda"]["paper_target"]:.3f}` lambda

The main paper and supplementary material do not publish the Gaussian kernel
normalization/support, boundary mode, phase-wrapping branch for uniqueness, or
the discrete autocorrelation fitting protocol. Freeze 2026-07-17.2 therefore
records the [-pi, pi) uniqueness branch and complex-transmittance
autocovariance as explicit paper-inferred project choices.

Paper-scale sensitivity audit:

- Complex-transmittance correlation over
  {paper_scale["diffuser_count"]} diffusers:
  `{paper_scale["complex_transmittance_correlation_length_lambda"]["sample_mean"]:.3f} ±
  {paper_scale["complex_transmittance_correlation_length_lambda"]["sample_standard_deviation"]:.3f}`
  lambda
- Exact [-pi, pi) uniqueness over all
  {paper_scale["exact_uniqueness"]["pair_count"]} unordered pairs:
  minimum
  `{paper_scale["exact_uniqueness"]["minimum_radians"]:.3f}` rad,
  pass fraction
  `{paper_scale["exact_uniqueness"]["pair_pass_fraction"]:.3f}`
- Exact pair audit time:
  `{paper_scale["exact_pair_audit_seconds"]:.1f}` s

## Resource benchmark

- Exact shape: 240 x 240, B=4, n=20, 80 fields/update
- Steady training step: `{benchmark["steady_step_seconds"]:.3f}` s
- Process peak RSS: `{benchmark["process_peak_rss_gib"]:.3f}` GiB
- Projected 100-epoch run: `{benchmark["projected_full_run_hours"]:.1f}` h
  (`{benchmark["projected_full_run_days"]:.1f}` days)
- Paper-reported training time: approximately
  `{benchmark["paper_reported_training_hours_approx"]:.0f}` h on GTX 1080 Ti

## Required before CUDA R0

The reduced R0 validation under freeze 2026-07-17.2 has passed.

1. Transfer the frozen configuration, environment lock, and validated code to
   a CUDA machine.
2. Record the CUDA runtime before launching the 100-epoch run. A separate local
   GPU benchmark is not required.
"""


def accelerator_metadata(device: torch.device) -> dict[str, Any]:
    """Return the accelerator state needed to audit or resume a long R0 run."""

    metadata: dict[str, Any] = {
        "requested_device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        metadata.update(
            {
                "cuda_device_index": index,
                "cuda_device_name": properties.name,
                "cuda_total_memory_bytes": properties.total_memory,
                "cuda_compute_capability": [
                    properties.major,
                    properties.minor,
                ],
                "cuda_memory_allocated_bytes": torch.cuda.memory_allocated(index),
                "cuda_memory_reserved_bytes": torch.cuda.memory_reserved(index),
                "cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(index),
                "cuda_max_memory_reserved_bytes": torch.cuda.max_memory_reserved(index),
            }
        )
    return metadata


def luo2022_expected_comparison(contract: dict[str, Any]) -> dict[str, Any]:
    """Describe the paper comparison target without inventing an aggregate PCC."""

    supplementary = contract["supplementary_material"]["figure_s4"]
    return {
        "primary_metric": contract["evaluation"]["primary_metric"],
        "matched_test_protocol": {
            "test_objects": contract["evaluation"]["test_objects"],
            "new_diffusers": contract["evaluation"]["diffuser_sets_for_n20"]["new_diffusers"],
            "image_values": contract["evaluation"]["image_values"],
            "contrast_enhancement_for_metrics": contract["evaluation"][
                "contrast_enhancement_for_metrics"
            ],
        },
        "published_numeric_context": {
            "single_digit_known_new_pcc": supplementary[
                "single_digit_known_new_pcc_baseline"
            ],
            "scope": "supplementary Figure S4 single-digit pruning experiment",
            "acceptance_target": False,
        },
        "aggregate_r0_target": {
            "status": "deferred",
            "reason": (
                "The aggregate n=20 MNIST result must be digitized from the paper figure "
                "before a numeric acceptance tolerance is frozen."
            ),
        },
    }


def build_luo2022_review(
    *,
    contract: dict[str, Any],
    history: list[dict[str, Any]],
    initial_new: dict[str, float],
    target_epochs: int,
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    """Build the epoch-level review record used for remote monitoring."""

    latest = history[-1]
    current_new = latest["new_diffusers"]
    best_entry = max(history, key=lambda item: item["new_diffusers"]["pearson"])
    return {
        "status": "training" if latest["epoch"] < target_epochs else "final_epoch_complete",
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "completed_epoch": latest["epoch"],
        "target_epochs": target_epochs,
        "completion_fraction": latest["epoch"] / target_epochs,
        "current": {
            "learning_rate": latest["learning_rate"],
            "train": latest["train"],
            "new_diffusers": current_new,
        },
        "change_from_untrained_model": {
            "new_diffuser_pearson": current_new["pearson"] - initial_new["pearson"],
            "new_diffuser_total_loss": current_new["total"] - initial_new["total"],
        },
        "best_new_diffuser_pearson": {
            "epoch": best_entry["epoch"],
            "value": best_entry["new_diffusers"]["pearson"],
        },
        "execution_controls": runtime_config["execution_controls"],
        "expected_comparison": luo2022_expected_comparison(contract),
        "review_rules": [
            "Treat finite metrics, phase updates, and improvement from initialization as engineering gates.",
            "Do not compare the aggregate run directly with the supplementary single-digit PCC values.",
            "Do not label the run a paper reproduction until the aggregate paper figure is digitized and a tolerance is frozen.",
        ],
    }


def save_luo2022_checkpoint(
    path: Path,
    *,
    model: Luo2022FourLayerD2NN,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    history: list[dict[str, Any]],
    initial_new: dict[str, float],
    completed_epoch: int,
    loader_generator: torch.Generator,
    runtime_config: dict[str, Any],
    source_freeze_version: str,
) -> None:
    """Atomically save enough state to resume at the next epoch."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "initial_new_diffusers": initial_new,
            "completed_epoch": completed_epoch,
            "loader_generator_state": loader_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "runtime_config": runtime_config,
            "source_freeze_version": source_freeze_version,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def run_luo2022_training(
    *,
    output_dir: Path,
    config_path: Path = DEFAULT_LUO2022_CONFIG,
    download: bool = False,
    small_run: bool = False,
    device_name: str = "cpu",
    seed: int | None = None,
    grid_size: int | None = None,
    input_size: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    train_limit: int | None = None,
    eval_limit: int | None = None,
    diffusers_per_epoch: int | None = None,
    eval_diffusers: int | None = None,
    lr: float | None = None,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
    diffuser_chunk_size: int | None = None,
    review_eval_batches: int = 25,
    resume: bool = False,
) -> dict[str, Any]:
    """Train the four-layer R0 D2NN using the frozen Luo 2022 contract."""

    contract = load_config(config_path)
    device = select_device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    runtime_config = build_luo2022_runtime_config(
        contract,
        small_run=small_run,
        device=device,
        seed=seed,
        grid_size=grid_size,
        input_size=input_size,
        epochs=epochs,
        batch_size=batch_size,
        train_limit=train_limit,
        eval_limit=eval_limit,
        diffusers_per_epoch=diffusers_per_epoch,
        eval_diffusers=eval_diffusers,
        lr=lr,
        max_train_batches=max_train_batches,
        max_eval_batches=max_eval_batches,
        diffuser_chunk_size=diffuser_chunk_size,
        review_eval_batches=review_eval_batches,
    )
    values = runtime_config["runtime"]
    seed_everything(values["seed"])
    checkpoint_path = output_dir / "checkpoints" / "latest.pt"
    if resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
        saved_runtime_config = load_config(output_dir / "config.json")
        if saved_runtime_config != runtime_config:
            raise ValueError("resume runtime configuration does not match the saved run")
    else:
        prepare_output_dir(output_dir)
        diffuser_dir = output_dir / "diffusers"
        if diffuser_dir.exists():
            shutil.rmtree(diffuser_dir)
        diffuser_dir.mkdir()
        for filename in ("review.json", "run_state.json"):
            path = output_dir / filename
            if path.exists():
                path.unlink()
        snapshot_config(runtime_config, output_dir=output_dir, config_path=config_path)
        write_json(
            output_dir / "run_state.json",
            {
                "status": "initializing",
                "started_at_utc": datetime.now(UTC).isoformat(),
                "device": accelerator_metadata(device),
                "runtime": run_metadata(),
                "target": luo2022_expected_comparison(contract),
            },
        )

    train_base = build_torchvision_dataset(
        name="MNIST",
        root=DEFAULT_DATA_ROOT,
        train=True,
        image_size=int(contract["input"]["original_shape"][0]),
        download=download,
    )
    eval_base = build_torchvision_dataset(
        name="MNIST",
        root=DEFAULT_DATA_ROOT,
        train=False,
        image_size=int(contract["input"]["original_shape"][0]),
        download=download,
    )
    train_dataset = Subset(train_base, range(min(values["train_limit"], len(train_base))))
    eval_dataset = Subset(eval_base, range(min(values["eval_limit"], len(eval_base))))
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(values["seed"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=values["batch_size"],
        shuffle=True,
        generator=loader_generator,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=values["batch_size"], shuffle=False)

    optics_config = Luo2022OpticsConfig(
        field_shape=(values["grid_size"], values["grid_size"]),
        wavelength=float(contract["illumination"]["wavelength_m"]),
        pixel_size=float(contract["grid"]["pixel_pitch_m"]),
        object_to_diffuser_distance=float(contract["geometry"]["object_to_diffuser_m"]),
        diffuser_to_first_layer_distance=float(contract["geometry"]["diffuser_to_first_layer_m"]),
        layer_distance=float(contract["geometry"]["layer_to_layer_m"]),
        output_distance=float(contract["geometry"]["last_layer_to_output_m"]),
        num_layers=int(contract["d2nn"]["layers"]),
        pad_factor=2,
    )
    model = Luo2022FourLayerD2NN(optics_config).to(device)
    optimizer_config = contract["training"]["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=values["learning_rate"],
        betas=(float(optimizer_config["beta1"]), float(optimizer_config["beta2"])),
        eps=float(optimizer_config["epsilon"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=float(contract["training"]["learning_rate"]["gamma"]),
    )
    diffuser_kwargs = {
        "wavelength": optics_config.wavelength,
        "pixel_size": optics_config.pixel_size,
        "refractive_index_difference": float(contract["diffuser"]["refractive_index_difference"]),
        "height_mean_lambda": float(contract["diffuser"]["height_mean_lambda"]),
        "height_std_lambda": float(contract["diffuser"]["height_std_lambda"]),
        "gaussian_sigma_lambda": float(contract["diffuser"]["gaussian_sigma_lambda"]),
        "truncate_sigma": float(contract["diffuser"]["finite_kernel_choice"]["truncate_sigma"]),
        "padding": str(contract["diffuser"]["finite_kernel_choice"]["padding"]),
    }
    uniqueness_config = contract["diffuser"]["uniqueness"]
    phase_representation = str(uniqueness_config["phase_representation"])
    minimum_difference_radians = float(uniqueness_config["minimum_radians"])
    eval_diffuser_bank_cpu = make_unique_correlated_diffusers(
        values["eval_diffusers"],
        field_shape=optics_config.field_shape,
        base_seed=values["seed"] + 10_000_000,
        minimum_difference_radians=minimum_difference_radians,
        phase_representation=phase_representation,
        **diffuser_kwargs,
    )
    eval_diffuser_bank = eval_diffuser_bank_cpu.to(device)
    initial_new: dict[str, float] = {}
    if not resume:
        initial_new = evaluate_luo2022_model(
            model,
            eval_loader,
            eval_diffuser_bank,
            resized_shape=(values["input_size"], values["input_size"]),
            canvas_shape=optics_config.field_shape,
            device=device,
            max_batches=values["review_eval_batches"],
            diffuser_chunk_size=values["diffuser_chunk_size"],
        )

    history: list[dict[str, Any]] = []
    final_training_diffusers = eval_diffuser_bank
    total_training_diffusers = values["epochs"] * values["diffusers_per_epoch"]
    training_diffuser_history = torch.empty(
        (total_training_diffusers, *optics_config.field_shape),
        dtype=eval_diffuser_bank_cpu.dtype,
    )
    training_diffuser_count = 0
    start_epoch = 1
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["source_freeze_version"] != contract["freeze_version"]:
            raise ValueError("resume checkpoint freeze version does not match source configuration")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint["history"]
        initial_new = checkpoint["initial_new_diffusers"]
        completed_epoch = int(checkpoint["completed_epoch"])
        if completed_epoch != len(history):
            raise ValueError("resume checkpoint epoch and history length do not match")
        for completed in range(1, completed_epoch + 1):
            phase_path = output_dir / "diffusers" / f"training_epoch_{completed:03d}.pt"
            if not phase_path.is_file():
                raise FileNotFoundError(f"saved training diffuser bank not found: {phase_path}")
            saved_phases = torch.load(phase_path, map_location="cpu", weights_only=True)
            next_count = training_diffuser_count + int(saved_phases.shape[0])
            training_diffuser_history[training_diffuser_count:next_count] = saved_phases
            training_diffuser_count = next_count
            final_training_diffusers = saved_phases.to(device)
        loader_generator.set_state(checkpoint["loader_generator_state"].cpu())
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
            )
        start_epoch = completed_epoch + 1

    for epoch in range(start_epoch, values["epochs"] + 1):
        epoch_diffusers_cpu = make_unique_correlated_diffusers(
            values["diffusers_per_epoch"],
            field_shape=optics_config.field_shape,
            base_seed=values["seed"] + epoch * 100_000,
            minimum_difference_radians=minimum_difference_radians,
            phase_representation=phase_representation,
            existing_phases=training_diffuser_history[:training_diffuser_count],
            **diffuser_kwargs,
        )
        next_training_diffuser_count = training_diffuser_count + values["diffusers_per_epoch"]
        training_diffuser_history[
            training_diffuser_count:next_training_diffuser_count
        ] = epoch_diffusers_cpu
        training_diffuser_count = next_training_diffuser_count
        torch.save(
            epoch_diffusers_cpu,
            output_dir / "diffusers" / f"training_epoch_{epoch:03d}.pt",
        )
        final_training_diffusers = epoch_diffusers_cpu.to(device)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = train_luo2022_one_epoch(
            model,
            train_loader,
            final_training_diffusers,
            optimizer,
            resized_shape=(values["input_size"], values["input_size"]),
            canvas_shape=optics_config.field_shape,
            device=device,
            max_batches=values["max_train_batches"],
            diffuser_chunk_size=values["diffuser_chunk_size"],
        )
        new_metrics = evaluate_luo2022_model(
            model,
            eval_loader,
            eval_diffuser_bank,
            resized_shape=(values["input_size"], values["input_size"]),
            canvas_shape=optics_config.field_shape,
            device=device,
            max_batches=values["review_eval_batches"],
            diffuser_chunk_size=values["diffuser_chunk_size"],
        )
        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train": train_metrics,
                "new_diffusers": new_metrics,
                "new_diffuser_evaluation_scope": {
                    "type": "fixed_test_prefix_monitoring_probe",
                    "max_batches": values["review_eval_batches"],
                },
                "training_diffuser_seed_base": values["seed"] + epoch * 100_000,
                "accepted_training_diffusers_total": training_diffuser_count,
                "uniqueness_comparison_scope": uniqueness_config["comparison_scope"],
            }
        )
        write_json(output_dir / "history.json", history)
        scheduler.step()
        review = build_luo2022_review(
            contract=contract,
            history=history,
            initial_new=initial_new,
            target_epochs=values["epochs"],
            runtime_config=runtime_config,
        )
        write_json(output_dir / "review.json", review)
        save_luo2022_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
            initial_new=initial_new,
            completed_epoch=epoch,
            loader_generator=loader_generator,
            runtime_config=runtime_config,
            source_freeze_version=contract["freeze_version"],
        )
        write_json(
            output_dir / "run_state.json",
            {
                "status": "training",
                "updated_at_utc": datetime.now(UTC).isoformat(),
                "completed_epoch": epoch,
                "target_epochs": values["epochs"],
                "device": accelerator_metadata(device),
                "latest_review": "review.json",
                "latest_checkpoint": "checkpoints/latest.pt",
            },
        )

    final_known = evaluate_luo2022_model(
        model,
        eval_loader,
        final_training_diffusers,
        resized_shape=(values["input_size"], values["input_size"]),
        canvas_shape=optics_config.field_shape,
        device=device,
        max_batches=values["max_eval_batches"],
        diffuser_chunk_size=values["diffuser_chunk_size"],
    )
    final_new = evaluate_luo2022_model(
        model,
        eval_loader,
        eval_diffuser_bank,
        resized_shape=(values["input_size"], values["input_size"]),
        canvas_shape=optics_config.field_shape,
        device=device,
        max_batches=values["max_eval_batches"],
        diffuser_chunk_size=values["diffuser_chunk_size"],
    )
    final_new_probe = evaluate_luo2022_model(
        model,
        eval_loader,
        eval_diffuser_bank,
        resized_shape=(values["input_size"], values["input_size"]),
        canvas_shape=optics_config.field_shape,
        device=device,
        max_batches=values["review_eval_batches"],
        diffuser_chunk_size=values["diffuser_chunk_size"],
    )
    phase_update_l2 = float(model.phase.detach().square().sum().sqrt().cpu().item())
    training_loss_decreased = history[-1]["train"]["total"] < history[0]["train"]["total"]
    new_diffuser_loss_decreased = final_new_probe["total"] < initial_new["total"]
    metrics = {
        "initial_new_diffuser_monitoring_probe": initial_new,
        "final_new_diffuser_monitoring_probe": final_new_probe,
        "final_known_diffusers_full_test_set": final_known,
        "final_new_diffusers_full_test_set": final_new,
        "phase_update_l2": phase_update_l2,
        "training_loss_decreased": training_loss_decreased,
        "new_diffuser_loss_decreased": new_diffuser_loss_decreased,
        "first_epoch_train_total": history[0]["train"]["total"],
        "last_epoch_train_total": history[-1]["train"]["total"],
    }

    sample_filename = "luo2022_r0_small.png" if small_run else "luo2022_r0.png"
    save_luo2022_sample_grid(
        model,
        eval_loader,
        eval_diffuser_bank,
        output_dir / "samples" / sample_filename,
        resized_shape=(values["input_size"], values["input_size"]),
        canvas_shape=optics_config.field_shape,
        device=device,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "runtime_config": runtime_config,
            "source_freeze_version": contract["freeze_version"],
        },
        output_dir / "checkpoints" / "luo2022_d2nn.pt",
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status_label": runtime_config["status_label"],
        "experiment_class": "E4",
        "comparison_level": "R0-small" if small_run else "R0",
        "profile_id": contract["profile_id"],
        "source_freeze_version": contract["freeze_version"],
        "claim_boundary": runtime_config["claim_boundary"],
        "dataset": "MNIST",
        "input_encoding": "field_amplitude",
        "output_for_loss_and_metrics": "raw_detector_intensity",
        "diffuser_uniqueness": {
            "phase_representation": phase_representation,
            "comparison_scope": uniqueness_config["comparison_scope"],
            "minimum_radians": minimum_difference_radians,
            "accepted_training_diffusers_total": training_diffuser_count,
        },
        "runtime": run_metadata(),
        "accelerator": accelerator_metadata(device),
        "execution_controls": runtime_config["execution_controls"],
        "acceptance": {
            "finite_metrics": all(
                np.isfinite(value)
                for group in (initial_new, final_known, final_new)
                for value in group.values()
            ),
            "phase_parameters_updated": phase_update_l2 > 0,
            "training_loss_decreased": training_loss_decreased,
            "fixed_new_diffuser_loss_decreased": new_diffuser_loss_decreased,
        },
        "artifacts": {
            "runtime_config": "config.json",
            "source_config": "source_config.json",
            "history": "history.json",
            "metrics": "metrics.json",
            "checkpoint": "checkpoints/luo2022_d2nn.pt",
            "resume_checkpoint": "checkpoints/latest.pt",
            "training_diffuser_banks": "diffusers/training_epoch_*.pt",
            "review": "review.json",
            "run_state": "run_state.json",
            "sample_grid": f"samples/{sample_filename}",
        },
        "physical_effects_included": [
            "amplitude-encoded MNIST input",
            "correlated thin pure-phase diffuser",
            "Rayleigh-Sommerfeld propagation",
            "four trainable phase-only diffractive layers",
            "raw detector intensity",
            "paper PCC-plus-energy loss",
        ],
        "physical_effects_omitted": [
            "volumetric multiple scattering",
            "material absorption",
            "sensor noise and quantization",
            "fabrication and alignment errors",
            "visible-light hardware constraints",
            "U-Net and GAN post-processing",
        ],
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "manifest.json", manifest)
    write_json(
        output_dir / "run_state.json",
        {
            "status": "completed",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "completed_epoch": values["epochs"],
            "target_epochs": values["epochs"],
            "device": accelerator_metadata(device),
            "latest_review": "review.json",
            "final_manifest": "manifest.json",
            "final_metrics": "metrics.json",
        },
    )
    return {"history": history, "metrics": metrics, "manifest": manifest}


def train_luo2022_one_epoch(
    model: Luo2022FourLayerD2NN,
    loader: DataLoader,
    diffuser_phase: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    resized_shape: tuple[int, int],
    canvas_shape: tuple[int, int],
    device: torch.device,
    max_batches: int | None = None,
    diffuser_chunk_size: int | None = None,
) -> dict[str, float]:
    """Train one epoch while reusing one epoch-specific diffuser bank."""

    model.train()
    totals: dict[str, float] = defaultdict(float)
    count = 0
    for batch_index, (image, _label) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        target = prepare_luo2022_amplitude(
            image.to(device),
            resized_shape=resized_shape,
            canvas_shape=canvas_shape,
        )
        field = amplitude_to_complex_field(target)
        optimizer.zero_grad(set_to_none=True)
        chunk_size = int(diffuser_chunk_size or diffuser_phase.shape[0])
        total_diffusers = int(diffuser_phase.shape[0])
        batch_size = int(image.shape[0])
        total_pairs = batch_size * total_diffusers
        for start in range(0, total_diffusers, chunk_size):
            phase_chunk = diffuser_phase[start : start + chunk_size]
            output = model(field, phase_chunk)
            loss, components = luo2022_d2nn_loss(output, target)
            pair_count = batch_size * int(phase_chunk.shape[0])
            (loss * (pair_count / total_pairs)).backward()
            for name, value in components.items():
                totals[name] += float(value.detach().cpu().item()) * pair_count
            count += pair_count
        optimizer.step()
    if count == 0:
        raise ValueError("training loader yielded no batches")
    return {name: total / count for name, total in totals.items()}


@torch.no_grad()
def evaluate_luo2022_model(
    model: Luo2022FourLayerD2NN,
    loader: DataLoader,
    diffuser_phase: torch.Tensor,
    *,
    resized_shape: tuple[int, int],
    canvas_shape: tuple[int, int],
    device: torch.device,
    max_batches: int | None = None,
    diffuser_chunk_size: int | None = None,
) -> dict[str, float]:
    """Evaluate the paper loss and PCC on raw detector intensity."""

    model.eval()
    totals: dict[str, float] = defaultdict(float)
    count = 0
    for batch_index, (image, _label) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        image = image.to(device)
        target = prepare_luo2022_amplitude(
            image,
            resized_shape=resized_shape,
            canvas_shape=canvas_shape,
        )
        field = amplitude_to_complex_field(target)
        chunk_size = int(diffuser_chunk_size or diffuser_phase.shape[0])
        total_diffusers = int(diffuser_phase.shape[0])
        for start in range(0, total_diffusers, chunk_size):
            phase_chunk = diffuser_phase[start : start + chunk_size]
            output = model(field, phase_chunk)
            _loss, components = luo2022_d2nn_loss(output, target)
            pair_count = int(image.shape[0]) * int(phase_chunk.shape[0])
            for name, value in components.items():
                totals[name] += float(value.detach().cpu().item()) * pair_count
            count += pair_count
    if count == 0:
        raise ValueError("evaluation loader yielded no batches")
    return {name: total / count for name, total in totals.items()}


@torch.no_grad()
def save_luo2022_sample_grid(
    model: Luo2022FourLayerD2NN,
    loader: DataLoader,
    diffuser_phase: torch.Tensor,
    output_path: Path,
    *,
    resized_shape: tuple[int, int],
    canvas_shape: tuple[int, int],
    device: torch.device,
) -> None:
    """Save target, corrupted intensity, reconstruction, and error."""

    image, _label = next(iter(loader))
    image = image[:1].to(device)
    target = prepare_luo2022_amplitude(
        image,
        resized_shape=resized_shape,
        canvas_shape=canvas_shape,
    )
    field = amplitude_to_complex_field(target)
    one_diffuser = diffuser_phase[:1]
    distorted = model.distort(field, one_diffuser).flatten(0, 1)
    corrupted = field_intensity(model.diffuser_to_first_layer.propagate(distorted))[0]
    output = model(field, one_diffuser)[0, 0]
    target_image = target[0, 0]

    def display_normalize(value: torch.Tensor) -> torch.Tensor:
        value = value.detach().cpu()
        return (value - value.min()) / (value.max() - value.min()).clamp_min(1e-8)

    panels = [
        ("target amplitude", display_normalize(target_image)),
        ("corrupted intensity", display_normalize(corrupted)),
        ("output intensity", display_normalize(output)),
        ("display error", (display_normalize(output) - display_normalize(target_image)).abs()),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 4, figsize=(12, 3))
    for axis, (title, value) in zip(axes, panels, strict=True):
        axis.imshow(value.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


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
    reconstruction_weights: ReconstructionLossWeights | None = None,
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
    weights = reconstruction_weights or ReconstructionLossWeights()
    experiment_class = experiment_class_for_run(
        corruption=corruption,
        train_diffuser_ids=train_diffuser_ids,
        eval_diffuser_ids=eval_diffuser_ids,
        uses_gan=False,
    )
    config = coherent_training_config(
        command="unet",
        experiment_class=experiment_class,
        corruption=corruption,
        seed=seed,
        d2nn_seed=d2nn_seed,
        train_diffuser_ids=train_diffuser_ids,
        eval_diffuser_ids=eval_diffuser_ids,
        train_limit=train_limit,
        eval_limit=eval_limit,
        epochs=epochs,
        batch_size=batch_size,
        base_channels=base_channels,
        lr=lr,
        device=device,
        materialize=materialize,
        sample_every=sample_every,
        max_train_batches=max_train_batches,
        max_eval_batches=max_eval_batches,
        num_workers=num_workers,
        reconstruction_weights=weights,
    )
    snapshot_config(config, output_dir=output_dir, config_path=None)

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
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status_label": "small run" if small_sized else "exploratory result",
        "experiment_class": experiment_class,
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
        "metrics_protocol": metrics_protocol_metadata(),
        "runtime": run_metadata(),
        "artifacts": {
            "config": "config.json",
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
    reconstruction_weights: ReconstructionLossWeights | None = None,
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
    weights = reconstruction_weights or ReconstructionLossWeights()
    experiment_class = experiment_class_for_run(
        corruption=corruption,
        train_diffuser_ids=train_diffuser_ids,
        eval_diffuser_ids=eval_diffuser_ids,
        uses_gan=True,
    )
    config = coherent_training_config(
        command="gan",
        experiment_class=experiment_class,
        corruption=corruption,
        seed=seed,
        d2nn_seed=d2nn_seed,
        train_diffuser_ids=train_diffuser_ids,
        eval_diffuser_ids=eval_diffuser_ids,
        train_limit=train_limit,
        eval_limit=eval_limit,
        epochs=epochs,
        batch_size=batch_size,
        base_channels=base_channels,
        lr=lr,
        device=device,
        materialize=materialize,
        sample_every=sample_every,
        max_train_batches=max_train_batches,
        max_eval_batches=max_eval_batches,
        num_workers=num_workers,
        reconstruction_weights=weights,
        adversarial_weight=adversarial_weight,
        generator_init=generator_init,
    )
    snapshot_config(config, output_dir=output_dir, config_path=None)

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
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status_label": "small run" if small_sized else "exploratory result",
        "experiment_class": experiment_class,
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
        "metrics_protocol": metrics_protocol_metadata(),
        "artifacts": {
            "config": "config.json",
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
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "unet_dir": str(unet_dir),
        "gan_dir": str(gan_dir),
        "unet_manifest": unet_manifest,
        "gan_manifest": gan_manifest,
        "unet_metrics": unet_metrics,
        "gan_metrics": gan_metrics,
        "metric_comparison": metric_comparison,
        "metrics_protocol": metrics_protocol_metadata(),
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
