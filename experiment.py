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
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import StringIO
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

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
    Luo2022CachedIntensityDataset,
    Luo2022IntensityCacheWriter,
    build_luo2022_fixed_depth_assignment,
    build_coherent_mnist_datasets,
    canonical_sha256,
    make_luo2022_frozen_scale,
    materialize_coherent_dataset,
    prepare_luo2022_amplitude,
    verify_luo2022_intensity_cache,
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
    represent_diffuser_phase,
    summarize_cross_diffuser_uniqueness,
    summarize_diffuser_bank_uniqueness,
)
from data import build_torchvision_dataset
from losses import (
    ReconstructionLossWeights,
    luo2022_d2nn_components_per_pair,
    luo2022_d2nn_energy_breakdown_per_pair,
    luo2022_d2nn_loss,
    masked_pearson_per_image,
    pearson_per_image,
    reconstruction_loss,
)
from metrics import (
    digit_group_statistics,
    pair_level_tail_statistics,
    paired_diffuser_delta_statistics,
    per_image_reconstruction_metrics,
    reconstruction_metrics,
    scalar_summary,
    two_level_diffuser_summary,
)
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
DEFAULT_LUO2022_BACKEND_CONFIG = Path("configs/luo2022_fixed4_backend.json")
LUO2022_BACKEND_SCHEMA_VERSION = 1
LUO2022_BACKEND_CACHE_PROTOCOL = "luo2022_fixed4_raw_intensity_cache_v1"
LUO2022_BACKEND_TRAINING_PROTOCOL = "luo2022_fixed4_backend_training_v1"
LUO2022_BACKEND_EVALUATION_PROTOCOL = "luo2022_fixed4_backend_evaluation_v1"
LUO2022_BACKEND_CONDITIONS = ("b0", "warmup", "r1", "r2")
LUO2022_BACKEND_VARIANTS = ("R0", "B0", "R1", "R2")
MANIFEST_SCHEMA_VERSION = 1
CONFIG_SCHEMA_VERSION = 1
METRICS_PROTOCOL_VERSION = 1
LOWER_IS_BETTER = {"l1", "mse"}
HIGHER_IS_BETTER = {"psnr", "ssim", "pearson"}
ORDERED_METRICS = ("l1", "mse", "psnr", "ssim", "pearson")
LUO2022_ROI_METRIC_FIELDS = (
    "roi_full_canvas_pearson",
    "roi_center_input_region_pearson",
    "roi_target_support_pearson",
    "roi_full_canvas_output_energy_fraction",
    "roi_center_input_region_output_energy_fraction",
    "roi_target_support_output_energy_fraction",
)
LUO2022_ROI_METRIC_PROTOCOL = "luo2022_r0_roi_pcc_v1"
LUO2022_ROI_REGRESSION_TOLERANCE = 1e-6
LUO2022_CONTROL_LADDER_METRIC_PROTOCOL = "luo2022_r0_optical_control_ladder_v1"
LUO2022_CONTROL_LADDER_IDS = (
    "direct_free_space_no_d2nn",
    "zero_phase_four_layer",
    "trained_four_layer",
)


def luo2022_diffuser_seed_schedule(
    *,
    seed: int,
    epochs: int,
    training_stride: int,
    evaluation_offset: int,
) -> dict[str, Any]:
    """Return and validate disjoint training/evaluation diffuser seed namespaces."""

    if epochs <= 0 or training_stride <= 0 or evaluation_offset <= 0:
        raise ValueError("diffuser seed schedule values must be positive")
    evaluation_base_seed = seed + evaluation_offset
    training_base_seeds = tuple(seed + epoch * training_stride for epoch in range(1, epochs + 1))
    if evaluation_base_seed in training_base_seeds:
        raise ValueError("evaluation diffuser seed overlaps a training epoch seed")
    return {
        "training_epoch_formula": "primary_seed_plus_epoch_times_stride",
        "training_stride": training_stride,
        "evaluation_formula": "primary_seed_plus_offset",
        "evaluation_offset": evaluation_offset,
        "evaluation_base_seed": evaluation_base_seed,
        "disjoint_for_configured_epochs": True,
    }


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
    d2nn_parser.add_argument(
        "--action",
        choices=(
            "inspect",
            "train",
            "assess",
            "scatter-audit",
            "evaluate",
            "diagnose",
            "control-ladder",
            "backend-cache",
            "backend-train",
            "backend-evaluate",
        ),
        default="inspect",
    )
    d2nn_parser.add_argument("--config-path", type=Path, default=DEFAULT_LUO2022_CONFIG)
    d2nn_parser.add_argument(
        "--backend-config-path",
        type=Path,
        default=DEFAULT_LUO2022_BACKEND_CONFIG,
        help="Public fixed-four-layer B0/R1/R2 ablation contract.",
    )
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
    d2nn_parser.add_argument(
        "--posthoc-output-dir",
        type=Path,
        default=None,
        help="Directory for resumable per-diffuser evidence; defaults inside OUTPUT_DIR.",
    )
    d2nn_parser.add_argument(
        "--posthoc-populations",
        choices=("training", "new", "no_diffuser"),
        nargs="+",
        default=("training", "new", "no_diffuser"),
        help="Frozen diffuser populations to evaluate with --action evaluate.",
    )
    d2nn_parser.add_argument(
        "--posthoc-training-epochs",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional training epochs to include in post-hoc evaluation. "
            "Defaults to all epochs when the training population is selected."
        ),
    )
    d2nn_parser.add_argument(
        "--posthoc-roi-metrics",
        action="store_true",
        help=(
            "Also record full-canvas, centered-input, and target-support PCC/energy "
            "per diffuser. This read-only mode requires an output directory outside RUN."
        ),
    )
    d2nn_parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Completed frozen R0 run directory for --action diagnose or control-ladder; "
            "defaults to --output-dir."
        ),
    )
    d2nn_parser.add_argument(
        "--diagnostic-output-dir",
        type=Path,
        default=Path("outputs/luo2022_r0_diagnosis"),
        help="Independent ignored evidence directory for --action diagnose.",
    )
    d2nn_parser.add_argument(
        "--diagnostic-batches",
        type=int,
        default=1,
        help="Fixed test-prefix batches used for the read-only R0 diagnosis.",
    )
    d2nn_parser.add_argument(
        "--diagnostic-diffusers",
        type=int,
        default=3,
        help="Known and unseen diffusers included in batch-level optical diagnostics.",
    )
    d2nn_parser.add_argument(
        "--diagnostic-pad-factors",
        type=int,
        nargs="+",
        default=(2, 3, 4),
        help="Padding factors for the discrete propagation sensitivity probe.",
    )
    d2nn_parser.add_argument(
        "--diagnostic-cross-bank-audit",
        action="store_true",
        help=(
            "Audit every new-versus-training diffuser pair. This is CPU-intensive and "
            "is therefore opt-in; the final-epoch cross-bank audit always runs."
        ),
    )
    d2nn_parser.add_argument(
        "--scatter-audit-output-dir",
        type=Path,
        default=Path("outputs/luo2022_r0_scatter_audit"),
        help="Independent JSON evidence directory for --action scatter-audit.",
    )
    d2nn_parser.add_argument(
        "--control-output-dir",
        type=Path,
        default=Path("outputs/luo2022_r0_optical_control_ladder"),
        help=(
            "Independent evidence directory for --action control-ladder. It must be "
            "outside the completed frozen run directory."
        ),
    )
    d2nn_parser.add_argument(
        "--control-populations",
        choices=("training", "new", "no_diffuser"),
        nargs="+",
        default=("training", "new", "no_diffuser"),
        help=(
            "Frozen diffuser populations for --action control-ladder. The default "
            "evaluates final known, new, and no-diffuser controls."
        ),
    )
    d2nn_parser.add_argument(
        "--control-controls",
        choices=LUO2022_CONTROL_LADDER_IDS,
        nargs="+",
        default=LUO2022_CONTROL_LADDER_IDS,
        help=(
            "Optical operators for --action control-ladder. Select "
            "direct_free_space_no_d2nn alone to measure the raw scattering "
            "baseline without evaluating phase-plate controls."
        ),
    )
    d2nn_parser.add_argument(
        "--control-training-epochs",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional known-diffuser epochs for --action control-ladder. Defaults "
            "to the final frozen epoch."
        ),
    )
    d2nn_parser.add_argument(
        "--backend-cache-dir",
        type=Path,
        default=Path("outputs/luo2022_fixed4_backend_cache"),
        help="Ignored read-only raw-intensity cache shared by backend runs.",
    )
    d2nn_parser.add_argument(
        "--backend-cache-operators",
        choices=("direct", "frozen_four_layer"),
        nargs="+",
        default=("direct", "frozen_four_layer"),
        help="Optical operators to materialize with --action backend-cache.",
    )
    d2nn_parser.add_argument(
        "--backend-condition",
        choices=("b0", "warmup", "r1", "r2"),
        default=None,
        help="Digital condition trained by --action backend-train.",
    )
    d2nn_parser.add_argument(
        "--backend-warmup-dir",
        type=Path,
        default=None,
        help="Completed common warm-up run required by R1 and R2.",
    )
    d2nn_parser.add_argument("--backend-b0-dir", type=Path, default=None)
    d2nn_parser.add_argument("--backend-r1-dir", type=Path, default=None)
    d2nn_parser.add_argument("--backend-r2-dir", type=Path, default=None)
    d2nn_parser.add_argument(
        "--backend-shard-size",
        type=int,
        default=512,
        help="Number of assigned raw-intensity samples per atomic cache shard.",
    )
    d2nn_parser.add_argument(
        "--backend-max-updates",
        type=int,
        default=None,
        help="Small-run-only cap on generator updates per epoch.",
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
            if args.action == "backend-cache":
                if args.run_dir is None:
                    raise ValueError("--run-dir is required for backend-cache")
                return run_luo2022_backend_cache(
                    run_dir=args.run_dir,
                    cache_dir=args.backend_cache_dir,
                    config_path=args.config_path,
                    backend_config_path=args.backend_config_path,
                    download=args.download,
                    device_name=args.device,
                    operators=tuple(args.backend_cache_operators),
                    shard_size=args.backend_shard_size,
                    small_run=args.small_run,
                    train_limit=args.train_limit,
                    validation_limit=args.eval_limit,
                )
            if args.action == "backend-train":
                if args.run_dir is None:
                    raise ValueError("--run-dir is required for backend-train")
                if args.backend_condition is None:
                    raise ValueError("--backend-condition is required for backend-train")
                return run_luo2022_backend_training(
                    run_dir=args.run_dir,
                    output_dir=Path(args.output_dir),
                    cache_dir=args.backend_cache_dir,
                    config_path=args.config_path,
                    backend_config_path=args.backend_config_path,
                    condition=args.backend_condition,
                    warmup_dir=args.backend_warmup_dir,
                    device_name=args.device,
                    small_run=args.small_run,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    max_updates_per_epoch=args.backend_max_updates,
                    resume=args.resume,
                    download=args.download,
                )
            if args.action == "backend-evaluate":
                if args.run_dir is None:
                    raise ValueError("--run-dir is required for backend-evaluate")
                required = {
                    "--backend-b0-dir": args.backend_b0_dir,
                    "--backend-r1-dir": args.backend_r1_dir,
                    "--backend-r2-dir": args.backend_r2_dir,
                }
                missing = [name for name, value in required.items() if value is None]
                if missing:
                    raise ValueError(
                        "backend-evaluate requires " + ", ".join(missing)
                    )
                return run_luo2022_backend_evaluation(
                    run_dir=args.run_dir,
                    output_dir=Path(args.output_dir),
                    cache_dir=args.backend_cache_dir,
                    config_path=args.config_path,
                    backend_config_path=args.backend_config_path,
                    b0_dir=args.backend_b0_dir,
                    r1_dir=args.backend_r1_dir,
                    r2_dir=args.backend_r2_dir,
                    device_name=args.device,
                    download=args.download,
                    max_eval_batches=args.max_eval_batches,
                )
            if args.action == "evaluate":
                return run_luo2022_posthoc_evaluation(
                    run_dir=Path(args.output_dir),
                    config_path=args.config_path,
                    download=args.download,
                    device_name=args.device,
                    diffuser_chunk_size=args.diffuser_chunk_size,
                    output_dir=args.posthoc_output_dir,
                    populations=tuple(args.posthoc_populations),
                    training_epochs=(
                        tuple(args.posthoc_training_epochs)
                        if args.posthoc_training_epochs is not None
                        else None
                    ),
                    include_roi_metrics=args.posthoc_roi_metrics,
                )
            if args.action == "diagnose":
                return run_luo2022_diagnosis(
                    run_dir=args.run_dir or Path(args.output_dir),
                    diagnostic_output_dir=args.diagnostic_output_dir,
                    config_path=args.config_path,
                    download=args.download,
                    device_name=args.device,
                    diagnostic_batches=args.diagnostic_batches,
                    diagnostic_diffusers=args.diagnostic_diffusers,
                    diagnostic_pad_factors=tuple(args.diagnostic_pad_factors),
                    cross_bank_audit=args.diagnostic_cross_bank_audit,
                )
            if args.action == "control-ladder":
                return run_luo2022_c0_optical_control_ladder(
                    run_dir=args.run_dir or Path(args.output_dir),
                    control_output_dir=args.control_output_dir,
                    config_path=args.config_path,
                    download=args.download,
                    device_name=args.device,
                    diffuser_chunk_size=args.diffuser_chunk_size,
                    populations=tuple(args.control_populations),
                    controls=tuple(args.control_controls),
                    training_epochs=(
                        tuple(args.control_training_epochs)
                        if args.control_training_epochs is not None
                        else None
                    ),
                )
            if args.action == "assess":
                return run_luo2022_readiness_assessment(
                    output_dir=Path(args.output_dir),
                    config_path=args.config_path,
                    device_name=args.device,
                    seed=args.seed,
                )
            if args.action == "scatter-audit":
                return run_luo2022_scatter_correlation_convention_audit(
                    output_dir=args.scatter_audit_output_dir,
                    config_path=args.config_path,
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
    training_seed_schedule = contract["training"]["diffuser_seed_schedule"]
    evaluation_seed_schedule = contract["evaluation"]["diffuser_seed_schedule"]
    diffuser_seed_schedule = luo2022_diffuser_seed_schedule(
        seed=resolved["seed"],
        epochs=resolved["epochs"],
        training_stride=int(training_seed_schedule["epoch_stride"]),
        evaluation_offset=int(evaluation_seed_schedule["offset"]),
    )
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
        "diffuser_seed_schedule": diffuser_seed_schedule,
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


def run_luo2022_scatter_correlation_convention_audit(
    *,
    output_dir: Path,
    config_path: Path = DEFAULT_LUO2022_CONFIG,
    seed: int | None = None,
    sample_count: int | None = None,
) -> dict[str, Any]:
    """Audit unpublished phase-autocorrelation conventions without changing R0.

    Luo et al. report ``sigma=4 lambda`` and an average phase-autocorrelation
    length of about ``10 lambda``, but do not publish the discrete estimator,
    phase branch, or fitting interval. This read-only evidence action holds
    every published diffuser parameter and the frozen finite-kernel choice
    fixed, then reports sensitivity to those unpublished measurement choices.

    ``sample_count`` exists for focused function tests only. The CLI always
    defaults to the paper-scale count frozen in the R0 contract.
    """

    contract = load_config(config_path)
    canonical_contract = load_config(DEFAULT_LUO2022_CONFIG)
    if contract != canonical_contract:
        raise ValueError("scatter audit requires the exact frozen R0 contract")
    if (
        str(contract["freeze_version"]) != "2026-07-19.3"
        or float(contract["diffuser"]["gaussian_sigma_lambda"]) != 4.0
        or float(contract["diffuser"]["expected_mean_correlation_length_lambda"]) != 10.0
        or float(contract["diffuser"]["finite_kernel_choice"]["truncate_sigma"]) != 4.0
        or str(contract["diffuser"]["finite_kernel_choice"]["padding"]) != "reflect"
    ):
        raise ValueError("scatter audit requires the published R0 diffuser parameters")

    expected_sample_count = int(contract["diffuser"]["training_correlation_validation_samples"])
    resolved_sample_count = expected_sample_count if sample_count is None else int(sample_count)
    if resolved_sample_count < 2:
        raise ValueError("scatter audit requires at least two diffuser samples")
    resolved_seed = int(contract["training"]["primary_seed"]["value"] if seed is None else seed)
    seed_everything(resolved_seed)
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
    diffuser_kwargs = _luo2022_diffuser_kwargs(optics_config, contract)
    target_length = float(contract["diffuser"]["expected_mean_correlation_length_lambda"])
    tolerance = float(contract["diffuser"]["correlation_estimator"]["acceptance_relative_error"])
    convention_specs = (
        {
            "id": "unwrapped_phase_frozen_fit",
            "observable": "phase_autocorrelation",
            "phase_representation": "unwrapped",
            "fit_range": (0.20, 0.95),
            "relation_to_frozen_r0": "sensitivity_only",
        },
        {
            "id": "zero_to_2pi_phase_frozen_fit",
            "observable": "phase_autocorrelation",
            "phase_representation": "zero_to_2pi",
            "fit_range": (0.20, 0.95),
            "relation_to_frozen_r0": "sensitivity_only",
        },
        {
            "id": "minus_pi_to_pi_phase_frozen_fit",
            "observable": "phase_autocorrelation",
            "phase_representation": "minus_pi_to_pi",
            "fit_range": (0.20, 0.95),
            "relation_to_frozen_r0": "sensitivity_only",
        },
        {
            "id": "minus_pi_to_pi_phase_low_correlation_fit",
            "observable": "phase_autocorrelation",
            "phase_representation": "minus_pi_to_pi",
            "fit_range": (0.05, 0.80),
            "relation_to_frozen_r0": "sensitivity_only",
        },
        {
            "id": "complex_transmittance_frozen_fit",
            "observable": "complex_transmittance_autocovariance",
            "phase_representation": None,
            "fit_range": tuple(
                float(value)
                for value in contract["diffuser"]["correlation_estimator"]["fit_correlation_range"]
            ),
            "relation_to_frozen_r0": "frozen_acceptance_estimator",
        },
    )
    audit_spec = {
        "schema_version": 1,
        "implementation_version": "luo2022_r0_scatter_correlation_convention_audit_v1",
        "read_only": True,
        "sample_count": resolved_sample_count,
        "seed": resolved_seed,
        "conventions": [
            {
                **spec,
                "fit_range": [float(value) for value in spec["fit_range"]],
            }
            for spec in convention_specs
        ],
    }
    fingerprint = {
        "source_config_sha256": _sha256_file(config_path),
        "source_freeze_version": str(contract["freeze_version"]),
        "audit_spec": audit_spec,
    }
    result_path = output_dir / "scatter_correlation_convention_audit.json"
    if result_path.is_file():
        saved = load_config(result_path)
        if (
            saved.get("status") == "completed"
            and saved.get("evidence_fingerprint") == fingerprint
        ):
            return saved
        raise ValueError("existing scatter audit does not match the requested frozen evidence")

    values_by_convention = {
        str(spec["id"]): np.empty(resolved_sample_count, dtype=np.float64)
        for spec in convention_specs
    }
    started = time.perf_counter()
    for index in range(resolved_sample_count):
        phase = make_correlated_diffuser_phase(
            optics_config.field_shape,
            seed=resolved_seed + index,
            **diffuser_kwargs,
        )
        for spec in convention_specs:
            convention_id = str(spec["id"])
            fit_range = tuple(float(value) for value in spec["fit_range"])
            if spec["observable"] == "phase_autocorrelation":
                represented = represent_diffuser_phase(
                    phase,
                    mode=str(spec["phase_representation"]),
                )
                value = estimate_phase_correlation_length(
                    represented,
                    pixel_size=optics_config.pixel_size,
                    wavelength=optics_config.wavelength,
                    fit_range=fit_range,
                )
            else:
                value = estimate_transmittance_correlation_length(
                    phase,
                    pixel_size=optics_config.pixel_size,
                    wavelength=optics_config.wavelength,
                    fit_range=fit_range,
                )
            values_by_convention[convention_id][index] = value
    generation_and_estimation_seconds = time.perf_counter() - started

    conventions: list[dict[str, Any]] = []
    for spec in convention_specs:
        convention_id = str(spec["id"])
        values = values_by_convention[convention_id]
        mean = float(values.mean())
        sample_standard_deviation = float(values.std(ddof=1))
        standard_error = sample_standard_deviation / float(np.sqrt(values.size))
        relative_error = abs(mean - target_length) / target_length
        conventions.append(
            {
                **spec,
                "fit_range": [float(value) for value in spec["fit_range"]],
                "sample_mean_correlation_length_lambda": mean,
                "sample_standard_deviation_lambda": sample_standard_deviation,
                "standard_error_lambda": standard_error,
                "ci95_normal_lambda": [
                    mean - 1.96 * standard_error,
                    mean + 1.96 * standard_error,
                ],
                "minimum_lambda": float(values.min()),
                "maximum_lambda": float(values.max()),
                "reported_target_lambda": target_length,
                "relative_error_to_reported_target": relative_error,
                "consistent_with_reported_L_under_this_convention": relative_error <= tolerance,
            }
        )

    result = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "implementation_version": audit_spec["implementation_version"],
        "read_only": True,
        "source_freeze_version": str(contract["freeze_version"]),
        "source_config_sha256": fingerprint["source_config_sha256"],
        "evidence_fingerprint": fingerprint,
        "paper_constraints": {
            "published_gaussian_sigma_lambda": 4.0,
            "configured_gaussian_sigma_lambda": float(
                contract["diffuser"]["gaussian_sigma_lambda"]
            ),
            "reported_target_correlation_length_lambda": target_length,
            "paper_observable_wording": "phase-autocorrelation",
            "discrete_estimator_published": False,
            "physical_parameters_unchanged": True,
        },
        "generation": {
            "sample_count": resolved_sample_count,
            "paper_scale_sample_count": expected_sample_count,
            "reduced_test_audit": resolved_sample_count != expected_sample_count,
            "candidate_seed_formula": "resolved_primary_seed_plus_index",
            "finite_kernel_and_padding": {
                "truncate_sigma": float(
                    contract["diffuser"]["finite_kernel_choice"]["truncate_sigma"]
                ),
                "padding": str(contract["diffuser"]["finite_kernel_choice"]["padding"]),
                "output_shape": str(
                    contract["diffuser"]["finite_kernel_choice"]["output_shape"]
                ),
            },
            "generation_and_estimation_seconds": generation_and_estimation_seconds,
        },
        "conventions": conventions,
        "claim_boundary": [
            (
                "A convention consistent with the reported L approximately 10 lambda does "
                "not identify the authors' unpublished estimator."
            ),
            "This audit does not alter the frozen R0 correlation acceptance criterion.",
            "This audit does not establish training or reconstruction superiority.",
            (
                "The raw sequential seed bank is used to audit phase-autocorrelation wording; "
                "it is not substituted for the training bank's uniqueness acceptance protocol."
            ),
        ],
        "runtime": run_metadata(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(result_path, result)
    return result


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
    source_config_sha256: str,
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
            "source_config_sha256": source_config_sha256,
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
    source_config_sha256 = _sha256_file(config_path)
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
    diffuser_seed_schedule = runtime_config["diffuser_seed_schedule"]
    training_diffuser_seed_stride = int(diffuser_seed_schedule["training_stride"])
    eval_diffuser_bank_cpu = make_unique_correlated_diffusers(
        values["eval_diffusers"],
        field_shape=optics_config.field_shape,
        base_seed=int(diffuser_seed_schedule["evaluation_base_seed"]),
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
        if (
            checkpoint.get("source_config_sha256") is not None
            and checkpoint["source_config_sha256"] != source_config_sha256
        ):
            raise ValueError("resume checkpoint source configuration hash does not match")
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
            base_seed=values["seed"] + epoch * training_diffuser_seed_stride,
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
                "training_diffuser_seed_base": (
                    values["seed"] + epoch * training_diffuser_seed_stride
                ),
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
            source_config_sha256=source_config_sha256,
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
            "source_config_sha256": source_config_sha256,
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
        "source_config_sha256": source_config_sha256,
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
def _evaluate_luo2022_forward_per_diffuser(
    forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    loader: DataLoader,
    diffuser_phase: torch.Tensor,
    *,
    resized_shape: tuple[int, int],
    canvas_shape: tuple[int, int],
    device: torch.device,
    max_batches: int | None = None,
    diffuser_chunk_size: int | None = None,
    include_roi_metrics: bool = False,
) -> list[dict[str, float | int]]:
    """Evaluate a frozen forward operator separately for every diffuser.

    Luo et al. first average PCC over the test objects for each diffuser and
    then summarize the diffuser distribution. The ordinary evaluator returns
    only the equal-weight global mean; this function preserves the diffuser
    axis needed for the published protocol.
    """

    if diffuser_phase.ndim != 3 or diffuser_phase.shape[0] == 0:
        raise ValueError("diffuser_phase must have shape (n, H, W) with n > 0")

    metric_names = ("total", "negative_pearson", "energy", "pearson")
    totals = {
        name: torch.zeros(int(diffuser_phase.shape[0]), dtype=torch.float64)
        for name in metric_names
    }
    roi_totals = {
        name: torch.zeros(int(diffuser_phase.shape[0]), dtype=torch.float64)
        for name in LUO2022_ROI_METRIC_FIELDS
    } if include_roi_metrics else {}
    if include_roi_metrics and resized_shape[0] != resized_shape[1]:
        raise ValueError("ROI post-hoc metrics require a square resized target")
    object_count = 0
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
            stop = min(start + chunk_size, total_diffusers)
            output = forward(field, diffuser_phase[start:stop])
            components = luo2022_d2nn_components_per_pair(output, target)
            for name in metric_names:
                totals[name][start:stop] += (
                    components[name].detach().sum(dim=0).to(device="cpu", dtype=torch.float64)
                )
            if include_roi_metrics:
                roi_components = _luo2022_roi_components_per_pair(
                    output,
                    target,
                    input_size=int(resized_shape[0]),
                    full_canvas_pearson=components["pearson"],
                )
                for roi_name, values in roi_components.items():
                    for metric_name in ("pearson", "output_energy_fraction"):
                        row_name = f"roi_{roi_name}_{metric_name}"
                        roi_totals[row_name][start:stop] += (
                            values[metric_name]
                            .detach()
                            .sum(dim=0)
                            .to(device="cpu", dtype=torch.float64)
                        )
        object_count += int(image.shape[0])

    if object_count == 0:
        raise ValueError("evaluation loader yielded no batches")
    rows: list[dict[str, float | int]] = []
    for index in range(int(diffuser_phase.shape[0])):
        row: dict[str, float | int] = {
            "object_count": object_count,
            **{
                name: float((totals[name][index] / object_count).item())
                for name in metric_names
            },
        }
        if include_roi_metrics:
            row.update(
                {
                    name: float((roi_totals[name][index] / object_count).item())
                    for name in LUO2022_ROI_METRIC_FIELDS
                }
            )
        rows.append(row)
    return rows


@torch.no_grad()
def evaluate_luo2022_model_per_diffuser(
    model: Luo2022FourLayerD2NN,
    loader: DataLoader,
    diffuser_phase: torch.Tensor,
    *,
    resized_shape: tuple[int, int],
    canvas_shape: tuple[int, int],
    device: torch.device,
    max_batches: int | None = None,
    diffuser_chunk_size: int | None = None,
    include_roi_metrics: bool = False,
) -> list[dict[str, float | int]]:
    """Evaluate the standard Luo 2022 model separately for every diffuser."""

    model.eval()
    return _evaluate_luo2022_forward_per_diffuser(
        model,
        loader,
        diffuser_phase,
        resized_shape=resized_shape,
        canvas_shape=canvas_shape,
        device=device,
        max_batches=max_batches,
        diffuser_chunk_size=diffuser_chunk_size,
        include_roi_metrics=include_roi_metrics,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(tensor: torch.Tensor) -> str:
    """Return a device-independent hash for a tensor used as evidence input."""

    normalized = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(normalized.dtype).encode("utf-8"))
    digest.update(str(tuple(normalized.shape)).encode("utf-8"))
    digest.update(normalized.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class Luo2022FrozenRunArtifacts:
    """Verified immutable inputs needed for a post-hoc R0 diagnostic."""

    run_dir: Path
    runtime_config: dict[str, Any]
    contract: dict[str, Any]
    manifest: dict[str, Any]
    checkpoint: dict[str, Any]
    checkpoint_path: Path
    checkpoint_sha256: str
    runtime_config_sha256: str
    source_config_sha256: str
    manifest_sha256: str
    run_state_sha256: str
    source_config_integrity: str


def _luo2022_runtime_contract_projection(contract: dict[str, Any]) -> dict[str, Any]:
    """Return every frozen-contract field copied into the runtime snapshot."""

    return {
        "profile_id": contract["profile_id"],
        "source_freeze_version": contract["freeze_version"],
        "experiment_class": contract["experiment_class"],
        "comparison_level": contract["comparison_level"],
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
    }


def _validate_luo2022_run_local_contract(
    *,
    runtime_config: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    """Reject a source contract that cannot have produced the runtime snapshot."""

    expected = _luo2022_runtime_contract_projection(contract)
    observed = {key: runtime_config.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            "run-local source configuration does not match immutable fields in the "
            "frozen runtime configuration"
        )


def _load_luo2022_frozen_run_artifacts(
    *,
    run_dir: Path,
    config_path: Path,
    device: torch.device,
) -> Luo2022FrozenRunArtifacts:
    """Load a completed R0 run after validating its frozen provenance."""

    runtime_config_path = run_dir / "config.json"
    source_config_path = run_dir / "source_config.json"
    manifest_path = run_dir / "manifest.json"
    run_state_path = run_dir / "run_state.json"
    final_checkpoint_path = run_dir / "checkpoints" / "luo2022_d2nn.pt"
    required_paths = (
        runtime_config_path,
        source_config_path,
        manifest_path,
        run_state_path,
        final_checkpoint_path,
    )
    if not all(path.is_file() for path in required_paths):
        raise FileNotFoundError(
            "completed Luo 2022 diagnosis requires runtime config, frozen source config, "
            "manifest, completed state, and final checkpoint"
        )

    runtime_config = load_config(runtime_config_path)
    manifest = load_config(manifest_path)
    run_state = load_config(run_state_path)
    if run_state.get("status") != "completed":
        raise ValueError("diagnosis requires a completed frozen R0 run")
    contract_path = source_config_path
    contract = load_config(contract_path)
    requested_contract = load_config(config_path)
    if requested_contract != contract:
        raise ValueError(
            "requested diagnosis config does not exactly match the run-local frozen source "
            "configuration"
        )
    _validate_luo2022_run_local_contract(
        runtime_config=runtime_config,
        contract=contract,
    )
    expected_epochs = int(runtime_config["runtime"]["epochs"])
    if int(run_state.get("target_epochs", -1)) != expected_epochs:
        raise ValueError("completed run state target_epochs does not match frozen runtime configuration")
    if int(run_state.get("completed_epoch", -1)) != expected_epochs:
        raise ValueError("completed run state epoch does not match frozen runtime configuration")
    freeze_version = str(contract["freeze_version"])
    if runtime_config.get("source_freeze_version") != freeze_version:
        raise ValueError("runtime configuration freeze version does not match source configuration")
    if manifest.get("source_freeze_version") != freeze_version:
        raise ValueError("manifest freeze version does not match source configuration")
    if manifest.get("profile_id") != contract.get("profile_id"):
        raise ValueError("manifest profile does not match source configuration")

    checkpoint = torch.load(final_checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("source_freeze_version") != freeze_version:
        raise ValueError("checkpoint freeze version does not match source configuration")
    if checkpoint.get("runtime_config") != runtime_config:
        raise ValueError("checkpoint runtime configuration does not match the frozen run")
    if "model" not in checkpoint:
        raise ValueError("frozen R0 checkpoint does not contain model parameters")
    source_config_sha256 = _sha256_file(contract_path)
    recorded_source_hashes = {
        name: value
        for name, value in {
            "manifest": manifest.get("source_config_sha256"),
            "checkpoint": checkpoint.get("source_config_sha256"),
        }.items()
        if value is not None
    }
    for artifact_name, recorded_hash in recorded_source_hashes.items():
        if str(recorded_hash) != source_config_sha256:
            raise ValueError(
                f"{artifact_name} source configuration hash does not match the run-local copy"
            )
    optics_config = _luo2022_optics_config_from_frozen_run(runtime_config, contract)
    try:
        checkpoint_model = Luo2022FourLayerD2NN(optics_config).to(device)
        checkpoint_model.load_state_dict(checkpoint["model"], strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "final checkpoint model state is incompatible with the frozen optical configuration"
        ) from exc
    source_config_integrity = (
        "sha256_bound_by_manifest_and_checkpoint"
        if set(recorded_source_hashes) == {"manifest", "checkpoint"}
        else (
            "sha256_bound_by_" + "_and_".join(sorted(recorded_source_hashes))
            if recorded_source_hashes
            else "requested_frozen_config_and_runtime_projection_verified"
        )
    )

    return Luo2022FrozenRunArtifacts(
        run_dir=run_dir,
        runtime_config=runtime_config,
        contract=contract,
        manifest=manifest,
        checkpoint=checkpoint,
        checkpoint_path=final_checkpoint_path,
        checkpoint_sha256=_sha256_file(final_checkpoint_path),
        runtime_config_sha256=_sha256_file(runtime_config_path),
        source_config_sha256=source_config_sha256,
        manifest_sha256=_sha256_file(manifest_path),
        run_state_sha256=_sha256_file(run_state_path),
        source_config_integrity=source_config_integrity,
    )


@dataclass
class Luo2022FrozenBackendFrontend:
    """Loaded R0 frontend plus immutable evidence captured around backend work."""

    artifacts: Luo2022FrozenRunArtifacts
    model: Luo2022FourLayerD2NN
    integrity_before: dict[str, Any]
    integrity_after: dict[str, Any] | None = None


def _load_luo2022_backend_contract(
    backend_config_path: Path = DEFAULT_LUO2022_BACKEND_CONFIG,
    *,
    artifacts: Luo2022FrozenRunArtifacts | None = None,
) -> dict[str, Any]:
    """Load and validate the public fixed-four-layer backend contract."""

    contract = load_config(backend_config_path)
    required_top_level = {
        "schema_version",
        "profile_id",
        "status",
        "experiment_class",
        "r0_reference",
        "assignment",
        "cache",
        "input_scaling",
        "model",
        "variants",
        "training",
        "evaluation",
        "claim_boundary",
    }
    missing = sorted(required_top_level.difference(contract))
    if missing:
        raise ValueError(f"backend contract is missing required fields: {', '.join(missing)}")
    if int(contract["schema_version"]) != 1:
        raise ValueError("unsupported fixed-depth backend contract schema")
    if contract["profile_id"] != "luo2022_fixed4_backend":
        raise ValueError("backend contract profile_id is not the fixed-four-layer profile")
    if contract["status"] != "exploratory fixed-depth backend ablation":
        raise ValueError("backend contract has an unsupported claim-status label")
    if int(contract["r0_reference"].get("optical_layers", -1)) != 4:
        raise ValueError("backend contract must reference exactly four frozen optical layers")
    if contract["r0_reference"].get("frozen") is not True:
        raise ValueError("backend contract must freeze the R0 optical frontend")
    if int(contract["model"].get("base_channels", -1)) != 4:
        raise ValueError("fixed-depth backend contract requires base_channels=4")
    scaling = contract["input_scaling"]
    if (
        scaling.get("method") != "per_operator_global_dataset_max"
        or scaling.get("fit_split") != "train_only"
        or scaling.get("per_image_normalization") is not False
        or scaling.get("clipping") is not False
    ):
        raise ValueError("backend input scaling must be the frozen train-only global-max rule")
    loss = contract["training"]["loss"]
    if (
        loss.get("reconstruction") != "L1"
        or float(loss.get("reconstruction_weight", -1.0)) != 1.0
        or float(loss.get("B0_adversarial_weight", -1.0)) != 0.0
        or float(loss.get("R1_adversarial_weight", -1.0)) != 0.0
    ):
        raise ValueError("backend contract must preserve the common unit-weight L1 loss")
    if artifacts is not None:
        reference = contract["r0_reference"]
        if reference.get("profile_id") != artifacts.contract.get("profile_id"):
            raise ValueError("backend contract R0 profile does not match the frozen run")
        if reference.get("freeze_version") != artifacts.contract.get("freeze_version"):
            raise ValueError("backend contract R0 freeze version does not match the frozen run")
        if int(artifacts.contract["d2nn"]["layers"]) != 4:
            raise ValueError("the verified R0 run is not a four-layer optical model")
    return contract


def _backend_r0_fingerprint(artifacts: Luo2022FrozenRunArtifacts) -> str:
    """Bind every immutable R0 input used by the backend experiment."""

    return canonical_sha256(
        {
            "profile_id": artifacts.contract["profile_id"],
            "freeze_version": artifacts.contract["freeze_version"],
            "checkpoint_sha256": artifacts.checkpoint_sha256,
            "runtime_config_sha256": artifacts.runtime_config_sha256,
            "source_config_sha256": artifacts.source_config_sha256,
            "manifest_sha256": artifacts.manifest_sha256,
            "run_state_sha256": artifacts.run_state_sha256,
            "runtime_contract_projection": _luo2022_runtime_contract_projection(
                artifacts.contract
            ),
        }
    )


def _stable_torch_value(value: Any) -> Any:
    """Convert tensor-bearing state into canonical JSON-compatible evidence."""

    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": _sha256_tensor(value),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _stable_torch_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_torch_value(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported checkpoint fingerprint value: {type(value).__name__}")


def _backend_state_sha256(state: Mapping[str, Any]) -> str:
    return canonical_sha256(_stable_torch_value(state))


def _backend_r0_integrity_snapshot(
    artifacts: Luo2022FrozenRunArtifacts,
    model: Luo2022FourLayerD2NN,
) -> dict[str, Any]:
    """Return current file/model hashes and the required frozen execution state."""

    file_paths = {
        "checkpoint_sha256": artifacts.checkpoint_path,
        "runtime_config_sha256": artifacts.run_dir / "config.json",
        "source_config_sha256": artifacts.run_dir / "source_config.json",
        "manifest_sha256": artifacts.run_dir / "manifest.json",
        "run_state_sha256": artifacts.run_dir / "run_state.json",
    }
    snapshot = {name: _sha256_file(path) for name, path in file_paths.items()}
    snapshot.update(
        {
            "phase_sha256": _sha256_tensor(model.phase),
            "model_state_sha256": _backend_state_sha256(model.state_dict()),
            "model_training": bool(model.training),
            "all_parameters_frozen": all(
                not parameter.requires_grad for parameter in model.parameters()
            ),
            "all_parameter_gradients_none": all(
                parameter.grad is None for parameter in model.parameters()
            ),
        }
    )
    return snapshot


def _load_luo2022_frozen_backend_model(
    *,
    run_dir: Path,
    config_path: Path = DEFAULT_LUO2022_CONFIG,
    device: torch.device,
) -> tuple[Luo2022FrozenRunArtifacts, Luo2022FourLayerD2NN, dict[str, Any]]:
    """Strictly load R0 for backend inference in eval mode with no trainable state."""

    artifacts = _load_luo2022_frozen_run_artifacts(
        run_dir=run_dir,
        config_path=config_path,
        device=device,
    )
    optics_config = _luo2022_optics_config_from_frozen_run(
        artifacts.runtime_config,
        artifacts.contract,
    )
    model = Luo2022FourLayerD2NN(optics_config).to(device)
    try:
        model.load_state_dict(artifacts.checkpoint["model"], strict=True)
    except RuntimeError as error:
        raise ValueError("frozen R0 model state is incompatible with its optical config") from error
    model.eval()
    set_requires_grad(model, False)
    for parameter in model.parameters():
        parameter.grad = None
    snapshot = _backend_r0_integrity_snapshot(artifacts, model)
    if snapshot["model_training"] or not snapshot["all_parameters_frozen"]:
        raise RuntimeError("frozen R0 frontend was not placed in eval/frozen state")
    if not snapshot["all_parameter_gradients_none"]:
        raise RuntimeError("frozen R0 frontend unexpectedly contains parameter gradients")
    if snapshot["checkpoint_sha256"] != artifacts.checkpoint_sha256:
        raise RuntimeError("frozen R0 checkpoint changed while it was being loaded")
    return artifacts, model, snapshot


def _assert_luo2022_backend_r0_unchanged(
    frontend: Luo2022FrozenBackendFrontend,
) -> dict[str, Any]:
    """Verify that a backend action did not alter R0 files, phases, or gradients."""

    after = _backend_r0_integrity_snapshot(frontend.artifacts, frontend.model)
    before = frontend.integrity_before
    immutable_fields = (
        "checkpoint_sha256",
        "runtime_config_sha256",
        "source_config_sha256",
        "manifest_sha256",
        "run_state_sha256",
        "phase_sha256",
        "model_state_sha256",
    )
    changed = [name for name in immutable_fields if before.get(name) != after.get(name)]
    if changed:
        raise RuntimeError("frozen R0 integrity changed: " + ", ".join(changed))
    if (
        after["model_training"]
        or not after["all_parameters_frozen"]
        or not after["all_parameter_gradients_none"]
    ):
        raise RuntimeError("frozen R0 execution-state invariant was violated")
    frontend.integrity_after = after
    return {
        "before": before,
        "after": after,
        "hashes_unchanged": True,
        "phase_gradients_none": True,
        "eval_mode": True,
        "requires_grad_false": True,
    }


@contextmanager
def _verified_luo2022_backend_frontend(
    *,
    run_dir: Path,
    config_path: Path,
    device: torch.device,
):
    """Yield a frozen R0 model and always audit it again on exit."""

    artifacts, model, before = _load_luo2022_frozen_backend_model(
        run_dir=run_dir,
        config_path=config_path,
        device=device,
    )
    frontend = Luo2022FrozenBackendFrontend(
        artifacts=artifacts,
        model=model,
        integrity_before=before,
    )
    try:
        yield frontend
    finally:
        _assert_luo2022_backend_r0_unchanged(frontend)


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON through an fsynced sibling and atomically replace the target."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_torch_save(path: Path, payload: Any) -> None:
    """Atomically save a torch payload without exposing partial checkpoints."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _is_path_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _require_backend_output_outside_r0(output_dir: Path, run_dir: Path) -> None:
    if _is_path_within(output_dir, run_dir) or _is_path_within(run_dir, output_dir):
        raise ValueError("backend output/cache directory must be separate from the frozen R0 run")


def _prepare_fresh_backend_output(output_dir: Path, *, resume: bool) -> None:
    """Create a fresh output, or require explicit resume for an existing one."""

    if output_dir.exists():
        contents = [path for path in output_dir.iterdir() if not path.name.startswith(".")]
        if contents and not resume:
            raise FileExistsError(
                "backend output directory is not fresh; use --resume only for a compatible run"
            )
        if resume and not contents:
            raise FileNotFoundError("cannot resume an empty backend output directory")
    elif resume:
        raise FileNotFoundError("cannot resume a backend output directory that does not exist")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "samples").mkdir(exist_ok=True)


def _backend_cache_manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "root_fingerprint"}
    return canonical_sha256(payload)


def _load_backend_cache_manifest(cache_dir: Path) -> dict[str, Any]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("backend cache manifest is missing")
    manifest = load_config(manifest_path)
    if manifest.get("protocol") != LUO2022_BACKEND_CACHE_PROTOCOL:
        raise ValueError("unsupported backend cache protocol")
    if manifest.get("root_fingerprint") != _backend_cache_manifest_fingerprint(manifest):
        raise ValueError("backend cache root fingerprint mismatch")
    if manifest.get("status") != "complete":
        raise ValueError("backend cache is not complete")
    for operator_id, split_map in manifest.get("caches", {}).items():
        if operator_id not in {"direct_no_d2nn", "r0_four_layer"}:
            raise ValueError("backend cache contains an unknown optical operator")
        train_manifest: dict[str, Any] | None = None
        for split, record in split_map.items():
            cache_root = cache_dir / record["path"]
            verified = verify_luo2022_intensity_cache(cache_root)
            if verified["root_fingerprint"] != record["root_fingerprint"]:
                raise ValueError("backend cache child fingerprint does not match its manifest")
            if verified.get("split") != split:
                raise ValueError("backend cache child split does not match its manifest")
            if split == "train":
                train_manifest = verified
        if train_manifest is None:
            raise ValueError(f"backend cache operator {operator_id} has no training split")
        for split, record in split_map.items():
            verified = verify_luo2022_intensity_cache(cache_dir / record["path"])
            if verified["scale"]["value"] != train_manifest["scale"]["value"]:
                raise ValueError("non-training cache does not reuse the frozen training scale")
            if verified["operator_id"] != train_manifest["operator_id"]:
                raise ValueError("cache operator IDs differ across splits")
    return manifest


def _backend_dataset(
    cache_dir: Path,
    *,
    operator_id: str,
    split: str = "train",
) -> Luo2022CachedIntensityDataset:
    manifest = _load_backend_cache_manifest(cache_dir)
    try:
        relative_path = manifest["caches"][operator_id][split]["path"]
    except KeyError as error:
        raise ValueError(f"backend cache is missing {operator_id}/{split}") from error
    return Luo2022CachedIntensityDataset(cache_dir / relative_path)


def _load_luo2022_training_diffuser_bank(
    artifacts: Luo2022FrozenRunArtifacts,
    *,
    small_run: bool,
    requested_count: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load the original immutable per-epoch diffuser banks in epoch/index order."""

    epoch_count = int(artifacts.runtime_config["runtime"]["epochs"])
    phase_banks: list[torch.Tensor] = []
    file_records: list[dict[str, Any]] = []
    for epoch in range(1, epoch_count + 1):
        phase_path = artifacts.run_dir / "diffusers" / f"training_epoch_{epoch:03d}.pt"
        if not phase_path.is_file():
            raise FileNotFoundError(f"frozen training diffuser bank is missing: {phase_path.name}")
        phase_bank = torch.load(phase_path, map_location="cpu", weights_only=True)
        if (
            phase_bank.ndim != 3
            or tuple(phase_bank.shape[-2:])
            != tuple(artifacts.runtime_config["runtime"]["grid_size"] for _ in range(2))
            or not torch.isfinite(phase_bank).all()
        ):
            raise ValueError(f"invalid frozen training diffuser bank: {phase_path.name}")
        phase_banks.append(phase_bank.to(dtype=torch.float32))
        file_records.append(
            {
                "epoch": epoch,
                "file": f"diffusers/{phase_path.name}",
                "sha256": _sha256_file(phase_path),
                "count": int(phase_bank.shape[0]),
            }
        )
    phases = torch.cat(phase_banks, dim=0)
    if small_run:
        resolved_count = min(int(requested_count), int(phases.shape[0]))
    else:
        resolved_count = int(requested_count)
        if int(phases.shape[0]) != resolved_count:
            raise ValueError(
                "full backend cache requires the exact configured frozen training diffuser count"
            )
    if resolved_count <= 0:
        raise ValueError("backend cache requires at least one frozen training diffuser")
    phases = phases[:resolved_count].contiguous()
    return phases, {
        "ordering": "training_epoch_ascending_then_within_epoch_index_ascending",
        "available_count": int(sum(row["count"] for row in file_records)),
        "used_count": resolved_count,
        "phase_sha256": _sha256_tensor(phases),
        "files": file_records,
    }


def _backend_unpack_image_label(item: Any) -> tuple[torch.Tensor, int]:
    if isinstance(item, tuple) and len(item) >= 2:
        image, label = item[0], int(item[1])
    elif isinstance(item, Mapping):
        image, label = item["image"], int(item.get("label", 0))
    else:
        image, label = item, 0
    if not isinstance(image, torch.Tensor):
        from torchvision.transforms.functional import to_tensor

        image = to_tensor(image)
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.ndim != 3:
        raise ValueError("backend MNIST image must have shape (C,H,W)")
    if image.shape[0] != 1:
        image = image.mean(dim=0, keepdim=True)
    return image.to(dtype=torch.float32).clamp(0.0, 1.0), label


@torch.no_grad()
def _backend_forward_assigned_diffusers(
    model: Luo2022FourLayerD2NN,
    object_field: torch.Tensor,
    diffuser_bank: torch.Tensor,
    diffuser_ids: Sequence[int] | torch.Tensor,
    *,
    operator_id: str,
) -> torch.Tensor:
    """Apply one assigned diffuser per object without forming a Cartesian product."""

    if isinstance(diffuser_ids, torch.Tensor):
        ids = [int(value) for value in diffuser_ids.detach().cpu().tolist()]
    else:
        ids = [int(value) for value in diffuser_ids]
    if len(ids) != int(object_field.shape[0]):
        raise ValueError("one diffuser ID is required for every backend object")
    if any(diffuser_id < 0 or diffuser_id >= int(diffuser_bank.shape[0]) for diffuser_id in ids):
        raise ValueError("backend assignment references a diffuser outside the frozen bank")
    if operator_id == "direct_no_d2nn":
        forward = model.forward_without_diffractive_layers
    elif operator_id == "r0_four_layer":
        forward = model.forward
    else:
        raise ValueError(f"unsupported backend optical operator {operator_id}")
    output = object_field.real.new_empty(
        (object_field.shape[0], 1, *object_field.shape[-2:])
    )
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, diffuser_id in enumerate(ids):
        grouped[diffuser_id].append(index)
    for diffuser_id, indices in grouped.items():
        index_tensor = torch.tensor(indices, device=object_field.device, dtype=torch.long)
        values = forward(
            object_field.index_select(0, index_tensor),
            diffuser_bank[diffuser_id : diffuser_id + 1],
        )
        output.index_copy_(0, index_tensor, values)
    return output


def _materialize_backend_cache_split(
    *,
    writer: Luo2022IntensityCacheWriter,
    model: Luo2022FourLayerD2NN,
    diffuser_bank: torch.Tensor,
    dataset: Any,
    source_indices: Sequence[int],
    assignment_rows: Sequence[Mapping[str, Any]],
    operator_id: str,
    resized_shape: tuple[int, int],
    canvas_shape: tuple[int, int],
    device: torch.device,
    shard_size: int,
) -> None:
    if len(source_indices) != len(assignment_rows):
        raise ValueError("backend cache source indices and assignment rows differ in length")
    # Keep the assignment identity unchanged while writing records in diffuser-major
    # order.  A paper-scale assignment has only 25 train objects per diffuser;
    # grouping them avoids tens of thousands of one-object optical forwards and
    # gives both operators the same byte-level cache row order.
    ordered_pairs = sorted(
        zip(source_indices, assignment_rows, strict=True),
        key=lambda pair: (int(pair[1]["diffuser_id"]), int(pair[1]["row_id"])),
    )
    ordered_source_indices = [int(source_index) for source_index, _row in ordered_pairs]
    ordered_assignment_rows = [row for _source_index, row in ordered_pairs]
    existing_rows = int(writer.manifest["row_count"])
    for start in range(existing_rows, len(ordered_assignment_rows), shard_size):
        stop = min(start + shard_size, len(ordered_assignment_rows))
        images: list[torch.Tensor] = []
        for source_index, row in zip(
            ordered_source_indices[start:stop],
            ordered_assignment_rows[start:stop],
            strict=True,
        ):
            image, label = _backend_unpack_image_label(dataset[int(source_index)])
            if label != int(row["label"]):
                raise ValueError("MNIST label does not match frozen backend assignment")
            images.append(image)
        image_batch = torch.stack(images).to(device)
        target = prepare_luo2022_amplitude(
            image_batch,
            resized_shape=resized_shape,
            canvas_shape=canvas_shape,
        )
        raw = _backend_forward_assigned_diffusers(
            model,
            amplitude_to_complex_field(target),
            diffuser_bank,
            [int(row["diffuser_id"]) for row in ordered_assignment_rows[start:stop]],
            operator_id=operator_id,
        )
        writer.append_shard(raw.detach().cpu(), ordered_assignment_rows[start:stop])


def _backend_cache_operator_name(cli_name: str) -> str:
    mapping = {"direct": "direct_no_d2nn", "frozen_four_layer": "r0_four_layer"}
    try:
        return mapping[cli_name]
    except KeyError as error:
        raise ValueError(f"unsupported backend cache operator {cli_name}") from error


def run_luo2022_backend_cache(
    *,
    run_dir: Path,
    cache_dir: Path,
    config_path: Path = DEFAULT_LUO2022_CONFIG,
    backend_config_path: Path = DEFAULT_LUO2022_BACKEND_CONFIG,
    download: bool = False,
    device_name: str = "cpu",
    operators: tuple[str, ...] = ("direct", "frozen_four_layer"),
    shard_size: int = 512,
    small_run: bool = False,
    train_limit: int | None = None,
    validation_limit: int | None = None,
) -> dict[str, Any]:
    """Build shared balanced raw-intensity caches from the original R0 diffuser banks."""

    run_dir = Path(run_dir)
    cache_dir = Path(cache_dir)
    _require_backend_output_outside_r0(cache_dir, run_dir)
    if shard_size <= 0:
        raise ValueError("backend shard_size must be positive")
    operator_ids = tuple(dict.fromkeys(_backend_cache_operator_name(name) for name in operators))
    if not operator_ids:
        raise ValueError("at least one backend cache operator is required")
    device = select_device(device_name)

    with _verified_luo2022_backend_frontend(
        run_dir=run_dir,
        config_path=config_path,
        device=device,
    ) as frontend:
        backend_contract = _load_luo2022_backend_contract(
            backend_config_path,
            artifacts=frontend.artifacts,
        )
        assignment_contract = backend_contract["assignment"]
        configured_train_count = (
            int(assignment_contract["train_object_ids"][1])
            - int(assignment_contract["train_object_ids"][0])
            + 1
        )
        configured_validation_count = (
            int(assignment_contract["validation_object_ids"][1])
            - int(assignment_contract["validation_object_ids"][0])
            + 1
        )
        if not small_run and (train_limit is not None or validation_limit is not None):
            raise ValueError("backend cache object-count overrides require --small-run")
        resolved_train_count = (
            min(8, configured_train_count) if small_run and train_limit is None
            else configured_train_count if train_limit is None
            else int(train_limit)
        )
        resolved_validation_count = (
            min(4, configured_validation_count) if small_run and validation_limit is None
            else configured_validation_count if validation_limit is None
            else int(validation_limit)
        )
        if resolved_train_count <= 0 or resolved_validation_count <= 0:
            raise ValueError("backend cache train and validation limits must be positive")
        requested_diffusers = int(assignment_contract["training_diffusers"])
        diffuser_bank, diffuser_provenance = _load_luo2022_training_diffuser_bank(
            frontend.artifacts,
            small_run=small_run,
            requested_count=requested_diffusers,
        )
        runtime_values = frontend.artifacts.runtime_config["runtime"]
        dataset = build_torchvision_dataset(
            name="MNIST",
            root=DEFAULT_DATA_ROOT,
            train=True,
            image_size=int(frontend.artifacts.contract["input"]["original_shape"][0]),
            download=download,
        )
        if len(dataset) < resolved_train_count:
            raise ValueError("MNIST training split is smaller than the backend train limit")
        validation_start = int(assignment_contract["validation_object_ids"][0])
        source_validation_start = validation_start
        if len(dataset) < validation_start + resolved_validation_count:
            if not small_run or len(dataset) < resolved_train_count + resolved_validation_count:
                raise ValueError("MNIST training split is smaller than the backend validation range")
            source_validation_start = resolved_train_count

        train_labels = [
            _backend_unpack_image_label(dataset[index])[1]
            for index in range(resolved_train_count)
        ]
        validation_source_indices = list(
            range(source_validation_start, source_validation_start + resolved_validation_count)
        )
        validation_labels = [
            _backend_unpack_image_label(dataset[index])[1]
            for index in validation_source_indices
        ]
        assignment_seed = int(assignment_contract["seed"])
        diffusers_per_epoch = int(runtime_values["diffusers_per_epoch"])
        train_assignment = build_luo2022_fixed_depth_assignment(
            train_labels,
            num_diffusers=int(diffuser_bank.shape[0]),
            diffusers_per_epoch=diffusers_per_epoch,
            seed=assignment_seed,
            object_id_offset=int(assignment_contract["train_object_ids"][0]),
        )
        validation_assignment = build_luo2022_fixed_depth_assignment(
            validation_labels,
            num_diffusers=int(diffuser_bank.shape[0]),
            diffusers_per_epoch=diffusers_per_epoch,
            seed=assignment_seed,
            object_id_offset=validation_start,
        )
        request = {
            "protocol": LUO2022_BACKEND_CACHE_PROTOCOL,
            "backend_contract_sha256": _sha256_file(backend_config_path),
            "r0_fingerprint": _backend_r0_fingerprint(frontend.artifacts),
            "operators": list(operator_ids),
            "small_run": bool(small_run),
            "train_count": resolved_train_count,
            "validation_count": resolved_validation_count,
            "shard_size": int(shard_size),
            "train_assignment_sha256": train_assignment["root_sha"],
            "validation_assignment_sha256": validation_assignment["root_sha"],
            "diffuser_phase_sha256": diffuser_provenance["phase_sha256"],
        }
        request_fingerprint = canonical_sha256(request)
        manifest_path = cache_dir / "manifest.json"
        if manifest_path.is_file():
            saved = load_config(manifest_path)
            if saved.get("request_fingerprint") != request_fingerprint:
                raise ValueError("existing backend cache is incompatible with this request")
            if saved.get("status") == "complete":
                verified = _load_backend_cache_manifest(cache_dir)
                return {"manifest": verified, "cache_dir": str(cache_dir)}
        elif cache_dir.exists() and any(
            path for path in cache_dir.iterdir() if not path.name.startswith(".")
        ):
            raise ValueError("backend cache directory contains files but no recoverable manifest")

        cache_dir.mkdir(parents=True, exist_ok=True)
        building_manifest = {
            "schema_version": LUO2022_BACKEND_SCHEMA_VERSION,
            "protocol": LUO2022_BACKEND_CACHE_PROTOCOL,
            "status": "building",
            "request": request,
            "request_fingerprint": request_fingerprint,
            "r0": {
                "fingerprint": request["r0_fingerprint"],
                "checkpoint_sha256": frontend.artifacts.checkpoint_sha256,
                "source_config_sha256": frontend.artifacts.source_config_sha256,
                "phase_sha256_before": frontend.integrity_before["phase_sha256"],
            },
            "assignment": {
                "rule": "deterministic_sha256_rank_round_robin_balanced_project_choice",
                "shared_across_operators": True,
                "train_sha256": train_assignment["root_sha"],
                "validation_sha256": validation_assignment["root_sha"],
                "train_count": resolved_train_count,
                "validation_count": resolved_validation_count,
            },
            "diffusers": diffuser_provenance,
            "input_scaling": {
                "method": "per_operator_global_dataset_max",
                "fit_split": "train_only",
                "per_image_normalization": False,
                "clipping": False,
            },
            "caches": {},
        }
        building_manifest["root_fingerprint"] = _backend_cache_manifest_fingerprint(
            building_manifest
        )
        _atomic_write_json(manifest_path, building_manifest)
        _atomic_write_json(
            cache_dir / "config.json",
            {
                "schema_version": LUO2022_BACKEND_SCHEMA_VERSION,
                "protocol": LUO2022_BACKEND_CACHE_PROTOCOL,
                "request": request,
                "request_fingerprint": request_fingerprint,
                "assignment": building_manifest["assignment"],
                "input_scaling": building_manifest["input_scaling"],
                "r0": building_manifest["r0"],
            },
        )
        _atomic_write_json(cache_dir / "train_assignment.json", train_assignment)
        _atomic_write_json(cache_dir / "validation_assignment.json", validation_assignment)

        canvas_shape = tuple(int(value) for value in diffuser_bank.shape[-2:])
        resized_shape = (int(runtime_values["input_size"]),) * 2
        # The full 2,000 x 240 x 240 bank is roughly 440 MiB.  Transfer it
        # exactly once and reuse it across shards, splits, and operators while
        # retaining the CPU tensor above for portable provenance hashing.
        diffuser_bank_device = diffuser_bank.to(device)
        caches: dict[str, Any] = {}
        for operator_id in operator_ids:
            split_manifests: dict[str, Any] = {}
            train_root = cache_dir / operator_id / "train"
            train_writer = Luo2022IntensityCacheWriter(
                train_root,
                operator_id=operator_id,
                split="train",
                assignment_sha=train_assignment["root_sha"],
                r0_fingerprint=request["r0_fingerprint"],
                expected_shape=(1, *canvas_shape),
                expected_rows=resolved_train_count,
            )
            _materialize_backend_cache_split(
                writer=train_writer,
                model=frontend.model,
                diffuser_bank=diffuser_bank_device,
                dataset=dataset,
                source_indices=[
                    int(row["object_id"])
                    - int(assignment_contract["train_object_ids"][0])
                    for row in train_assignment["rows"]
                ],
                assignment_rows=train_assignment["rows"],
                operator_id=operator_id,
                resized_shape=resized_shape,
                canvas_shape=canvas_shape,
                device=device,
                shard_size=shard_size,
            )
            train_manifest = train_writer.finalize()
            split_manifests["train"] = {
                "path": f"{operator_id}/train",
                "root_fingerprint": train_manifest["root_fingerprint"],
                "row_count": train_manifest["row_count"],
                "scale": train_manifest["scale"],
            }

            validation_root = cache_dir / operator_id / "validation"
            validation_writer = Luo2022IntensityCacheWriter(
                validation_root,
                operator_id=operator_id,
                split="validation",
                assignment_sha=validation_assignment["root_sha"],
                r0_fingerprint=request["r0_fingerprint"],
                expected_shape=(1, *canvas_shape),
                expected_rows=resolved_validation_count,
            )
            _materialize_backend_cache_split(
                writer=validation_writer,
                model=frontend.model,
                diffuser_bank=diffuser_bank_device,
                dataset=dataset,
                source_indices=[
                    source_validation_start
                    + int(row["object_id"])
                    - validation_start
                    for row in validation_assignment["rows"]
                ],
                assignment_rows=validation_assignment["rows"],
                operator_id=operator_id,
                resized_shape=resized_shape,
                canvas_shape=canvas_shape,
                device=device,
                shard_size=shard_size,
            )
            validation_manifest = validation_writer.finalize(
                frozen_scale=make_luo2022_frozen_scale(train_manifest)
            )
            split_manifests["validation"] = {
                "path": f"{operator_id}/validation",
                "root_fingerprint": validation_manifest["root_fingerprint"],
                "row_count": validation_manifest["row_count"],
                "scale": validation_manifest["scale"],
            }
            caches[operator_id] = split_manifests

        integrity = _assert_luo2022_backend_r0_unchanged(frontend)
        complete_manifest = {
            **building_manifest,
            "status": "complete",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "caches": caches,
            "r0": {
                **building_manifest["r0"],
                "phase_sha256_after": integrity["after"]["phase_sha256"],
                "hashes_unchanged": integrity["hashes_unchanged"],
                "eval_no_grad": True,
            },
            "runtime": run_metadata(),
            "artifacts": {
                "config": "config.json",
                "train_assignment": "train_assignment.json",
                "validation_assignment": "validation_assignment.json",
            },
        }
        complete_manifest["root_fingerprint"] = _backend_cache_manifest_fingerprint(
            complete_manifest
        )
        _atomic_write_json(manifest_path, complete_manifest)
        verified = _load_backend_cache_manifest(cache_dir)
        return {"manifest": verified, "cache_dir": str(cache_dir)}


def backend_epoch_permutation(*, sample_count: int, seed: int, global_epoch: int) -> list[int]:
    """Return the fixed object order shared by matched backend conditions."""

    if sample_count <= 0 or global_epoch <= 0:
        raise ValueError("sample_count and global_epoch must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(global_epoch) * 1_000_003 + 4_202_022)
    return torch.randperm(sample_count, generator=generator).tolist()


def _backend_epoch_order_record(
    *,
    sample_count: int,
    seed: int,
    global_epoch: int,
    batch_size: int,
    max_updates: int | None,
) -> tuple[list[int], dict[str, Any]]:
    full_order = backend_epoch_permutation(
        sample_count=sample_count,
        seed=seed,
        global_epoch=global_epoch,
    )
    update_budget = math.ceil(sample_count / batch_size)
    if max_updates is not None:
        if max_updates <= 0:
            raise ValueError("backend max_updates_per_epoch must be positive")
        update_budget = min(update_budget, int(max_updates))
    consumed_order = full_order[: min(sample_count, update_budget * batch_size)]
    return consumed_order, {
        "global_epoch": int(global_epoch),
        "algorithm": "torch_randperm_seed_plus_global_epoch_domain_v1",
        "full_order_sha256": canonical_sha256(full_order),
        "consumed_order_sha256": canonical_sha256(consumed_order),
        "sample_count": int(sample_count),
        "consumed_count": len(consumed_order),
        "generator_update_budget": int(update_budget),
    }


def _initialized_backend_generator(*, seed: int, device: torch.device) -> UNetReconstructor:
    """Construct the frozen base-4 U-Net without perturbing caller RNG state."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        generator = UNetReconstructor(base_channels=4)
    return generator.to(device)


def _initialized_backend_discriminator(
    *, seed: int, device: torch.device
) -> PatchDiscriminator:
    """Construct the base-4 PatchGAN in a branch-local RNG domain."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) + 8_402_031)
        discriminator = PatchDiscriminator(base_channels=4)
    return discriminator.to(device)


def backend_supervised_train_step(
    generator: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    source: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Perform the common unit-weight L1 generator update used by B0/R1/warm-up."""

    generator.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = generator(source)
    loss = F.l1_loss(prediction, target)
    if not torch.isfinite(loss):
        raise FloatingPointError("backend supervised loss is not finite")
    loss.backward()
    if not any(parameter.grad is not None for parameter in generator.parameters()):
        raise RuntimeError("backend generator received no gradients")
    optimizer.step()
    return {"l1": float(loss.detach()), "generator_total": float(loss.detach())}


def backend_gan_train_step(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    adversarial_weight: float,
) -> dict[str, float]:
    """Perform one matched-budget R2 update: one D step and one G step."""

    if adversarial_weight <= 0:
        raise ValueError("R2 adversarial_weight must be positive")
    generator.train()
    discriminator.train()
    # Reuse one generator forward for both losses.  Besides avoiding duplicate
    # compute, this keeps U-Net BatchNorm running-stat updates matched one-for-
    # one with R1; R2's only generator-side change is then the adversarial term.
    fake = generator(source)
    discriminator_optimizer.zero_grad(set_to_none=True)
    discriminator_real = adversarial_loss(
        discriminator(source, target), target_is_real=True
    )
    discriminator_fake = adversarial_loss(
        discriminator(source, fake.detach()), target_is_real=False
    )
    discriminator_total = 0.5 * (discriminator_real + discriminator_fake)
    if not torch.isfinite(discriminator_total):
        raise FloatingPointError("backend discriminator loss is not finite")
    discriminator_total.backward()
    if not any(parameter.grad is not None for parameter in discriminator.parameters()):
        raise RuntimeError("backend discriminator received no gradients")
    discriminator_optimizer.step()

    generator_optimizer.zero_grad(set_to_none=True)
    l1 = F.l1_loss(fake, target)
    set_requires_grad(discriminator, False)
    try:
        adversarial = adversarial_loss(
            discriminator(source, fake), target_is_real=True
        )
        generator_total = l1 + float(adversarial_weight) * adversarial
        if not torch.isfinite(generator_total):
            raise FloatingPointError("backend R2 generator loss is not finite")
        generator_total.backward()
        if not any(parameter.grad is not None for parameter in generator.parameters()):
            raise RuntimeError("backend R2 generator received no gradients")
        generator_optimizer.step()
    finally:
        set_requires_grad(discriminator, True)
    return {
        "l1": float(l1.detach()),
        "adversarial": float(adversarial.detach()),
        "generator_total": float(generator_total.detach()),
        "discriminator_total": float(discriminator_total.detach()),
        "discriminator_real": float(discriminator_real.detach()),
        "discriminator_fake": float(discriminator_fake.detach()),
    }


def _backend_training_batch(
    cached_dataset: Luo2022CachedIntensityDataset,
    mnist_dataset: Any,
    indices: Sequence[int],
    *,
    resized_shape: tuple[int, int],
    canvas_shape: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    samples = [cached_dataset[int(index)] for index in indices]
    source = torch.stack([sample["input_intensity"] for sample in samples]).to(device)
    images: list[torch.Tensor] = []
    for sample in samples:
        object_id = int(sample["object_id"])
        source_index = object_id
        if source_index >= len(mnist_dataset):
            # Tiny test fixtures do not contain the physical MNIST indices
            # 50000--59999.  Their validation rows remain globally identified
            # but are read from the final split-local fixture entries.
            source_index = (
                len(mnist_dataset)
                - len(cached_dataset)
                + object_id
                - int(cached_dataset.object_id_min)
            )
        image, label = _backend_unpack_image_label(mnist_dataset[source_index])
        if label != int(sample["label"]):
            raise ValueError("cached backend label does not match MNIST source object")
        images.append(image)
    target = prepare_luo2022_amplitude(
        torch.stack(images).to(device),
        resized_shape=resized_shape,
        canvas_shape=canvas_shape,
    )
    return source, target


@torch.no_grad()
def _evaluate_backend_cached_generator(
    generator: torch.nn.Module,
    cached_dataset: Luo2022CachedIntensityDataset,
    mnist_dataset: Any,
    *,
    batch_size: int,
    resized_shape: tuple[int, int],
    canvas_shape: tuple[int, int],
    device: torch.device,
) -> dict[str, float]:
    generator.eval()
    totals: dict[str, float] = defaultdict(float)
    count = 0
    for start in range(0, len(cached_dataset), batch_size):
        stop = min(start + batch_size, len(cached_dataset))
        source, target = _backend_training_batch(
            cached_dataset,
            mnist_dataset,
            list(range(start, stop)),
            resized_shape=resized_shape,
            canvas_shape=canvas_shape,
            device=device,
        )
        prediction = generator(source)
        batch_metrics = reconstruction_metrics(prediction, target)
        for name, value in batch_metrics.items():
            totals[name] += float(value) * int(target.shape[0])
        count += int(target.shape[0])
    if count == 0:
        raise ValueError("backend validation cache is empty")
    return {name: value / count for name, value in totals.items()}


def _backend_condition_spec(
    backend_contract: Mapping[str, Any],
    condition: str,
    *,
    small_run: bool,
    epochs: int | None,
) -> dict[str, Any]:
    normalized = condition.lower()
    if normalized not in LUO2022_BACKEND_CONDITIONS:
        raise ValueError(f"unknown backend condition {condition}")
    if not small_run and epochs is not None:
        raise ValueError("backend epoch overrides require --small-run")
    if normalized == "b0":
        configured_epochs = int(backend_contract["variants"]["B0"]["epochs"]["total"])
        operator_id = "direct_no_d2nn"
    elif normalized == "warmup":
        configured_epochs = int(
            backend_contract["variants"]["R1"]["epochs"]["shared_supervised_warmup"]
        )
        operator_id = "r0_four_layer"
    else:
        field = "supervised_continuation" if normalized == "r1" else "adversarial_continuation"
        configured_epochs = int(backend_contract["variants"][normalized.upper()]["epochs"][field])
        operator_id = "r0_four_layer"
    resolved_epochs = int(epochs) if epochs is not None else (1 if small_run else configured_epochs)
    if resolved_epochs <= 0:
        raise ValueError("backend training epochs must be positive")
    return {
        "condition": normalized,
        "operator_id": operator_id,
        "stage_epochs": resolved_epochs,
        "configured_stage_epochs": configured_epochs,
        "uses_discriminator": normalized == "r2",
        "requires_warmup": normalized in {"r1", "r2"},
    }


def _backend_checkpoint_payload_fingerprint(checkpoint: Mapping[str, Any]) -> str:
    payload = {
        "protocol": checkpoint.get("protocol"),
        "compatibility_fingerprint": checkpoint.get("compatibility_fingerprint"),
        "condition": checkpoint.get("condition"),
        "completed_epoch": checkpoint.get("completed_epoch"),
        "global_completed_epoch": checkpoint.get("global_completed_epoch"),
        "generator_update_count": checkpoint.get("generator_update_count"),
        "generator_state_sha256": _backend_state_sha256(checkpoint.get("generator", {})),
        "generator_optimizer_sha256": _backend_state_sha256(
            checkpoint.get("generator_optimizer", {})
        ),
        "discriminator_state_sha256": (
            _backend_state_sha256(checkpoint["discriminator"])
            if checkpoint.get("discriminator") is not None
            else None
        ),
        "discriminator_optimizer_sha256": (
            _backend_state_sha256(checkpoint["discriminator_optimizer"])
            if checkpoint.get("discriminator_optimizer") is not None
            else None
        ),
        "history_sha256": canonical_sha256(checkpoint.get("history", [])),
        "epoch_order_sha256": canonical_sha256(checkpoint.get("epoch_orders", [])),
        "branch_start": checkpoint.get("branch_start"),
        "torch_rng_state_sha256": (
            _sha256_tensor(checkpoint["torch_rng_state"])
            if isinstance(checkpoint.get("torch_rng_state"), torch.Tensor)
            else None
        ),
        "cuda_rng_state_sha256": (
            [_sha256_tensor(state) for state in checkpoint["cuda_rng_state_all"]]
            if checkpoint.get("cuda_rng_state_all") is not None
            else None
        ),
    }
    return canonical_sha256(payload)


def _validate_backend_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    expected_compatibility_fingerprint: str | None = None,
) -> None:
    if checkpoint.get("protocol") != LUO2022_BACKEND_TRAINING_PROTOCOL:
        raise ValueError("unsupported backend checkpoint protocol")
    if (
        expected_compatibility_fingerprint is not None
        and checkpoint.get("compatibility_fingerprint")
        != expected_compatibility_fingerprint
    ):
        raise ValueError("resume checkpoint compatibility fingerprint mismatch")
    completed_epoch = int(checkpoint.get("completed_epoch", -1))
    history = checkpoint.get("history")
    epoch_orders = checkpoint.get("epoch_orders")
    if completed_epoch < 0 or not isinstance(history, list) or len(history) != completed_epoch:
        raise ValueError("backend checkpoint epoch and history are inconsistent")
    if not isinstance(epoch_orders, list) or len(epoch_orders) != completed_epoch:
        raise ValueError("backend checkpoint epoch-order history is inconsistent")
    expected_updates = sum(int(row["generator_updates"]) for row in history)
    if int(checkpoint.get("generator_update_count", -1)) != expected_updates:
        raise ValueError("backend checkpoint generator update count is inconsistent")
    if checkpoint.get("payload_fingerprint") != _backend_checkpoint_payload_fingerprint(
        checkpoint
    ):
        raise ValueError("backend checkpoint payload fingerprint mismatch")


def _load_backend_training_checkpoint(
    path: Path,
    *,
    device: torch.device,
    expected_compatibility_fingerprint: str | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"backend checkpoint is missing: {path.name}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    _validate_backend_checkpoint(
        checkpoint,
        expected_compatibility_fingerprint=expected_compatibility_fingerprint,
    )
    return checkpoint


def _save_backend_epoch_checkpoint(
    *,
    output_dir: Path,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    checkpoint["payload_fingerprint"] = _backend_checkpoint_payload_fingerprint(checkpoint)
    epoch = int(checkpoint["completed_epoch"])
    epoch_path = output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
    latest_path = output_dir / "checkpoints" / "latest.pt"
    _atomic_torch_save(epoch_path, checkpoint)
    _atomic_torch_save(latest_path, checkpoint)
    return checkpoint


def _backend_restore_rng(checkpoint: Mapping[str, Any], device: torch.device) -> None:
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if device.type == "cuda" and checkpoint.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
        )


def _completed_backend_run_checkpoint(
    run_dir: Path,
    *,
    device: torch.device,
    expected_condition: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_path = run_dir / "config.json"
    state_path = run_dir / "run_state.json"
    manifest_path = run_dir / "manifest.json"
    if not all(path.is_file() for path in (config_path, state_path, manifest_path)):
        raise FileNotFoundError("completed backend run is missing config/state/manifest")
    config = load_config(config_path)
    run_state = load_config(state_path)
    manifest = load_config(manifest_path)
    if run_state.get("status") != "completed" or manifest.get("status") != "completed":
        raise ValueError("backend branch requires a completed source run")
    if expected_condition is not None and config.get("condition") != expected_condition:
        raise ValueError(f"backend run is not the required {expected_condition} condition")
    checkpoint_path = run_dir / "checkpoints" / "latest.pt"
    checkpoint = _load_backend_training_checkpoint(
        checkpoint_path,
        device=device,
        expected_compatibility_fingerprint=config.get("compatibility_fingerprint"),
    )
    if int(checkpoint["completed_epoch"]) != int(config["stage_epochs"]):
        raise ValueError("completed backend checkpoint does not cover its configured stage")
    if run_state.get("latest_checkpoint_sha256") != _sha256_file(checkpoint_path):
        raise ValueError("completed backend checkpoint hash does not match run state")
    return config, checkpoint, manifest


def run_luo2022_backend_training(
    *,
    run_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    config_path: Path = DEFAULT_LUO2022_CONFIG,
    backend_config_path: Path = DEFAULT_LUO2022_BACKEND_CONFIG,
    condition: str,
    warmup_dir: Path | None = None,
    device_name: str = "cpu",
    small_run: bool = False,
    epochs: int | None = None,
    batch_size: int | None = None,
    max_updates_per_epoch: int | None = None,
    resume: bool = False,
    download: bool = False,
) -> dict[str, Any]:
    """Train B0/common warm-up/R1/R2 with fixed orders and strict resume."""

    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir)
    _require_backend_output_outside_r0(output_dir, run_dir)
    _require_backend_output_outside_r0(cache_dir, run_dir)
    if _is_path_within(output_dir, cache_dir) or _is_path_within(cache_dir, output_dir):
        raise ValueError("backend training output and shared cache must be separate")
    device = select_device(device_name)
    cache_manifest = _load_backend_cache_manifest(cache_dir)

    with _verified_luo2022_backend_frontend(
        run_dir=run_dir,
        config_path=config_path,
        device=device,
    ) as frontend:
        backend_contract = _load_luo2022_backend_contract(
            backend_config_path,
            artifacts=frontend.artifacts,
        )
        if cache_manifest["request"]["r0_fingerprint"] != _backend_r0_fingerprint(
            frontend.artifacts
        ):
            raise ValueError("backend cache does not belong to the verified frozen R0 run")
        spec = _backend_condition_spec(
            backend_contract,
            condition,
            small_run=small_run,
            epochs=epochs,
        )
        if max_updates_per_epoch is not None and not small_run:
            raise ValueError("backend update caps require --small-run")
        training_contract = backend_contract["training"]
        configured_batch_size = int(training_contract["batch_size"])
        if not small_run and batch_size is not None:
            raise ValueError("backend batch-size overrides require --small-run")
        resolved_batch_size = int(batch_size or (2 if small_run else configured_batch_size))
        if resolved_batch_size <= 0:
            raise ValueError("backend batch_size must be positive")
        seed = int(training_contract["seed"])
        seed_everything(seed)

        train_cache = _backend_dataset(
            cache_dir,
            operator_id=spec["operator_id"],
            split="train",
        )
        validation_cache = _backend_dataset(
            cache_dir,
            operator_id=spec["operator_id"],
            split="validation",
        )
        if len(train_cache) == 0:
            raise ValueError("backend training cache is empty")
        base_dataset = build_torchvision_dataset(
            name="MNIST",
            root=DEFAULT_DATA_ROOT,
            train=True,
            image_size=int(frontend.artifacts.contract["input"]["original_shape"][0]),
            download=download,
        )
        runtime_values = frontend.artifacts.runtime_config["runtime"]
        resized_shape = (int(runtime_values["input_size"]),) * 2
        canvas_shape = (int(runtime_values["grid_size"]),) * 2

        generator_optimizer_contract = training_contract["generator_optimizer"]
        discriminator_optimizer_contract = training_contract["discriminator_optimizer"]
        generator = _initialized_backend_generator(seed=seed, device=device)
        fresh_generator_initial_state_sha256 = _backend_state_sha256(
            generator.state_dict()
        )
        generator_optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=float(generator_optimizer_contract["learning_rate"]),
            betas=tuple(float(value) for value in generator_optimizer_contract["betas"]),
        )
        discriminator: PatchDiscriminator | None = None
        discriminator_optimizer: torch.optim.Optimizer | None = None
        global_epoch_offset = 0
        warmup_provenance: dict[str, Any] | None = None
        branch_start: dict[str, Any] | None = None
        branch_rng: dict[str, Any] | None = None
        if spec["requires_warmup"]:
            if warmup_dir is None:
                raise ValueError("R1 and R2 require --backend-warmup-dir")
            warmup_path = Path(warmup_dir)
            warmup_config, warmup_checkpoint, _warmup_manifest = (
                _completed_backend_run_checkpoint(
                    warmup_path,
                    device=device,
                    expected_condition="warmup",
                )
            )
            compatibility_fields = (
                "backend_contract_sha256",
                "r0_fingerprint",
                "cache_root_fingerprint",
                "cache_assignment_sha256",
                "seed",
                "batch_size",
                "base_channels",
                "fresh_generator_initial_state_sha256",
                "generator_optimizer",
                "reconstruction_loss",
                "max_updates_per_epoch",
                "small_run",
            )
            requested_values = {
                "backend_contract_sha256": _sha256_file(backend_config_path),
                "r0_fingerprint": _backend_r0_fingerprint(frontend.artifacts),
                "cache_root_fingerprint": cache_manifest["root_fingerprint"],
                "cache_assignment_sha256": cache_manifest["assignment"]["train_sha256"],
                "seed": seed,
                "batch_size": resolved_batch_size,
                "base_channels": 4,
                "fresh_generator_initial_state_sha256": (
                    fresh_generator_initial_state_sha256
                ),
                "generator_optimizer": generator_optimizer_contract,
                "reconstruction_loss": {"name": "L1", "weight": 1.0},
                "max_updates_per_epoch": max_updates_per_epoch,
                "small_run": bool(small_run),
            }
            mismatched = [
                name
                for name in compatibility_fields
                if warmup_config.get(name) != requested_values[name]
            ]
            if mismatched:
                raise ValueError(
                    "warm-up run is incompatible with the requested branch: "
                    + ", ".join(mismatched)
                )
            generator.load_state_dict(warmup_checkpoint["generator"], strict=True)
            generator_optimizer.load_state_dict(warmup_checkpoint["generator_optimizer"])
            global_epoch_offset = int(warmup_checkpoint["global_completed_epoch"])
            warmup_checkpoint_path = warmup_path / "checkpoints" / "latest.pt"
            warmup_provenance = {
                "checkpoint_sha256": _sha256_file(warmup_checkpoint_path),
                "compatibility_fingerprint": warmup_checkpoint["compatibility_fingerprint"],
                "completed_epoch": int(warmup_checkpoint["completed_epoch"]),
                "global_completed_epoch": global_epoch_offset,
                "generator_update_count": int(warmup_checkpoint["generator_update_count"]),
            }
            branch_start = {
                "warmup_checkpoint_sha256": warmup_provenance["checkpoint_sha256"],
                "generator_state_sha256": _backend_state_sha256(generator.state_dict()),
                "generator_optimizer_sha256": _backend_state_sha256(
                    generator_optimizer.state_dict()
                ),
                "warmup_generator_update_count": warmup_provenance[
                    "generator_update_count"
                ],
            }
            branch_rng = {
                "torch_rng_state": warmup_checkpoint["torch_rng_state"].clone(),
                "cuda_rng_state_all": warmup_checkpoint.get("cuda_rng_state_all"),
            }
        if spec["uses_discriminator"]:
            discriminator = _initialized_backend_discriminator(seed=seed, device=device)
            discriminator_optimizer = torch.optim.Adam(
                discriminator.parameters(),
                lr=float(discriminator_optimizer_contract["learning_rate"]),
                betas=tuple(
                    float(value) for value in discriminator_optimizer_contract["betas"]
                ),
            )
        if branch_rng is not None:
            torch.set_rng_state(branch_rng["torch_rng_state"].cpu())
            if device.type == "cuda" and branch_rng["cuda_rng_state_all"] is not None:
                torch.cuda.set_rng_state_all(
                    [state.cpu() for state in branch_rng["cuda_rng_state_all"]]
                )

        config = {
            "schema_version": LUO2022_BACKEND_SCHEMA_VERSION,
            "protocol": LUO2022_BACKEND_TRAINING_PROTOCOL,
            "status_label": backend_contract["status"],
            "experiment_class": backend_contract["experiment_class"],
            "condition": spec["condition"],
            "operator_id": spec["operator_id"],
            "backend_contract_sha256": _sha256_file(backend_config_path),
            "r0_fingerprint": _backend_r0_fingerprint(frontend.artifacts),
            "cache_root_fingerprint": cache_manifest["root_fingerprint"],
            "cache_assignment_sha256": cache_manifest["assignment"]["train_sha256"],
            "cache_scale": train_cache.manifest["scale"],
            "stage_epochs": int(spec["stage_epochs"]),
            "global_epoch_offset": int(global_epoch_offset),
            "seed": seed,
            "batch_size": resolved_batch_size,
            "base_channels": 4,
            "fresh_generator_initial_state_sha256": (
                fresh_generator_initial_state_sha256
            ),
            "generator_optimizer": generator_optimizer_contract,
            "discriminator_optimizer": (
                discriminator_optimizer_contract if spec["uses_discriminator"] else None
            ),
            "reconstruction_loss": {"name": "L1", "weight": 1.0},
            "adversarial_weight": (
                float(training_contract["loss"]["R2_adversarial_weight"])
                if spec["uses_discriminator"]
                else 0.0
            ),
            "max_updates_per_epoch": max_updates_per_epoch,
            "small_run": bool(small_run),
            "warmup": warmup_provenance,
            "input_scaling": {
                "raw_detector_intensity": True,
                "method": "operator_train_dataset_global_max",
                "scale": float(train_cache.scale),
                "per_image_normalization": False,
                "clipping": False,
            },
            "device": str(device),
        }
        compatibility_payload = {
            key: value for key, value in config.items() if key not in {"device"}
        }
        config["compatibility_fingerprint"] = canonical_sha256(compatibility_payload)

        _prepare_fresh_backend_output(output_dir, resume=resume)
        history: list[dict[str, Any]] = []
        epoch_orders: list[dict[str, Any]] = []
        completed_epoch = 0
        generator_update_count = 0
        if resume:
            saved_config = load_config(output_dir / "config.json")
            if saved_config != config:
                raise ValueError("resume config does not exactly match the requested backend run")
            checkpoint = _load_backend_training_checkpoint(
                output_dir / "checkpoints" / "latest.pt",
                device=device,
                expected_compatibility_fingerprint=config["compatibility_fingerprint"],
            )
            if checkpoint.get("condition") != spec["condition"]:
                raise ValueError("resume checkpoint condition mismatch")
            generator.load_state_dict(checkpoint["generator"], strict=True)
            generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
            if spec["uses_discriminator"]:
                assert discriminator is not None and discriminator_optimizer is not None
                if checkpoint.get("discriminator") is None:
                    raise ValueError("R2 resume checkpoint is missing the discriminator")
                discriminator.load_state_dict(checkpoint["discriminator"], strict=True)
                discriminator_optimizer.load_state_dict(
                    checkpoint["discriminator_optimizer"]
                )
            elif checkpoint.get("discriminator") is not None:
                raise ValueError("supervised backend checkpoint unexpectedly contains a discriminator")
            if checkpoint.get("branch_start") != branch_start:
                raise ValueError("resume checkpoint warm-up branch provenance mismatch")
            history = list(checkpoint["history"])
            epoch_orders = list(checkpoint["epoch_orders"])
            completed_epoch = int(checkpoint["completed_epoch"])
            generator_update_count = int(checkpoint["generator_update_count"])
            _backend_restore_rng(checkpoint, device)
        else:
            _atomic_write_json(output_dir / "config.json", config)
            _atomic_write_json(output_dir / "source_backend_config.json", backend_contract)
            _atomic_write_json(output_dir / "history.json", history)
            _atomic_write_json(
                output_dir / "run_state.json",
                {
                    "status": "running",
                    "completed_epoch": 0,
                    "target_epochs": int(spec["stage_epochs"]),
                    "compatibility_fingerprint": config["compatibility_fingerprint"],
                },
            )

        for local_epoch in range(completed_epoch + 1, int(spec["stage_epochs"]) + 1):
            global_epoch = global_epoch_offset + local_epoch
            order, order_record = _backend_epoch_order_record(
                sample_count=len(train_cache),
                seed=seed,
                global_epoch=global_epoch,
                batch_size=resolved_batch_size,
                max_updates=max_updates_per_epoch,
            )
            totals: dict[str, float] = defaultdict(float)
            seen = 0
            updates_this_epoch = 0
            for start in range(0, len(order), resolved_batch_size):
                indices = order[start : start + resolved_batch_size]
                source, target = _backend_training_batch(
                    train_cache,
                    base_dataset,
                    indices,
                    resized_shape=resized_shape,
                    canvas_shape=canvas_shape,
                    device=device,
                )
                if spec["uses_discriminator"]:
                    assert discriminator is not None and discriminator_optimizer is not None
                    values = backend_gan_train_step(
                        generator,
                        discriminator,
                        generator_optimizer,
                        discriminator_optimizer,
                        source,
                        target,
                        adversarial_weight=float(config["adversarial_weight"]),
                    )
                else:
                    values = backend_supervised_train_step(
                        generator,
                        generator_optimizer,
                        source,
                        target,
                    )
                batch_count = len(indices)
                for name, value in values.items():
                    totals[name] += float(value) * batch_count
                seen += batch_count
                updates_this_epoch += 1
            if seen == 0 or updates_this_epoch != order_record["generator_update_budget"]:
                raise RuntimeError("backend epoch did not consume its fixed update budget")
            generator_update_count += updates_this_epoch
            validation_metrics = _evaluate_backend_cached_generator(
                generator,
                validation_cache,
                base_dataset,
                batch_size=resolved_batch_size,
                resized_shape=resized_shape,
                canvas_shape=canvas_shape,
                device=device,
            )
            epoch_record = {
                "epoch": local_epoch,
                "global_epoch": global_epoch,
                "generator_updates": updates_this_epoch,
                "generator_updates_total": generator_update_count,
                "order": order_record,
                "train": {name: value / seen for name, value in totals.items()},
                "validation": validation_metrics,
            }
            if not all(
                math.isfinite(float(value))
                for section in (epoch_record["train"], validation_metrics)
                for value in section.values()
            ):
                raise FloatingPointError("backend epoch produced non-finite metrics")
            history.append(epoch_record)
            epoch_orders.append(order_record)
            checkpoint = {
                "protocol": LUO2022_BACKEND_TRAINING_PROTOCOL,
                "compatibility_fingerprint": config["compatibility_fingerprint"],
                "condition": spec["condition"],
                "completed_epoch": local_epoch,
                "global_completed_epoch": global_epoch,
                "generator_update_count": generator_update_count,
                "generator": generator.state_dict(),
                "generator_optimizer": generator_optimizer.state_dict(),
                "discriminator": discriminator.state_dict() if discriminator is not None else None,
                "discriminator_optimizer": (
                    discriminator_optimizer.state_dict()
                    if discriminator_optimizer is not None
                    else None
                ),
                "history": history,
                "epoch_orders": epoch_orders,
                "branch_start": branch_start,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                ),
            }
            checkpoint = _save_backend_epoch_checkpoint(
                output_dir=output_dir,
                checkpoint=checkpoint,
            )
            _atomic_write_json(output_dir / "history.json", history)
            _atomic_write_json(
                output_dir / "run_state.json",
                {
                    "status": "running",
                    "completed_epoch": local_epoch,
                    "target_epochs": int(spec["stage_epochs"]),
                    "generator_update_count": generator_update_count,
                    "latest_checkpoint": "checkpoints/latest.pt",
                    "latest_checkpoint_sha256": _sha256_file(
                        output_dir / "checkpoints" / "latest.pt"
                    ),
                    "compatibility_fingerprint": config["compatibility_fingerprint"],
                },
            )

        final_checkpoint_path = output_dir / "checkpoints" / "latest.pt"
        final_checkpoint = _load_backend_training_checkpoint(
            final_checkpoint_path,
            device=device,
            expected_compatibility_fingerprint=config["compatibility_fingerprint"],
        )
        integrity = _assert_luo2022_backend_r0_unchanged(frontend)
        training_metrics = {
            "schema_version": LUO2022_BACKEND_SCHEMA_VERSION,
            "protocol": LUO2022_BACKEND_TRAINING_PROTOCOL,
            "condition": spec["condition"],
            "completed_epochs": int(spec["stage_epochs"]),
            "generator_update_count": int(final_checkpoint["generator_update_count"]),
            "final_train": history[-1]["train"],
            "final_validation": history[-1]["validation"],
            "epoch_order_sha256": canonical_sha256(epoch_orders),
        }
        _atomic_write_json(output_dir / "metrics.json", training_metrics)
        manifest = {
            "schema_version": LUO2022_BACKEND_SCHEMA_VERSION,
            "protocol": LUO2022_BACKEND_TRAINING_PROTOCOL,
            "status": "completed",
            "status_label": backend_contract["status"],
            "condition": spec["condition"],
            "operator_id": spec["operator_id"],
            "generator_parameter_count": sum(
                parameter.numel() for parameter in generator.parameters()
            ),
            "discriminator_parameter_count": (
                sum(parameter.numel() for parameter in discriminator.parameters())
                if discriminator is not None
                else 0
            ),
            "generator_update_count": int(final_checkpoint["generator_update_count"]),
            "stage_epochs": int(spec["stage_epochs"]),
            "global_epoch_offset": int(global_epoch_offset),
            "branch_start": branch_start,
            "r0_integrity": integrity,
            "runtime": run_metadata(),
            "artifacts": {
                "config": "config.json",
                "history": "history.json",
                "metrics": "metrics.json",
                "latest_checkpoint": "checkpoints/latest.pt",
                "run_state": "run_state.json",
            },
        }
        _atomic_write_json(output_dir / "manifest.json", manifest)
        _atomic_write_json(
            output_dir / "run_state.json",
            {
                "status": "completed",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "completed_epoch": int(spec["stage_epochs"]),
                "target_epochs": int(spec["stage_epochs"]),
                "generator_update_count": int(final_checkpoint["generator_update_count"]),
                "latest_checkpoint": "checkpoints/latest.pt",
                "latest_checkpoint_sha256": _sha256_file(final_checkpoint_path),
                "final_metrics": "metrics.json",
                "compatibility_fingerprint": config["compatibility_fingerprint"],
            },
        )
        return {
            "history": history,
            "metrics": training_metrics,
            "manifest": manifest,
            "config": config,
        }


def _load_backend_generator_for_evaluation(
    run_dir: Path,
    *,
    condition: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any], dict[str, Any]]:
    config, checkpoint, manifest = _completed_backend_run_checkpoint(
        run_dir,
        device=device,
        expected_condition=condition,
    )
    generator = _initialized_backend_generator(seed=int(config["seed"]), device=device)
    generator.load_state_dict(checkpoint["generator"], strict=True)
    generator.eval()
    return generator, config, checkpoint, manifest


def _validate_backend_comparison_runs(
    *,
    cache_manifest: Mapping[str, Any],
    b0_config: Mapping[str, Any],
    b0_checkpoint: Mapping[str, Any],
    r1_config: Mapping[str, Any],
    r1_checkpoint: Mapping[str, Any],
    r2_config: Mapping[str, Any],
    r2_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject an evaluator input set that is not a controlled B0/R1/R2 comparison."""

    configs = {"B0": b0_config, "R1": r1_config, "R2": r2_config}
    common_fields = (
        "backend_contract_sha256",
        "r0_fingerprint",
        "cache_root_fingerprint",
        "cache_assignment_sha256",
        "seed",
        "batch_size",
        "base_channels",
        "fresh_generator_initial_state_sha256",
        "generator_optimizer",
        "reconstruction_loss",
        "max_updates_per_epoch",
        "small_run",
    )
    for name, config in configs.items():
        if config["cache_root_fingerprint"] != cache_manifest["root_fingerprint"]:
            raise ValueError(f"{name} was not trained from the requested backend cache")
    for field in common_fields:
        values = {canonical_sha256(config.get(field)) for config in configs.values()}
        if len(values) != 1:
            raise ValueError(f"backend comparison field {field} differs across B0/R1/R2")
    if r1_config.get("operator_id") != "r0_four_layer" or r2_config.get(
        "operator_id"
    ) != "r0_four_layer":
        raise ValueError("R1 and R2 must use the same frozen four-layer operator")
    if b0_config.get("operator_id") != "direct_no_d2nn":
        raise ValueError("B0 must use the direct no-D2NN operator")
    if float(r1_config.get("adversarial_weight", -1.0)) != 0.0:
        raise ValueError("R1 must not contain adversarial loss")
    if float(r2_config.get("adversarial_weight", 0.0)) <= 0.0:
        raise ValueError("R2 must add a positive adversarial loss")
    branch_start = r1_checkpoint.get("branch_start")
    if not isinstance(branch_start, dict) or r2_checkpoint.get("branch_start") != branch_start:
        raise ValueError("R1 and R2 do not share the exact warm-up branch state")
    if r1_config.get("global_epoch_offset") != r2_config.get("global_epoch_offset"):
        raise ValueError("R1 and R2 continuation epoch offsets differ")
    r1_orders = [row["consumed_order_sha256"] for row in r1_checkpoint["epoch_orders"]]
    r2_orders = [row["consumed_order_sha256"] for row in r2_checkpoint["epoch_orders"]]
    if r1_orders != r2_orders:
        raise ValueError("R1 and R2 did not consume the same object order")
    if int(r1_checkpoint["generator_update_count"]) != int(
        r2_checkpoint["generator_update_count"]
    ):
        raise ValueError("R1 and R2 generator update budgets differ")
    total_r1_updates = int(branch_start["warmup_generator_update_count"]) + int(
        r1_checkpoint["generator_update_count"]
    )
    if int(b0_checkpoint["generator_update_count"]) != total_r1_updates:
        raise ValueError("B0 and warm-up+R1 total generator update budgets differ")
    return {
        "shared_warmup_checkpoint_sha256": branch_start["warmup_checkpoint_sha256"],
        "shared_generator_start_sha256": branch_start["generator_state_sha256"],
        "shared_generator_optimizer_start_sha256": branch_start[
            "generator_optimizer_sha256"
        ],
        "continuation_order_sha256": r1_orders,
        "continuation_generator_updates": int(r1_checkpoint["generator_update_count"]),
        "b0_total_generator_updates": int(b0_checkpoint["generator_update_count"]),
        "r1_total_generator_updates": total_r1_updates,
        "r2_total_generator_updates": int(branch_start["warmup_generator_update_count"])
        + int(r2_checkpoint["generator_update_count"]),
        "matched": True,
    }


def _backend_evaluation_populations(
    frontend: Luo2022FrozenBackendFrontend,
) -> dict[str, tuple[torch.Tensor, list[int] | None]]:
    runtime_values = frontend.artifacts.runtime_config["runtime"]
    epoch = int(runtime_values["epochs"])
    known_path = frontend.artifacts.run_dir / "diffusers" / f"training_epoch_{epoch:03d}.pt"
    if not known_path.is_file():
        raise FileNotFoundError("final known-diffuser bank is missing from the frozen R0 run")
    known = torch.load(known_path, map_location="cpu", weights_only=True).to(dtype=torch.float32)
    optics_config = frontend.model.config
    diffuser_kwargs = _luo2022_diffuser_kwargs(
        optics_config,
        frontend.artifacts.contract,
    )
    seed_schedule, _provenance = _luo2022_frozen_diffuser_seed_schedule(
        frontend.artifacts.runtime_config,
        frontend.artifacts.contract,
    )
    uniqueness = frontend.artifacts.contract["diffuser"]["uniqueness"]
    unseen = make_unique_correlated_diffusers(
        int(runtime_values["eval_diffusers"]),
        field_shape=optics_config.field_shape,
        base_seed=int(seed_schedule["evaluation_base_seed"]),
        minimum_difference_radians=float(uniqueness["minimum_radians"]),
        phase_representation=str(uniqueness["phase_representation"]),
        **diffuser_kwargs,
    ).to(dtype=torch.float32)
    no_diffuser = torch.zeros((1, *optics_config.field_shape), dtype=torch.float32)
    known_start = (epoch - 1) * int(runtime_values["diffusers_per_epoch"])
    return {
        "final_epoch_known": (
            known,
            list(range(known_start, known_start + int(known.shape[0]))),
        ),
        "seed_disjoint_unseen": (
            unseen,
            list(range(int(unseen.shape[0]))),
        ),
        "no_diffuser": (no_diffuser, None),
    }


def _backend_pair_summary(
    store: Mapping[str, Any],
    *,
    no_diffuser: bool,
) -> dict[str, Any]:
    diffuser_ids = None if no_diffuser else store["diffuser_ids"]
    metric_summaries = {
        name: {
            "distribution": two_level_diffuser_summary(
                values,
                diffuser_ids=diffuser_ids,
            ),
            "per_digit": digit_group_statistics(
                values,
                store["digits"],
                diffuser_ids=diffuser_ids,
            ),
        }
        for name, values in store["metrics"].items()
    }
    target_pcc = store["metrics"]["pearson_target_support"]
    metric_summaries["pearson_target_support"]["worst_5_percent"] = (
        pair_level_tail_statistics(target_pcc)
    )
    return {
        "pair_count": len(store["object_ids"]),
        "object_id_coverage": {
            "unique_count": len(set(store["object_ids"])),
            "minimum": min(store["object_ids"]),
            "maximum": max(store["object_ids"]),
        },
        "metrics": metric_summaries,
    }


def _save_fixed_backend_sample_grid(
    samples: Mapping[int, Mapping[str, torch.Tensor]],
    *,
    requested_ids: Sequence[int],
    output_path: Path,
    title: str,
) -> None:
    ordered_ids = [object_id for object_id in requested_ids if object_id in samples]
    if not ordered_ids:
        return
    panels = (
        "clean_target",
        "direct_scattered_intensity",
        "frozen_four_layer_output",
        "B0_reconstruction",
        "R1_reconstruction",
        "R2_reconstruction",
        "B0_absolute_error",
        "R1_absolute_error",
        "R2_absolute_error",
    )
    fig, axes = plt.subplots(
        len(ordered_ids),
        len(panels),
        figsize=(1.7 * len(panels), 1.7 * len(ordered_ids)),
        squeeze=False,
    )
    for row_index, object_id in enumerate(ordered_ids):
        for column_index, panel in enumerate(panels):
            axis = axes[row_index][column_index]
            image = samples[object_id][panel].detach().cpu().squeeze().numpy()
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            if row_index == 0:
                axis.set_title(panel.replace("_", " "), fontsize=7)
            if column_index == 0:
                axis.set_ylabel(str(object_id), fontsize=7)
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_backend_metric_plot(metrics: Mapping[str, Any], output_path: Path) -> None:
    conditions = ("final_epoch_known", "seed_disjoint_unseen", "no_diffuser")
    metric_names = ("pearson_target_support", "psnr", "ssim")
    fig, axes = plt.subplots(1, len(metric_names), figsize=(12.0, 3.6), squeeze=False)
    x = np.arange(len(conditions), dtype=np.float64)
    width = 0.18
    for metric_index, metric_name in enumerate(metric_names):
        axis = axes[0][metric_index]
        for variant_index, variant in enumerate(LUO2022_BACKEND_VARIANTS):
            values = [
                metrics["conditions"][condition][variant]["metrics"][metric_name][
                    "distribution"
                ]["statistics"]["mean"]
                for condition in conditions
            ]
            axis.bar(
                x + (variant_index - 1.5) * width,
                values,
                width=width,
                label=variant,
            )
        axis.set_title(metric_name)
        axis.set_xticks(x, ["known", "unseen", "none"])
        axis.grid(axis="y", alpha=0.25)
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_backend_cost_plot(
    metrics: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(5.2, 4.0))
    for variant in LUO2022_BACKEND_VARIANTS:
        x = float(model_metadata[variant]["digital_parameter_count"])
        y = float(
            metrics["conditions"]["seed_disjoint_unseen"][variant]["metrics"][
                "pearson_target_support"
            ]["distribution"]["statistics"]["mean"]
        )
        axis.scatter([x], [y], s=55)
        axis.annotate(variant, (x, y))
    axis.set_xlabel("digital parameter count")
    axis.set_ylabel("unseen target-support PCC")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_backend_training_plot(
    histories: Mapping[str, Sequence[Mapping[str, Any]]],
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(6.2, 4.0))
    for variant, history in histories.items():
        if not history:
            continue
        axis.plot(
            [int(row["global_epoch"]) for row in history],
            [float(row["train"]["l1"]) for row in history],
            marker="o",
            label=variant,
        )
    axis.set_xlabel("global epoch")
    axis.set_ylabel("training L1")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_luo2022_backend_evaluation(
    *,
    run_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    config_path: Path = DEFAULT_LUO2022_CONFIG,
    backend_config_path: Path = DEFAULT_LUO2022_BACKEND_CONFIG,
    b0_dir: Path,
    r1_dir: Path,
    r2_dir: Path,
    device_name: str = "cpu",
    download: bool = False,
    max_eval_batches: int | None = None,
) -> dict[str, Any]:
    """Stream unified R0/B0/R1/R2 metrics over known/unseen/no-diffuser controls."""

    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir)
    _require_backend_output_outside_r0(output_dir, run_dir)
    _require_backend_output_outside_r0(cache_dir, run_dir)
    if max_eval_batches is not None and max_eval_batches <= 0:
        raise ValueError("max_eval_batches must be positive")
    device = select_device(device_name)
    cache_manifest = _load_backend_cache_manifest(cache_dir)
    _prepare_fresh_backend_output(output_dir, resume=False)

    with _verified_luo2022_backend_frontend(
        run_dir=run_dir,
        config_path=config_path,
        device=device,
    ) as frontend:
        backend_contract = _load_luo2022_backend_contract(
            backend_config_path,
            artifacts=frontend.artifacts,
        )
        if cache_manifest["request"]["r0_fingerprint"] != _backend_r0_fingerprint(
            frontend.artifacts
        ):
            raise ValueError("backend evaluation cache does not match the frozen R0 run")
        b0, b0_config, b0_checkpoint, b0_manifest = _load_backend_generator_for_evaluation(
            Path(b0_dir), condition="b0", device=device
        )
        r1, r1_config, r1_checkpoint, r1_manifest = _load_backend_generator_for_evaluation(
            Path(r1_dir), condition="r1", device=device
        )
        r2, r2_config, r2_checkpoint, r2_manifest = _load_backend_generator_for_evaluation(
            Path(r2_dir), condition="r2", device=device
        )
        fairness = _validate_backend_comparison_runs(
            cache_manifest=cache_manifest,
            b0_config=b0_config,
            b0_checkpoint=b0_checkpoint,
            r1_config=r1_config,
            r1_checkpoint=r1_checkpoint,
            r2_config=r2_config,
            r2_checkpoint=r2_checkpoint,
        )
        direct_scale = float(
            verify_luo2022_intensity_cache(
                cache_dir / cache_manifest["caches"]["direct_no_d2nn"]["train"]["path"]
            )["scale"]["value"]
        )
        frozen_scale = float(
            verify_luo2022_intensity_cache(
                cache_dir / cache_manifest["caches"]["r0_four_layer"]["train"]["path"]
            )["scale"]["value"]
        )
        runtime_values = frontend.artifacts.runtime_config["runtime"]
        eval_limit = int(runtime_values["eval_limit"])
        batch_size = int(b0_config["batch_size"])
        eval_dataset = build_torchvision_dataset(
            name="MNIST",
            root=DEFAULT_DATA_ROOT,
            train=False,
            image_size=int(frontend.artifacts.contract["input"]["original_shape"][0]),
            download=download,
        )
        eval_count = min(eval_limit, len(eval_dataset))
        if max_eval_batches is not None:
            eval_count = min(eval_count, max_eval_batches * batch_size)
        if eval_count <= 0:
            raise ValueError("backend evaluation dataset is empty")
        populations = _backend_evaluation_populations(frontend)
        requested_sample_ids = [
            int(value) for value in backend_contract["evaluation"]["fixed_example_object_ids"]
        ]
        stores: dict[str, dict[str, dict[str, Any]]] = {
            condition_name: {
                variant: {
                    "object_ids": [],
                    "digits": [],
                    "diffuser_ids": [],
                    "metrics": defaultdict(list),
                }
                for variant in LUO2022_BACKEND_VARIANTS
            }
            for condition_name in populations
        }
        fixed_samples: dict[str, dict[int, dict[str, torch.Tensor]]] = {
            condition_name: {} for condition_name in populations
        }
        inference_seconds: dict[str, float] = defaultdict(float)
        inference_items: dict[str, int] = defaultdict(int)
        per_object_path = output_dir / "per_object_metrics.csv"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".per_object_metrics.", suffix=".tmp", dir=output_dir
        )
        os.close(descriptor)
        temporary_csv = Path(temporary_name)
        fieldnames = [
            "condition",
            "variant",
            "object_id",
            "label",
            "diffuser_id",
            "pearson_target_support",
            "pearson_full_canvas",
            "psnr",
            "ssim",
        ]
        try:
            with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for condition_name, (phases_cpu, public_diffuser_ids) in populations.items():
                    phases = phases_cpu.to(device)
                    for object_start in range(0, eval_count, batch_size):
                        object_stop = min(object_start + batch_size, eval_count)
                        batch_items = [
                            _backend_unpack_image_label(eval_dataset[index])
                            for index in range(object_start, object_stop)
                        ]
                        images = torch.stack([item[0] for item in batch_items]).to(device)
                        labels = [int(item[1]) for item in batch_items]
                        target = prepare_luo2022_amplitude(
                            images,
                            resized_shape=(int(runtime_values["input_size"]),) * 2,
                            canvas_shape=(int(runtime_values["grid_size"]),) * 2,
                        )
                        field = amplitude_to_complex_field(target)
                        for diffuser_index in range(int(phases.shape[0])):
                            one_phase = phases[diffuser_index : diffuser_index + 1]
                            with torch.inference_mode():
                                if device.type == "cuda":
                                    torch.cuda.synchronize(device)
                                optical_start = time.perf_counter()
                                direct_raw = frontend.model.forward_without_diffractive_layers(
                                    field, one_phase
                                )[:, 0].unsqueeze(1)
                                frozen_raw = frontend.model(field, one_phase)[:, 0].unsqueeze(1)
                                if device.type == "cuda":
                                    torch.cuda.synchronize(device)
                                inference_seconds["R0_optical"] += time.perf_counter() - optical_start
                                direct_input = direct_raw / direct_scale
                                frozen_input = frozen_raw / frozen_scale
                                digital_predictions: dict[str, torch.Tensor] = {}
                                for variant, generator, source in (
                                    ("B0", b0, direct_input),
                                    ("R1", r1, frozen_input),
                                    ("R2", r2, frozen_input),
                                ):
                                    if device.type == "cuda":
                                        torch.cuda.synchronize(device)
                                    digital_start = time.perf_counter()
                                    digital_predictions[variant] = generator(source)
                                    if device.type == "cuda":
                                        torch.cuda.synchronize(device)
                                    inference_seconds[variant] += time.perf_counter() - digital_start
                                    inference_items[variant] += int(target.shape[0])
                                predictions = {
                                    "R0": frozen_input,
                                    **digital_predictions,
                                }
                                inference_items["R0"] += int(target.shape[0])
                                for variant, prediction in predictions.items():
                                    per_image = per_image_reconstruction_metrics(
                                        prediction,
                                        target,
                                        target_support=target > 0,
                                    )
                                    store = stores[condition_name][variant]
                                    diffuser_id = (
                                        None
                                        if public_diffuser_ids is None
                                        else int(public_diffuser_ids[diffuser_index])
                                    )
                                    for batch_index, object_id in enumerate(
                                        range(object_start, object_stop)
                                    ):
                                        row_metrics = {
                                            name: float(values[batch_index].detach().cpu())
                                            for name, values in per_image.items()
                                        }
                                        store["object_ids"].append(object_id)
                                        store["digits"].append(labels[batch_index])
                                        if diffuser_id is not None:
                                            store["diffuser_ids"].append(diffuser_id)
                                        for name, value in row_metrics.items():
                                            store["metrics"][name].append(value)
                                        writer.writerow(
                                            {
                                                "condition": condition_name,
                                                "variant": variant,
                                                "object_id": object_id,
                                                "label": labels[batch_index],
                                                "diffuser_id": "" if diffuser_id is None else diffuser_id,
                                                **row_metrics,
                                            }
                                        )
                                if diffuser_index == 0:
                                    for batch_index, object_id in enumerate(
                                        range(object_start, object_stop)
                                    ):
                                        if object_id not in requested_sample_ids:
                                            continue
                                        clean = target[batch_index].detach().cpu()
                                        direct_image = direct_input[batch_index].detach().cpu()
                                        frozen_image = frozen_input[batch_index].detach().cpu()
                                        b0_image = digital_predictions["B0"][batch_index].detach().cpu()
                                        r1_image = digital_predictions["R1"][batch_index].detach().cpu()
                                        r2_image = digital_predictions["R2"][batch_index].detach().cpu()
                                        fixed_samples[condition_name][object_id] = {
                                            "clean_target": clean,
                                            "direct_scattered_intensity": direct_image,
                                            "frozen_four_layer_output": frozen_image,
                                            "B0_reconstruction": b0_image,
                                            "R1_reconstruction": r1_image,
                                            "R2_reconstruction": r2_image,
                                            "B0_absolute_error": (b0_image - clean).abs(),
                                            "R1_absolute_error": (r1_image - clean).abs(),
                                            "R2_absolute_error": (r2_image - clean).abs(),
                                        }
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_csv, per_object_path)
        finally:
            if temporary_csv.exists():
                temporary_csv.unlink()

        condition_metrics: dict[str, Any] = {}
        for condition_name, variants in stores.items():
            no_diffuser = condition_name == "no_diffuser"
            expected_pairs = eval_count * int(populations[condition_name][0].shape[0])
            condition_metrics[condition_name] = {}
            for variant, store in variants.items():
                if len(store["object_ids"]) != expected_pairs:
                    raise RuntimeError("backend evaluator pair coverage is incomplete")
                if not no_diffuser and len(store["diffuser_ids"]) != expected_pairs:
                    raise RuntimeError("backend evaluator diffuser IDs are incomplete")
                condition_metrics[condition_name][variant] = _backend_pair_summary(
                    store,
                    no_diffuser=no_diffuser,
                )
        metrics: dict[str, Any] = {
            "schema_version": LUO2022_BACKEND_SCHEMA_VERSION,
            "protocol": LUO2022_BACKEND_EVALUATION_PROTOCOL,
            "status_label": backend_contract["status"],
            "conditions": condition_metrics,
        }
        causal_pairs = {
            "digital_reconstruction_total_gain_R1_minus_R0": ("R1", "R0"),
            "optical_frontend_net_contribution_R1_minus_B0": ("R1", "B0"),
            "gan_marginal_contribution_R2_minus_R1": ("R2", "R1"),
        }
        metrics["causal_deltas"] = {
            question: {
                condition_name: {
                    metric_name: (
                        float(
                            condition_metrics[condition_name][comparison]["metrics"][
                                metric_name
                            ]["distribution"]["statistics"]["mean"]
                        )
                        - float(
                            condition_metrics[condition_name][reference]["metrics"][
                                metric_name
                            ]["distribution"]["statistics"]["mean"]
                        )
                    )
                    for metric_name in (
                        "pearson_target_support",
                        "pearson_full_canvas",
                        "psnr",
                        "ssim",
                    )
                }
                for condition_name in condition_metrics
            }
            for question, (comparison, reference) in causal_pairs.items()
        }
        metrics["causal_delta_inference"] = {}
        for question, (comparison, reference) in causal_pairs.items():
            question_statistics: dict[str, Any] = {}
            for condition_name in condition_metrics:
                metric_statistics: dict[str, Any] = {}
                for metric_name in (
                    "pearson_target_support",
                    "pearson_full_canvas",
                    "psnr",
                    "ssim",
                ):
                    if condition_name == "no_diffuser":
                        reference_values = stores[condition_name][reference]["metrics"][
                            metric_name
                        ]
                        comparison_values = stores[condition_name][comparison]["metrics"][
                            metric_name
                        ]
                        metric_statistics[metric_name] = {
                            "aggregation_unit": "object",
                            "delta_definition": "comparison_minus_reference",
                            "statistics": scalar_summary(
                                [
                                    float(comparison_value) - float(reference_value)
                                    for reference_value, comparison_value in zip(
                                        reference_values,
                                        comparison_values,
                                        strict=True,
                                    )
                                ]
                            ),
                        }
                    else:
                        reference_rows = condition_metrics[condition_name][reference][
                            "metrics"
                        ][metric_name]["distribution"]["per_diffuser"]
                        comparison_rows = condition_metrics[condition_name][comparison][
                            "metrics"
                        ][metric_name]["distribution"]["per_diffuser"]
                        metric_statistics[metric_name] = {
                            "aggregation_unit": "matched_diffuser",
                            **paired_diffuser_delta_statistics(
                                {
                                    row["diffuser_id"]: float(row["mean"])
                                    for row in reference_rows
                                },
                                {
                                    row["diffuser_id"]: float(row["mean"])
                                    for row in comparison_rows
                                },
                            ),
                        }
                question_statistics[condition_name] = metric_statistics
            metrics["causal_delta_inference"][question] = question_statistics
        model_metadata = {
            "R0": {
                "digital_parameter_count": 0,
                "optical_layers": 4,
                "mean_digital_inference_seconds_per_object": 0.0,
            },
            "B0": {
                "digital_parameter_count": int(b0_manifest["generator_parameter_count"]),
                "optical_layers": 0,
            },
            "R1": {
                "digital_parameter_count": int(r1_manifest["generator_parameter_count"]),
                "optical_layers": 4,
            },
            "R2": {
                "digital_parameter_count": int(r2_manifest["generator_parameter_count"])
                + int(r2_manifest["discriminator_parameter_count"]),
                "generator_parameter_count": int(r2_manifest["generator_parameter_count"]),
                "discriminator_parameter_count": int(
                    r2_manifest["discriminator_parameter_count"]
                ),
                "optical_layers": 4,
            },
        }
        for variant in ("B0", "R1", "R2"):
            model_metadata[variant]["mean_digital_inference_seconds_per_object"] = (
                inference_seconds[variant] / max(1, inference_items[variant])
            )
        metrics["model_and_budget"] = {
            "models": model_metadata,
            "training_fairness": fairness,
        }
        metrics["input_scaling"] = {
            "quantity": "raw_detector_intensity",
            "direct_no_d2nn_train_global_max": direct_scale,
            "r0_four_layer_train_global_max": frozen_scale,
            "R0_metric_input": "raw_frozen_four_layer_intensity_divided_by_r0_operator_train_global_max",
            "digital_inputs": "raw_operator_intensity_divided_by_matching_operator_train_global_max",
            "fit_split": "train_only",
            "per_image_normalization": False,
            "clipping": False,
        }
        _atomic_write_json(output_dir / "metrics.json", metrics)
        for condition_name, samples in fixed_samples.items():
            _save_fixed_backend_sample_grid(
                samples,
                requested_ids=requested_sample_ids,
                output_path=output_dir / "samples" / f"{condition_name}.png",
                title=condition_name,
            )
        _save_backend_metric_plot(metrics, output_dir / "metrics_comparison.png")
        _save_backend_cost_plot(metrics, model_metadata, output_dir / "cost.png")
        _save_backend_training_plot(
            {
                "B0": b0_checkpoint["history"],
                "R1": r1_checkpoint["history"],
                "R2": r2_checkpoint["history"],
            },
            output_dir / "training_curves.png",
        )
        integrity = _assert_luo2022_backend_r0_unchanged(frontend)
        evaluation_config = {
            "schema_version": LUO2022_BACKEND_SCHEMA_VERSION,
            "protocol": LUO2022_BACKEND_EVALUATION_PROTOCOL,
            "backend_contract_sha256": _sha256_file(backend_config_path),
            "r0_fingerprint": _backend_r0_fingerprint(frontend.artifacts),
            "cache_root_fingerprint": cache_manifest["root_fingerprint"],
            "evaluation_object_count": eval_count,
            "max_eval_batches": max_eval_batches,
            "diffuser_conditions": list(populations),
            "fixed_example_object_ids": requested_sample_ids,
            "input_scaling": metrics["input_scaling"],
            "fairness": fairness,
        }
        _atomic_write_json(output_dir / "config.json", evaluation_config)
        manifest = {
            "schema_version": LUO2022_BACKEND_SCHEMA_VERSION,
            "protocol": LUO2022_BACKEND_EVALUATION_PROTOCOL,
            "status": "completed",
            "status_label": backend_contract["status"],
            "r0_integrity": integrity,
            "runtime": run_metadata(),
            "artifacts": {
                "config": "config.json",
                "metrics": "metrics.json",
                "per_object_metrics": "per_object_metrics.csv",
                "metric_plot": "metrics_comparison.png",
                "cost_plot": "cost.png",
                "training_curves": "training_curves.png",
                "samples": [
                    f"samples/{condition_name}.png"
                    for condition_name, samples in fixed_samples.items()
                    if samples
                ],
            },
        }
        _atomic_write_json(output_dir / "manifest.json", manifest)
        return {"metrics": metrics, "manifest": manifest, "config": evaluation_config}


def _luo2022_optics_config_from_frozen_run(
    runtime_config: dict[str, Any],
    contract: dict[str, Any],
    *,
    pad_factor: int = 2,
) -> Luo2022OpticsConfig:
    values = runtime_config["runtime"]
    return Luo2022OpticsConfig(
        field_shape=(int(values["grid_size"]), int(values["grid_size"])),
        wavelength=float(contract["illumination"]["wavelength_m"]),
        pixel_size=float(contract["grid"]["pixel_pitch_m"]),
        object_to_diffuser_distance=float(contract["geometry"]["object_to_diffuser_m"]),
        diffuser_to_first_layer_distance=float(contract["geometry"]["diffuser_to_first_layer_m"]),
        layer_distance=float(contract["geometry"]["layer_to_layer_m"]),
        output_distance=float(contract["geometry"]["last_layer_to_output_m"]),
        num_layers=int(contract["d2nn"]["layers"]),
        pad_factor=pad_factor,
    )


def _luo2022_diffuser_kwargs(
    optics_config: Luo2022OpticsConfig,
    contract: dict[str, Any],
) -> dict[str, float | str]:
    return {
        "wavelength": optics_config.wavelength,
        "pixel_size": optics_config.pixel_size,
        "refractive_index_difference": float(contract["diffuser"]["refractive_index_difference"]),
        "height_mean_lambda": float(contract["diffuser"]["height_mean_lambda"]),
        "height_std_lambda": float(contract["diffuser"]["height_std_lambda"]),
        "gaussian_sigma_lambda": float(contract["diffuser"]["gaussian_sigma_lambda"]),
        "truncate_sigma": float(contract["diffuser"]["finite_kernel_choice"]["truncate_sigma"]),
        "padding": str(contract["diffuser"]["finite_kernel_choice"]["padding"]),
    }


def _luo2022_frozen_diffuser_seed_schedule(
    runtime_config: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Validate or derive the frozen non-overlapping diffuser seed schedule.

    The run-level copy was introduced after earlier R0 smoke artifacts existed.
    A completed run is accepted without that copy only when its own frozen
    source configuration declares both the training stride and evaluation
    offset. Runs that predate the isolation policy are rejected instead of
    silently being assigned a modern schedule.
    """

    training_schedule = contract.get("training", {}).get("diffuser_seed_schedule")
    evaluation_schedule = contract.get("evaluation", {}).get("diffuser_seed_schedule")
    if not isinstance(training_schedule, dict) or not isinstance(evaluation_schedule, dict):
        raise ValueError(
            "frozen run predates the diffuser seed-isolation policy; "
            "its unseen-diffuser status cannot be certified"
        )
    try:
        expected = luo2022_diffuser_seed_schedule(
            seed=int(runtime_config["runtime"]["seed"]),
            epochs=int(runtime_config["runtime"]["epochs"]),
            training_stride=int(training_schedule["epoch_stride"]),
            evaluation_offset=int(evaluation_schedule["offset"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "frozen source configuration does not define a valid diffuser seed-isolation schedule"
        ) from exc

    recorded = runtime_config.get("diffuser_seed_schedule")
    if recorded is None:
        return expected, "derived_from_frozen_source_config"
    if not isinstance(recorded, dict):
        raise ValueError("runtime diffuser seed schedule must be a mapping")
    for key, expected_value in expected.items():
        if recorded.get(key) != expected_value:
            raise ValueError(
                f"runtime diffuser seed schedule field {key!r} does not match frozen source config"
            )
    return expected, "validated_runtime_copy"


def _luo2022_tensor_summary(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().to(device="cpu", dtype=torch.float64).flatten()
    if values.numel() == 0:
        raise ValueError("cannot summarize an empty tensor")
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "sample_std": float(values.std(unbiased=True)) if values.numel() > 1 else 0.0,
    }


def _luo2022_complex_intensity(field: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(field):
        raise TypeError("field must be complex")
    return field.real.square() + field.imag.square()


def _luo2022_edge_mask(
    field_shape: tuple[int, int],
    *,
    fraction: float = 0.1,
    device: torch.device,
) -> torch.Tensor:
    if not 0 < fraction < 0.5:
        raise ValueError("edge fraction must be between zero and one half")
    height, width = field_shape
    border_y = max(1, int(round(height * fraction)))
    border_x = max(1, int(round(width * fraction)))
    mask = torch.ones(field_shape, dtype=torch.bool, device=device)
    mask[border_y : height - border_y, border_x : width - border_x] = False
    return mask


def _luo2022_field_summary(
    field: torch.Tensor,
    *,
    pixel_size: float,
) -> dict[str, Any]:
    """Summarize field energy and border occupancy without assuming conservation."""

    intensity = _luo2022_complex_intensity(field)
    energy = intensity.flatten(start_dim=-2).sum(dim=-1) * pixel_size**2
    edge_mask = _luo2022_edge_mask(
        tuple(int(value) for value in field.shape[-2:]),
        device=field.device,
    )
    edge_energy = intensity[..., edge_mask].sum(dim=-1) * pixel_size**2
    edge_fraction = edge_energy / energy.clamp_min(torch.finfo(intensity.dtype).eps)
    return {
        "field_count": int(energy.numel()),
        "integrated_energy": _luo2022_tensor_summary(energy),
        "edge_energy_fraction": _luo2022_tensor_summary(edge_fraction),
        "mean_intensity": _luo2022_tensor_summary(intensity.mean(dim=(-2, -1))),
        "speckle_contrast": _luo2022_tensor_summary(
            intensity.std(dim=(-2, -1), unbiased=False)
            / intensity.mean(dim=(-2, -1)).clamp_min(torch.finfo(intensity.dtype).eps)
        ),
    }


def _luo2022_phase_multiply_energy_error(
    before: torch.Tensor,
    after: torch.Tensor,
) -> float:
    before_energy = _luo2022_complex_intensity(before).flatten(start_dim=-2).sum(dim=-1)
    after_energy = _luo2022_complex_intensity(after).flatten(start_dim=-2).sum(dim=-1)
    while before_energy.ndim < after_energy.ndim:
        before_energy = before_energy.unsqueeze(-1)
    difference = (after_energy - before_energy).abs() / before_energy.abs().clamp_min(
        torch.finfo(before_energy.dtype).eps
    )
    return float(difference.mean().detach().cpu())


def _luo2022_center_mask(
    target: torch.Tensor,
    *,
    input_size: int,
) -> torch.Tensor:
    height, width = target.shape[-2:]
    if input_size <= 0 or input_size > min(height, width):
        raise ValueError("input_size must fit within the diagnostic canvas")
    top = (height - input_size) // 2
    left = (width - input_size) // 2
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[..., top : top + input_size, left : left + input_size] = True
    return mask


def _luo2022_roi_metrics(
    output_intensity: torch.Tensor,
    target_amplitude: torch.Tensor,
    *,
    input_size: int,
) -> dict[str, Any]:
    """Summarize explicit spatial metrics without changing the frozen loss."""

    full_components = luo2022_d2nn_components_per_pair(output_intensity, target_amplitude)
    per_pair = _luo2022_roi_components_per_pair(
        output_intensity,
        target_amplitude,
        input_size=input_size,
        full_canvas_pearson=full_components["pearson"],
    )
    metrics: dict[str, Any] = {}
    for name, components in per_pair.items():
        metrics[name] = {
            "pcc": _luo2022_tensor_summary(components["pearson"]),
            "output_energy_fraction": _luo2022_tensor_summary(
                components["output_energy_fraction"]
            ),
            "roi_pixel_count": int(components["roi_pixel_count"]),
        }
    metrics["full_canvas"]["pcc_matches_frozen_metric_abs_error"] = float(
        abs(
            metrics["full_canvas"]["pcc"]["mean"]
            - float(full_components["pearson"].mean().detach().cpu())
        )
    )
    return metrics


def _luo2022_roi_components_per_pair(
    output_intensity: torch.Tensor,
    target_amplitude: torch.Tensor,
    *,
    input_size: int,
    full_canvas_pearson: torch.Tensor | None = None,
) -> dict[str, dict[str, torch.Tensor | int]]:
    """Return per-object, per-diffuser ROI metrics for read-only post-hoc use.

    ``full_canvas_pearson`` is accepted from the frozen loss computation so
    that the full-canvas ROI value remains exactly the metric used by R0.
    """

    if target_amplitude.ndim == 4 and target_amplitude.shape[1] == 1:
        target_amplitude = target_amplitude[:, 0]
    if output_intensity.ndim != 4 or target_amplitude.ndim != 3:
        raise ValueError("ROI metrics require (B, n, H, W) output and (B, H, W) target")
    expanded_target = target_amplitude[:, None].expand_as(output_intensity)
    masks = {
        "full_canvas": torch.ones_like(target_amplitude, dtype=torch.bool),
        "center_input_region": _luo2022_center_mask(target_amplitude, input_size=input_size),
        "target_support": target_amplitude > 0,
    }
    output_energy = output_intensity.sum(dim=(-2, -1)).clamp_min(
        torch.finfo(output_intensity.dtype).eps
    )
    metrics: dict[str, dict[str, torch.Tensor | int]] = {}
    for name, mask in masks.items():
        expanded_mask = mask[:, None].expand_as(output_intensity)
        if name == "full_canvas" and full_canvas_pearson is not None:
            pcc = full_canvas_pearson
        else:
            flat_output = output_intensity.reshape(-1, *output_intensity.shape[-2:])
            flat_target = expanded_target.reshape_as(flat_output)
            flat_mask = expanded_mask.reshape_as(flat_output)
            pcc = masked_pearson_per_image(flat_output, flat_target, flat_mask).reshape(
                output_intensity.shape[:2]
            )
        energy_fraction = (
            (output_intensity * expanded_mask).sum(dim=(-2, -1)) / output_energy
        )
        metrics[name] = {
            "pearson": pcc,
            "output_energy_fraction": energy_fraction,
            "roi_pixel_count": int(mask[0].sum().item()),
        }
    return metrics


def _luo2022_loss_scale_summary(
    output_intensity: torch.Tensor,
    target_amplitude: torch.Tensor,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for scale in (0.1, 1.0, 10.0):
        components = luo2022_d2nn_components_per_pair(output_intensity * scale, target_amplitude)
        results[str(scale)] = {
            name: float(value.mean().detach().cpu())
            for name, value in components.items()
        }
    return {
        "scales": results,
        "interpretation": (
            "For positive output scaling, PCC should be invariant while the equation (12) "
            "energy term scales linearly; this is a loss-property diagnostic, not a metric change."
        ),
    }


def _luo2022_diffuser_statistics(
    phases: torch.Tensor,
    *,
    optics_config: Luo2022OpticsConfig,
) -> dict[str, Any]:
    if phases.ndim != 3 or phases.shape[0] == 0:
        raise ValueError("diffuser statistics require a nonempty (count, H, W) bank")
    phase_lengths: list[float] = []
    wrapped_phase_lengths: list[float] = []
    transmittance_lengths: list[float] = []
    for phase in phases:
        phase_lengths.append(
            estimate_phase_correlation_length(
                phase,
                pixel_size=optics_config.pixel_size,
                wavelength=optics_config.wavelength,
            )
        )
        wrapped_phase_lengths.append(
            estimate_phase_correlation_length(
                torch.angle(torch.exp(1j * phase)),
                pixel_size=optics_config.pixel_size,
                wavelength=optics_config.wavelength,
            )
        )
        transmittance_lengths.append(
            estimate_transmittance_correlation_length(
                phase,
                pixel_size=optics_config.pixel_size,
                wavelength=optics_config.wavelength,
            )
        )
    height, width = phases.shape[-2:]
    edge_mask = _luo2022_edge_mask((height, width), device=phases.device)
    center_mask = ~edge_mask
    return {
        "count": int(phases.shape[0]),
        "unwrapped_phase_correlation_length_lambda": _luo2022_tensor_summary(
            torch.tensor(phase_lengths)
        ),
        "wrapped_phase_correlation_length_lambda": _luo2022_tensor_summary(
            torch.tensor(wrapped_phase_lengths)
        ),
        "complex_transmittance_correlation_length_lambda": _luo2022_tensor_summary(
            torch.tensor(transmittance_lengths)
        ),
        "phase_standard_deviation_radians": _luo2022_tensor_summary(
            phases.std(dim=(-2, -1), unbiased=False)
        ),
        "phase_center_mean_radians": _luo2022_tensor_summary(
            phases[..., center_mask].mean(dim=-1)
        ),
        "phase_edge_mean_radians": _luo2022_tensor_summary(
            phases[..., edge_mask].mean(dim=-1)
        ),
        "phase_center_standard_deviation_radians": _luo2022_tensor_summary(
            phases[..., center_mask].std(dim=-1, unbiased=False)
        ),
        "phase_edge_standard_deviation_radians": _luo2022_tensor_summary(
            phases[..., edge_mask].std(dim=-1, unbiased=False)
        ),
    }


def _luo2022_merge_cross_bank_summaries(
    summaries: list[dict[str, float | int | str]],
) -> dict[str, float | int | str]:
    if not summaries:
        raise ValueError("at least one cross-bank summary is required")
    pair_count = sum(int(summary["pair_count"]) for summary in summaries)
    if pair_count == 0:
        raise ValueError("cross-bank summaries contain no pairs")
    return {
        "phase_representation": str(summaries[0]["phase_representation"]),
        "pair_count": pair_count,
        "minimum_radians": min(float(summary["minimum_radians"]) for summary in summaries),
        "mean_radians": (
            sum(
                int(summary["pair_count"]) * float(summary["mean_radians"])
                for summary in summaries
            )
            / pair_count
        ),
        "maximum_radians": max(float(summary["maximum_radians"]) for summary in summaries),
        "pass_count": sum(int(summary["pass_count"]) for summary in summaries),
        "pair_pass_fraction": (
            sum(int(summary["pass_count"]) for summary in summaries) / pair_count
        ),
    }


def _luo2022_cross_bank_audit_record(
    summary: dict[str, float | int | str],
    *,
    expected_pair_count: int,
    coverage: str,
) -> dict[str, Any]:
    """State audit coverage and certification without overstating seed isolation."""

    observed_pair_count = int(summary["pair_count"])
    if observed_pair_count != expected_pair_count:
        raise RuntimeError("cross-bank audit pair count does not match the expected coverage")
    all_pairs_pass_threshold = int(summary["pass_count"]) == observed_pair_count
    if coverage == "all_training":
        certification = (
            "certified_against_all_training_diffusers"
            if all_pairs_pass_threshold
            else "not_certified_threshold_violation"
        )
    elif coverage == "final_epoch_only":
        certification = (
            "not_certified_final_epoch_only"
            if all_pairs_pass_threshold
            else "not_certified_threshold_violation"
        )
    else:
        raise ValueError(f"unknown cross-bank audit coverage: {coverage}")
    return {
        "status": "completed",
        "audit_coverage": coverage,
        "expected_pair_count": expected_pair_count,
        "all_pairs_pass_threshold": all_pairs_pass_threshold,
        "unseen_certification": certification,
        **summary,
    }


def _luo2022_learning_rate_audit(
    *,
    run_dir: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    history_path = run_dir / "history.json"
    if not history_path.is_file():
        return {"status": "missing_history"}
    history = load_config(history_path)
    if not history:
        return {"status": "empty_history"}
    learning_rate = contract["training"]["learning_rate"]
    initial = float(learning_rate["initial"])
    gamma = float(learning_rate["gamma"])
    discrepancies: list[float] = []
    entries: list[dict[str, float | int]] = []
    for entry in history:
        epoch = int(entry["epoch"])
        expected = initial * gamma ** (epoch - 1)
        observed = float(entry["learning_rate"])
        discrepancy = observed - expected
        discrepancies.append(abs(discrepancy))
        entries.append(
            {
                "epoch": epoch,
                "expected": expected,
                "observed": observed,
                "observed_minus_expected": discrepancy,
            }
        )
    return {
        "status": "audited",
        "formula": "initial_times_gamma_power_zero_based_epoch",
        "update_interval": str(learning_rate["update_interval"]),
        "entry_count": len(entries),
        "maximum_absolute_error": max(discrepancies),
        "first": entries[0],
        "last": entries[-1],
    }


def _luo2022_trace_summary(
    trace: dict[str, torch.Tensor],
    *,
    pixel_size: float,
) -> dict[str, Any]:
    fields = {
        name: _luo2022_field_summary(field, pixel_size=pixel_size)
        for name, field in trace.items()
    }
    phase_multiply_errors = {
        "diffuser": _luo2022_phase_multiply_energy_error(
            trace["before_diffuser"],
            trace["after_diffuser"],
        ),
    }
    layer_count = sum(name.startswith("after_layer_") for name in trace)
    for layer_index in range(1, layer_count + 1):
        phase_multiply_errors[f"layer_{layer_index}"] = _luo2022_phase_multiply_energy_error(
            trace[f"before_layer_{layer_index}"],
            trace[f"after_layer_{layer_index}"],
        )
    return {
        "fields": fields,
        "phase_multiply_mean_relative_energy_error": phase_multiply_errors,
        "interpretation": (
            "Propagation energy changes are reported rather than asserted to be zero because "
            "the frozen FFT implementation center-crops after each propagation segment."
        ),
    }


def _luo2022_padding_sensitivity(
    *,
    model: Luo2022FourLayerD2NN,
    field: torch.Tensor,
    diffusers: torch.Tensor,
    base_output: torch.Tensor,
    runtime_config: dict[str, Any],
    contract: dict[str, Any],
    device: torch.device,
    pad_factors: tuple[int, ...],
) -> dict[str, Any]:
    if not pad_factors or any(factor < 2 for factor in pad_factors):
        raise ValueError("diagnostic pad factors must all be at least two")
    reference_norm = base_output.flatten(start_dim=1).norm(dim=1).clamp_min(
        torch.finfo(base_output.dtype).eps
    )
    records: dict[str, Any] = {}
    original_pad_factor = model.config.pad_factor
    for pad_factor in tuple(dict.fromkeys(pad_factors)):
        if pad_factor == original_pad_factor:
            output = base_output
        else:
            optics_config = _luo2022_optics_config_from_frozen_run(
                runtime_config,
                contract,
                pad_factor=pad_factor,
            )
            probe_model = Luo2022FourLayerD2NN(optics_config).to(device)
            with torch.no_grad():
                probe_model.phase.copy_(model.phase)
                output = probe_model(field, diffusers)
        relative_l2 = (
            (output - base_output).flatten(start_dim=1).norm(dim=1) / reference_norm
        )
        output_pcc = pearson_per_image(
            output.reshape(-1, *output.shape[-2:]),
            base_output.reshape(-1, *base_output.shape[-2:]),
        )
        records[str(pad_factor)] = {
            "relative_l2_to_pad_factor_" + str(original_pad_factor): _luo2022_tensor_summary(
                relative_l2
            ),
            "pcc_to_pad_factor_" + str(original_pad_factor): _luo2022_tensor_summary(output_pcc),
        }
    return {
        "reference_pad_factor": original_pad_factor,
        "records": records,
        "interpretation": (
            "This is a discrete propagation-window sensitivity probe. It does not select "
            "a replacement padding policy for the frozen run."
        ),
    }


def _luo2022_semigroup_probe(
    field: torch.Tensor,
    *,
    optics_config: Luo2022OpticsConfig,
) -> dict[str, Any]:
    """Compare two 2 mm segments with one 4 mm segment under the frozen discretization."""

    segment = RayleighSommerfeldPropagator(
        field_shape=optics_config.field_shape,
        wavelength=optics_config.wavelength,
        pixel_size=optics_config.pixel_size,
        distance=optics_config.layer_distance,
        pad_factor=optics_config.pad_factor,
    )
    combined = RayleighSommerfeldPropagator(
        field_shape=optics_config.field_shape,
        wavelength=optics_config.wavelength,
        pixel_size=optics_config.pixel_size,
        distance=2.0 * optics_config.layer_distance,
        pad_factor=optics_config.pad_factor,
    )
    two_step = segment.propagate(segment.propagate(field))
    one_step = combined.propagate(field)
    denominator = one_step.flatten(start_dim=1).norm(dim=1).clamp_min(
        torch.finfo(one_step.real.dtype).eps
    )
    relative_l2 = (two_step - one_step).flatten(start_dim=1).norm(dim=1) / denominator
    intensity_pcc = pearson_per_image(
        _luo2022_complex_intensity(two_step),
        _luo2022_complex_intensity(one_step),
    )
    return {
        "two_segment_distance_m": optics_config.layer_distance,
        "single_segment_distance_m": 2.0 * optics_config.layer_distance,
        "relative_complex_field_l2": _luo2022_tensor_summary(relative_l2),
        "intensity_pcc": _luo2022_tensor_summary(intensity_pcc),
        "interpretation": (
            "A finite-window, center-cropped discrete propagation need not satisfy the "
            "continuous free-space semigroup identity exactly."
        ),
    }


def run_luo2022_diagnosis(
    *,
    run_dir: Path,
    diagnostic_output_dir: Path,
    config_path: Path = DEFAULT_LUO2022_CONFIG,
    download: bool = False,
    device_name: str = "cpu",
    diagnostic_batches: int = 1,
    diagnostic_diffusers: int = 3,
    diagnostic_pad_factors: tuple[int, ...] = (2, 3, 4),
    cross_bank_audit: bool = False,
) -> dict[str, Any]:
    """Generate read-only physical and numerical evidence for a frozen R0 run.

    The function deliberately does not prepare, mutate, or resume ``run_dir``.
    It writes independent diagnostics only after the config, final checkpoint,
    and completed-run manifest pass an exact provenance check.
    """

    if diagnostic_batches <= 0 or diagnostic_diffusers <= 0:
        raise ValueError("diagnostic batches and diffusers must be positive")
    if not diagnostic_pad_factors or any(factor < 2 for factor in diagnostic_pad_factors):
        raise ValueError("diagnostic pad factors must be integers of at least two")
    run_resolved = run_dir.resolve()
    diagnostic_resolved = diagnostic_output_dir.resolve()
    if diagnostic_resolved == run_resolved or run_resolved in diagnostic_resolved.parents:
        raise ValueError("diagnostic output directory must be outside the frozen run directory")

    device = select_device(device_name)
    frozen = _load_luo2022_frozen_run_artifacts(
        run_dir=run_dir,
        config_path=config_path,
        device=device,
    )
    values = frozen.runtime_config["runtime"]
    target_epochs = int(values["epochs"])
    requested_diffuser_count = int(diagnostic_diffusers)
    optics_config = _luo2022_optics_config_from_frozen_run(
        frozen.runtime_config,
        frozen.contract,
    )
    final_training_path = run_dir / "diffusers" / f"training_epoch_{target_epochs:03d}.pt"
    if not final_training_path.is_file():
        raise FileNotFoundError("final training diffuser bank is required for diagnosis")
    final_training_diffuser_sha256 = _sha256_file(final_training_path)
    final_training_diffusers = torch.load(
        final_training_path,
        map_location="cpu",
        weights_only=True,
    )
    expected_training_count = int(values["diffusers_per_epoch"])
    expected_diffuser_shape = (expected_training_count, *optics_config.field_shape)
    if tuple(final_training_diffusers.shape) != expected_diffuser_shape:
        raise ValueError("final training diffuser bank shape does not match frozen runtime configuration")

    diffuser_kwargs = _luo2022_diffuser_kwargs(optics_config, frozen.contract)
    uniqueness = frozen.contract["diffuser"]["uniqueness"]
    seed_schedule, seed_schedule_provenance = _luo2022_frozen_diffuser_seed_schedule(
        frozen.runtime_config,
        frozen.contract,
    )
    new_diffusers = make_unique_correlated_diffusers(
        int(values["eval_diffusers"]),
        field_shape=optics_config.field_shape,
        base_seed=int(seed_schedule["evaluation_base_seed"]),
        minimum_difference_radians=float(uniqueness["minimum_radians"]),
        phase_representation=str(uniqueness["phase_representation"]),
        **diffuser_kwargs,
    )
    evaluation_seed_diffuser_sha256 = _sha256_tensor(new_diffusers)

    all_training_diffuser_banks_sha256: str | None = None
    if cross_bank_audit:
        all_training_digest = hashlib.sha256()
        for epoch in range(1, target_epochs + 1):
            phase_path = run_dir / "diffusers" / f"training_epoch_{epoch:03d}.pt"
            if not phase_path.is_file():
                raise FileNotFoundError(
                    f"training diffuser bank is missing for cross-bank audit epoch {epoch}"
                )
            all_training_digest.update(
                f"{epoch}:{_sha256_file(phase_path)}\n".encode("utf-8")
            )
        all_training_diffuser_banks_sha256 = all_training_digest.hexdigest()
    history_path = run_dir / "history.json"
    history_sha256 = _sha256_file(history_path) if history_path.is_file() else None
    diagnostic_spec = {
        "schema_version": 2,
        "implementation_version": "luo2022_r0_diagnosis_v2",
        "diagnostic_batches": int(diagnostic_batches),
        "diagnostic_diffusers": requested_diffuser_count,
        "diagnostic_pad_factors": [int(value) for value in diagnostic_pad_factors],
        "cross_bank_audit": bool(cross_bank_audit),
        "read_only": True,
    }
    fingerprint = {
        "profile_id": frozen.contract["profile_id"],
        "source_freeze_version": frozen.contract["freeze_version"],
        "checkpoint_sha256": frozen.checkpoint_sha256,
        "runtime_config_sha256": frozen.runtime_config_sha256,
        "source_config_sha256": frozen.source_config_sha256,
        "manifest_sha256": frozen.manifest_sha256,
        "run_state_sha256": frozen.run_state_sha256,
        "final_training_diffuser_sha256": final_training_diffuser_sha256,
        "all_training_diffuser_banks_sha256": all_training_diffuser_banks_sha256,
        "evaluation_seed_diffuser_sha256": evaluation_seed_diffuser_sha256,
        "history_sha256": history_sha256,
        "diagnostic_spec": diagnostic_spec,
    }
    state_path = diagnostic_output_dir / "diagnostic_state.json"
    result_path = diagnostic_output_dir / "diagnosis.json"
    if state_path.is_file():
        saved_state = load_config(state_path)
        if saved_state.get("evidence_fingerprint") != fingerprint:
            raise ValueError("existing diagnostic state does not match the frozen run or request")
        if saved_state.get("status") == "completed" and result_path.is_file():
            saved_result = load_config(result_path)
            if (
                saved_result.get("status") != "completed"
                or saved_result.get("read_only") is not True
                or saved_result.get("evidence_fingerprint") != fingerprint
            ):
                raise ValueError("completed diagnostic result does not match its saved state")
            return saved_result
        raise ValueError("existing diagnostic state is incomplete; do not overwrite evidence")
    if result_path.is_file():
        raise ValueError("diagnostic result exists without a matching diagnostic state")

    model = Luo2022FourLayerD2NN(optics_config).to(device)
    model.load_state_dict(frozen.checkpoint["model"], strict=True)
    model.eval()
    phase_before = model.phase.detach().clone()
    diagnostic_output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        state_path,
        {
            "schema_version": 2,
            "status": "running",
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "read_only": True,
            "evidence_fingerprint": fingerprint,
        },
    )
    selected_count = min(
        requested_diffuser_count,
        int(final_training_diffusers.shape[0]),
        int(new_diffusers.shape[0]),
    )
    if selected_count <= 0:
        raise ValueError("frozen run contains no diffusers for diagnosis")
    selected_known = final_training_diffusers[:selected_count]
    selected_new = new_diffusers[:selected_count]
    no_diffuser = torch.zeros((1, *optics_config.field_shape), dtype=torch.float32)
    evaluation_diffusers_device = new_diffusers.to(device)

    final_cross_summary = summarize_cross_diffuser_uniqueness(
        evaluation_diffusers_device,
        final_training_diffusers.to(device),
        phase_representation=str(uniqueness["phase_representation"]),
        threshold_radians=float(uniqueness["minimum_radians"]),
    )
    expected_final_cross_pairs = int(values["eval_diffusers"]) * expected_training_count
    final_cross_audit = _luo2022_cross_bank_audit_record(
        final_cross_summary,
        expected_pair_count=expected_final_cross_pairs,
        coverage="final_epoch_only",
    )
    all_training_cross_summary: dict[str, Any]
    if cross_bank_audit:
        summaries: list[dict[str, float | int | str]] = []
        for epoch in range(1, target_epochs + 1):
            phase_path = run_dir / "diffusers" / f"training_epoch_{epoch:03d}.pt"
            training_bank = torch.load(phase_path, map_location="cpu", weights_only=True)
            if tuple(training_bank.shape) != expected_diffuser_shape:
                raise ValueError(
                    f"training diffuser bank at epoch {epoch} does not match frozen runtime shape"
                )
            summaries.append(
                summarize_cross_diffuser_uniqueness(
                    evaluation_diffusers_device,
                    training_bank.to(device),
                    phase_representation=str(uniqueness["phase_representation"]),
                    threshold_radians=float(uniqueness["minimum_radians"]),
                )
            )
        merged_cross_summary = _luo2022_merge_cross_bank_summaries(summaries)
        expected_pair_count = (
            int(values["eval_diffusers"]) * target_epochs * expected_training_count
        )
        all_training_cross_summary = _luo2022_cross_bank_audit_record(
            merged_cross_summary,
            expected_pair_count=expected_pair_count,
            coverage="all_training",
        )
    else:
        all_training_cross_summary = {
            "status": "not_requested",
            "audit_coverage": "not_requested",
            "all_pairs_pass_threshold": None,
            "unseen_certification": "not_certified_full_training_audit_not_requested",
            "reason": (
                "Seed namespaces are disjoint, but the new population must not be described "
                "as phase-audited unseen until every evaluation-to-training diffuser pair is checked."
            ),
        }

    seed_everything(int(values["seed"]))
    eval_base = build_torchvision_dataset(
        name="MNIST",
        root=DEFAULT_DATA_ROOT,
        train=False,
        image_size=int(frozen.contract["input"]["original_shape"][0]),
        download=download,
    )
    eval_dataset = Subset(eval_base, range(min(int(values["eval_limit"]), len(eval_base))))
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(values["batch_size"]),
        shuffle=False,
    )

    populations = {
        "final_epoch_known_sample": selected_known,
        "evaluation_seed_population_sample": selected_new,
        "no_diffuser": no_diffuser,
    }
    population_batches: dict[str, list[dict[str, Any]]] = {
        name: [] for name in populations
    }
    first_new_field: torch.Tensor | None = None
    first_new_output: torch.Tensor | None = None
    first_new_trace: dict[str, torch.Tensor] | None = None
    first_new_target: torch.Tensor | None = None

    with torch.no_grad():
        for batch_index, (image, _label) in enumerate(eval_loader):
            if batch_index >= diagnostic_batches:
                break
            target = prepare_luo2022_amplitude(
                image.to(device),
                resized_shape=(int(values["input_size"]), int(values["input_size"])),
                canvas_shape=optics_config.field_shape,
            )
            field = amplitude_to_complex_field(target)
            for population, phases_cpu in populations.items():
                phases = phases_cpu.to(device)
                if population == "evaluation_seed_population_sample" and first_new_trace is None:
                    output, trace = model.forward_with_trace(field, phases)
                    first_new_field = field
                    first_new_output = output
                    first_new_trace = trace
                    first_new_target = target
                else:
                    output = model(field, phases)
                components = luo2022_d2nn_components_per_pair(output, target)
                energy_breakdown = luo2022_d2nn_energy_breakdown_per_pair(output, target)
                population_batches[population].append(
                    {
                        "batch_index": batch_index,
                        "object_count": int(target.shape[0]),
                        "diffuser_count": int(phases.shape[0]),
                        "frozen_loss_components": {
                            name: float(value.mean().detach().cpu())
                            for name, value in components.items()
                        },
                        "energy_breakdown": {
                            name: _luo2022_tensor_summary(value)
                            for name, value in energy_breakdown.items()
                        },
                        "roi_metrics": _luo2022_roi_metrics(
                            output,
                            target,
                            input_size=int(values["input_size"]),
                        ),
                        "loss_scale_sensitivity": _luo2022_loss_scale_summary(output, target),
                    }
                )

        if (
            first_new_field is None
            or first_new_output is None
            or first_new_trace is None
            or first_new_target is None
        ):
            raise ValueError("evaluation loader yielded no diagnostic batches")
        zero_phase_model = Luo2022FourLayerD2NN(optics_config).to(device).eval()
        zero_phase_output = zero_phase_model(first_new_field, selected_new.to(device))
        zero_phase_components = luo2022_d2nn_components_per_pair(
            zero_phase_output,
            first_new_target,
        )
        zero_phase_reference = {
            "frozen_checkpoint_phase_l2": float(model.phase.detach().square().sum().sqrt().cpu()),
            "components": {
                name: float(value.mean().detach().cpu())
                for name, value in zero_phase_components.items()
            },
            "roi_metrics": _luo2022_roi_metrics(
                zero_phase_output,
                first_new_target,
                input_size=int(values["input_size"]),
            ),
            "claim_boundary": (
                "This is a zero-phase four-layer network control, not an ideal free-space "
                "reference: it retains the frozen sequence of finite-window propagations."
            ),
        }
        padding_sensitivity = _luo2022_padding_sensitivity(
            model=model,
            field=first_new_field,
            diffusers=selected_new.to(device),
            base_output=first_new_output,
            runtime_config=frozen.runtime_config,
            contract=frozen.contract,
            device=device,
            pad_factors=diagnostic_pad_factors,
        )
        semigroup = _luo2022_semigroup_probe(
            first_new_field[:1],
            optics_config=optics_config,
        )
        trace_summary = _luo2022_trace_summary(
            first_new_trace,
            pixel_size=optics_config.pixel_size,
        )

    result = {
        "schema_version": 2,
        "status": "completed",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "read_only": True,
        "evidence_fingerprint": fingerprint,
        "diagnostic_spec": diagnostic_spec,
        "source_run": {
            "profile_id": frozen.manifest["profile_id"],
            "source_freeze_version": frozen.contract["freeze_version"],
            "checkpoint": "checkpoints/luo2022_d2nn.pt",
            "git": frozen.manifest.get("runtime", {}).get("git"),
        },
        "artifact_integrity": {
            "completed_run_state_required": True,
            "checkpoint_sha256": frozen.checkpoint_sha256,
            "runtime_config_sha256": frozen.runtime_config_sha256,
            "source_config_sha256": frozen.source_config_sha256,
            "manifest_sha256": frozen.manifest_sha256,
            "run_state_sha256": frozen.run_state_sha256,
            "final_training_diffuser_sha256": final_training_diffuser_sha256,
            "all_training_diffuser_banks_sha256": all_training_diffuser_banks_sha256,
            "evaluation_seed_diffuser_sha256": evaluation_seed_diffuser_sha256,
            "history_sha256": history_sha256,
            "source_config_integrity": frozen.source_config_integrity,
            "model_phase_unchanged": bool(torch.equal(model.phase.detach(), phase_before)),
        },
        "diffusers": {
            "frozen_contract": {
                "gaussian_sigma_lambda": frozen.contract["diffuser"]["gaussian_sigma_lambda"],
                "finite_kernel": frozen.contract["diffuser"]["finite_kernel_choice"],
                "uniqueness": uniqueness,
                "r0_acceptance_correlation_estimator": frozen.contract["diffuser"][
                    "correlation_estimator"
                ],
                "seed_schedule": seed_schedule,
                "seed_schedule_provenance": seed_schedule_provenance,
            },
            "selection": {
                "batch_forward_diagnostic_requested_count_per_population": requested_diffuser_count,
                "batch_forward_diagnostic_selected_count_per_population": selected_count,
                "batch_forward_known_source": f"final training epoch {target_epochs}",
                "batch_forward_evaluation_source": "frozen evaluation seed schedule",
                "full_final_epoch_training_diffuser_count": int(
                    final_training_diffusers.shape[0]
                ),
                "full_evaluation_seed_diffuser_count": int(new_diffusers.shape[0]),
            },
            "known_final_epoch": _luo2022_diffuser_statistics(
                final_training_diffusers,
                optics_config=optics_config,
            ),
            "evaluation_seed_population": _luo2022_diffuser_statistics(
                new_diffusers,
                optics_config=optics_config,
            ),
            "known_final_epoch_internal_uniqueness": (
                summarize_diffuser_bank_uniqueness(
                    final_training_diffusers,
                    phase_representation=str(uniqueness["phase_representation"]),
                    threshold_radians=float(uniqueness["minimum_radians"]),
                )
                if final_training_diffusers.shape[0] > 1
                else {"status": "not_applicable_for_one_diffuser"}
            ),
            "evaluation_seed_population_internal_uniqueness": (
                summarize_diffuser_bank_uniqueness(
                    new_diffusers,
                    phase_representation=str(uniqueness["phase_representation"]),
                    threshold_radians=float(uniqueness["minimum_radians"]),
                )
                if new_diffusers.shape[0] > 1
                else {"status": "not_applicable_for_one_diffuser"}
            ),
            "evaluation_seed_vs_final_epoch_training": final_cross_audit,
            "evaluation_seed_vs_all_training": all_training_cross_summary,
            "interpretation": (
                "The unwrapped phase autocorrelation is the closest available numerical "
                "reading of the paper's wording. The complex-transmittance autocorrelation "
                "is separately reported because it is the frozen R0 acceptance estimator."
            ),
        },
        "batch_level_forward_diagnostics": {
            "scope": (
                "These are fixed-prefix forward diagnostics over selected diffuser samples; "
                "they are not the full-population performance evaluation."
            ),
            "populations": population_batches,
        },
        "trace_evaluation_seed_population_first_batch": trace_summary,
        "zero_phase_four_layer_reference": zero_phase_reference,
        "propagation_window_sensitivity": {
            "padding": padding_sensitivity,
            "two_step_vs_one_step": semigroup,
        },
        "learning_rate_audit": _luo2022_learning_rate_audit(
            run_dir=run_dir,
            contract=frozen.contract,
        ),
        "claim_boundary": (
            "Read-only diagnostics for a frozen digital R0 checkpoint. The output identifies "
            "implementation sensitivities and does not itself establish a closer paper reproduction, "
            "physical hardware validity, or a performance improvement."
        ),
    }
    write_json(result_path, result)
    write_json(
        state_path,
        {
            "schema_version": 2,
            "status": "completed",
            "completed_at_utc": result["completed_at_utc"],
            "read_only": True,
            "evidence_fingerprint": fingerprint,
            "result": "diagnosis.json",
        },
    )
    return result


def _load_luo2022_posthoc_rows(
    path: Path,
    *,
    checkpoint_sha256: str,
    metric_protocol: str | None = None,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            diffuser_id = str(row["diffuser_id"])
            if row.get("checkpoint_sha256") != checkpoint_sha256:
                raise ValueError(
                    f"post-hoc row {line_number} was produced by a different checkpoint"
                )
            if metric_protocol is not None and row.get("metric_protocol") != metric_protocol:
                raise ValueError(
                    f"post-hoc row {line_number} does not use metric protocol {metric_protocol!r}"
                )
            if diffuser_id in rows:
                raise ValueError(f"duplicate post-hoc diffuser_id: {diffuser_id}")
            rows[diffuser_id] = row
    return rows


def _append_luo2022_posthoc_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
        handle.flush()


def _luo2022_metric_distribution(
    rows: list[dict[str, Any]],
    *,
    metric_names: tuple[str, ...] = ("total", "negative_pearson", "energy", "pearson"),
) -> dict[str, Any]:
    summary: dict[str, Any] = {"diffuser_count": len(rows)}
    if not rows:
        return summary
    object_counts = {int(row["object_count"]) for row in rows}
    if len(object_counts) != 1:
        raise ValueError("post-hoc rows do not share one object count")
    summary["objects_per_diffuser"] = object_counts.pop()
    summary["metrics"] = {}
    for name in metric_names:
        if any(name not in row for row in rows):
            raise ValueError(f"post-hoc rows do not all contain metric {name!r}")
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        mean = float(values.mean())
        sample_std = float(values.std(ddof=1)) if len(values) > 1 else None
        standard_error = sample_std / float(np.sqrt(len(values))) if sample_std is not None else None
        summary["metrics"][name] = {
            "mean": mean,
            "sample_std": sample_std,
            "standard_error": standard_error,
            "ci95_normal": (
                [mean - 1.96 * standard_error, mean + 1.96 * standard_error]
                if standard_error is not None
                else None
            ),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return summary


def _luo2022_posthoc_population_groups(
    rows: list[dict[str, Any]],
    *,
    target_epochs: int,
) -> dict[str, list[dict[str, Any]]]:
    training_rows = [row for row in rows if row["population"] == "training"]
    return {
        "all_training_diffusers": training_rows,
        "epochs_1_to_penultimate_training_diffusers": [
            row for row in training_rows if int(row["training_epoch"]) < target_epochs
        ],
        "last_10_epoch_training_diffusers": [
            row
            for row in training_rows
            if int(row["training_epoch"]) >= max(1, target_epochs - 9)
        ],
        "final_epoch_known_diffusers": [
            row for row in training_rows if int(row["training_epoch"]) == target_epochs
        ],
        "new_unseen_diffusers": [row for row in rows if row["population"] == "new"],
        "no_diffuser_control": [row for row in rows if row["population"] == "no_diffuser"],
    }


def summarize_luo2022_posthoc_rows(
    rows: list[dict[str, Any]],
    *,
    target_epochs: int,
) -> dict[str, Any]:
    """Summarize the paper's known and unseen diffuser populations."""

    groups = _luo2022_posthoc_population_groups(rows, target_epochs=target_epochs)
    return {name: _luo2022_metric_distribution(group_rows) for name, group_rows in groups.items()}


def _write_luo2022_posthoc_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    include_roi_metrics: bool = False,
) -> None:
    fieldnames = (
        "diffuser_id",
        "population",
        "training_epoch",
        "within_epoch_index",
        "object_count",
        "pearson",
        "negative_pearson",
        "energy",
        "total",
        "checkpoint_sha256",
        "source_freeze_version",
    )
    if include_roi_metrics:
        fieldnames += (
            *LUO2022_ROI_METRIC_FIELDS,
            "metric_protocol",
            "roi_full_canvas_metric_abs_error",
            "legacy_pearson_abs_error",
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def run_luo2022_posthoc_evaluation(
    *,
    run_dir: Path,
    config_path: Path = DEFAULT_LUO2022_CONFIG,
    download: bool = False,
    device_name: str = "cpu",
    diffuser_chunk_size: int | None = None,
    output_dir: Path | None = None,
    populations: tuple[str, ...] = ("training", "new", "no_diffuser"),
    training_epochs: tuple[int, ...] | None = None,
    include_roi_metrics: bool = False,
) -> dict[str, Any]:
    """Collect resumable per-diffuser evidence from a frozen R0 checkpoint."""

    if include_roi_metrics:
        return run_luo2022_roi_posthoc_evaluation(
            run_dir=run_dir,
            config_path=config_path,
            download=download,
            device_name=device_name,
            diffuser_chunk_size=diffuser_chunk_size,
            output_dir=output_dir,
            populations=populations,
            training_epochs=training_epochs,
        )
    if training_epochs is not None:
        raise ValueError("--posthoc-training-epochs requires --posthoc-roi-metrics")

    allowed_populations = {"training", "new", "no_diffuser"}
    requested_populations = tuple(dict.fromkeys(populations))
    if not requested_populations or not set(requested_populations) <= allowed_populations:
        raise ValueError("post-hoc populations must be training, new, or no_diffuser")

    runtime_config_path = run_dir / "config.json"
    manifest_path = run_dir / "manifest.json"
    if not runtime_config_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("completed Luo 2022 run config and manifest are required")
    runtime_config = load_config(runtime_config_path)
    manifest = load_config(manifest_path)
    source_config_path = run_dir / "source_config.json"
    contract = load_config(source_config_path if source_config_path.is_file() else config_path)
    freeze_version = str(contract["freeze_version"])
    if runtime_config["source_freeze_version"] != freeze_version:
        raise ValueError("runtime configuration freeze version does not match source configuration")
    if manifest["source_freeze_version"] != freeze_version:
        raise ValueError("manifest freeze version does not match source configuration")

    final_checkpoint_path = run_dir / "checkpoints" / "luo2022_d2nn.pt"
    latest_checkpoint_path = run_dir / "checkpoints" / "latest.pt"
    checkpoint_path = (
        final_checkpoint_path if final_checkpoint_path.is_file() else latest_checkpoint_path
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError("completed Luo 2022 checkpoint is required")
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    runtime_config_sha256 = _sha256_file(runtime_config_path)
    evidence_dir = output_dir or (run_dir / "posthoc_evaluation")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    rows_path = evidence_dir / "per_diffuser_metrics.jsonl"
    state_path = evidence_dir / "posthoc_state.json"
    fingerprint = {
        "checkpoint_sha256": checkpoint_sha256,
        "runtime_config_sha256": runtime_config_sha256,
        "source_freeze_version": freeze_version,
    }
    if state_path.is_file():
        saved_state = load_config(state_path)
        if saved_state["evidence_fingerprint"] != fingerprint:
            raise ValueError("post-hoc state does not match the frozen run")
    rows_by_id = _load_luo2022_posthoc_rows(
        rows_path,
        checkpoint_sha256=checkpoint_sha256,
    )

    values = runtime_config["runtime"]
    target_epochs = int(values["epochs"])
    expected_counts = {
        "training": target_epochs * int(values["diffusers_per_epoch"]),
        "new": int(values["eval_diffusers"]),
        "no_diffuser": 1,
    }
    device = select_device(device_name)
    seed_everything(int(values["seed"]))
    eval_base = build_torchvision_dataset(
        name="MNIST",
        root=DEFAULT_DATA_ROOT,
        train=False,
        image_size=int(contract["input"]["original_shape"][0]),
        download=download,
    )
    eval_dataset = Subset(eval_base, range(min(int(values["eval_limit"]), len(eval_base))))
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(values["batch_size"]),
        shuffle=False,
    )
    optics_config = Luo2022OpticsConfig(
        field_shape=(int(values["grid_size"]), int(values["grid_size"])),
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint["source_freeze_version"] != freeze_version:
        raise ValueError("checkpoint freeze version does not match source configuration")
    if checkpoint["runtime_config"] != runtime_config:
        raise ValueError("checkpoint runtime configuration does not match the frozen run")
    model.load_state_dict(checkpoint["model"])

    effective_chunk_size = int(
        diffuser_chunk_size or values["diffuser_chunk_size"] or values["diffusers_per_epoch"]
    )
    if effective_chunk_size <= 0:
        raise ValueError("diffuser chunk size must be positive")
    resized_shape = (int(values["input_size"]), int(values["input_size"]))
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

    def save_progress(stage: str) -> None:
        population_counts = {
            population: sum(
                row["population"] == population for row in rows_by_id.values()
            )
            for population in allowed_populations
        }
        write_json(
            state_path,
            {
                "schema_version": 1,
                "status": "running",
                "stage": stage,
                "updated_at_utc": datetime.now(UTC).isoformat(),
                "requested_populations": list(requested_populations),
                "completed_population_counts": population_counts,
                "expected_population_counts": expected_counts,
                "objects_per_diffuser": len(eval_dataset),
                "evidence_fingerprint": fingerprint,
            },
        )

    def evaluate_and_record(
        phases_cpu: torch.Tensor,
        row_metadata: list[dict[str, Any]],
        *,
        stage: str,
    ) -> None:
        missing = [
            (index, metadata)
            for index, metadata in enumerate(row_metadata)
            if metadata["diffuser_id"] not in rows_by_id
        ]
        if not missing:
            save_progress(stage)
            return
        missing_indices = [index for index, _metadata in missing]
        phases = phases_cpu[missing_indices].to(device)
        metrics = evaluate_luo2022_model_per_diffuser(
            model,
            eval_loader,
            phases,
            resized_shape=resized_shape,
            canvas_shape=optics_config.field_shape,
            device=device,
            max_batches=values["max_eval_batches"],
            diffuser_chunk_size=effective_chunk_size,
        )
        new_rows: list[dict[str, Any]] = []
        for (_index, metadata), metric in zip(missing, metrics, strict=True):
            row = {
                **metadata,
                **metric,
                "checkpoint_sha256": checkpoint_sha256,
                "source_freeze_version": freeze_version,
            }
            new_rows.append(row)
            rows_by_id[str(row["diffuser_id"])] = row
        _append_luo2022_posthoc_rows(rows_path, new_rows)
        save_progress(stage)

    save_progress("initializing")
    if "training" in requested_populations:
        for epoch in range(1, target_epochs + 1):
            phase_path = run_dir / "diffusers" / f"training_epoch_{epoch:03d}.pt"
            if not phase_path.is_file():
                raise FileNotFoundError(f"saved training diffuser bank is missing for epoch {epoch}")
            phases_cpu = torch.load(phase_path, map_location="cpu", weights_only=True)
            if int(phases_cpu.shape[0]) != int(values["diffusers_per_epoch"]):
                raise ValueError(f"training diffuser count mismatch for epoch {epoch}")
            metadata = [
                {
                    "diffuser_id": f"training:e{epoch:03d}:i{index:02d}",
                    "population": "training",
                    "training_epoch": epoch,
                    "within_epoch_index": index,
                }
                for index in range(int(phases_cpu.shape[0]))
            ]
            evaluate_and_record(phases_cpu, metadata, stage=f"training_epoch_{epoch:03d}")

    if "new" in requested_populations:
        diffuser_seed_schedule = runtime_config["diffuser_seed_schedule"]
        uniqueness = contract["diffuser"]["uniqueness"]
        phases_cpu = make_unique_correlated_diffusers(
            int(values["eval_diffusers"]),
            field_shape=optics_config.field_shape,
            base_seed=int(diffuser_seed_schedule["evaluation_base_seed"]),
            minimum_difference_radians=float(uniqueness["minimum_radians"]),
            phase_representation=str(uniqueness["phase_representation"]),
            **diffuser_kwargs,
        )
        metadata = [
            {
                "diffuser_id": f"new:i{index:02d}",
                "population": "new",
                "training_epoch": None,
                "within_epoch_index": index,
            }
            for index in range(int(phases_cpu.shape[0]))
        ]
        evaluate_and_record(phases_cpu, metadata, stage="new_unseen_diffusers")

    if "no_diffuser" in requested_populations:
        phases_cpu = torch.zeros((1, *optics_config.field_shape), dtype=torch.float32)
        evaluate_and_record(
            phases_cpu,
            [
                {
                    "diffuser_id": "no_diffuser",
                    "population": "no_diffuser",
                    "training_epoch": None,
                    "within_epoch_index": 0,
                }
            ],
            stage="no_diffuser_control",
        )

    rows = sorted(
        rows_by_id.values(),
        key=lambda row: (
            {"training": 0, "new": 1, "no_diffuser": 2}[str(row["population"])],
            int(row["training_epoch"] or 0),
            int(row["within_epoch_index"]),
        ),
    )
    summaries = summarize_luo2022_posthoc_rows(rows, target_epochs=target_epochs)
    _write_luo2022_posthoc_csv(evidence_dir / "per_diffuser_metrics.csv", rows)
    completed_counts = {
        population: sum(row["population"] == population for row in rows)
        for population in allowed_populations
    }
    requested_complete = all(
        completed_counts[population] == expected_counts[population]
        for population in requested_populations
    )
    summary = {
        "schema_version": 1,
        "status": "completed" if requested_complete else "incomplete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "requested_populations": list(requested_populations),
        "completed_population_counts": completed_counts,
        "expected_population_counts": expected_counts,
        "aggregation_protocol": (
            "mean over test objects per diffuser, then distribution across diffusers"
        ),
        "confidence_interval": "normal approximation using sample standard error",
        "groups": summaries,
        "evidence_fingerprint": fingerprint,
        "source_run": {
            "profile_id": manifest["profile_id"],
            "source_freeze_version": freeze_version,
            "git": manifest["runtime"]["git"],
        },
        "runtime": run_metadata(),
        "artifacts": {
            "per_diffuser_jsonl": "per_diffuser_metrics.jsonl",
            "per_diffuser_csv": "per_diffuser_metrics.csv",
            "state": "posthoc_state.json",
        },
        "claim_boundary": (
            "Post-hoc numerical evidence from the frozen checkpoint; paper-level "
            "acceptance still depends on published-value uncertainty and implementation audit."
        ),
    }
    write_json(evidence_dir / "posthoc_summary.json", summary)
    write_json(
        state_path,
        {
            "schema_version": 1,
            "status": summary["status"],
            "completed_at_utc": summary["completed_at_utc"],
            "requested_populations": list(requested_populations),
            "completed_population_counts": completed_counts,
            "expected_population_counts": expected_counts,
            "objects_per_diffuser": len(eval_dataset),
            "evidence_fingerprint": fingerprint,
            "summary": "posthoc_summary.json",
        },
    )
    return summary


def run_luo2022_roi_posthoc_evaluation(
    *,
    run_dir: Path,
    config_path: Path = DEFAULT_LUO2022_CONFIG,
    download: bool = False,
    device_name: str = "cpu",
    diffuser_chunk_size: int | None = None,
    output_dir: Path | None,
    populations: tuple[str, ...] = ("training", "new", "no_diffuser"),
    training_epochs: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Collect independent spatial-ROI evidence from a completed frozen R0 run.

    This is deliberately a separate evidence protocol from the original
    full-canvas post-hoc evaluator. It never writes under ``run_dir`` and
    validates every ROI full-canvas value against both the frozen loss and the
    already-completed full-population post-hoc evidence.
    """

    if output_dir is None:
        raise ValueError("ROI post-hoc evaluation requires --posthoc-output-dir")
    run_resolved = run_dir.resolve()
    evidence_resolved = output_dir.resolve()
    if evidence_resolved == run_resolved or run_resolved in evidence_resolved.parents:
        raise ValueError("ROI post-hoc output directory must be outside the frozen run directory")

    allowed_populations = {"training", "new", "no_diffuser"}
    requested_populations = tuple(dict.fromkeys(populations))
    if not requested_populations or not set(requested_populations) <= allowed_populations:
        raise ValueError("post-hoc populations must be training, new, or no_diffuser")
    if training_epochs is not None and "training" not in requested_populations:
        raise ValueError("training epoch selection requires the training population")

    device = select_device(device_name)
    frozen = _load_luo2022_frozen_run_artifacts(
        run_dir=run_dir,
        config_path=config_path,
        device=device,
    )
    values = frozen.runtime_config["runtime"]
    target_epochs = int(values["epochs"])
    if training_epochs is None:
        selected_training_epochs = tuple(range(1, target_epochs + 1))
    else:
        selected_training_epochs = tuple(sorted(set(int(epoch) for epoch in training_epochs)))
        if not selected_training_epochs:
            raise ValueError("training epoch selection must not be empty")
        if any(epoch < 1 or epoch > target_epochs for epoch in selected_training_epochs):
            raise ValueError("selected training epochs must lie within the frozen run")
    if "training" not in requested_populations:
        selected_training_epochs = ()

    optics_config = _luo2022_optics_config_from_frozen_run(
        frozen.runtime_config,
        frozen.contract,
    )
    expected_training_count = int(values["diffusers_per_epoch"])
    final_training_banks: dict[int, Path] = {}
    selected_training_bank_sha256: dict[str, str] = {}
    for epoch in selected_training_epochs:
        phase_path = run_dir / "diffusers" / f"training_epoch_{epoch:03d}.pt"
        if not phase_path.is_file():
            raise FileNotFoundError(f"saved training diffuser bank is missing for epoch {epoch}")
        final_training_banks[epoch] = phase_path
        selected_training_bank_sha256[str(epoch)] = _sha256_file(phase_path)

    diffuser_kwargs = _luo2022_diffuser_kwargs(optics_config, frozen.contract)
    seed_schedule, seed_schedule_provenance = _luo2022_frozen_diffuser_seed_schedule(
        frozen.runtime_config,
        frozen.contract,
    )
    new_diffusers: torch.Tensor | None = None
    new_diffusers_sha256: str | None = None
    if "new" in requested_populations:
        uniqueness = frozen.contract["diffuser"]["uniqueness"]
        new_diffusers = make_unique_correlated_diffusers(
            int(values["eval_diffusers"]),
            field_shape=optics_config.field_shape,
            base_seed=int(seed_schedule["evaluation_base_seed"]),
            minimum_difference_radians=float(uniqueness["minimum_radians"]),
            phase_representation=str(uniqueness["phase_representation"]),
            **diffuser_kwargs,
        )
        new_diffusers_sha256 = _sha256_tensor(new_diffusers)
    no_diffuser = torch.zeros((1, *optics_config.field_shape), dtype=torch.float32)

    expected_metadata: dict[str, dict[str, Any]] = {}
    if "training" in requested_populations:
        for epoch in selected_training_epochs:
            for index in range(expected_training_count):
                metadata = {
                    "diffuser_id": f"training:e{epoch:03d}:i{index:02d}",
                    "population": "training",
                    "training_epoch": epoch,
                    "within_epoch_index": index,
                }
                expected_metadata[str(metadata["diffuser_id"])] = metadata
    if "new" in requested_populations:
        for index in range(int(values["eval_diffusers"])):
            metadata = {
                "diffuser_id": f"new:i{index:02d}",
                "population": "new",
                "training_epoch": None,
                "within_epoch_index": index,
            }
            expected_metadata[str(metadata["diffuser_id"])] = metadata
    if "no_diffuser" in requested_populations:
        expected_metadata["no_diffuser"] = {
            "diffuser_id": "no_diffuser",
            "population": "no_diffuser",
            "training_epoch": None,
            "within_epoch_index": 0,
        }

    expected_counts = {
        "training": len(selected_training_epochs) * expected_training_count
        if "training" in requested_populations
        else 0,
        "new": int(values["eval_diffusers"]) if "new" in requested_populations else 0,
        "no_diffuser": 1 if "no_diffuser" in requested_populations else 0,
    }
    if sum(expected_counts.values()) != len(expected_metadata):
        raise RuntimeError("ROI post-hoc metadata count does not match the requested scope")

    legacy_rows_path = run_dir / "posthoc_evaluation" / "per_diffuser_metrics.jsonl"
    legacy_rows_by_id = _load_luo2022_posthoc_rows(
        legacy_rows_path,
        checkpoint_sha256=frozen.checkpoint_sha256,
    )
    missing_legacy_rows = sorted(set(expected_metadata) - set(legacy_rows_by_id))
    if missing_legacy_rows:
        raise ValueError(
            "completed full-canvas post-hoc evidence is missing requested diffuser rows"
        )

    evidence_spec = {
        "schema_version": 1,
        "implementation_version": LUO2022_ROI_METRIC_PROTOCOL,
        "read_only": True,
        "requested_populations": list(requested_populations),
        "requested_training_epochs": list(selected_training_epochs),
        "roi_definitions": {
            "full_canvas": (
                f"all {optics_config.field_shape[0]}x{optics_config.field_shape[1]} detector pixels"
            ),
            "center_input_region": (
                f"centered {int(values['input_size'])}x{int(values['input_size'])} input footprint"
            ),
            "target_support": "prepared target amplitude strictly greater than zero per object",
        },
        "aggregation_protocol": (
            "per-object PCC or energy fraction, then mean over test objects per diffuser, "
            "then distribution across diffusers"
        ),
        "regression_tolerance": LUO2022_ROI_REGRESSION_TOLERANCE,
    }
    fingerprint = {
        "metric_protocol": LUO2022_ROI_METRIC_PROTOCOL,
        "evidence_spec": evidence_spec,
        "checkpoint_sha256": frozen.checkpoint_sha256,
        "runtime_config_sha256": frozen.runtime_config_sha256,
        "source_config_sha256": frozen.source_config_sha256,
        "manifest_sha256": frozen.manifest_sha256,
        "run_state_sha256": frozen.run_state_sha256,
        "selected_training_diffuser_banks_sha256": selected_training_bank_sha256,
        "evaluation_seed_diffusers_sha256": new_diffusers_sha256,
        "no_diffuser_phase_sha256": _sha256_tensor(no_diffuser),
        "diffuser_seed_schedule": seed_schedule,
        "diffuser_seed_schedule_provenance": seed_schedule_provenance,
        "evaluation_object_count": int(values["eval_limit"]),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "roi_per_diffuser_metrics.jsonl"
    state_path = output_dir / "posthoc_roi_state.json"
    summary_path = output_dir / "posthoc_roi_summary.json"
    if state_path.is_file():
        saved_state = load_config(state_path)
        if saved_state.get("evidence_fingerprint") != fingerprint:
            raise ValueError("ROI post-hoc state does not match the frozen evidence inputs")
        if saved_state.get("metric_protocol") != LUO2022_ROI_METRIC_PROTOCOL:
            raise ValueError("ROI post-hoc state uses a different metric protocol")
        if saved_state.get("status") not in {"running", "completed", "incomplete"}:
            raise ValueError("ROI post-hoc state has an unsupported status")
    rows_by_id = _load_luo2022_posthoc_rows(
        rows_path,
        checkpoint_sha256=frozen.checkpoint_sha256,
        metric_protocol=LUO2022_ROI_METRIC_PROTOCOL,
    )
    unexpected_rows = sorted(set(rows_by_id) - set(expected_metadata))
    if unexpected_rows:
        raise ValueError("ROI post-hoc evidence contains diffuser rows outside its frozen scope")

    seed_everything(int(values["seed"]))
    eval_base = build_torchvision_dataset(
        name="MNIST",
        root=DEFAULT_DATA_ROOT,
        train=False,
        image_size=int(frozen.contract["input"]["original_shape"][0]),
        download=download,
    )
    eval_dataset = Subset(eval_base, range(min(int(values["eval_limit"]), len(eval_base))))
    if len(eval_dataset) != int(values["eval_limit"]):
        raise ValueError("frozen evaluation object count is unavailable from the requested dataset")
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(values["batch_size"]),
        shuffle=False,
    )

    model = Luo2022FourLayerD2NN(optics_config).to(device)
    model.load_state_dict(frozen.checkpoint["model"], strict=True)
    phase_before = model.phase.detach().clone()
    phase_before_sha256 = _sha256_tensor(phase_before)
    effective_chunk_size = int(
        diffuser_chunk_size or values["diffuser_chunk_size"] or expected_training_count
    )
    if effective_chunk_size <= 0:
        raise ValueError("diffuser chunk size must be positive")
    resized_shape = (int(values["input_size"]), int(values["input_size"]))

    def completed_counts() -> dict[str, int]:
        return {
            population: sum(
                row["population"] == population for row in rows_by_id.values()
            )
            for population in sorted(allowed_populations)
        }

    def save_progress(stage: str, *, status: str = "running") -> None:
        write_json(
            state_path,
            {
                "schema_version": 1,
                "status": status,
                "stage": stage,
                "updated_at_utc": datetime.now(UTC).isoformat(),
                "read_only": True,
                "metric_protocol": LUO2022_ROI_METRIC_PROTOCOL,
                "requested_populations": list(requested_populations),
                "requested_training_epochs": list(selected_training_epochs),
                "completed_population_counts": completed_counts(),
                "expected_population_counts": expected_counts,
                "objects_per_diffuser": len(eval_dataset),
                "evidence_fingerprint": fingerprint,
            },
        )

    def validate_roi_row(row: dict[str, Any]) -> None:
        diffuser_id = str(row["diffuser_id"])
        full_canvas_error = abs(
            float(row["roi_full_canvas_pearson"]) - float(row["pearson"])
        )
        if full_canvas_error > LUO2022_ROI_REGRESSION_TOLERANCE:
            raise ValueError(
                f"ROI full-canvas PCC does not reproduce frozen PCC for {diffuser_id}"
            )
        legacy_row = legacy_rows_by_id[diffuser_id]
        if int(legacy_row["object_count"]) != int(row["object_count"]):
            raise ValueError(
                f"ROI object count does not match legacy post-hoc evidence for {diffuser_id}"
            )
        legacy_error = abs(float(row["pearson"]) - float(legacy_row["pearson"]))
        if legacy_error > LUO2022_ROI_REGRESSION_TOLERANCE:
            raise ValueError(
                f"ROI PCC does not reproduce legacy post-hoc evidence for {diffuser_id}"
            )
        if str(row["population"]) != str(legacy_row["population"]):
            raise ValueError(f"ROI population does not match legacy evidence for {diffuser_id}")
        if row.get("training_epoch") != legacy_row.get("training_epoch"):
            raise ValueError(f"ROI epoch does not match legacy evidence for {diffuser_id}")

    for existing_row in rows_by_id.values():
        validate_roi_row(existing_row)

    def evaluate_and_record(
        phases_cpu: torch.Tensor,
        row_metadata: list[dict[str, Any]],
        *,
        stage: str,
    ) -> None:
        missing = [
            (index, metadata)
            for index, metadata in enumerate(row_metadata)
            if str(metadata["diffuser_id"]) not in rows_by_id
        ]
        if not missing:
            save_progress(stage)
            return
        missing_indices = [index for index, _metadata in missing]
        metrics = evaluate_luo2022_model_per_diffuser(
            model,
            eval_loader,
            phases_cpu[missing_indices].to(device),
            resized_shape=resized_shape,
            canvas_shape=optics_config.field_shape,
            device=device,
            max_batches=values["max_eval_batches"],
            diffuser_chunk_size=effective_chunk_size,
            include_roi_metrics=True,
        )
        new_rows: list[dict[str, Any]] = []
        for (_index, metadata), metric in zip(missing, metrics, strict=True):
            row = {
                **metadata,
                **metric,
                "checkpoint_sha256": frozen.checkpoint_sha256,
                "source_freeze_version": str(frozen.contract["freeze_version"]),
                "metric_protocol": LUO2022_ROI_METRIC_PROTOCOL,
            }
            row["roi_full_canvas_metric_abs_error"] = abs(
                float(row["roi_full_canvas_pearson"]) - float(row["pearson"])
            )
            row["legacy_pearson_abs_error"] = abs(
                float(row["pearson"])
                - float(legacy_rows_by_id[str(row["diffuser_id"])]["pearson"])
            )
            validate_roi_row(row)
            new_rows.append(row)
        _append_luo2022_posthoc_rows(rows_path, new_rows)
        for row in new_rows:
            rows_by_id[str(row["diffuser_id"])] = row
        save_progress(stage)

    save_progress("initializing")
    if "training" in requested_populations:
        for epoch in selected_training_epochs:
            phases_cpu = torch.load(
                final_training_banks[epoch],
                map_location="cpu",
                weights_only=True,
            )
            expected_shape = (expected_training_count, *optics_config.field_shape)
            if tuple(phases_cpu.shape) != expected_shape:
                raise ValueError(
                    f"training diffuser bank shape does not match frozen runtime for epoch {epoch}"
                )
            evaluate_and_record(
                phases_cpu,
                [
                    expected_metadata[f"training:e{epoch:03d}:i{index:02d}"]
                    for index in range(expected_training_count)
                ],
                stage=f"training_epoch_{epoch:03d}",
            )
    if "new" in requested_populations:
        if new_diffusers is None:
            raise RuntimeError("new diffuser population was requested but not generated")
        evaluate_and_record(
            new_diffusers,
            [
                expected_metadata[f"new:i{index:02d}"]
                for index in range(int(new_diffusers.shape[0]))
            ],
            stage="new_unseen_diffusers",
        )
    if "no_diffuser" in requested_populations:
        evaluate_and_record(
            no_diffuser,
            [expected_metadata["no_diffuser"]],
            stage="no_diffuser_control",
        )

    if set(rows_by_id) != set(expected_metadata):
        raise RuntimeError("ROI post-hoc evidence is incomplete after evaluation")
    phase_after = model.phase.detach().clone()
    model_phase_unchanged = bool(torch.equal(phase_after, phase_before))
    if not model_phase_unchanged:
        raise RuntimeError("read-only ROI post-hoc evaluation changed the frozen model phase")
    frozen_hashes_after = {
        "checkpoint_sha256": _sha256_file(frozen.checkpoint_path),
        "runtime_config_sha256": _sha256_file(run_dir / "config.json"),
        "source_config_sha256": _sha256_file(run_dir / "source_config.json"),
        "manifest_sha256": _sha256_file(run_dir / "manifest.json"),
        "run_state_sha256": _sha256_file(run_dir / "run_state.json"),
        "selected_training_diffuser_banks_sha256": {
            str(epoch): _sha256_file(path)
            for epoch, path in final_training_banks.items()
        },
    }
    frozen_hashes_before = {
        "checkpoint_sha256": frozen.checkpoint_sha256,
        "runtime_config_sha256": frozen.runtime_config_sha256,
        "source_config_sha256": frozen.source_config_sha256,
        "manifest_sha256": frozen.manifest_sha256,
        "run_state_sha256": frozen.run_state_sha256,
        "selected_training_diffuser_banks_sha256": selected_training_bank_sha256,
    }
    if frozen_hashes_after != frozen_hashes_before:
        raise RuntimeError("frozen R0 inputs changed during ROI post-hoc evaluation")

    rows = sorted(
        rows_by_id.values(),
        key=lambda row: (
            {"training": 0, "new": 1, "no_diffuser": 2}[str(row["population"])],
            int(row["training_epoch"] or 0),
            int(row["within_epoch_index"]),
        ),
    )
    population_groups = _luo2022_posthoc_population_groups(rows, target_epochs=target_epochs)
    roi_groups = {
        name: _luo2022_metric_distribution(
            group_rows,
            metric_names=LUO2022_ROI_METRIC_FIELDS,
        )
        for name, group_rows in population_groups.items()
    }
    pearson_groups = {
        name: _luo2022_metric_distribution(group_rows, metric_names=("pearson",))
        for name, group_rows in population_groups.items()
    }
    max_full_canvas_error = max(
        float(row["roi_full_canvas_metric_abs_error"]) for row in rows
    )
    max_legacy_error = max(float(row["legacy_pearson_abs_error"]) for row in rows)
    legacy_group_means = {
        name: (
            float(np.mean([float(legacy_rows_by_id[str(row["diffuser_id"])]["pearson"]) for row in group_rows]))
            if group_rows
            else None
        )
        for name, group_rows in population_groups.items()
    }
    completed_population_counts = completed_counts()
    requested_complete = all(
        completed_population_counts[population] == expected_counts[population]
        for population in requested_populations
    )
    if not requested_complete:
        raise RuntimeError("ROI post-hoc evaluation did not complete its requested diffuser scope")

    _write_luo2022_posthoc_csv(
        output_dir / "roi_per_diffuser_metrics.csv",
        rows,
        include_roi_metrics=True,
    )
    summary = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "read_only": True,
        "metric_protocol": LUO2022_ROI_METRIC_PROTOCOL,
        "requested_populations": list(requested_populations),
        "requested_training_epochs": list(selected_training_epochs),
        "completed_population_counts": completed_population_counts,
        "expected_population_counts": expected_counts,
        "objects_per_diffuser": len(eval_dataset),
        "evidence_spec": evidence_spec,
        "groups": pearson_groups,
        "roi_groups": roi_groups,
        "full_canvas_regression": {
            "tolerance": LUO2022_ROI_REGRESSION_TOLERANCE,
            "max_abs_roi_full_canvas_minus_frozen_pearson": max_full_canvas_error,
            "max_abs_roi_pearson_minus_legacy_pearson": max_legacy_error,
            "legacy_group_pearson_means": legacy_group_means,
        },
        "evidence_fingerprint": fingerprint,
        "source_run": {
            "profile_id": frozen.manifest["profile_id"],
            "source_freeze_version": frozen.contract["freeze_version"],
            "checkpoint": "checkpoints/luo2022_d2nn.pt",
            "git": frozen.manifest.get("runtime", {}).get("git"),
        },
        "artifact_integrity": {
            **frozen_hashes_before,
            "source_config_integrity": frozen.source_config_integrity,
            "evaluation_seed_diffusers_sha256": new_diffusers_sha256,
            "no_diffuser_phase_sha256": _sha256_tensor(no_diffuser),
            "model_phase_before_sha256": phase_before_sha256,
            "model_phase_after_sha256": _sha256_tensor(phase_after),
            "model_phase_unchanged": model_phase_unchanged,
            "frozen_input_hashes_unchanged": True,
        },
        "artifacts": {
            "per_diffuser_jsonl": "roi_per_diffuser_metrics.jsonl",
            "per_diffuser_csv": "roi_per_diffuser_metrics.csv",
            "state": "posthoc_roi_state.json",
            "summary": "posthoc_roi_summary.json",
        },
        "claim_boundary": (
            "Read-only ROI sensitivity evidence for the frozen digital R0 checkpoint. "
            "Only full-canvas PCC is the frozen R0 metric; the centered and target-support "
            "ROIs are implementation diagnostics, not paper-published acceptance domains."
        ),
    }
    write_json(summary_path, summary)
    save_progress("completed", status="completed")
    return summary


def _write_luo2022_control_ladder_text_atomically(path: Path, contents: str) -> None:
    """Atomically replace a C0 evidence text artifact on the same filesystem."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_luo2022_control_ladder_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a C0 state or summary after fully serializing it."""

    _write_luo2022_control_ladder_text_atomically(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _append_luo2022_control_ladder_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Durably append complete C0 JSONL rows so an interrupted row can be retried."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())


def _load_luo2022_control_ladder_rows(
    path: Path,
    *,
    checkpoint_sha256: str,
    metric_protocol: str,
) -> dict[str, dict[str, Any]]:
    """Load resumable C0 records keyed by their control-plus-diffuser identity."""

    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    valid_lines: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            valid_lines.append(line)
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if line_number == len(lines) and not line.endswith("\n"):
                _write_luo2022_control_ladder_text_atomically(path, "".join(valid_lines))
                break
            raise ValueError(
                f"control-ladder row {line_number} is not valid JSON evidence"
            ) from exc
        valid_lines.append(line)
        if not isinstance(row, dict):
            raise ValueError(f"control-ladder row {line_number} must be a JSON object")
        try:
            record_id = str(row.get("record_id", ""))
            control_id = str(row.get("control_id", ""))
            diffuser_id = str(row.get("diffuser_id", ""))
            if not record_id or not control_id or not diffuser_id:
                raise ValueError(f"control-ladder row {line_number} lacks a stable identity")
            if control_id not in LUO2022_CONTROL_LADDER_IDS:
                raise ValueError(f"control-ladder row {line_number} has an unknown control")
            if row.get("checkpoint_sha256") != checkpoint_sha256:
                raise ValueError(
                    f"control-ladder row {line_number} was produced by a different checkpoint"
                )
            if row.get("metric_protocol") != metric_protocol:
                raise ValueError(
                    f"control-ladder row {line_number} uses a different metric protocol"
                )
            if record_id != f"{control_id}:{diffuser_id}":
                raise ValueError(
                    f"control-ladder row {line_number} record_id does not match its control "
                    "and diffuser"
                )
            if record_id in rows:
                raise ValueError(f"duplicate control-ladder record_id: {record_id}")
            rows[record_id] = row
        except (TypeError, ValueError):
            raise
    if valid_lines and not valid_lines[-1].endswith("\n"):
        _write_luo2022_control_ladder_text_atomically(path, "".join(valid_lines) + "\n")
    return rows


def _write_luo2022_control_ladder_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the C0 records in a table that preserves the control axis."""

    fieldnames = (
        "record_id",
        "control_id",
        "operator",
        "phase_dependency",
        "control_phase_sha256",
        "post_diffuser_distance_m",
        "explicit_physical_aperture",
        "finite_numerical_window",
        "post_diffuser_window_applications",
        "diffuser_id",
        "population",
        "training_epoch",
        "within_epoch_index",
        "object_count",
        "pearson",
        "negative_pearson",
        "energy",
        "total",
        *LUO2022_ROI_METRIC_FIELDS,
        "roi_full_canvas_metric_abs_error",
        "legacy_pearson_abs_error",
        "checkpoint_sha256",
        "source_freeze_version",
        "metric_protocol",
    )
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name) for name in fieldnames})
    _write_luo2022_control_ladder_text_atomically(path, buffer.getvalue())


def _luo2022_scalar_distribution(values: list[float]) -> dict[str, float | int | list[float] | None]:
    """Summarize paired control differences without assuming multiple samples."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty control-ladder metric distribution")
    mean = float(array.mean())
    sample_std = float(array.std(ddof=1)) if array.size > 1 else None
    standard_error = (
        sample_std / float(np.sqrt(array.size))
        if sample_std is not None
        else None
    )
    return {
        "diffuser_count": int(array.size),
        "mean": mean,
        "sample_std": sample_std,
        "standard_error": standard_error,
        "ci95_normal": (
            [mean - 1.96 * standard_error, mean + 1.96 * standard_error]
            if standard_error is not None
            else None
        ),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _summarize_luo2022_control_ladder_rows(
    rows: list[dict[str, Any]],
    *,
    requested_populations: tuple[str, ...],
    requested_controls: tuple[str, ...],
) -> dict[str, Any]:
    """Group C0 evidence by optical operator and matched diffuser population."""

    metric_names = (
        "total",
        "negative_pearson",
        "energy",
        "pearson",
        *LUO2022_ROI_METRIC_FIELDS,
    )
    population_names = {
        "training": "selected_known_diffusers",
        "new": "new_unseen_diffusers",
        "no_diffuser": "no_diffuser_control",
    }
    by_control = {
        control_id: [row for row in rows if row["control_id"] == control_id]
        for control_id in requested_controls
    }
    groups = {
        control_id: {
            population_names[population]: _luo2022_metric_distribution(
                [
                    row
                    for row in control_rows
                    if row["population"] == population
                ],
                metric_names=metric_names,
            )
            for population in requested_populations
        }
        for control_id, control_rows in by_control.items()
    }
    pairwise: dict[str, dict[str, Any]] = {}
    comparisons = (
        ("zero_phase_four_layer", "direct_free_space_no_d2nn"),
        ("trained_four_layer", "zero_phase_four_layer"),
        ("trained_four_layer", "direct_free_space_no_d2nn"),
    )
    for minuend_control, subtrahend_control in comparisons:
        if (
            minuend_control not in requested_controls
            or subtrahend_control not in requested_controls
        ):
            continue
        comparison_id = f"{minuend_control}_minus_{subtrahend_control}"
        by_population: dict[str, Any] = {}
        for population in requested_populations:
            minuend_rows = {
                str(row["diffuser_id"]): row
                for row in by_control[minuend_control]
                if row["population"] == population
            }
            subtrahend_rows = {
                str(row["diffuser_id"]): row
                for row in by_control[subtrahend_control]
                if row["population"] == population
            }
            if set(minuend_rows) != set(subtrahend_rows):
                raise ValueError(
                    "control-ladder controls do not share an identical diffuser set for "
                    f"{population}"
                )
            by_population[population_names[population]] = {
                "metrics": {
                    metric: _luo2022_scalar_distribution(
                        [
                            float(minuend_rows[diffuser_id][metric])
                            - float(subtrahend_rows[diffuser_id][metric])
                            for diffuser_id in sorted(minuend_rows)
                        ]
                    )
                    for metric in metric_names
                }
            }
        pairwise[comparison_id] = by_population
    return {
        "by_control": groups,
        "paired_differences": pairwise,
    }


def _validate_luo2022_control_ladder_operator_bindings(
    forward_operators: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    *,
    trained_model: Luo2022FourLayerD2NN,
    zero_phase_model: Luo2022FourLayerD2NN,
    requested_controls: tuple[str, ...],
) -> None:
    """Reject a C0 run if a named control is bound to the wrong optical operator."""

    if set(forward_operators) != set(requested_controls):
        raise RuntimeError("control-ladder operator bindings do not cover requested controls")

    if "direct_free_space_no_d2nn" in requested_controls:
        direct_operator = forward_operators["direct_free_space_no_d2nn"]
        if (
            getattr(direct_operator, "__self__", None) is not trained_model
            or getattr(direct_operator, "__func__", None)
            is not Luo2022FourLayerD2NN.forward_without_diffractive_layers
        ):
            raise RuntimeError(
                "direct_free_space_no_d2nn is not bound to the trained model's "
                "direct-propagation operator"
            )

    for control_id, expected_model in (
        ("zero_phase_four_layer", zero_phase_model),
        ("trained_four_layer", trained_model),
    ):
        if control_id not in requested_controls:
            continue
        forward_operator = forward_operators[control_id]
        if forward_operator is not expected_model:
            raise RuntimeError(
                f"{control_id} is not bound to its expected four-layer optical operator"
            )


def run_luo2022_c0_optical_control_ladder(
    *,
    run_dir: Path,
    control_output_dir: Path,
    config_path: Path = DEFAULT_LUO2022_CONFIG,
    download: bool = False,
    device_name: str = "cpu",
    diffuser_chunk_size: int | None = None,
    populations: tuple[str, ...] = ("training", "new", "no_diffuser"),
    controls: tuple[str, ...] = LUO2022_CONTROL_LADDER_IDS,
    training_epochs: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Evaluate direct, zero-phase, and frozen four-layer optical controls.

    The routine is read-only with respect to ``run_dir``. It evaluates the
    same frozen diffuser populations under three operators: a single direct
    post-diffuser propagation with no D2NN layers, the sampled four-layer
    geometry with zero phase plates, and the exact trained checkpoint.
    """

    allowed_populations = {"training", "new", "no_diffuser"}
    requested_populations = tuple(dict.fromkeys(populations))
    if not requested_populations or not set(requested_populations) <= allowed_populations:
        raise ValueError("control-ladder populations must be training, new, or no_diffuser")
    requested_controls = tuple(dict.fromkeys(controls))
    if not requested_controls or not set(requested_controls) <= set(LUO2022_CONTROL_LADDER_IDS):
        raise ValueError("control-ladder controls must be known optical control identifiers")
    if training_epochs is not None and "training" not in requested_populations:
        raise ValueError("control training epoch selection requires the training population")
    run_resolved = run_dir.resolve()
    evidence_resolved = control_output_dir.resolve()
    if evidence_resolved == run_resolved or run_resolved in evidence_resolved.parents:
        raise ValueError("control-ladder output directory must be outside the frozen run directory")

    device = select_device(device_name)
    frozen = _load_luo2022_frozen_run_artifacts(
        run_dir=run_dir,
        config_path=config_path,
        device=device,
    )
    values = frozen.runtime_config["runtime"]
    if values.get("max_eval_batches") is not None:
        raise ValueError(
            "control-ladder requires full frozen evaluation; max_eval_batches must be null"
        )
    target_epochs = int(values["epochs"])
    if training_epochs is None:
        selected_training_epochs = (target_epochs,)
    else:
        selected_training_epochs = tuple(sorted(set(int(epoch) for epoch in training_epochs)))
        if not selected_training_epochs:
            raise ValueError("control training epoch selection must not be empty")
        if any(epoch < 1 or epoch > target_epochs for epoch in selected_training_epochs):
            raise ValueError("control training epochs must lie within the frozen run")
    if "training" not in requested_populations:
        selected_training_epochs = ()

    optics_config = _luo2022_optics_config_from_frozen_run(
        frozen.runtime_config,
        frozen.contract,
    )
    expected_training_count = int(values["diffusers_per_epoch"])
    selected_training_banks: dict[int, Path] = {}
    selected_training_bank_sha256: dict[str, str] = {}
    phase_groups: list[tuple[str, torch.Tensor, list[dict[str, Any]]]] = []
    if "training" in requested_populations:
        for epoch in selected_training_epochs:
            bank_path = run_dir / "diffusers" / f"training_epoch_{epoch:03d}.pt"
            if not bank_path.is_file():
                raise FileNotFoundError(
                    f"saved training diffuser bank is missing for epoch {epoch}"
                )
            phases = torch.load(bank_path, map_location="cpu", weights_only=True)
            expected_shape = (expected_training_count, *optics_config.field_shape)
            if tuple(phases.shape) != expected_shape:
                raise ValueError(
                    f"training diffuser bank shape does not match frozen runtime for epoch {epoch}"
                )
            selected_training_banks[epoch] = bank_path
            selected_training_bank_sha256[str(epoch)] = _sha256_file(bank_path)
            phase_groups.append(
                (
                    "training",
                    phases,
                    [
                        {
                            "diffuser_id": f"training:e{epoch:03d}:i{index:02d}",
                            "population": "training",
                            "training_epoch": epoch,
                            "within_epoch_index": index,
                        }
                        for index in range(expected_training_count)
                    ],
                )
            )

    new_diffusers: torch.Tensor | None = None
    new_diffusers_sha256: str | None = None
    seed_schedule: dict[str, Any] | None = None
    seed_schedule_provenance: str | None = None
    if "new" in requested_populations:
        diffuser_kwargs = _luo2022_diffuser_kwargs(optics_config, frozen.contract)
        uniqueness = frozen.contract["diffuser"]["uniqueness"]
        seed_schedule, seed_schedule_provenance = _luo2022_frozen_diffuser_seed_schedule(
            frozen.runtime_config,
            frozen.contract,
        )
        new_diffusers = make_unique_correlated_diffusers(
            int(values["eval_diffusers"]),
            field_shape=optics_config.field_shape,
            base_seed=int(seed_schedule["evaluation_base_seed"]),
            minimum_difference_radians=float(uniqueness["minimum_radians"]),
            phase_representation=str(uniqueness["phase_representation"]),
            **diffuser_kwargs,
        )
        new_diffusers_sha256 = _sha256_tensor(new_diffusers)
        phase_groups.append(
            (
                "new",
                new_diffusers,
                [
                    {
                        "diffuser_id": f"new:i{index:02d}",
                        "population": "new",
                        "training_epoch": None,
                        "within_epoch_index": index,
                    }
                    for index in range(int(new_diffusers.shape[0]))
                ],
            )
        )

    no_diffuser: torch.Tensor | None = None
    no_diffuser_phase_sha256: str | None = None
    if "no_diffuser" in requested_populations:
        no_diffuser = torch.zeros((1, *optics_config.field_shape), dtype=torch.float32)
        no_diffuser_phase_sha256 = _sha256_tensor(no_diffuser)
        phase_groups.append(
            (
                "no_diffuser",
                no_diffuser,
                [
                    {
                        "diffuser_id": "no_diffuser",
                        "population": "no_diffuser",
                        "training_epoch": None,
                        "within_epoch_index": 0,
                    }
                ],
            )
        )

    expected_population_counts = {
        "training": (
            len(selected_training_epochs) * expected_training_count
            if "training" in requested_populations
            else 0
        ),
        "new": int(values["eval_diffusers"]) if "new" in requested_populations else 0,
        "no_diffuser": 1 if "no_diffuser" in requested_populations else 0,
    }
    source_metadata = {
        str(metadata["diffuser_id"]): metadata
        for _population, _phases, rows in phase_groups
        for metadata in rows
    }
    if len(source_metadata) != sum(expected_population_counts.values()):
        raise RuntimeError("control-ladder diffuser metadata count does not match requested scope")

    legacy_rows_path = run_dir / "posthoc_evaluation" / "per_diffuser_metrics.jsonl"
    legacy_rows_by_id: dict[str, dict[str, Any]] | None = None
    legacy_posthoc_per_diffuser_sha256: str | None = None
    if legacy_rows_path.is_file():
        candidate_legacy_rows = _load_luo2022_posthoc_rows(
            legacy_rows_path,
            checkpoint_sha256=frozen.checkpoint_sha256,
        )
        missing_legacy_rows = sorted(set(source_metadata) - set(candidate_legacy_rows))
        if missing_legacy_rows:
            raise ValueError(
                "existing full-canvas post-hoc evidence is missing requested C0 diffuser rows"
            )
        legacy_rows_by_id = candidate_legacy_rows
        legacy_posthoc_per_diffuser_sha256 = _sha256_file(legacy_rows_path)

    trained_model = Luo2022FourLayerD2NN(optics_config).to(device).eval()
    trained_model.load_state_dict(frozen.checkpoint["model"], strict=True)
    trained_phase_before = trained_model.phase.detach().clone()
    trained_phase_sha256 = _sha256_tensor(trained_phase_before)
    zero_phase_model = Luo2022FourLayerD2NN(optics_config).to(device).eval()
    zero_phase_before = zero_phase_model.phase.detach().clone()
    if not torch.equal(zero_phase_before, torch.zeros_like(zero_phase_before)):
        raise RuntimeError("zero-phase C0 control did not initialize with zero phase plates")
    zero_phase_sha256 = _sha256_tensor(zero_phase_before)
    post_diffuser_distance_m = (
        optics_config.diffuser_to_first_layer_distance
        + (optics_config.num_layers - 1) * optics_config.layer_distance
        + optics_config.output_distance
    )
    sampled_post_diffuser_window_applications = optics_config.num_layers + 1
    all_control_definitions = {
        "direct_free_space_no_d2nn": {
            "operator": "single_direct_propagation",
            "phase_dependency": "none",
            "control_phase_sha256": None,
            "post_diffuser_distance_m": post_diffuser_distance_m,
            "explicit_physical_aperture": "none",
            "finite_numerical_window": (
                "one center-cropped finite Rayleigh-Sommerfeld propagation grid"
            ),
            "post_diffuser_window_applications": 1,
            "description": (
                "One Rayleigh-Sommerfeld propagation from the field immediately after the "
                "diffuser to the unchanged detector plane; no diffractive layer plane is kept. "
                "The direct control applies one finite numerical propagation window."
            ),
        },
        "zero_phase_four_layer": {
            "operator": "four_layer_zero_phase",
            "phase_dependency": "zero_phase_plates",
            "control_phase_sha256": zero_phase_sha256,
            "post_diffuser_distance_m": post_diffuser_distance_m,
            "explicit_physical_aperture": "none",
            "finite_numerical_window": (
                "center-cropped finite Rayleigh-Sommerfeld grid after every post-diffuser "
                "propagation segment"
            ),
            "post_diffuser_window_applications": sampled_post_diffuser_window_applications,
            "description": (
                "The frozen sampled four-layer geometry with four phase plates fixed to zero; "
                "it applies a finite numerical propagation window at each sampled plane."
            ),
        },
        "trained_four_layer": {
            "operator": "four_layer_checkpoint_phase",
            "phase_dependency": "frozen_checkpoint_phase_plates",
            "control_phase_sha256": trained_phase_sha256,
            "post_diffuser_distance_m": post_diffuser_distance_m,
            "explicit_physical_aperture": "none",
            "finite_numerical_window": (
                "center-cropped finite Rayleigh-Sommerfeld grid after every post-diffuser "
                "propagation segment"
            ),
            "post_diffuser_window_applications": sampled_post_diffuser_window_applications,
            "description": (
                "The exact frozen four-layer checkpoint with no parameter or optimizer update; "
                "it applies a finite numerical propagation window at each sampled plane."
            ),
        },
    }
    control_definitions = {
        control_id: all_control_definitions[control_id]
        for control_id in requested_controls
    }
    evidence_spec = {
        "schema_version": 1,
        "implementation_version": LUO2022_CONTROL_LADDER_METRIC_PROTOCOL,
        "read_only": True,
        "requested_populations": list(requested_populations),
        "requested_training_epochs": list(selected_training_epochs),
        "requested_controls": list(requested_controls),
        "controls": control_definitions,
        "dataset": {
            "name": "MNIST",
            "split": "test",
            "object_indices": [0, int(values["eval_limit"]) - 1],
            "objects_per_diffuser": int(values["eval_limit"]),
        },
        "roi_definitions": {
            "full_canvas": (
                f"all {optics_config.field_shape[0]}x{optics_config.field_shape[1]} detector pixels"
            ),
            "center_input_region": (
                f"centered {int(values['input_size'])}x{int(values['input_size'])} input footprint"
            ),
            "target_support": "prepared target amplitude strictly greater than zero per object",
        },
        "aggregation_protocol": (
            "per-object metrics, then mean over test objects per diffuser; controls are "
            "compared by matched diffuser identities"
        ),
        "padding_factor": int(optics_config.pad_factor),
        "dtype": str(trained_model.phase.dtype),
    }
    fingerprint = {
        "profile_id": frozen.contract["profile_id"],
        "source_freeze_version": frozen.contract["freeze_version"],
        "checkpoint_sha256": frozen.checkpoint_sha256,
        "runtime_config_sha256": frozen.runtime_config_sha256,
        "source_config_sha256": frozen.source_config_sha256,
        "manifest_sha256": frozen.manifest_sha256,
        "run_state_sha256": frozen.run_state_sha256,
        "selected_training_diffuser_banks_sha256": selected_training_bank_sha256,
        "evaluation_seed_diffusers_sha256": new_diffusers_sha256,
        "no_diffuser_phase_sha256": no_diffuser_phase_sha256,
        "legacy_posthoc_per_diffuser_sha256": legacy_posthoc_per_diffuser_sha256,
        "legacy_posthoc_regression_required": legacy_rows_by_id is not None,
        "diffuser_seed_schedule": seed_schedule,
        "diffuser_seed_schedule_provenance": seed_schedule_provenance,
        "trained_phase_sha256": trained_phase_sha256,
        "zero_phase_sha256": zero_phase_sha256,
        "implementation_source_sha256": {
            "experiment": _sha256_file(Path(__file__)),
            "d2nn": _sha256_file(Path(__file__).with_name("d2nn.py")),
        },
        "evidence_spec": evidence_spec,
    }

    rows_path = control_output_dir / "control_ladder_per_diffuser_metrics.jsonl"
    csv_path = control_output_dir / "control_ladder_per_diffuser_metrics.csv"
    state_path = control_output_dir / "control_ladder_state.json"
    summary_path = control_output_dir / "control_ladder_summary.json"
    if not state_path.is_file() and (
        rows_path.is_file() or csv_path.is_file() or summary_path.is_file()
    ):
        raise ValueError("control-ladder evidence exists without a matching state")
    saved_state = load_config(state_path) if state_path.is_file() else None
    if saved_state is not None:
        if saved_state.get("evidence_fingerprint") != fingerprint:
            raise ValueError("control-ladder state does not match the frozen inputs or request")
        if saved_state.get("metric_protocol") != LUO2022_CONTROL_LADDER_METRIC_PROTOCOL:
            raise ValueError("control-ladder state uses a different metric protocol")
        if saved_state.get("status") not in {
            "running",
            "finalizing",
            "completed",
            "incomplete",
        }:
            raise ValueError("control-ladder state has an unsupported status")
    rows_by_id = _load_luo2022_control_ladder_rows(
        rows_path,
        checkpoint_sha256=frozen.checkpoint_sha256,
        metric_protocol=LUO2022_CONTROL_LADDER_METRIC_PROTOCOL,
    )
    expected_rows = {
        f"{control_id}:{diffuser_id}": {
            "record_id": f"{control_id}:{diffuser_id}",
            "control_id": control_id,
            **metadata,
        }
        for control_id in LUO2022_CONTROL_LADDER_IDS
        if control_id in requested_controls
        for diffuser_id, metadata in source_metadata.items()
    }
    unexpected_rows = sorted(set(rows_by_id) - set(expected_rows))
    if unexpected_rows:
        raise ValueError("control-ladder evidence contains rows outside its frozen scope")

    def completed_counts() -> dict[str, dict[str, int]]:
        return {
            control_id: {
                population: sum(
                    row["control_id"] == control_id and row["population"] == population
                    for row in rows_by_id.values()
                )
                for population in sorted(allowed_populations)
            }
            for control_id in requested_controls
        }

    metric_names = (
        "total",
        "negative_pearson",
        "energy",
        "pearson",
        *LUO2022_ROI_METRIC_FIELDS,
    )

    def finite_row_value(row: dict[str, Any], name: str, record_id: str) -> float:
        try:
            value = float(row[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"control-ladder row has no numeric {name!r} value for {record_id}"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"control-ladder row has a non-finite {name!r} value for {record_id}"
            )
        return value

    def validate_control_row(row: dict[str, Any]) -> None:
        record_id = str(row["record_id"])
        expected = expected_rows.get(record_id)
        if expected is None:
            raise ValueError(f"control-ladder row is outside its frozen scope: {record_id}")
        for name in (
            "control_id",
            "diffuser_id",
            "population",
            "training_epoch",
            "within_epoch_index",
        ):
            if row.get(name) != expected.get(name):
                raise ValueError(f"control-ladder row metadata mismatch for {record_id}: {name}")
        try:
            object_count = int(row["object_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"control-ladder object count is invalid for {record_id}"
            ) from exc
        if object_count != int(values["eval_limit"]):
            raise ValueError(f"control-ladder object count mismatch for {record_id}")
        values_by_name = {
            name: finite_row_value(row, name, record_id)
            for name in metric_names
        }
        full_canvas_error = abs(
            values_by_name["roi_full_canvas_pearson"] - values_by_name["pearson"]
        )
        reported_full_canvas_error = finite_row_value(
            row,
            "roi_full_canvas_metric_abs_error",
            record_id,
        )
        if (
            abs(reported_full_canvas_error - full_canvas_error)
            > LUO2022_ROI_REGRESSION_TOLERANCE
        ):
            raise ValueError(
                f"control-ladder full-canvas PCC error provenance mismatch for {record_id}"
            )
        if full_canvas_error > LUO2022_ROI_REGRESSION_TOLERANCE:
            raise ValueError(f"control-ladder full-canvas PCC mismatch for {record_id}")
        control_id = str(row["control_id"])
        definition = control_definitions[control_id]
        for name in (
            "operator",
            "phase_dependency",
            "control_phase_sha256",
            "explicit_physical_aperture",
            "finite_numerical_window",
            "post_diffuser_window_applications",
        ):
            if row.get(name) != definition[name]:
                raise ValueError(
                    f"control-ladder control provenance mismatch for {record_id}: {name}"
                )
        try:
            post_diffuser_distance = float(row["post_diffuser_distance_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"control-ladder post-diffuser distance is invalid for {record_id}"
            ) from exc
        if not math.isfinite(post_diffuser_distance) or not math.isclose(
            post_diffuser_distance,
            float(definition["post_diffuser_distance_m"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"control-ladder post-diffuser distance mismatch for {record_id}")
        if row.get("source_freeze_version") != frozen.contract["freeze_version"]:
            raise ValueError(f"control-ladder freeze version mismatch for {record_id}")
        if row.get("metric_protocol") != LUO2022_CONTROL_LADDER_METRIC_PROTOCOL:
            raise ValueError(f"control-ladder metric protocol mismatch for {record_id}")
        if row.get("checkpoint_sha256") != frozen.checkpoint_sha256:
            raise ValueError(f"control-ladder checkpoint mismatch for {record_id}")

        reported_legacy_error = row.get("legacy_pearson_abs_error")
        if control_id != "trained_four_layer":
            if reported_legacy_error is not None:
                raise ValueError(
                    f"non-trained control has legacy regression evidence for {record_id}"
                )
            return
        if legacy_rows_by_id is None:
            if reported_legacy_error is not None:
                raise ValueError(
                    f"trained control has unexpected legacy regression evidence for {record_id}"
                )
            return
        legacy_row = legacy_rows_by_id[str(row["diffuser_id"])]
        if int(legacy_row["object_count"]) != object_count:
            raise ValueError(
                f"trained control object count does not match legacy evidence for {record_id}"
            )
        for name in ("population", "training_epoch", "within_epoch_index"):
            if legacy_row.get(name) != row.get(name):
                raise ValueError(
                    f"trained control metadata does not match legacy evidence for "
                    f"{record_id}: {name}"
                )
        legacy_pearson = finite_row_value(
            legacy_row,
            "pearson",
            f"legacy:{row['diffuser_id']}",
        )
        expected_legacy_error = abs(values_by_name["pearson"] - legacy_pearson)
        actual_legacy_error = finite_row_value(
            row,
            "legacy_pearson_abs_error",
            record_id,
        )
        if (
            abs(actual_legacy_error - expected_legacy_error)
            > LUO2022_ROI_REGRESSION_TOLERANCE
        ):
            raise ValueError(
                f"trained control legacy PCC error provenance mismatch for {record_id}"
            )
        if expected_legacy_error > LUO2022_ROI_REGRESSION_TOLERANCE:
            raise ValueError(
                f"trained control PCC does not reproduce legacy post-hoc evidence for "
                f"{record_id}"
            )

    for existing_row in rows_by_id.values():
        validate_control_row(existing_row)

    def validate_completed_summary(summary: dict[str, Any]) -> None:
        if (
            summary.get("status") != "completed"
            or summary.get("read_only") is not True
            or summary.get("metric_protocol") != LUO2022_CONTROL_LADDER_METRIC_PROTOCOL
            or summary.get("evidence_fingerprint") != fingerprint
        ):
            raise ValueError("completed control-ladder summary does not match its saved state")
        if summary.get("expected_record_count") != len(expected_rows):
            raise ValueError("completed control-ladder summary has an incorrect record count")
        if summary.get("objects_per_diffuser") != int(values["eval_limit"]):
            raise ValueError("completed control-ladder summary has an incorrect object count")
        if summary.get("completed_population_counts") != completed_counts():
            raise ValueError("completed control-ladder summary has incorrect group counts")
        expected_groups = _summarize_luo2022_control_ladder_rows(
            list(rows_by_id.values()),
            requested_populations=requested_populations,
            requested_controls=requested_controls,
        )
        if summary.get("groups") != expected_groups:
            raise ValueError("completed control-ladder summary does not match its evidence rows")
        legacy_errors = [
            float(row["legacy_pearson_abs_error"])
            for row in rows_by_id.values()
            if row["control_id"] == "trained_four_layer"
            and row["legacy_pearson_abs_error"] is not None
        ]
        expected_regression = {
            "status": (
                "verified_against_run_local_posthoc"
                if (
                    "trained_four_layer" in requested_controls
                    and legacy_rows_by_id is not None
                )
                else (
                    "not_requested"
                    if "trained_four_layer" not in requested_controls
                    else "not_available_run_local_posthoc_absent"
                )
            ),
            "tolerance": LUO2022_ROI_REGRESSION_TOLERANCE,
            "max_abs_error": max(legacy_errors) if legacy_errors else None,
        }
        if summary.get("trained_four_layer_legacy_full_canvas_regression") != (
            expected_regression
        ):
            raise ValueError(
                "completed control-ladder summary does not match trained-control "
                "legacy regression evidence"
            )

    if saved_state is not None and saved_state.get("status") == "completed":
        if set(rows_by_id) != set(expected_rows):
            raise ValueError("completed control-ladder state has incomplete evidence rows")
        if not summary_path.is_file():
            raise ValueError("completed control-ladder state lacks its summary")
        saved_summary = load_config(summary_path)
        validate_completed_summary(saved_summary)
        return saved_summary
    if saved_state is not None and saved_state.get("status") == "finalizing" and summary_path.is_file():
        if set(rows_by_id) != set(expected_rows):
            raise ValueError("finalizing control-ladder state has incomplete evidence rows")
        saved_summary = load_config(summary_path)
        validate_completed_summary(saved_summary)
        _write_luo2022_control_ladder_json(
            state_path,
            {
                **saved_state,
                "status": "completed",
                "stage": "completed",
                "updated_at_utc": datetime.now(UTC).isoformat(),
                "completed_at_utc": saved_summary.get("completed_at_utc"),
            },
        )
        return saved_summary
    if summary_path.is_file():
        raise ValueError("control-ladder summary exists before its state is completed")

    seed_everything(int(values["seed"]))
    eval_base = build_torchvision_dataset(
        name="MNIST",
        root=DEFAULT_DATA_ROOT,
        train=False,
        image_size=int(frozen.contract["input"]["original_shape"][0]),
        download=download,
    )
    eval_dataset = Subset(eval_base, range(min(int(values["eval_limit"]), len(eval_base))))
    if len(eval_dataset) != int(values["eval_limit"]):
        raise ValueError("frozen evaluation object count is unavailable from the requested dataset")
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(values["batch_size"]),
        shuffle=False,
    )
    effective_chunk_size = int(
        diffuser_chunk_size or values["diffuser_chunk_size"] or expected_training_count
    )
    if effective_chunk_size <= 0:
        raise ValueError("diffuser chunk size must be positive")
    resized_shape = (int(values["input_size"]), int(values["input_size"]))

    control_output_dir.mkdir(parents=True, exist_ok=True)

    def save_progress(stage: str, *, status: str = "running") -> None:
        _write_luo2022_control_ladder_json(
            state_path,
            {
                "schema_version": 1,
                "status": status,
                "stage": stage,
                "updated_at_utc": datetime.now(UTC).isoformat(),
                "read_only": True,
                "metric_protocol": LUO2022_CONTROL_LADDER_METRIC_PROTOCOL,
                "requested_populations": list(requested_populations),
                "requested_training_epochs": list(selected_training_epochs),
                "completed_population_counts": completed_counts(),
                "expected_population_counts_per_control": expected_population_counts,
                "expected_record_count": len(expected_rows),
                "objects_per_diffuser": len(eval_dataset),
                "evidence_fingerprint": fingerprint,
            },
        )

    def evaluate_and_record(
        control_id: str,
        forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        phases_cpu: torch.Tensor,
        metadata_rows: list[dict[str, Any]],
        *,
        stage: str,
    ) -> None:
        missing = [
            (index, metadata)
            for index, metadata in enumerate(metadata_rows)
            if f"{control_id}:{metadata['diffuser_id']}" not in rows_by_id
        ]
        if not missing:
            save_progress(stage)
            return
        missing_indices = [index for index, _metadata in missing]
        metrics = _evaluate_luo2022_forward_per_diffuser(
            forward,
            eval_loader,
            phases_cpu[missing_indices].to(device),
            resized_shape=resized_shape,
            canvas_shape=optics_config.field_shape,
            device=device,
            max_batches=values["max_eval_batches"],
            diffuser_chunk_size=effective_chunk_size,
            include_roi_metrics=True,
        )
        new_rows: list[dict[str, Any]] = []
        for (_index, metadata), metric in zip(missing, metrics, strict=True):
            record_id = f"{control_id}:{metadata['diffuser_id']}"
            row = {
                "record_id": record_id,
                "control_id": control_id,
                **control_definitions[control_id],
                **metadata,
                **metric,
                "checkpoint_sha256": frozen.checkpoint_sha256,
                "source_freeze_version": str(frozen.contract["freeze_version"]),
                "metric_protocol": LUO2022_CONTROL_LADDER_METRIC_PROTOCOL,
            }
            row["roi_full_canvas_metric_abs_error"] = abs(
                float(row["roi_full_canvas_pearson"]) - float(row["pearson"])
            )
            if control_id == "trained_four_layer" and legacy_rows_by_id is not None:
                row["legacy_pearson_abs_error"] = abs(
                    float(row["pearson"])
                    - float(legacy_rows_by_id[str(row["diffuser_id"])]["pearson"])
                )
                if (
                    float(row["legacy_pearson_abs_error"])
                    > LUO2022_ROI_REGRESSION_TOLERANCE
                ):
                    raise ValueError(
                        "trained-four-layer C0 PCC does not reproduce existing post-hoc "
                        f"evidence for {record_id}"
                    )
            else:
                row["legacy_pearson_abs_error"] = None
            validate_control_row(row)
            new_rows.append(row)
        _append_luo2022_control_ladder_rows(rows_path, new_rows)
        for row in new_rows:
            rows_by_id[str(row["record_id"])] = row
        save_progress(stage)

    all_forward_operators: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
        "direct_free_space_no_d2nn": trained_model.forward_without_diffractive_layers,
        "zero_phase_four_layer": zero_phase_model,
        "trained_four_layer": trained_model,
    }
    forward_operators = {
        control_id: all_forward_operators[control_id]
        for control_id in requested_controls
    }
    _validate_luo2022_control_ladder_operator_bindings(
        forward_operators,
        trained_model=trained_model,
        zero_phase_model=zero_phase_model,
        requested_controls=requested_controls,
    )
    save_progress("initializing")
    for control_id in requested_controls:
        for population, phases_cpu, metadata_rows in phase_groups:
            evaluate_and_record(
                control_id,
                forward_operators[control_id],
                phases_cpu,
                metadata_rows,
                stage=f"{control_id}:{population}",
            )

    if set(rows_by_id) != set(expected_rows):
        raise RuntimeError("control-ladder evaluation did not complete its requested scope")
    trained_phase_after = trained_model.phase.detach().clone()
    zero_phase_after = zero_phase_model.phase.detach().clone()
    if not torch.equal(trained_phase_after, trained_phase_before):
        raise RuntimeError("read-only control-ladder evaluation changed the frozen model phase")
    if not torch.equal(zero_phase_after, zero_phase_before):
        raise RuntimeError("read-only control-ladder evaluation changed the zero-phase control")

    frozen_hashes_before = {
        "checkpoint_sha256": frozen.checkpoint_sha256,
        "runtime_config_sha256": frozen.runtime_config_sha256,
        "source_config_sha256": frozen.source_config_sha256,
        "manifest_sha256": frozen.manifest_sha256,
        "run_state_sha256": frozen.run_state_sha256,
        "selected_training_diffuser_banks_sha256": selected_training_bank_sha256,
        "legacy_posthoc_per_diffuser_sha256": legacy_posthoc_per_diffuser_sha256,
    }
    frozen_hashes_after = {
        "checkpoint_sha256": _sha256_file(frozen.checkpoint_path),
        "runtime_config_sha256": _sha256_file(run_dir / "config.json"),
        "source_config_sha256": _sha256_file(run_dir / "source_config.json"),
        "manifest_sha256": _sha256_file(run_dir / "manifest.json"),
        "run_state_sha256": _sha256_file(run_dir / "run_state.json"),
        "selected_training_diffuser_banks_sha256": {
            str(epoch): _sha256_file(path)
            for epoch, path in selected_training_banks.items()
        },
        "legacy_posthoc_per_diffuser_sha256": (
            _sha256_file(legacy_rows_path) if legacy_rows_path.is_file() else None
        ),
    }
    if frozen_hashes_after != frozen_hashes_before:
        raise RuntimeError("frozen R0 inputs changed during control-ladder evaluation")

    control_order = {name: index for index, name in enumerate(LUO2022_CONTROL_LADDER_IDS)}
    population_order = {"training": 0, "new": 1, "no_diffuser": 2}
    rows = sorted(
        rows_by_id.values(),
        key=lambda row: (
            control_order[str(row["control_id"])],
            population_order[str(row["population"])],
            int(row["training_epoch"] or 0),
            int(row["within_epoch_index"]),
        ),
    )
    summary_groups = _summarize_luo2022_control_ladder_rows(
        rows,
        requested_populations=requested_populations,
        requested_controls=requested_controls,
    )
    legacy_errors = [
        float(row["legacy_pearson_abs_error"])
        for row in rows
        if row["control_id"] == "trained_four_layer"
        and row["legacy_pearson_abs_error"] is not None
    ]
    if (
        "trained_four_layer" in requested_controls
        and legacy_rows_by_id is not None
        and len(legacy_errors) != len(source_metadata)
    ):
        raise RuntimeError(
            "trained control does not contain one validated legacy regression error per diffuser"
        )
    summary = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "read_only": True,
        "metric_protocol": LUO2022_CONTROL_LADDER_METRIC_PROTOCOL,
        "requested_populations": list(requested_populations),
        "requested_training_epochs": list(selected_training_epochs),
        "requested_controls": list(requested_controls),
        "completed_population_counts": completed_counts(),
        "expected_population_counts_per_control": expected_population_counts,
        "expected_record_count": len(expected_rows),
        "objects_per_diffuser": len(eval_dataset),
        "evidence_spec": evidence_spec,
        "groups": summary_groups,
        "trained_four_layer_legacy_full_canvas_regression": {
            "status": (
                "verified_against_run_local_posthoc"
                if (
                    "trained_four_layer" in requested_controls
                    and legacy_rows_by_id is not None
                )
                else (
                    "not_requested"
                    if "trained_four_layer" not in requested_controls
                    else "not_available_run_local_posthoc_absent"
                )
            ),
            "tolerance": LUO2022_ROI_REGRESSION_TOLERANCE,
            "max_abs_error": max(legacy_errors) if legacy_errors else None,
        },
        "evidence_fingerprint": fingerprint,
        "source_run": {
            "profile_id": frozen.manifest["profile_id"],
            "source_freeze_version": frozen.contract["freeze_version"],
            "checkpoint": "checkpoints/luo2022_d2nn.pt",
            "git": frozen.manifest.get("runtime", {}).get("git"),
        },
        "artifact_integrity": {
            **frozen_hashes_before,
            "source_config_integrity": frozen.source_config_integrity,
            "evaluation_seed_diffusers_sha256": new_diffusers_sha256,
            "no_diffuser_phase_sha256": no_diffuser_phase_sha256,
            "trained_phase_before_sha256": trained_phase_sha256,
            "trained_phase_after_sha256": _sha256_tensor(trained_phase_after),
            "zero_phase_before_sha256": zero_phase_sha256,
            "zero_phase_after_sha256": _sha256_tensor(zero_phase_after),
            "trained_phase_unchanged": True,
            "zero_phase_unchanged": True,
            "frozen_input_hashes_unchanged": True,
        },
        "artifacts": {
            "per_diffuser_jsonl": "control_ladder_per_diffuser_metrics.jsonl",
            "per_diffuser_csv": "control_ladder_per_diffuser_metrics.csv",
            "state": "control_ladder_state.json",
            "summary": "control_ladder_summary.json",
        },
        "claim_boundary": (
            "Read-only numerical controls for a frozen digital R0 checkpoint. The direct "
            "condition is a project-defined single-propagation analogue of the paper's "
            "supplementary no-diffractive-layer wording; the paper does not disclose the "
            "exact numerical discretization, ROI, or aggregate sample protocol for Figure S4."
        ),
    }
    save_progress("finalizing", status="finalizing")
    _write_luo2022_control_ladder_csv(csv_path, rows)
    _write_luo2022_control_ladder_json(summary_path, summary)
    save_progress("completed", status="completed")
    return summary


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
