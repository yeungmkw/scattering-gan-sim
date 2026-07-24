"""Image reconstruction metrics and fixed-depth evaluation summaries.

The summary helpers keep the experimental sampling unit explicit.  A metric
measured for several object/diffuser pairs is first averaged over objects for
each diffuser and only then summarized across diffusers.  A no-diffuser
control has no such second level, so its objects remain the sampling units.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any

import torch
from torch.nn import functional as F

from losses import masked_pearson_per_image, pearson_per_image, ssim_index


ScalarValues = torch.Tensor | Iterable[float]


def _as_float64_vector(values: ScalarValues, *, name: str = "values") -> torch.Tensor:
    """Return detached, finite scalar values on CPU without changing order."""

    if isinstance(values, torch.Tensor):
        vector = values.detach().to(device="cpu", dtype=torch.float64).flatten()
    else:
        try:
            vector = torch.tensor([float(value) for value in values], dtype=torch.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain scalar numeric values") from error
    if not bool(torch.isfinite(vector).all()):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _as_id_list(values: Sequence[Hashable] | torch.Tensor, *, name: str) -> list[Hashable]:
    if isinstance(values, torch.Tensor):
        identifiers: list[Hashable] = values.detach().to(device="cpu").flatten().tolist()
    else:
        identifiers = list(values)
    for identifier in identifiers:
        try:
            hash(identifier)
        except TypeError as error:
            raise ValueError(f"{name} must contain hashable identifiers") from error
    return identifiers


def _empty_scalar_summary() -> dict[str, float | int | list[float] | None]:
    return {
        "count": 0,
        "mean": None,
        "sample_std": None,
        "standard_error": None,
        "ci95_normal": None,
        "minimum": None,
        "maximum": None,
    }


def scalar_summary(values: ScalarValues) -> dict[str, float | int | list[float] | None]:
    """Summarize scalar sampling units with a normal-approximation 95% CI.

    ``sample_std`` uses Bessel's correction (``ddof=1``).  A single sampling
    unit has a well-defined mean/minimum/maximum but cannot estimate sampling
    uncertainty, so its standard deviation, standard error, and CI are
    returned as ``None``.
    """

    vector = _as_float64_vector(values)
    if vector.numel() == 0:
        raise ValueError("cannot summarize an empty metric distribution")
    count = int(vector.numel())
    mean = float(vector.mean())
    sample_std = float(vector.std(unbiased=True)) if count > 1 else None
    standard_error = sample_std / math.sqrt(count) if sample_std is not None else None
    return {
        "count": count,
        "mean": mean,
        "sample_std": sample_std,
        "standard_error": standard_error,
        "ci95_normal": (
            [mean - 1.96 * standard_error, mean + 1.96 * standard_error]
            if standard_error is not None
            else None
        ),
        "minimum": float(vector.min()),
        "maximum": float(vector.max()),
    }


def two_level_diffuser_summary(
    values: ScalarValues,
    *,
    diffuser_ids: Sequence[Hashable] | torch.Tensor | None = None,
) -> dict[str, Any]:
    """Summarize pair metrics at the correct independent sampling level.

    When ``diffuser_ids`` is supplied, its entries align with ``values``.
    Objects are averaged within each diffuser, and the returned
    ``statistics`` describe the distribution of diffuser means.  Diffusers
    retain first-appearance order.  With ``diffuser_ids=None``, values are a
    no-diffuser control and are summarized directly as object-level units.
    """

    vector = _as_float64_vector(values)
    if vector.numel() == 0:
        raise ValueError("cannot summarize an empty metric distribution")
    pair_count = int(vector.numel())
    if diffuser_ids is None:
        return {
            "aggregation_unit": "object",
            "pair_count": pair_count,
            "unit_count": pair_count,
            "per_diffuser": [],
            "statistics": scalar_summary(vector),
        }

    identifiers = _as_id_list(diffuser_ids, name="diffuser_ids")
    if len(identifiers) != pair_count:
        raise ValueError("diffuser_ids and values must have the same length")
    grouped_indices: dict[Hashable, list[int]] = {}
    for index, diffuser_id in enumerate(identifiers):
        grouped_indices.setdefault(diffuser_id, []).append(index)

    per_diffuser: list[dict[str, Any]] = []
    diffuser_means: list[float] = []
    for diffuser_id, indices in grouped_indices.items():
        group_values = vector[indices]
        group_summary = scalar_summary(group_values)
        diffuser_mean = float(group_summary["mean"])
        diffuser_means.append(diffuser_mean)
        per_diffuser.append(
            {
                "diffuser_id": diffuser_id,
                "object_count": len(indices),
                "mean": diffuser_mean,
                "object_statistics": group_summary,
            }
        )
    return {
        "aggregation_unit": "diffuser",
        "pair_count": pair_count,
        "unit_count": len(per_diffuser),
        "per_diffuser": per_diffuser,
        "statistics": scalar_summary(diffuser_means),
    }


def pair_level_tail_statistics(values: ScalarValues) -> dict[str, Any]:
    """Return pair-level lower-tail statistics at five percent.

    ``cvar5_mean`` is the mean of the lowest ``ceil(0.05 * n)`` pairs and the
    fifth percentile uses linear interpolation.  Python's sort is stable, so
    equal-valued pairs retain caller-provided order; callers can therefore
    establish an explicit tie-break by ordering the input first.
    """

    vector = _as_float64_vector(values)
    if vector.numel() == 0:
        raise ValueError("cannot summarize an empty pair distribution")
    pair_count = int(vector.numel())
    bottom_count = max(1, math.ceil(0.05 * pair_count))
    ordered_indices = sorted(range(pair_count), key=lambda index: float(vector[index]))
    bottom_indices = ordered_indices[:bottom_count]
    bottom_values = vector[bottom_indices]
    return {
        "pair_count": pair_count,
        "bottom_count": bottom_count,
        "cvar5_mean": float(bottom_values.mean()),
        "percentile_5": float(torch.quantile(vector, 0.05, interpolation="linear")),
        "bottom_indices": bottom_indices,
        "bottom_values": [float(value) for value in bottom_values],
    }


def digit_group_statistics(
    values: ScalarValues,
    digits: Sequence[int] | torch.Tensor,
    *,
    diffuser_ids: Sequence[Hashable] | torch.Tensor | None = None,
) -> dict[str, dict[str, Any]]:
    """Return metrics for every digit from 0 through 9.

    Nonempty digit groups use :func:`two_level_diffuser_summary`; consequently
    they remain diffuser-level summaries when diffuser IDs are supplied and
    object-level summaries for a no-diffuser control.  Empty groups are kept
    explicitly with ``count=0`` fields so report schemas do not depend on the
    sampled labels.
    """

    vector = _as_float64_vector(values)
    labels = _as_id_list(digits, name="digits")
    if len(labels) != int(vector.numel()):
        raise ValueError("digits and values must have the same length")
    if any(
        isinstance(label, bool) or not isinstance(label, int) or not 0 <= label <= 9
        for label in labels
    ):
        raise ValueError("digits must contain integer labels from 0 through 9")
    identifiers = (
        _as_id_list(diffuser_ids, name="diffuser_ids") if diffuser_ids is not None else None
    )
    if identifiers is not None and len(identifiers) != int(vector.numel()):
        raise ValueError("diffuser_ids and values must have the same length")

    grouped: dict[str, dict[str, Any]] = {}
    for digit in range(10):
        indices = [index for index, label in enumerate(labels) if label == digit]
        if not indices:
            grouped[str(digit)] = {
                "aggregation_unit": "diffuser" if identifiers is not None else "object",
                "pair_count": 0,
                "unit_count": 0,
                "per_diffuser": [],
                "statistics": _empty_scalar_summary(),
            }
            continue
        group_diffuser_ids = (
            [identifiers[index] for index in indices] if identifiers is not None else None
        )
        grouped[str(digit)] = two_level_diffuser_summary(
            vector[indices],
            diffuser_ids=group_diffuser_ids,
        )
    return grouped


def paired_diffuser_delta_statistics(
    reference_by_diffuser: Mapping[Hashable, float],
    comparison_by_diffuser: Mapping[Hashable, float],
    *,
    require_exact_match: bool = True,
) -> dict[str, Any]:
    """Return comparison-minus-reference deltas for matched diffusers.

    Matching uses mapping keys and preserves reference mapping order.  Exact
    key matching is required by default so a missing diffuser cannot silently
    turn a paired comparison into a different population comparison.
    """

    reference_ids = list(reference_by_diffuser)
    comparison_ids = list(comparison_by_diffuser)
    reference_set = set(reference_ids)
    comparison_set = set(comparison_ids)
    if require_exact_match and reference_set != comparison_set:
        missing_from_comparison = [
            diffuser_id for diffuser_id in reference_ids if diffuser_id not in comparison_set
        ]
        missing_from_reference = [
            diffuser_id for diffuser_id in comparison_ids if diffuser_id not in reference_set
        ]
        raise ValueError(
            "paired diffuser IDs do not match: "
            f"missing from comparison={missing_from_comparison!r}, "
            f"missing from reference={missing_from_reference!r}"
        )
    matched_ids = [diffuser_id for diffuser_id in reference_ids if diffuser_id in comparison_set]
    if not matched_ids:
        raise ValueError("paired diffuser comparison has no matched IDs")

    reference_values = _as_float64_vector(
        [reference_by_diffuser[diffuser_id] for diffuser_id in matched_ids],
        name="reference values",
    )
    comparison_values = _as_float64_vector(
        [comparison_by_diffuser[diffuser_id] for diffuser_id in matched_ids],
        name="comparison values",
    )
    deltas = comparison_values - reference_values
    return {
        "delta_definition": "comparison_minus_reference",
        "diffuser_ids": matched_ids,
        "reference_values": [float(value) for value in reference_values],
        "comparison_values": [float(value) for value in comparison_values],
        "deltas": [float(value) for value in deltas],
        "statistics": scalar_summary(deltas),
    }


def psnr_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    max_value: float = 1.0,
) -> torch.Tensor:
    """Return one peak signal-to-noise ratio value per image."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have matching shapes")
    if prediction.ndim < 2:
        raise ValueError("prediction and target must include a batch dimension")
    if max_value <= 0:
        raise ValueError("max_value must be positive")
    mse_per_image = (prediction - target).square().flatten(start_dim=1).mean(dim=1)
    scale = torch.as_tensor(max_value**2, device=prediction.device, dtype=prediction.dtype)
    return 10.0 * torch.log10(scale / mse_per_image.clamp_min(1e-12))


def per_image_reconstruction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    target_support: torch.Tensor | None = None,
    max_value: float = 1.0,
    ssim_window_size: int = 7,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Return aligned reconstruction metrics without reducing the batch axis.

    The default target support is ``target > 0``.  Passing an explicit mask is
    useful when zero-valued pixels can belong to the intended object support.
    """

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have matching shapes")
    if prediction.ndim != 4:
        raise ValueError("prediction and target must have shape (B, C, H, W)")
    if target_support is None:
        target_support = target > 0
    elif target_support.shape != target.shape:
        raise ValueError("target_support must have the same shape as prediction and target")
    return {
        "psnr": psnr_per_image(prediction, target, max_value=max_value),
        "ssim": ssim_index(
            prediction,
            target,
            window_size=ssim_window_size,
            max_value=max_value,
        ),
        "pearson_full_canvas": pearson_per_image(prediction, target, eps=eps),
        "pearson_target_support": masked_pearson_per_image(
            prediction,
            target,
            target_support,
            eps=eps,
        ),
    }


def pearson_correlation(prediction: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    return pearson_per_image(prediction, target, eps=eps).mean()


def psnr(prediction: torch.Tensor, target: torch.Tensor, *, max_value: float = 1.0) -> torch.Tensor:
    """Return mean per-image PSNR for a batched reconstruction tensor.

    Computing PSNR from one batch-wide MSE makes a data-set result depend on
    how the loader partitions samples. Per-image PSNR followed by a batch mean
    composes correctly with the sample-count weighting in
    ``evaluate_reconstructor``.
    """

    return psnr_per_image(prediction, target, max_value=max_value).mean()


def reconstruction_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        return {
            "l1": float(F.l1_loss(prediction, target).item()),
            "mse": float(F.mse_loss(prediction, target).item()),
            "psnr": float(psnr(prediction, target).item()),
            "ssim": float(ssim_index(prediction, target).mean().item()),
            "pearson": float(pearson_correlation(prediction, target).item()),
        }


def huang2026_pcc_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return Huang per-image PCC with a harmless singleton-channel alignment."""

    if (
        prediction.ndim + 1 == target.ndim
        and target.ndim >= 4
        and target.shape[1] == 1
    ):
        target = target[:, 0]
    elif (
        target.ndim + 1 == prediction.ndim
        and prediction.ndim >= 4
        and prediction.shape[1] == 1
    ):
        prediction = prediction[:, 0]
    return pearson_per_image(prediction, target, eps=eps)


def per_image_pcc(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Generic alias for :func:`huang2026_pcc_per_image`."""

    return huang2026_pcc_per_image(prediction, target, eps=eps)


def huang2026_dataset_statistics(
    values: ScalarValues,
) -> dict[str, float | int | list[float] | None]:
    """Return mean/sample-SD/SE/95%-CI/min/max for image-level values."""

    return scalar_summary(values)


def huang2026_pcc_statistics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> dict[str, float | int | list[float] | None]:
    """Summarize per-image PCC without adding PCC to the training loss."""

    return huang2026_dataset_statistics(
        huang2026_pcc_per_image(prediction, target, eps=eps)
    )


def huang2026_grouped_statistics(
    values: ScalarValues,
    *,
    diffuser_seeds: Sequence[Hashable] | torch.Tensor | None = None,
    correlation_lengths: Sequence[Hashable] | torch.Tensor | None = None,
    wavelengths: Sequence[Hashable] | torch.Tensor | None = None,
    illumination_modes: Sequence[Hashable] | torch.Tensor | None = None,
    misalignments: Sequence[Hashable] | torch.Tensor | None = None,
) -> dict[str, Any]:
    """Summarize Huang 2026 metrics globally and by declared conditions.

    Every supplied grouping vector aligns one-to-one with ``values``.  Group
    rows preserve first-appearance order and retain the original scalar or
    hashable condition value, which keeps numeric wavelengths and explicit
    misalignment tuples machine-readable.  The dataset-level sampling units
    are images; each group likewise summarizes its image distribution.
    """

    vector = _as_float64_vector(values)
    if vector.numel() == 0:
        raise ValueError("cannot summarize an empty metric distribution")
    return {
        "dataset": huang2026_dataset_statistics(vector),
        "per_diffuser": _huang2026_group_rows(
            vector,
            diffuser_seeds,
            field_name="diffuser_seed",
        ),
        "per_correlation_length": _huang2026_group_rows(
            vector,
            correlation_lengths,
            field_name="correlation_length",
        ),
        "per_wavelength": _huang2026_group_rows(
            vector,
            wavelengths,
            field_name="wavelength",
        ),
        "per_illumination_mode": _huang2026_group_rows(
            vector,
            illumination_modes,
            field_name="illumination_mode",
        ),
        "per_misalignment": _huang2026_group_rows(
            vector,
            misalignments,
            field_name="misalignment",
        ),
    }


def _huang2026_group_rows(
    values: torch.Tensor,
    identifiers: Sequence[Hashable] | torch.Tensor | None,
    *,
    field_name: str,
) -> list[dict[str, Any]]:
    if identifiers is None:
        return []
    normalized_ids = _as_id_list(identifiers, name=field_name)
    if len(normalized_ids) != int(values.numel()):
        raise ValueError(f"{field_name} identifiers and values must have the same length")
    grouped_indices: dict[Hashable, list[int]] = {}
    for index, identifier in enumerate(normalized_ids):
        grouped_indices.setdefault(identifier, []).append(index)
    return [
        {
            field_name: identifier,
            "statistics": huang2026_dataset_statistics(values[indices]),
        }
        for identifier, indices in grouped_indices.items()
    ]
