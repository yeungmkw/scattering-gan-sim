"""Reconstruction losses for the non-GAN E0/E1 baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ReconstructionLossWeights:
    """Weights for supervised reconstruction losses."""

    l1: float = 1.0
    negative_pearson: float = 0.0
    ssim: float = 0.0
    fourier: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "ReconstructionLossWeights":
        return cls(
            l1=float(values.get("l1", 1.0)),
            negative_pearson=float(values.get("negative_pearson", 0.0)),
            ssim=float(values.get("ssim", 0.0)),
            fourier=float(values.get("fourier", 0.0)),
        )

    def __post_init__(self) -> None:
        for name, value in (
            ("l1", self.l1),
            ("negative_pearson", self.negative_pearson),
            ("ssim", self.ssim),
            ("fourier", self.fourier),
        ):
            if value < 0:
                raise ValueError(f"{name} weight must be non-negative")


def negative_pearson_loss(prediction: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Return ``1 - Pearson`` averaged over the batch."""

    pred_flat = prediction.flatten(start_dim=1)
    target_flat = target.flatten(start_dim=1)
    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    numerator = (pred_centered * target_centered).sum(dim=1)
    denominator = (
        pred_centered.square().sum(dim=1).sqrt() * target_centered.square().sum(dim=1).sqrt()
    ).clamp_min(eps)
    return (1.0 - numerator / denominator).mean()


def pearson_per_image(prediction: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Return one Pearson coefficient per aligned image pair."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have matching shapes")
    pred_flat = prediction.flatten(start_dim=1)
    target_flat = target.flatten(start_dim=1)
    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    numerator = (pred_centered * target_centered).sum(dim=1)
    denominator = (
        pred_centered.square().sum(dim=1).sqrt() * target_centered.square().sum(dim=1).sqrt()
    ).clamp_min(eps)
    return numerator / denominator


def masked_pearson_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return one Pearson coefficient per pair over an explicit binary ROI."""

    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have matching shapes")
    if prediction.ndim < 2:
        raise ValueError("prediction, target, and mask must include a spatial dimension")
    weights = mask.to(dtype=prediction.dtype)
    flat_prediction = prediction.flatten(start_dim=1)
    flat_target = target.flatten(start_dim=1)
    flat_weights = weights.flatten(start_dim=1)
    weight_sum = flat_weights.sum(dim=1, keepdim=True).clamp_min(eps)
    prediction_mean = (flat_prediction * flat_weights).sum(dim=1, keepdim=True) / weight_sum
    target_mean = (flat_target * flat_weights).sum(dim=1, keepdim=True) / weight_sum
    prediction_centered = (flat_prediction - prediction_mean) * flat_weights
    target_centered = (flat_target - target_mean) * flat_weights
    numerator = (prediction_centered * target_centered).sum(dim=1)
    denominator = (
        prediction_centered.square().sum(dim=1).sqrt()
        * target_centered.square().sum(dim=1).sqrt()
    ).clamp_min(eps)
    return numerator / denominator


def _normalize_luo2022_target(
    output_intensity: torch.Tensor,
    target_amplitude: torch.Tensor,
    *,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    if output_intensity.ndim != 4:
        raise ValueError("output_intensity must have shape (B, n, H, W)")
    if target_amplitude.ndim == 4 and target_amplitude.shape[1] == 1:
        target_amplitude = target_amplitude[:, 0]
    if target_amplitude.ndim != 3:
        raise ValueError("target_amplitude must have shape (B, H, W) or (B, 1, H, W)")
    if output_intensity.shape[0] != target_amplitude.shape[0]:
        raise ValueError("output and target batch dimensions must match")
    if output_intensity.shape[-2:] != target_amplitude.shape[-2:]:
        raise ValueError("output and target spatial dimensions must match")
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be non-negative")
    return target_amplitude


def luo2022_d2nn_energy_breakdown_per_pair(
    output_intensity: torch.Tensor,
    target_amplitude: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 0.5,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Return equation (12) terms for each object-diffuser pair.

    This diagnostic helper preserves the historical four-key public result of
    :func:`luo2022_d2nn_components_per_pair` while making its energy balance
    inspectable.
    """

    target_amplitude = _normalize_luo2022_target(
        output_intensity,
        target_amplitude,
        alpha=alpha,
        beta=beta,
    )
    support = (target_amplitude > 0).to(dtype=output_intensity.dtype)
    support = support[:, None].expand_as(output_intensity)
    support_pixels = support.sum(dim=(-2, -1)).clamp_min(eps)
    outside_sum = ((1.0 - support) * output_intensity).sum(dim=(-2, -1))
    inside_sum = (support * output_intensity).sum(dim=(-2, -1))
    outside_per_support_pixel = outside_sum / support_pixels
    inside_per_support_pixel = inside_sum / support_pixels
    return {
        "outside_per_support_pixel": outside_per_support_pixel,
        "inside_per_support_pixel": inside_per_support_pixel,
        "support_pixels": support_pixels,
        "energy": (alpha * outside_sum - beta * inside_sum) / support_pixels,
    }


def luo2022_d2nn_loss(
    output_intensity: torch.Tensor,
    target_amplitude: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 0.5,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Implement Luo et al. equations (1) and (11)-(13).

    ``output_intensity`` has shape ``(B, n, H, W)`` and remains raw: no
    contrast enhancement or per-image normalization is applied.
    """

    per_pair = luo2022_d2nn_components_per_pair(
        output_intensity,
        target_amplitude,
        alpha=alpha,
        beta=beta,
        eps=eps,
    )
    components = {name: value.mean() for name, value in per_pair.items()}
    return components["total"], components


def luo2022_d2nn_components_per_pair(
    output_intensity: torch.Tensor,
    target_amplitude: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 0.5,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Return Luo et al. loss components for every object-diffuser pair.

    Each returned tensor has shape ``(B, n)``. Keeping this axis is required
    for the paper's two-level evaluation: average over objects for each
    diffuser, then summarize the diffuser distribution.
    """

    if output_intensity.ndim != 4:
        raise ValueError("output_intensity must have shape (B, n, H, W)")
    if target_amplitude.ndim == 4 and target_amplitude.shape[1] == 1:
        target_amplitude = target_amplitude[:, 0]
    if target_amplitude.ndim != 3:
        raise ValueError("target_amplitude must have shape (B, H, W) or (B, 1, H, W)")
    if output_intensity.shape[0] != target_amplitude.shape[0]:
        raise ValueError("output and target batch dimensions must match")
    if output_intensity.shape[-2:] != target_amplitude.shape[-2:]:
        raise ValueError("output and target spatial dimensions must match")
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be non-negative")

    batch_size, diffuser_count = output_intensity.shape[:2]
    expanded_target = target_amplitude[:, None].expand_as(output_intensity)
    flat_output = output_intensity.reshape(batch_size * diffuser_count, *output_intensity.shape[-2:])
    flat_target = expanded_target.reshape_as(flat_output)
    pearson = pearson_per_image(flat_output, flat_target, eps=eps)

    energy = luo2022_d2nn_energy_breakdown_per_pair(
        output_intensity,
        target_amplitude,
        alpha=alpha,
        beta=beta,
        eps=eps,
    )["energy"]
    negative_pearson = -pearson.reshape(batch_size, diffuser_count)
    return {
        "total": negative_pearson + energy,
        "negative_pearson": negative_pearson,
        "energy": energy,
        "pearson": pearson.reshape(batch_size, diffuser_count),
    }


def ssim_index(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int = 7,
    max_value: float = 1.0,
) -> torch.Tensor:
    """Compute a simple differentiable SSIM estimate per image."""

    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    padding = window_size // 2
    c1 = (0.01 * max_value) ** 2
    c2 = (0.03 * max_value) ** 2
    mu_x = F.avg_pool2d(prediction, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(prediction.square(), window_size, stride=1, padding=padding) - mu_x.square()
    sigma_y = F.avg_pool2d(target.square(), window_size, stride=1, padding=padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, window_size, stride=1, padding=padding) - mu_x * mu_y
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator.clamp_min(1e-8)).mean(dim=(1, 2, 3))


def ssim_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - ssim_index(prediction, target)).mean()


def fourier_magnitude_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_fft = torch.fft.rfft2(prediction, norm="ortho").abs()
    target_fft = torch.fft.rfft2(target, norm="ortho").abs()
    return F.l1_loss(pred_fft, target_fft)


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: ReconstructionLossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return weighted total reconstruction loss and detached components."""

    components: dict[str, torch.Tensor] = {}
    total = prediction.new_tensor(0.0)
    if weights.l1:
        components["l1"] = F.l1_loss(prediction, target)
        total = total + weights.l1 * components["l1"]
    if weights.negative_pearson:
        components["negative_pearson"] = negative_pearson_loss(prediction, target)
        total = total + weights.negative_pearson * components["negative_pearson"]
    if weights.ssim:
        components["ssim"] = ssim_loss(prediction, target)
        total = total + weights.ssim * components["ssim"]
    if weights.fourier:
        components["fourier"] = fourier_magnitude_loss(prediction, target)
        total = total + weights.fourier * components["fourier"]
    components["total"] = total
    return total, components


def huang2026_intensity_mse(
    output_intensity: torch.Tensor,
    target_intensity: torch.Tensor,
    *,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Raw detector-intensity MSE from main-text equation (11).

    No per-image normalization, PCC term, clipping, or contrast transform is
    applied.  A singleton channel axis is accepted on either tensor so the
    optical ``(B,H,W)`` output can be compared with a dataset ``(B,1,H,W)``
    target without implicit broadcasting.
    """

    output, target = _huang2026_aligned_intensities(
        output_intensity,
        target_intensity,
    )
    loss = F.mse_loss(output, target)
    if not return_components:
        return loss
    return loss, {"intensity_mse": loss, "total": loss}


def huang2026_incoherent_mse(
    output_intensity: torch.Tensor,
    target_intensity: torch.Tensor,
    *,
    input_is_ensemble: bool = True,
    ensemble_dim: int = 1,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """IC-DONN loss from Supporting equations (S11) and (S14).

    With ``input_is_ensemble=True``, ``output_intensity`` contains the
    independent coherent-realization intensities and is averaged exactly once
    along ``ensemble_dim`` before MSE.  With ``False``, the caller explicitly
    declares that the tensor is already the full-``Nr`` ensemble average.
    Chunk orchestration must therefore sum all chunk intensities and divide by
    the total ``Nr`` before calling the averaged form; averaging per-chunk
    losses is not equivalent.
    """

    if not isinstance(input_is_ensemble, bool):
        raise TypeError("input_is_ensemble must be a bool")
    if input_is_ensemble:
        if output_intensity.ndim < 3:
            raise ValueError(
                "ensemble output_intensity must include batch, realization, and "
                "spatial dimensions"
            )
        normalized_dim = _huang2026_normalized_dim(
            ensemble_dim,
            output_intensity.ndim,
            name="ensemble_dim",
        )
        if normalized_dim == 0:
            raise ValueError("ensemble_dim must not be the batch dimension")
        if output_intensity.shape[normalized_dim] <= 0:
            raise ValueError("ensemble dimension must not be empty")
        averaged_intensity = output_intensity.mean(dim=normalized_dim)
    else:
        averaged_intensity = output_intensity
    loss = huang2026_intensity_mse(
        averaged_intensity,
        target_intensity,
        return_components=False,
    )
    if not return_components:
        return loss
    return loss, {"incoherent_intensity_mse": loss, "total": loss}


def huang2026_multiwavelength_mse(
    output_intensities: torch.Tensor
    | Sequence[torch.Tensor]
    | Mapping[Any, torch.Tensor],
    target_intensities: torch.Tensor
    | Sequence[torch.Tensor]
    | Mapping[Any, torch.Tensor],
    *,
    wavelength_dim: int = 1,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sum wavelength-resolved MSE terms from Supporting equation (S24).

    ``output_intensities`` may be a tensor with an explicit wavelength axis, a
    sequence, or a mapping keyed by wavelength.  ``target_intensities`` may
    use the same container or be one common target tensor broadcast across
    wavelengths.  The wavelength terms are summed, never averaged or
    PCC-weighted.
    """

    output_items, tensor_wavelength_dim = _huang2026_wavelength_items(
        output_intensities,
        wavelength_dim=wavelength_dim,
        name="output_intensities",
    )
    target_items = _huang2026_targets_for_wavelengths(
        target_intensities,
        output_items=output_items,
        output_container=output_intensities,
        tensor_wavelength_dim=tensor_wavelength_dim,
        wavelength_dim=wavelength_dim,
    )
    per_wavelength: list[tuple[Any, torch.Tensor]] = []
    for (output_key, output), (target_key, target) in zip(
        output_items,
        target_items,
        strict=True,
    ):
        if output_key != target_key:
            raise ValueError("output and target wavelength keys do not match")
        per_wavelength.append(
            (
                output_key,
                huang2026_intensity_mse(
                    output,
                    target,
                    return_components=False,
                ),
            )
        )
    if not per_wavelength:
        raise ValueError("multi-wavelength loss requires at least one wavelength")
    total = torch.stack([loss for _key, loss in per_wavelength]).sum()
    if not return_components:
        return total
    components = {
        f"wavelength_{_huang2026_component_key(key)}_mse": loss
        for key, loss in per_wavelength
    }
    components["total"] = total
    return total, components


def _huang2026_aligned_intensities(
    output_intensity: torch.Tensor,
    target_intensity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(output_intensity, torch.Tensor) or not isinstance(
        target_intensity,
        torch.Tensor,
    ):
        raise TypeError("output_intensity and target_intensity must be tensors")
    if torch.is_complex(output_intensity) or torch.is_complex(target_intensity):
        raise TypeError("intensity tensors must be real-valued")
    output = output_intensity
    target = target_intensity
    if output.ndim + 1 == target.ndim and target.shape[1] == 1:
        target = target.squeeze(1)
    elif target.ndim + 1 == output.ndim and output.shape[1] == 1:
        output = output.squeeze(1)
    if output.shape != target.shape:
        raise ValueError(
            "output_intensity and target_intensity must have matching shapes "
            "apart from one singleton channel axis"
        )
    if output.ndim < 2:
        raise ValueError("intensity tensors must include batch and spatial dimensions")
    return output, target.to(device=output.device, dtype=output.dtype)


def _huang2026_normalized_dim(dim: int, ndim: int, *, name: str) -> int:
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise TypeError(f"{name} must be an integer")
    normalized = dim + ndim if dim < 0 else dim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"{name} is out of range for a {ndim}-dimensional tensor")
    return normalized


def _huang2026_wavelength_items(
    values: torch.Tensor | Sequence[torch.Tensor] | Mapping[Any, torch.Tensor],
    *,
    wavelength_dim: int,
    name: str,
) -> tuple[list[tuple[Any, torch.Tensor]], int | None]:
    if isinstance(values, torch.Tensor):
        if values.ndim < 3:
            raise ValueError(f"{name} tensor must include a wavelength dimension")
        normalized_dim = _huang2026_normalized_dim(
            wavelength_dim,
            values.ndim,
            name="wavelength_dim",
        )
        if normalized_dim == 0:
            raise ValueError("wavelength_dim must not be the batch dimension")
        return list(enumerate(values.unbind(dim=normalized_dim))), normalized_dim
    if isinstance(values, Mapping):
        if not values:
            raise ValueError(f"{name} mapping must not be empty")
        items = list(values.items())
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        if not values:
            raise ValueError(f"{name} sequence must not be empty")
        items = list(enumerate(values))
    else:
        raise TypeError(f"{name} must be a tensor, sequence, or mapping")
    if any(not isinstance(value, torch.Tensor) for _key, value in items):
        raise TypeError(f"{name} entries must be tensors")
    return items, None


def _huang2026_targets_for_wavelengths(
    targets: torch.Tensor | Sequence[torch.Tensor] | Mapping[Any, torch.Tensor],
    *,
    output_items: list[tuple[Any, torch.Tensor]],
    output_container: torch.Tensor | Sequence[torch.Tensor] | Mapping[Any, torch.Tensor],
    tensor_wavelength_dim: int | None,
    wavelength_dim: int,
) -> list[tuple[Any, torch.Tensor]]:
    output_keys = [key for key, _value in output_items]
    if isinstance(targets, Mapping):
        if list(targets) != output_keys:
            if set(targets) != set(output_keys):
                raise ValueError("output and target wavelength mapping keys must match")
        return [(key, targets[key]) for key in output_keys]
    if isinstance(targets, Sequence) and not isinstance(
        targets,
        (str, bytes, torch.Tensor),
    ):
        if len(targets) != len(output_items):
            raise ValueError("output and target wavelength sequences must have equal length")
        if any(not isinstance(value, torch.Tensor) for value in targets):
            raise TypeError("target_intensities entries must be tensors")
        return [
            (key, target)
            for (key, _output), target in zip(output_items, targets, strict=True)
        ]
    if not isinstance(targets, torch.Tensor):
        raise TypeError("target_intensities must be a tensor, sequence, or mapping")

    if isinstance(output_container, torch.Tensor):
        assert tensor_wavelength_dim is not None
        if targets.shape == output_container.shape:
            return list(
                enumerate(targets.unbind(dim=tensor_wavelength_dim))
            )
        if targets.ndim == output_container.ndim:
            normalized_target_dim = _huang2026_normalized_dim(
                wavelength_dim,
                targets.ndim,
                name="wavelength_dim",
            )
            if targets.shape[normalized_target_dim] == 1:
                common_target = targets.select(normalized_target_dim, 0)
                return [(key, common_target) for key in output_keys]
    return [(key, targets) for key in output_keys]


def _huang2026_component_key(key: Any) -> str:
    value = str(key)
    return "".join(character if character.isalnum() else "_" for character in value)
