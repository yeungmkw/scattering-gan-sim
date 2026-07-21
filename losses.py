"""Reconstruction losses for the non-GAN E0/E1 baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

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

    support = (target_amplitude > 0).to(dtype=output_intensity.dtype)
    support = support[:, None].expand_as(output_intensity)
    support_pixels = support.sum(dim=(-2, -1)).clamp_min(eps)
    energy = (
        alpha * ((1.0 - support) * output_intensity).sum(dim=(-2, -1))
        - beta * (support * output_intensity).sum(dim=(-2, -1))
    ) / support_pixels
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
