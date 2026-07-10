"""Image reconstruction metrics."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from losses import ssim_index


def pearson_correlation(prediction: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    pred_flat = prediction.flatten(start_dim=1)
    target_flat = target.flatten(start_dim=1)
    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    numerator = (pred_centered * target_centered).sum(dim=1)
    denominator = (
        pred_centered.square().sum(dim=1).sqrt() * target_centered.square().sum(dim=1).sqrt()
    ).clamp_min(eps)
    return (numerator / denominator).mean()


def psnr(prediction: torch.Tensor, target: torch.Tensor, *, max_value: float = 1.0) -> torch.Tensor:
    """Return mean per-image PSNR for a batched reconstruction tensor.

    Computing PSNR from one batch-wide MSE makes a data-set result depend on
    how the loader partitions samples. Per-image PSNR followed by a batch mean
    composes correctly with the sample-count weighting in
    ``evaluate_reconstructor``.
    """

    mse_per_image = (prediction - target).square().flatten(start_dim=1).mean(dim=1)
    scale = torch.as_tensor(max_value**2, device=prediction.device, dtype=prediction.dtype)
    return (10.0 * torch.log10(scale / mse_per_image.clamp_min(1e-12))).mean()


def reconstruction_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        return {
            "l1": float(F.l1_loss(prediction, target).item()),
            "mse": float(F.mse_loss(prediction, target).item()),
            "psnr": float(psnr(prediction, target).item()),
            "ssim": float(ssim_index(prediction, target).mean().item()),
            "pearson": float(pearson_correlation(prediction, target).item()),
        }
