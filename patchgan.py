"""Conditional PatchGAN discriminator for E2 adversarial reconstruction."""

from __future__ import annotations

import torch
from torch import nn


class PatchDiscriminator(nn.Module):
    """Patch-level discriminator conditioned on the corrupted observation.

    The discriminator receives ``[corrupted, candidate]`` channel-wise, so the
    adversarial signal scores whether a reconstruction is plausible for the
    specific scattering observation instead of only whether it looks like a
    clean digit in isolation.
    """

    def __init__(
        self,
        *,
        condition_channels: int = 1,
        image_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        if condition_channels <= 0 or image_channels <= 0 or base_channels <= 0:
            raise ValueError("channel counts must be positive")
        in_channels = condition_channels + image_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            _disc_block(base_channels, base_channels * 2, stride=2),
            _disc_block(base_channels * 2, base_channels * 4, stride=2),
            nn.Conv2d(base_channels * 4, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, corrupted: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        if corrupted.shape != candidate.shape:
            raise ValueError("corrupted and candidate must have matching shape")
        if corrupted.ndim != 4:
            raise ValueError("inputs must have shape (batch, channels, height, width)")
        return self.net(torch.cat([corrupted, candidate], dim=1))


def _disc_block(in_channels: int, out_channels: int, *, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )
