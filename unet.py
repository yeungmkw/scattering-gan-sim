"""U-Net reconstructor for scattering-corrupted intensity images."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Two-convolution block used by the compact U-Net baseline."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.block(image)


class UNetReconstructor(nn.Module):
    """Small U-Net mapping corrupted observations ``y`` to reconstructions."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        output_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or base_channels <= 0:
            raise ValueError("channel counts must be positive")
        if output_activation not in {"sigmoid", "identity"}:
            raise ValueError("output_activation must be 'sigmoid' or 'identity'")
        self.output_activation = output_activation

        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
        self.down = nn.MaxPool2d(kernel_size=2)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, corrupted: torch.Tensor) -> torch.Tensor:
        if corrupted.ndim != 4:
            raise ValueError("corrupted must have shape (batch, channels, height, width)")
        enc1 = self.enc1(corrupted)
        enc2 = self.enc2(self.down(enc1))
        bottleneck = self.bottleneck(self.down(enc2))

        up2 = _resize_like(self.up2(bottleneck), enc2)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = _resize_like(self.up1(dec2), enc1)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        output = self.out(dec1)
        if self.output_activation == "sigmoid":
            output = torch.sigmoid(output)
        return output


def _resize_like(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if image.shape[-2:] == reference.shape[-2:]:
        return image
    return F.interpolate(image, size=reference.shape[-2:], mode="bilinear", align_corners=False)
