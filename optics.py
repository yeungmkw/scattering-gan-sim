"""Random-PSF scattering channel for the first reconstruction baseline.

The model here is intentionally simpler than a coherent phase-screen simulator:
it treats a diffuser as a fixed positive PSF, corrupts an object by convolution,
then applies multiplicative speckle gain and additive sensor/read noise. This
matches the E0/E1 role of a controllable PSF/noise forward model before moving
to Fresnel or angular-spectrum propagation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RandomPSFConfig:
    """Configuration for the random-PSF forward model."""

    kernel_size: int = 17
    num_train_diffusers: int = 16
    num_test_diffusers: int = 4
    gaussian_noise_std: float = 0.02
    speckle_contrast: float = 0.25
    seed: int = 42

    @property
    def num_diffusers(self) -> int:
        return self.num_train_diffusers + self.num_test_diffusers

    def __post_init__(self) -> None:
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if self.num_train_diffusers <= 0:
            raise ValueError("num_train_diffusers must be positive")
        if self.num_test_diffusers < 0:
            raise ValueError("num_test_diffusers must be non-negative")
        if self.gaussian_noise_std < 0:
            raise ValueError("gaussian_noise_std must be non-negative")
        if self.speckle_contrast < 0:
            raise ValueError("speckle_contrast must be non-negative")


@dataclass(frozen=True)
class DiffuserSplit:
    """Deterministic train/test diffuser ids."""

    train_ids: tuple[int, ...]
    test_ids: tuple[int, ...]


def diffuser_split(num_train_diffusers: int, num_test_diffusers: int) -> DiffuserSplit:
    """Return contiguous seen and unseen diffuser ids."""

    if num_train_diffusers <= 0:
        raise ValueError("num_train_diffusers must be positive")
    if num_test_diffusers < 0:
        raise ValueError("num_test_diffusers must be non-negative")
    train_ids = tuple(range(num_train_diffusers))
    test_ids = tuple(range(num_train_diffusers, num_train_diffusers + num_test_diffusers))
    return DiffuserSplit(train_ids=train_ids, test_ids=test_ids)


def make_random_psfs(
    *,
    num_diffusers: int,
    kernel_size: int,
    seed: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create normalized positive PSFs with diffuser-specific random texture.

    Each kernel is a random nonnegative field under a Gaussian envelope, then
    normalized to unit sum so that convolution roughly preserves total energy.
    """

    if num_diffusers <= 0:
        raise ValueError("num_diffusers must be positive")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if not torch.is_floating_point(torch.empty((), dtype=dtype)):
        raise TypeError("dtype must be floating point")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    axis = torch.linspace(-1.0, 1.0, kernel_size, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    radius2 = grid_x.square() + grid_y.square()
    random_width = 0.25 + 0.35 * torch.rand((num_diffusers, 1, 1, 1), generator=generator, dtype=dtype)
    envelope = torch.exp(-radius2.view(1, 1, kernel_size, kernel_size) / (2.0 * random_width.square()))
    texture = torch.rand((num_diffusers, 1, kernel_size, kernel_size), generator=generator, dtype=dtype)
    kernels = (0.05 + texture.square()) * envelope
    kernels = kernels / kernels.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return kernels.to(device=device) if device is not None else kernels


class RandomPSFChannel(nn.Module):
    """Image-domain scattering corruption channel backed by fixed PSFs."""

    def __init__(
        self,
        psfs: torch.Tensor,
        *,
        gaussian_noise_std: float = 0.02,
        speckle_contrast: float = 0.25,
        normalize_output: bool = True,
    ) -> None:
        super().__init__()
        if psfs.ndim != 4 or psfs.shape[1] != 1:
            raise ValueError("psfs must have shape (num_diffusers, 1, kernel, kernel)")
        if psfs.shape[-1] != psfs.shape[-2] or psfs.shape[-1] % 2 == 0:
            raise ValueError("psfs must use odd square kernels")
        if gaussian_noise_std < 0:
            raise ValueError("gaussian_noise_std must be non-negative")
        if speckle_contrast < 0:
            raise ValueError("speckle_contrast must be non-negative")
        psfs = psfs.detach().clone().to(dtype=torch.float32)
        psfs = psfs / psfs.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        self.register_buffer("psfs", psfs)
        self.gaussian_noise_std = float(gaussian_noise_std)
        self.speckle_contrast = float(speckle_contrast)
        self.normalize_output = bool(normalize_output)

    @classmethod
    def from_config(cls, config: RandomPSFConfig) -> "RandomPSFChannel":
        psfs = make_random_psfs(
            num_diffusers=config.num_diffusers,
            kernel_size=config.kernel_size,
            seed=config.seed,
        )
        return cls(
            psfs,
            gaussian_noise_std=config.gaussian_noise_std,
            speckle_contrast=config.speckle_contrast,
        )

    @property
    def num_diffusers(self) -> int:
        return int(self.psfs.shape[0])

    @property
    def kernel_size(self) -> int:
        return int(self.psfs.shape[-1])

    def forward(
        self,
        clean: torch.Tensor,
        diffuser_ids: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        add_noise: bool = True,
    ) -> torch.Tensor:
        """Return corrupted observations for a batch of clean images."""

        validate_image_batch(clean, name="clean")
        if diffuser_ids.ndim != 1 or diffuser_ids.shape[0] != clean.shape[0]:
            raise ValueError("diffuser_ids must have shape (batch,)")

        diffuser_ids = diffuser_ids.to(device=clean.device, dtype=torch.long)
        if diffuser_ids.amin().item() < 0 or diffuser_ids.amax().item() >= self.num_diffusers:
            raise ValueError("diffuser_ids contain ids outside the PSF bank")

        psfs = self.psfs.to(device=clean.device, dtype=clean.dtype)
        padding = self.kernel_size // 2
        unique_diffuser_ids = torch.unique(diffuser_ids)
        if unique_diffuser_ids.numel() == 1:
            kernel = psfs[int(unique_diffuser_ids.item())].unsqueeze(0)
            corrupted = F.conv2d(clean, kernel, padding=padding)
        elif unique_diffuser_ids.numel() == clean.shape[0]:
            kernels = psfs.index_select(0, diffuser_ids)
            grouped_clean = clean.permute(1, 0, 2, 3)
            corrupted = F.conv2d(
                grouped_clean,
                kernels,
                padding=padding,
                groups=int(clean.shape[0]),
            ).permute(1, 0, 2, 3)
        else:
            corrupted = torch.empty_like(clean)
            for diffuser_id in unique_diffuser_ids.tolist():
                mask = diffuser_ids == int(diffuser_id)
                kernel = psfs[int(diffuser_id)].unsqueeze(0)
                corrupted[mask] = F.conv2d(clean[mask], kernel, padding=padding)

        if add_noise and self.speckle_contrast > 0:
            speckle = torch.exp(
                self.speckle_contrast * _randn_like(corrupted, generator=generator)
                - 0.5 * self.speckle_contrast**2
            )
            corrupted = corrupted * speckle

        if add_noise and self.gaussian_noise_std > 0:
            corrupted = corrupted + self.gaussian_noise_std * _randn_like(corrupted, generator=generator)

        corrupted = corrupted.clamp_min(0.0)
        if self.normalize_output:
            corrupted = normalize_minmax_per_sample(corrupted)
        return corrupted


def normalize_minmax_per_sample(image: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Normalize each sample independently to the [0, 1] range."""

    validate_image_batch(image, name="image")
    low = image.amin(dim=(-2, -1), keepdim=True)
    high = image.amax(dim=(-2, -1), keepdim=True)
    return (image - low) / (high - low).clamp_min(eps)


def forward_sanity_metrics(clean: torch.Tensor, corrupted: torch.Tensor) -> dict[str, float]:
    """Return scalar diagnostics for a generated forward-model batch."""

    validate_image_batch(clean, name="clean")
    validate_image_batch(corrupted, name="corrupted")
    if clean.shape != corrupted.shape:
        raise ValueError("clean and corrupted must have matching shape")
    corrupted_mean = corrupted.mean()
    return {
        "clean_mean": float(clean.mean().item()),
        "clean_std": float(clean.std(unbiased=False).item()),
        "corrupted_mean": float(corrupted_mean.item()),
        "corrupted_std": float(corrupted.std(unbiased=False).item()),
        "corrupted_contrast": float((corrupted.std(unbiased=False) / corrupted_mean.clamp_min(1e-8)).item()),
        "corrupted_min": float(corrupted.amin().item()),
        "corrupted_max": float(corrupted.amax().item()),
    }


def validate_image_batch(image: torch.Tensor, *, name: str) -> None:
    if image.ndim != 4:
        raise ValueError(f"{name} must have shape (batch, channels, height, width)")
    if image.shape[1] != 1:
        raise ValueError(f"{name} must be single-channel")
    if torch.is_complex(image):
        raise TypeError(f"{name} must be real-valued")
    if not torch.is_floating_point(image):
        raise TypeError(f"{name} must use a floating-point dtype")


def _randn_like(image: torch.Tensor, *, generator: torch.Generator | None) -> torch.Tensor:
    return torch.randn(
        image.shape,
        generator=generator,
        device=image.device,
        dtype=image.dtype,
    )
