"""Minimal coherent-field and single-layer D2NN inspection primitives.

This module is the first coherent-path checkpoint after the image-domain
random-PSF baseline. It supports a single MNIST image converted to a complex
field, random phase-screen or amplitude-particle corruption, one phase-only
D2NN layer, and output-plane intensity inspection. It is a compact simulator,
not a calibrated optical system.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CoherentOpticsConfig:
    """SI-unit parameters for a square sampled coherent field."""

    field_shape: tuple[int, int] = (64, 64)
    wavelength: float = 532e-9
    pixel_size: float = 8e-6
    propagation_distance: float = 0.02
    pad_factor: int = 1

    def __post_init__(self) -> None:
        height, width = self.field_shape
        if height <= 0 or width <= 0:
            raise ValueError("field_shape values must be positive")
        for name in ("wavelength", "pixel_size", "propagation_distance"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if type(self.pad_factor) is not int or self.pad_factor < 1:
            raise ValueError("pad_factor must be a positive integer")


def image_to_complex_field(image: torch.Tensor, *, eps: float = 0.0) -> torch.Tensor:
    """Map a grayscale image tensor to a zero-phase complex field.

    The image is interpreted as intensity, so the field amplitude is
    ``sqrt(image)``. Accepted shapes are ``(H, W)``, ``(1, H, W)``, and
    ``(B, 1, H, W)``. The returned field shape is ``(B, H, W)``.
    """

    image = _as_batched_single_channel(image)
    amplitude = image.clamp_min(eps).sqrt()
    return torch.complex(amplitude[:, 0], torch.zeros_like(amplitude[:, 0]))


def field_intensity(field: torch.Tensor) -> torch.Tensor:
    """Return real-valued intensity ``|field|^2``."""

    validate_complex_field(field)
    return field.real.square() + field.imag.square()


def field_phase(field: torch.Tensor) -> torch.Tensor:
    """Return wrapped phase in radians."""

    validate_complex_field(field)
    return torch.angle(field)


def make_random_phase_screen(
    field_shape: tuple[int, int],
    *,
    seed: int,
    phase_range: float = 2 * torch.pi,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a deterministic random phase screen in radians."""

    _validate_field_shape(field_shape)
    if phase_range < 0:
        raise ValueError("phase_range must be non-negative")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return (torch.rand(field_shape, generator=generator, dtype=dtype) - 0.5) * float(phase_range)


def apply_phase_screen(field: torch.Tensor, phase_screen: torch.Tensor) -> torch.Tensor:
    """Apply a phase-only scattering screen to a complex field."""

    validate_complex_field(field)
    if tuple(phase_screen.shape) != tuple(field.shape[-2:]):
        raise ValueError("phase_screen shape must match field spatial shape")
    phase = phase_screen.to(device=field.device, dtype=field.real.dtype)
    return field * torch.exp(1j * phase)


def make_amplitude_particles(
    field_shape: tuple[int, int],
    *,
    seed: int,
    num_particles: int = 12,
    radius_range: tuple[int, int] = (2, 6),
    attenuation: float = 0.15,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a deterministic amplitude mask with dark circular particles."""

    height, width = _validate_field_shape(field_shape)
    if num_particles < 0:
        raise ValueError("num_particles must be non-negative")
    low_radius, high_radius = radius_range
    if low_radius <= 0 or high_radius < low_radius:
        raise ValueError("radius_range must be positive and ordered")
    if attenuation < 0 or attenuation > 1:
        raise ValueError("attenuation must be in [0, 1]")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    mask = torch.ones(field_shape, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, dtype=dtype),
        torch.arange(width, dtype=dtype),
        indexing="ij",
    )
    for _ in range(num_particles):
        center_y = torch.randint(0, height, (1,), generator=generator).item()
        center_x = torch.randint(0, width, (1,), generator=generator).item()
        radius = torch.randint(low_radius, high_radius + 1, (1,), generator=generator).item()
        particle = (grid_y - float(center_y)).square() + (grid_x - float(center_x)).square() <= float(radius**2)
        mask = torch.where(particle, torch.full_like(mask, float(attenuation)), mask)
    return mask


def apply_amplitude_particles(field: torch.Tensor, amplitude_mask: torch.Tensor) -> torch.Tensor:
    """Apply an amplitude particle mask to a complex field."""

    validate_complex_field(field)
    if tuple(amplitude_mask.shape) != tuple(field.shape[-2:]):
        raise ValueError("amplitude_mask shape must match field spatial shape")
    mask = amplitude_mask.to(device=field.device, dtype=field.real.dtype)
    return field * mask


class AngularSpectrumPropagator:
    """Angular-spectrum propagator for batched complex fields."""

    def __init__(self, config: CoherentOpticsConfig) -> None:
        self.config = config
        self._transfer_cache: dict[tuple[tuple[int, int], str, torch.dtype, torch.dtype], torch.Tensor] = {}

    def propagate(self, field: torch.Tensor) -> torch.Tensor:
        validate_complex_field(field, expected_shape=self.config.field_shape)
        original_ndim = field.ndim
        batched = field.unsqueeze(0) if field.ndim == 2 else field
        padded_shape = tuple(dimension * self.config.pad_factor for dimension in self.config.field_shape)
        padded = _center_pad(batched, padded_shape)
        transfer = self._cached_transfer_function(
            shape=padded_shape,
            device=field.device,
            complex_dtype=field.dtype,
            real_dtype=field.real.dtype,
        )
        propagated = torch.fft.ifft2(torch.fft.fft2(padded) * transfer)
        cropped = _center_crop(propagated, self.config.field_shape)
        return cropped[0] if original_ndim == 2 else cropped

    def _cached_transfer_function(
        self,
        *,
        shape: tuple[int, int],
        device: torch.device,
        complex_dtype: torch.dtype,
        real_dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (shape, str(device), complex_dtype, real_dtype)
        transfer = self._transfer_cache.get(key)
        if transfer is None:
            transfer = self._transfer_function(
                shape=shape,
                device=device,
                complex_dtype=complex_dtype,
                real_dtype=real_dtype,
            )
            self._transfer_cache[key] = transfer
        return transfer

    def _transfer_function(
        self,
        *,
        shape: tuple[int, int],
        device: torch.device,
        complex_dtype: torch.dtype,
        real_dtype: torch.dtype,
    ) -> torch.Tensor:
        height, width = shape
        fy = torch.fft.fftfreq(height, d=self.config.pixel_size, device=device, dtype=real_dtype)
        fx = torch.fft.fftfreq(width, d=self.config.pixel_size, device=device, dtype=real_dtype)
        grid_fy, grid_fx = torch.meshgrid(fy, fx, indexing="ij")
        wave_number = 2 * torch.pi / self.config.wavelength
        ky = 2 * torch.pi * grid_fy
        kx = 2 * torch.pi * grid_fx
        kz_squared = (wave_number**2 - kx.square() - ky.square()).to(dtype=complex_dtype)
        kz = torch.sqrt(kz_squared)
        return torch.exp(1j * self.config.propagation_distance * kz)


class SingleLayerD2NN(nn.Module):
    """Single phase-only D2NN layer followed by free-space propagation."""

    def __init__(
        self,
        config: CoherentOpticsConfig,
        *,
        seed: int = 0,
        trainable: bool = False,
        phase_range: float = 2 * torch.pi,
    ) -> None:
        super().__init__()
        phase = make_random_phase_screen(config.field_shape, seed=seed, phase_range=phase_range)
        if trainable:
            self.phase = nn.Parameter(phase)
        else:
            self.register_buffer("phase", phase)
        self.config = config
        self.propagator = AngularSpectrumPropagator(config)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        validate_complex_field(field, expected_shape=self.config.field_shape)
        phase = self.phase.to(device=field.device, dtype=field.real.dtype)
        modulated = field * torch.exp(1j * phase)
        return self.propagator.propagate(modulated)


def validate_complex_field(field: torch.Tensor, expected_shape: tuple[int, int] | None = None) -> None:
    if not torch.is_complex(field):
        raise TypeError("field must be a complex tensor")
    if field.ndim not in {2, 3}:
        raise ValueError("field must have shape (height, width) or (batch, height, width)")
    if expected_shape is not None and tuple(field.shape[-2:]) != tuple(expected_shape):
        raise ValueError(f"field spatial shape {tuple(field.shape[-2:])} does not match {expected_shape}")


def _as_batched_single_channel(image: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(image):
        raise TypeError("image must be real-valued")
    if not torch.is_floating_point(image):
        image = image.to(dtype=torch.float32)
    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 3:
        if image.shape[0] != 1:
            raise ValueError("3D image tensors must have shape (1, height, width)")
        image = image.unsqueeze(0)
    elif image.ndim == 4:
        if image.shape[1] != 1:
            raise ValueError("4D image tensors must have shape (batch, 1, height, width)")
    else:
        raise ValueError("image must have shape (H, W), (1, H, W), or (B, 1, H, W)")
    return image.clamp(0.0, 1.0)


def _validate_field_shape(field_shape: tuple[int, int]) -> tuple[int, int]:
    if len(field_shape) != 2:
        raise ValueError("field_shape must have two dimensions")
    height, width = int(field_shape[0]), int(field_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("field_shape values must be positive")
    return height, width


def _center_pad(field: torch.Tensor, padded_shape: tuple[int, int]) -> torch.Tensor:
    height, width = field.shape[-2:]
    padded_height, padded_width = padded_shape
    if padded_height < height or padded_width < width:
        raise ValueError("padded shape must be at least as large as field shape")
    top = (padded_height - height) // 2
    left = (padded_width - width) // 2
    padded = field.new_zeros((*field.shape[:-2], padded_height, padded_width))
    padded[..., top : top + height, left : left + width] = field
    return padded


def _center_crop(field: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
    height, width = field.shape[-2:]
    target_height, target_width = target_shape
    if target_height > height or target_width > width:
        raise ValueError("target shape must fit inside field shape")
    top = (height - target_height) // 2
    left = (width - target_width) // 2
    return field[..., top : top + target_height, left : left + target_width]
