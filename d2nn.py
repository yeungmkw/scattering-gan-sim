"""Coherent optical propagation and D2NN primitives.

The legacy path retains the compact single-layer inspection prototype. The
Luo 2022 R0 path adds correlated thin phase diffusers, Rayleigh-Sommerfeld
propagation, and a four-layer trainable phase-only D2NN. Neither path is a
calibrated hardware model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


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


def amplitude_to_complex_field(image: torch.Tensor) -> torch.Tensor:
    """Encode an image directly as zero-phase field amplitude.

    This is the Luo et al. 2022 R0 input convention used by paper equation
    (6). It intentionally differs from :func:`image_to_complex_field`, which
    interprets the image as intensity for the legacy prototype.
    """

    image = _as_batched_single_channel(image)
    amplitude = image[:, 0]
    return torch.complex(amplitude, torch.zeros_like(amplitude))


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


@dataclass(frozen=True)
class Luo2022OpticsConfig:
    """Optical geometry for the four-layer Luo et al. 2022 R0 path."""

    field_shape: tuple[int, int] = (240, 240)
    wavelength: float = 0.75e-3
    pixel_size: float = 0.3e-3
    object_to_diffuser_distance: float = 40e-3
    diffuser_to_first_layer_distance: float = 2e-3
    layer_distance: float = 2e-3
    output_distance: float = 7e-3
    num_layers: int = 4
    pad_factor: int = 2

    def __post_init__(self) -> None:
        _validate_field_shape(self.field_shape)
        for name in (
            "wavelength",
            "pixel_size",
            "object_to_diffuser_distance",
            "diffuser_to_first_layer_distance",
            "layer_distance",
            "output_distance",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if type(self.num_layers) is not int or self.num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        if type(self.pad_factor) is not int or self.pad_factor < 2:
            raise ValueError("pad_factor must be an integer of at least 2 for linear convolution")


class RayleighSommerfeldPropagator:
    """FFT linear-convolution implementation of paper equation (7).

    The sampled Rayleigh-Sommerfeld kernel includes the ``pixel_size**2``
    quadrature factor. The input is center-padded before convolution and the
    central same-sized field is returned.
    """

    def __init__(
        self,
        *,
        field_shape: tuple[int, int],
        wavelength: float,
        pixel_size: float,
        distance: float,
        pad_factor: int = 2,
    ) -> None:
        self.field_shape = _validate_field_shape(field_shape)
        self.wavelength = float(wavelength)
        self.pixel_size = float(pixel_size)
        self.distance = float(distance)
        self.pad_factor = int(pad_factor)
        if self.wavelength <= 0 or self.pixel_size <= 0 or self.distance <= 0:
            raise ValueError("wavelength, pixel_size, and distance must be positive")
        if self.pad_factor < 2:
            raise ValueError("pad_factor must be at least 2 for linear convolution")
        self._kernel_cache: dict[tuple[tuple[int, int], str, torch.dtype], torch.Tensor] = {}

    def propagate(self, field: torch.Tensor) -> torch.Tensor:
        validate_complex_field(field, expected_shape=self.field_shape)
        original_ndim = field.ndim
        batched = field.unsqueeze(0) if field.ndim == 2 else field
        padded_shape = tuple(size * self.pad_factor for size in self.field_shape)
        padded = _center_pad(batched, padded_shape)
        kernel = self._cached_kernel(padded_shape, field.device, field.dtype)
        propagated = torch.fft.ifft2(torch.fft.fft2(padded) * kernel)
        cropped = _center_crop(propagated, self.field_shape)
        return cropped[0] if original_ndim == 2 else cropped

    def _cached_kernel(
        self,
        shape: tuple[int, int],
        device: torch.device,
        complex_dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (shape, str(device), complex_dtype)
        kernel = self._kernel_cache.get(key)
        if kernel is None:
            kernel = self._frequency_kernel(shape, device=device, complex_dtype=complex_dtype)
            self._kernel_cache[key] = kernel
        return kernel

    def _frequency_kernel(
        self,
        shape: tuple[int, int],
        *,
        device: torch.device,
        complex_dtype: torch.dtype,
    ) -> torch.Tensor:
        real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32
        height, width = shape
        y = (torch.arange(height, device=device, dtype=real_dtype) - height // 2) * self.pixel_size
        x = (torch.arange(width, device=device, dtype=real_dtype) - width // 2) * self.pixel_size
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        distance = torch.as_tensor(self.distance, device=device, dtype=real_dtype)
        radius = torch.sqrt(grid_x.square() + grid_y.square() + distance.square())
        real_term = 1.0 / (2.0 * torch.pi * radius)
        bracket = torch.complex(real_term, torch.full_like(real_term, -1.0 / self.wavelength))
        phase = 2.0 * torch.pi * radius / self.wavelength
        phasor = torch.complex(torch.cos(phase), torch.sin(phase))
        spatial_kernel = distance / radius.square() * bracket * phasor
        spatial_kernel = spatial_kernel.to(dtype=complex_dtype) * self.pixel_size**2
        return torch.fft.fft2(torch.fft.ifftshift(spatial_kernel))


def gaussian_kernel_1d(
    sigma_pixels: float,
    *,
    truncate_sigma: float = 4.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return one axis of the normalized Gaussian used in equation (3)."""

    if sigma_pixels <= 0 or truncate_sigma <= 0:
        raise ValueError("sigma_pixels and truncate_sigma must be positive")
    radius = int(math.ceil(float(sigma_pixels) * float(truncate_sigma)))
    coordinate = torch.arange(-radius, radius + 1, dtype=dtype)
    kernel_1d = torch.exp(-0.5 * (coordinate / float(sigma_pixels)).square())
    return kernel_1d / kernel_1d.sum()


def gaussian_kernel_2d(
    sigma_pixels: float,
    *,
    truncate_sigma: float = 4.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the normalized Gaussian smoothing kernel used in equation (3)."""

    kernel_1d = gaussian_kernel_1d(
        sigma_pixels,
        truncate_sigma=truncate_sigma,
        dtype=dtype,
    )
    return torch.outer(kernel_1d, kernel_1d)


def make_correlated_diffuser_phase(
    field_shape: tuple[int, int],
    *,
    seed: int,
    wavelength: float,
    pixel_size: float,
    refractive_index_difference: float = 0.74,
    height_mean_lambda: float = 25.0,
    height_std_lambda: float = 8.0,
    gaussian_sigma_lambda: float = 4.0,
    truncate_sigma: float = 4.0,
    padding: str = "reflect",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a correlated thin phase diffuser from paper equations (2)-(4)."""

    height, width = _validate_field_shape(field_shape)
    if wavelength <= 0 or pixel_size <= 0:
        raise ValueError("wavelength and pixel_size must be positive")
    if refractive_index_difference <= 0 or height_std_lambda < 0:
        raise ValueError("diffuser parameters are invalid")
    sigma_pixels = float(gaussian_sigma_lambda) * float(wavelength) / float(pixel_size)
    kernel_1d = gaussian_kernel_1d(sigma_pixels, truncate_sigma=truncate_sigma, dtype=dtype)
    radius = kernel_1d.shape[-1] // 2
    if padding == "reflect" and (radius >= height or radius >= width):
        raise ValueError("reflect padding requires the Gaussian radius to be smaller than the field")
    if padding not in {"reflect", "constant", "circular"}:
        raise ValueError("padding must be 'reflect', 'constant', or 'circular'")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    mean_height = float(height_mean_lambda) * float(wavelength)
    std_height = float(height_std_lambda) * float(wavelength)
    white_height = torch.randn(field_shape, generator=generator, dtype=dtype) * std_height + mean_height
    field = white_height[None, None]
    padded_x = F.pad(field, (radius, radius, 0, 0), mode=padding)
    correlated_x = F.conv2d(padded_x, kernel_1d[None, None, None, :])
    padded_y = F.pad(correlated_x, (0, 0, radius, radius), mode=padding)
    correlated_height = F.conv2d(padded_y, kernel_1d[None, None, :, None])[0, 0]
    phase_scale = 2.0 * torch.pi * float(refractive_index_difference) / float(wavelength)
    return correlated_height * phase_scale


def represent_diffuser_phase(phase: torch.Tensor, *, mode: str) -> torch.Tensor:
    """Represent diffuser phase for the paper's mean-centered uniqueness test.

    The paper does not state whether equation (2)'s phase is compared before or
    after wrapping. R0 freeze ``2026-07-17.2`` selects ``minus_pi_to_pi`` as an
    explicitly inferred reproduction choice.
    """

    if torch.is_complex(phase):
        raise ValueError("diffuser phase must be real")
    if mode == "unwrapped":
        return phase
    if mode == "zero_to_2pi":
        return torch.remainder(phase, 2.0 * torch.pi)
    if mode == "minus_pi_to_pi":
        return torch.angle(torch.exp(1j * phase))
    raise ValueError(f"unsupported phase representation {mode}")


def diffuser_phase_difference(
    phase_a: torch.Tensor,
    phase_b: torch.Tensor,
    *,
    phase_representation: str = "unwrapped",
) -> torch.Tensor:
    """Return the paper's mean-centered absolute phase difference."""

    if phase_a.shape != phase_b.shape:
        raise ValueError("diffuser phases must have matching shapes")
    represented_a = represent_diffuser_phase(phase_a, mode=phase_representation)
    represented_b = represent_diffuser_phase(phase_b, mode=phase_representation)
    centered_a = represented_a - represented_a.mean()
    centered_b = represented_b - represented_b.mean()
    return (centered_a - centered_b).abs().mean()


def summarize_diffuser_bank_uniqueness(
    phases: torch.Tensor,
    *,
    phase_representation: str,
    threshold_radians: float,
    block_size: int = 32,
) -> dict[str, float | int | str]:
    """Exactly audit every unordered pair in a diffuser bank.

    Pair distances are evaluated in blocks with ``torch.cdist(p=1)`` so the
    full paper-scale bank can be checked without materializing an
    ``N x N x H x W`` tensor.
    """

    if phases.ndim != 3 or torch.is_complex(phases):
        raise ValueError("phases must have shape (count, height, width)")
    if phases.shape[0] < 2:
        raise ValueError("at least two diffuser phases are required")
    if threshold_radians < 0 or block_size <= 0:
        raise ValueError("uniqueness audit settings are invalid")

    represented = represent_diffuser_phase(phases, mode=phase_representation)
    represented = represented - represented.mean(dim=(-2, -1), keepdim=True)
    vectors = represented.flatten(start_dim=1).contiguous()
    pixel_count = int(vectors.shape[1])
    pair_count = 0
    pass_count = 0
    difference_sum = 0.0
    minimum = float("inf")
    maximum = float("-inf")

    for left_start in range(0, int(vectors.shape[0]), block_size):
        left = vectors[left_start : left_start + block_size]
        for right_start in range(left_start, int(vectors.shape[0]), block_size):
            right = vectors[right_start : right_start + block_size]
            differences = torch.cdist(left, right, p=1) / pixel_count
            if left_start == right_start:
                row, column = torch.triu_indices(
                    differences.shape[0],
                    differences.shape[1],
                    offset=1,
                    device=differences.device,
                )
                differences = differences[row, column]
            else:
                differences = differences.flatten()
            if differences.numel() == 0:
                continue
            pair_count += int(differences.numel())
            pass_count += int((differences > threshold_radians).sum())
            difference_sum += float(differences.sum(dtype=torch.float64))
            minimum = min(minimum, float(differences.min()))
            maximum = max(maximum, float(differences.max()))

    expected_pair_count = int(phases.shape[0]) * (int(phases.shape[0]) - 1) // 2
    if pair_count != expected_pair_count:
        raise RuntimeError("pairwise diffuser audit did not cover every unordered pair")
    return {
        "phase_representation": phase_representation,
        "pair_count": pair_count,
        "minimum_radians": minimum,
        "mean_radians": difference_sum / pair_count,
        "maximum_radians": maximum,
        "pass_count": pass_count,
        "pair_pass_fraction": pass_count / pair_count,
    }


def summarize_cross_diffuser_uniqueness(
    left_phases: torch.Tensor,
    right_phases: torch.Tensor,
    *,
    phase_representation: str,
    threshold_radians: float,
    block_size: int = 32,
) -> dict[str, float | int | str]:
    """Exactly summarize paper-style phase distances between two diffuser banks.

    Unlike :func:`summarize_diffuser_bank_uniqueness`, every cross-bank pair is
    included. This supports a post-hoc audit of whether held-out evaluation
    diffusers are distinct from all training diffusers under the frozen phase
    representation and threshold.
    """

    if (
        left_phases.ndim != 3
        or right_phases.ndim != 3
        or torch.is_complex(left_phases)
        or torch.is_complex(right_phases)
        or tuple(left_phases.shape[-2:]) != tuple(right_phases.shape[-2:])
    ):
        raise ValueError("diffuser banks must be real tensors with matching (count, height, width)")
    if left_phases.shape[0] == 0 or right_phases.shape[0] == 0:
        raise ValueError("diffuser banks must each contain at least one phase")
    if left_phases.device != right_phases.device:
        raise ValueError("cross-bank diffuser tensors must be on the same device")
    if left_phases.dtype != right_phases.dtype:
        raise ValueError("cross-bank diffuser tensors must use the same dtype")
    if threshold_radians < 0 or block_size <= 0:
        raise ValueError("uniqueness settings are invalid")

    def centered_vectors(phases: torch.Tensor) -> torch.Tensor:
        represented = represent_diffuser_phase(phases, mode=phase_representation)
        centered = represented - represented.mean(dim=(-2, -1), keepdim=True)
        return centered.flatten(start_dim=1).contiguous()

    left_vectors = centered_vectors(left_phases)
    right_vectors = centered_vectors(right_phases)
    pixel_count = int(left_vectors.shape[1])
    pair_count = 0
    pass_count = 0
    difference_sum = 0.0
    minimum = float("inf")
    maximum = float("-inf")

    for left_start in range(0, int(left_vectors.shape[0]), block_size):
        left = left_vectors[left_start : left_start + block_size]
        for right_start in range(0, int(right_vectors.shape[0]), block_size):
            right = right_vectors[right_start : right_start + block_size]
            differences = torch.cdist(left, right, p=1).flatten() / pixel_count
            pair_count += int(differences.numel())
            pass_count += int((differences > threshold_radians).sum())
            difference_sum += float(differences.sum(dtype=torch.float64))
            minimum = min(minimum, float(differences.min()))
            maximum = max(maximum, float(differences.max()))

    expected_pair_count = int(left_phases.shape[0]) * int(right_phases.shape[0])
    if pair_count != expected_pair_count:
        raise RuntimeError("cross-bank diffuser audit did not cover every pair")
    return {
        "phase_representation": phase_representation,
        "pair_count": pair_count,
        "minimum_radians": minimum,
        "mean_radians": difference_sum / pair_count,
        "maximum_radians": maximum,
        "pass_count": pass_count,
        "pair_pass_fraction": pass_count / pair_count,
    }


def estimate_phase_correlation_length(
    phase: torch.Tensor,
    *,
    pixel_size: float,
    wavelength: float,
    fit_range: tuple[float, float] = (0.2, 0.95),
) -> float:
    """Fit paper equation (5) to a radially averaged phase autocorrelation.

    The returned correlation length is expressed in wavelengths. This is an
    assessment helper because the paper does not publish its discrete
    autocorrelation estimator or fitting window.
    """

    if phase.ndim != 2 or torch.is_complex(phase):
        raise ValueError("phase must be a real tensor with shape (height, width)")
    if pixel_size <= 0 or wavelength <= 0:
        raise ValueError("pixel_size and wavelength must be positive")
    lower, upper = fit_range
    if not 0 < lower < upper < 1:
        raise ValueError("fit_range must satisfy 0 < lower < upper < 1")

    centered = phase.to(dtype=torch.float64) - phase.to(dtype=torch.float64).mean()
    return _estimate_correlation_length_from_centered_field(
        centered,
        pixel_size=pixel_size,
        wavelength=wavelength,
        fit_range=fit_range,
    )


def estimate_transmittance_correlation_length(
    phase: torch.Tensor,
    *,
    pixel_size: float,
    wavelength: float,
    fit_range: tuple[float, float] = (0.2, 0.95),
) -> float:
    """Fit equation (5) to the complex diffuser transmittance autocovariance."""

    if phase.ndim != 2 or torch.is_complex(phase):
        raise ValueError("phase must be a real tensor with shape (height, width)")
    transmittance = torch.exp(1j * phase.to(dtype=torch.float64)).to(dtype=torch.complex128)
    centered = transmittance - transmittance.mean()
    return _estimate_correlation_length_from_centered_field(
        centered,
        pixel_size=pixel_size,
        wavelength=wavelength,
        fit_range=fit_range,
    )


def _estimate_correlation_length_from_centered_field(
    centered: torch.Tensor,
    *,
    pixel_size: float,
    wavelength: float,
    fit_range: tuple[float, float],
) -> float:
    if pixel_size <= 0 or wavelength <= 0:
        raise ValueError("pixel_size and wavelength must be positive")
    lower, upper = fit_range
    if not 0 < lower < upper < 1:
        raise ValueError("fit_range must satisfy 0 < lower < upper < 1")

    height, width = centered.shape
    spectrum = torch.fft.fft2(centered)
    autocorrelation = torch.fft.fftshift(torch.fft.ifft2(spectrum.abs().square()).real)
    center_y, center_x = height // 2, width // 2
    peak = autocorrelation[center_y, center_x]
    if not torch.isfinite(peak) or float(peak) <= 0:
        raise ValueError("phase variance must be positive and finite")
    autocorrelation = autocorrelation / peak

    coordinate_y = torch.arange(height, dtype=torch.float64) - center_y
    coordinate_x = torch.arange(width, dtype=torch.float64) - center_x
    grid_y, grid_x = torch.meshgrid(coordinate_y, coordinate_x, indexing="ij")
    radius_bins = torch.floor(torch.sqrt(grid_x.square() + grid_y.square())).to(dtype=torch.long)
    max_radius = min(height, width) // 2
    radial_sum = torch.bincount(
        radius_bins.flatten(),
        weights=autocorrelation.flatten(),
        minlength=max_radius + 1,
    )[:max_radius]
    radial_count = torch.bincount(
        radius_bins.flatten(),
        minlength=max_radius + 1,
    )[:max_radius]
    radial_autocorrelation = radial_sum / radial_count
    radius_lambda = (
        torch.arange(max_radius, dtype=torch.float64) * float(pixel_size) / float(wavelength)
    )
    fit_mask = (
        (torch.arange(max_radius) > 0)
        & (radial_autocorrelation > lower)
        & (radial_autocorrelation < upper)
        & torch.isfinite(radial_autocorrelation)
    )
    if int(fit_mask.sum()) < 3:
        raise ValueError("not enough autocorrelation samples inside fit_range")

    radius_squared = radius_lambda[fit_mask].square()
    negative_log_correlation = -torch.log(radial_autocorrelation[fit_mask])
    slope = (radius_squared * negative_log_correlation).sum() / radius_squared.square().sum()
    if not torch.isfinite(slope) or float(slope) <= 0:
        raise ValueError("phase autocorrelation fit did not produce a positive slope")
    return float(torch.sqrt(torch.as_tensor(torch.pi, dtype=torch.float64) / slope))


def make_unique_correlated_diffusers(
    count: int,
    *,
    field_shape: tuple[int, int],
    base_seed: int,
    minimum_difference_radians: float = float(torch.pi / 2),
    max_attempts_per_diffuser: int = 1000,
    phase_representation: str = "unwrapped",
    existing_phases: torch.Tensor | None = None,
    comparison_chunk_size: int = 64,
    **diffuser_kwargs: float | str | torch.dtype,
) -> torch.Tensor:
    """Generate diffusers that pass comparison with every accepted diffuser.

    ``existing_phases`` lets epoch-specific generation compare against all
    diffusers accepted in earlier epochs, matching the paper's stated
    all-existing-diffusers rule.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    if (
        minimum_difference_radians < 0
        or max_attempts_per_diffuser <= 0
        or comparison_chunk_size <= 0
    ):
        raise ValueError("uniqueness settings are invalid")
    height, width = _validate_field_shape(field_shape)
    if existing_phases is None:
        existing_phases = torch.empty((0, height, width), dtype=torch.float32)
    if (
        existing_phases.ndim != 3
        or tuple(existing_phases.shape[-2:]) != (height, width)
        or torch.is_complex(existing_phases)
    ):
        raise ValueError("existing_phases must have shape (count, height, width)")
    if existing_phases.device.type != "cpu":
        raise ValueError("existing_phases must remain on CPU during diffuser generation")

    existing_represented = represent_diffuser_phase(
        existing_phases,
        mode=phase_representation,
    )
    existing_vectors = (
        existing_represented
        - existing_represented.mean(dim=(-2, -1), keepdim=True)
    ).flatten(start_dim=1)
    accepted_phases: torch.Tensor | None = None
    accepted_vectors: torch.Tensor | None = None
    accepted_count = 0
    candidate_seed = int(base_seed)
    for _ in range(count):
        for _attempt in range(max_attempts_per_diffuser):
            candidate = make_correlated_diffuser_phase(
                field_shape,
                seed=candidate_seed,
                **diffuser_kwargs,
            )
            candidate_seed += 1
            represented = represent_diffuser_phase(candidate, mode=phase_representation)
            candidate_vector = (represented - represented.mean()).flatten()
            if accepted_phases is None:
                accepted_phases = torch.empty(
                    (count, height, width),
                    dtype=candidate.dtype,
                )
                accepted_vectors = torch.empty(
                    (count, height * width),
                    dtype=candidate_vector.dtype,
                )
            comparison_banks = [existing_vectors]
            if accepted_count:
                assert accepted_vectors is not None
                comparison_banks.append(accepted_vectors[:accepted_count])
            passes = True
            for bank in comparison_banks:
                for start in range(0, int(bank.shape[0]), comparison_chunk_size):
                    differences = (
                        torch.cdist(
                            candidate_vector[None],
                            bank[start : start + comparison_chunk_size],
                            p=1,
                        )[0]
                        / candidate_vector.numel()
                    )
                    if bool(torch.any(differences <= minimum_difference_radians)):
                        passes = False
                        break
                if not passes:
                    break
            if passes:
                assert accepted_phases is not None and accepted_vectors is not None
                accepted_phases[accepted_count] = candidate
                accepted_vectors[accepted_count] = candidate_vector
                accepted_count += 1
                break
        else:
            raise RuntimeError("could not generate a diffuser satisfying the uniqueness threshold")
    assert accepted_phases is not None
    return accepted_phases


class Luo2022FourLayerD2NN(nn.Module):
    """Trainable phase-only D2NN implementing paper equations (6)-(10)."""

    def __init__(
        self,
        config: Luo2022OpticsConfig,
        *,
        phase_initialization: str = "zero",
        phase_seed: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        if phase_initialization == "zero":
            initial_phase = torch.zeros((config.num_layers, *config.field_shape))
        elif phase_initialization == "uniform_0_to_2pi":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(phase_seed))
            initial_phase = 2.0 * torch.pi * torch.rand(
                (config.num_layers, *config.field_shape),
                generator=generator,
            )
        else:
            raise ValueError("phase_initialization must be 'zero' or 'uniform_0_to_2pi'")
        self.phase = nn.Parameter(initial_phase)
        propagation_kwargs = {
            "field_shape": config.field_shape,
            "wavelength": config.wavelength,
            "pixel_size": config.pixel_size,
            "pad_factor": config.pad_factor,
        }
        self.object_to_diffuser = RayleighSommerfeldPropagator(
            **propagation_kwargs,
            distance=config.object_to_diffuser_distance,
        )
        self.diffuser_to_first_layer = RayleighSommerfeldPropagator(
            **propagation_kwargs,
            distance=config.diffuser_to_first_layer_distance,
        )
        self.between_layers = RayleighSommerfeldPropagator(
            **propagation_kwargs,
            distance=config.layer_distance,
        )
        self.last_layer_to_output = RayleighSommerfeldPropagator(
            **propagation_kwargs,
            distance=config.output_distance,
        )
        post_diffuser_to_output_distance = (
            config.diffuser_to_first_layer_distance
            + (config.num_layers - 1) * config.layer_distance
            + config.output_distance
        )
        self.post_diffuser_to_output_direct = RayleighSommerfeldPropagator(
            **propagation_kwargs,
            distance=post_diffuser_to_output_distance,
        )

    def distort(self, object_field: torch.Tensor, diffuser_phase: torch.Tensor) -> torch.Tensor:
        """Apply equation (6) and return fields immediately after the diffuser."""

        validate_complex_field(object_field, expected_shape=self.config.field_shape)
        if diffuser_phase.ndim != 3 or tuple(diffuser_phase.shape[-2:]) != self.config.field_shape:
            raise ValueError("diffuser_phase must have shape (diffusers, height, width)")
        incident = self.object_to_diffuser.propagate(object_field)
        phase = diffuser_phase.to(device=object_field.device, dtype=object_field.real.dtype)
        return incident[:, None] * torch.exp(1j * phase[None])

    def forward(self, object_field: torch.Tensor, diffuser_phase: torch.Tensor) -> torch.Tensor:
        """Return raw detector intensity with shape ``(B, n, H, W)``."""

        output, _trace = self._forward_fields(
            object_field,
            diffuser_phase,
            collect_trace=False,
        )
        return output

    def forward_without_diffractive_layers(
        self,
        object_field: torch.Tensor,
        diffuser_phase: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate directly from the diffuser to the detector without D2NN layers.

        This is a numerical control for the Luo et al. supplementary
        no-diffractive-layer condition. It retains the object-to-diffuser path,
        phase diffuser, detector sampling, and total post-diffuser distance,
        while replacing the finite-window sequence of layer planes with one
        direct propagation. It is intentionally distinct from a zero-phase
        four-layer model, which still has intermediate sampled propagations.
        """

        distorted = self.distort(object_field, diffuser_phase)
        batch_size, diffuser_count = distorted.shape[:2]
        output_field = self.post_diffuser_to_output_direct.propagate(
            distorted.flatten(0, 1)
        )
        return field_intensity(output_field).reshape(
            batch_size,
            diffuser_count,
            *self.config.field_shape,
        )

    def forward_with_trace(
        self,
        object_field: torch.Tensor,
        diffuser_phase: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return raw intensity and intermediate fields from the exact forward path.

        This is a diagnostic helper. It retains no extra model state and does
        not modify the frozen R0 configuration or parameters.
        """

        return self._forward_fields(
            object_field,
            diffuser_phase,
            collect_trace=True,
        )

    def _forward_fields(
        self,
        object_field: torch.Tensor,
        diffuser_phase: torch.Tensor,
        *,
        collect_trace: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        validate_complex_field(object_field, expected_shape=self.config.field_shape)
        if diffuser_phase.ndim != 3 or tuple(diffuser_phase.shape[-2:]) != self.config.field_shape:
            raise ValueError("diffuser_phase must have shape (diffusers, height, width)")

        trace: dict[str, torch.Tensor] = {}
        incident = self.object_to_diffuser.propagate(object_field)
        phase = diffuser_phase.to(device=object_field.device, dtype=object_field.real.dtype)
        distorted = incident[:, None] * torch.exp(1j * phase[None])
        if collect_trace:
            trace["object_field"] = object_field
            trace["before_diffuser"] = incident
            trace["after_diffuser"] = distorted

        batch_size, diffuser_count = distorted.shape[:2]
        field = distorted.flatten(0, 1)
        field = self.diffuser_to_first_layer.propagate(field)
        if collect_trace:
            trace["before_layer_1"] = field
        layer_phase = self.phase.to(device=field.device, dtype=field.real.dtype)
        for layer_index in range(self.config.num_layers):
            field = field * torch.exp(1j * layer_phase[layer_index])
            if collect_trace:
                trace[f"after_layer_{layer_index + 1}"] = field
            if layer_index + 1 < self.config.num_layers:
                field = self.between_layers.propagate(field)
                if collect_trace:
                    trace[f"before_layer_{layer_index + 2}"] = field
        output_field = self.last_layer_to_output.propagate(field)
        if collect_trace:
            trace["detector_field"] = output_field
        output = field_intensity(output_field).reshape(
            batch_size,
            diffuser_count,
            *self.config.field_shape,
        )
        return output, trace


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
