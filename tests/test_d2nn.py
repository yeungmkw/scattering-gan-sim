"""Core checks for coherent optical propagation and D2NN artifacts."""

from pathlib import Path

from PIL import Image
import torch

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
    gaussian_kernel_2d,
    image_to_complex_field,
    make_amplitude_particles,
    make_correlated_diffuser_phase,
    make_random_phase_screen,
    make_unique_correlated_diffusers,
    summarize_diffuser_bank_uniqueness,
)
from experiment import run_d2nn_inspection


def test_image_to_complex_field_preserves_intensity_contract() -> None:
    image = torch.rand(1, 16, 16)

    field = image_to_complex_field(image)
    intensity = field_intensity(field)

    assert field.shape == (1, 16, 16)
    assert torch.is_complex(field)
    assert torch.allclose(intensity[0], image[0], atol=1e-6)


def test_image_to_complex_field_shape_and_dtype_variants() -> None:
    cases = [
        (torch.rand(8, 8, dtype=torch.float32), (1, 8, 8), torch.complex64),
        (torch.rand(1, 8, 8, dtype=torch.float64), (1, 8, 8), torch.complex128),
        (torch.rand(3, 1, 8, 8, dtype=torch.float32), (3, 8, 8), torch.complex64),
    ]

    for image, expected_shape, expected_dtype in cases:
        field = image_to_complex_field(image)
        intensity = field_intensity(field)

        assert field.shape == expected_shape
        assert field.dtype == expected_dtype
        assert intensity.dtype == image.dtype


def test_luo2022_amplitude_encoding_does_not_take_square_root() -> None:
    image = torch.full((2, 1, 8, 8), 0.25)

    field = amplitude_to_complex_field(image)

    assert torch.allclose(field.real, torch.full((2, 8, 8), 0.25))
    assert torch.allclose(field_intensity(field), torch.full((2, 8, 8), 0.0625))


def test_random_phase_screen_is_deterministic_and_applies_phase() -> None:
    field = image_to_complex_field(torch.ones(1, 8, 8))
    phase_a = make_random_phase_screen((8, 8), seed=7)
    phase_b = make_random_phase_screen((8, 8), seed=7)

    dirty = apply_phase_screen(field, phase_a)

    assert torch.allclose(phase_a, phase_b)
    assert dirty.shape == field.shape
    assert torch.allclose(field_intensity(dirty), field_intensity(field), atol=1e-6)
    assert field_phase(dirty).shape == (1, 8, 8)


def test_phase_and_particles_preserve_complex_dtype() -> None:
    field = image_to_complex_field(torch.rand(1, 8, 8, dtype=torch.float64))
    phase = make_random_phase_screen((8, 8), seed=2, dtype=torch.float64)
    particles = make_amplitude_particles((8, 8), seed=2, dtype=torch.float64)

    phase_dirty = apply_phase_screen(field, phase)
    particle_dirty = apply_amplitude_particles(field, particles)

    assert phase_dirty.dtype == torch.complex128
    assert particle_dirty.dtype == torch.complex128
    assert field_intensity(phase_dirty).dtype == torch.float64
    assert field_intensity(particle_dirty).dtype == torch.float64


def test_amplitude_particles_are_deterministic_and_attenuate() -> None:
    field = image_to_complex_field(torch.ones(1, 16, 16))
    mask_a = make_amplitude_particles((16, 16), seed=3, num_particles=3, radius_range=(2, 3))
    mask_b = make_amplitude_particles((16, 16), seed=3, num_particles=3, radius_range=(2, 3))

    dirty = apply_amplitude_particles(field, mask_a)

    assert torch.allclose(mask_a, mask_b)
    assert float(mask_a.min()) < 1.0
    assert float(mask_a.max()) <= 1.0
    assert field_intensity(dirty).mean() < field_intensity(field).mean()


def test_single_layer_d2nn_outputs_intensity() -> None:
    image = torch.rand(1, 16, 16)
    field = image_to_complex_field(image)
    config = CoherentOpticsConfig(field_shape=(16, 16), pad_factor=1)
    model = SingleLayerD2NN(config, seed=4)

    output = model(field)
    intensity = field_intensity(output)

    assert output.shape == field.shape
    assert intensity.shape == (1, 16, 16)
    assert torch.all(intensity >= 0)


def test_single_layer_d2nn_handles_batched_complex128_field_with_padding() -> None:
    real = torch.rand(2, 10, 12, dtype=torch.float64)
    imag = torch.rand(2, 10, 12, dtype=torch.float64) * 0.1
    field = torch.complex(real, imag)
    config = CoherentOpticsConfig(field_shape=(10, 12), pad_factor=2)
    model = SingleLayerD2NN(config, seed=5)

    output = model(field)
    intensity = field_intensity(output)

    assert output.shape == field.shape
    assert output.dtype == torch.complex128
    assert intensity.shape == (2, 10, 12)
    assert intensity.dtype == torch.float64
    assert torch.isfinite(intensity).all()


def test_angular_spectrum_propagator_preserves_float64_shape_and_dtype() -> None:
    real = torch.rand(9, 7, dtype=torch.float64)
    imag = torch.rand(9, 7, dtype=torch.float64) * 0.05
    field = torch.complex(real, imag)
    config = CoherentOpticsConfig(field_shape=(9, 7), pad_factor=2)
    propagator = AngularSpectrumPropagator(config)

    output = propagator.propagate(field)

    assert output.shape == field.shape
    assert output.dtype == torch.complex128
    assert field_intensity(output).dtype == torch.float64


def test_angular_spectrum_propagator_reuses_transfer_function_cache() -> None:
    field = image_to_complex_field(torch.rand(1, 10, 10))
    config = CoherentOpticsConfig(field_shape=(10, 10), pad_factor=2)
    propagator = AngularSpectrumPropagator(config)

    output_a = propagator.propagate(field)
    cache_size = len(propagator._transfer_cache)
    output_b = propagator.propagate(field)

    assert cache_size == 1
    assert len(propagator._transfer_cache) == cache_size
    assert torch.allclose(output_a, output_b)


def test_single_layer_d2nn_is_differentiable_for_input_and_phase() -> None:
    image = (torch.rand(1, 1, 12, 12, dtype=torch.float32) * 0.8 + 0.1).detach().requires_grad_()
    field = image_to_complex_field(image)
    config = CoherentOpticsConfig(field_shape=(12, 12), pad_factor=1)
    model = SingleLayerD2NN(config, seed=6, trainable=True)

    loss = field_intensity(model(field)).mean()
    loss.backward()

    assert image.grad is not None
    assert model.phase.grad is not None
    assert torch.isfinite(image.grad).all()
    assert torch.isfinite(model.phase.grad).all()
    assert float(image.grad.abs().sum()) > 0.0
    assert float(model.phase.grad.abs().sum()) > 0.0


def test_rayleigh_sommerfeld_fft_matches_direct_discrete_sum() -> None:
    height, width = 5, 6
    wavelength = 0.75e-3
    pixel_size = 0.3e-3
    distance = 2e-3
    real = torch.randn(height, width, dtype=torch.float64)
    imag = torch.randn(height, width, dtype=torch.float64)
    field = torch.complex(real, imag)
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
                * (1 / (2 * torch.pi * radius) + 1 / (1j * wavelength))
                * torch.exp(1j * 2 * torch.pi * radius / wavelength)
                * pixel_size**2
            )
            direct_output[output_y, output_x] = (field * kernel).sum()

    assert torch.allclose(fft_output, direct_output, atol=1e-12, rtol=1e-12)


def test_correlated_diffusers_are_deterministic_and_unique() -> None:
    kwargs = {
        "field_shape": (48, 48),
        "wavelength": 0.75e-3,
        "pixel_size": 0.3e-3,
    }
    phase_a = make_correlated_diffuser_phase(seed=17, **kwargs)
    phase_b = make_correlated_diffuser_phase(seed=17, **kwargs)
    bank = make_unique_correlated_diffusers(
        2,
        base_seed=21,
        phase_representation="minus_pi_to_pi",
        **kwargs,
    )

    assert torch.allclose(phase_a, phase_b)
    assert bank.shape == (2, 48, 48)
    assert float(
        diffuser_phase_difference(
            bank[0],
            bank[1],
            phase_representation="minus_pi_to_pi",
        )
    ) > float(torch.pi / 2)


def test_unique_diffuser_generation_checks_all_existing_phases() -> None:
    kwargs = {
        "field_shape": (48, 48),
        "wavelength": 0.75e-3,
        "pixel_size": 0.3e-3,
        "phase_representation": "minus_pi_to_pi",
    }
    existing = make_unique_correlated_diffusers(3, base_seed=101, **kwargs)
    new = make_unique_correlated_diffusers(
        2,
        base_seed=201,
        existing_phases=existing,
        **kwargs,
    )
    combined = torch.cat((existing, new))
    summary = summarize_diffuser_bank_uniqueness(
        combined,
        phase_representation="minus_pi_to_pi",
        threshold_radians=float(torch.pi / 2),
        block_size=2,
    )

    assert summary["pair_count"] == 10
    assert summary["pair_pass_fraction"] == 1.0
    assert summary["minimum_radians"] > float(torch.pi / 2)


def test_separable_correlated_diffuser_matches_2d_gaussian_reference() -> None:
    field_shape = (48, 48)
    wavelength = 0.75e-3
    pixel_size = 0.3e-3
    seed = 19
    sigma_pixels = 4.0 * wavelength / pixel_size
    kernel = gaussian_kernel_2d(sigma_pixels, dtype=torch.float64)
    radius = kernel.shape[-1] // 2
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    white_height = torch.randn(field_shape, generator=generator, dtype=torch.float64) * (
        8.0 * wavelength
    )
    white_height = white_height + 25.0 * wavelength
    padded = torch.nn.functional.pad(
        white_height[None, None],
        (radius, radius, radius, radius),
        mode="reflect",
    )
    reference_height = torch.nn.functional.conv2d(padded, kernel[None, None])[0, 0]
    reference_phase = reference_height * (2.0 * torch.pi * 0.74 / wavelength)

    phase = make_correlated_diffuser_phase(
        field_shape,
        seed=seed,
        wavelength=wavelength,
        pixel_size=pixel_size,
        dtype=torch.float64,
    )

    assert torch.allclose(phase, reference_phase, atol=1e-12, rtol=1e-12)


def test_phase_correlation_length_estimator_returns_finite_scale() -> None:
    phase = make_correlated_diffuser_phase(
        (240, 240),
        seed=23,
        wavelength=0.75e-3,
        pixel_size=0.3e-3,
    )

    correlation_length = estimate_phase_correlation_length(
        phase,
        pixel_size=0.3e-3,
        wavelength=0.75e-3,
    )

    assert 8.0 < correlation_length < 20.0

    transmittance_correlation_length = estimate_transmittance_correlation_length(
        phase,
        pixel_size=0.3e-3,
        wavelength=0.75e-3,
    )

    assert 8.0 < transmittance_correlation_length < 14.0


def test_luo2022_four_layer_path_updates_all_phase_layers() -> None:
    config = Luo2022OpticsConfig(field_shape=(48, 48))
    model = Luo2022FourLayerD2NN(config)
    field = amplitude_to_complex_field(torch.rand(2, 1, 48, 48))
    diffusers = make_unique_correlated_diffusers(
        2,
        field_shape=config.field_shape,
        base_seed=31,
        wavelength=config.wavelength,
        pixel_size=config.pixel_size,
    )

    output = model(field, diffusers)
    output.mean().backward()

    assert output.shape == (2, 2, 48, 48)
    assert torch.isfinite(output).all()
    assert model.phase.shape == (4, 48, 48)
    assert model.phase.grad is not None
    assert torch.all(model.phase.grad.flatten(start_dim=1).abs().sum(dim=1) > 0)


def test_d2nn_inspection_saves_required_artifacts_for_phase_and_particles(tmp_path: Path) -> None:
    image = torch.rand(1, 16, 16)

    for corruption in ("phase", "particles"):
        output_dir = tmp_path / corruption
        manifest = run_d2nn_inspection(
            image,
            output_dir=output_dir,
            label=5,
            image_index=0,
            seed=11,
            corruption=corruption,
        )

        assert manifest["corruption"] == corruption
        expected_class = "E0-inspection" if corruption == "phase" else "E3-inspection"
        assert manifest["experiment_class"] == expected_class
        assert manifest["schema_version"] == 1
        assert manifest["artifacts"] == {
            "input": "input.png",
            "dirty_intensity": "dirty_intensity.png",
            "dirty_phase": "dirty_phase.png",
            "output_intensity": "output_intensity.png",
        }
        assert manifest["forward_model"]["optics"]["field_shape"] == [16, 16]
        assert manifest["runtime"]["dependencies"]["torch"] is not None
        for filename in manifest["artifacts"].values():
            path = output_dir / filename
            assert path.is_file()
            assert path.stat().st_size > 0
            with Image.open(path) as artifact:
                assert artifact.size == (16, 16)
                assert artifact.convert("L").getbbox() is not None
        assert (output_dir / "manifest.json").is_file()


def test_phase_dirty_intensity_is_propagated_observation(tmp_path: Path) -> None:
    image = torch.zeros(1, 16, 16)
    image[:, 4:12, 5:11] = 1.0

    run_d2nn_inspection(
        image,
        output_dir=tmp_path,
        label=1,
        image_index=0,
        seed=13,
        corruption="phase",
    )

    with Image.open(tmp_path / "input.png") as input_image:
        input_pixels = input_image.convert("L").tobytes()
    with Image.open(tmp_path / "dirty_intensity.png") as dirty_image:
        dirty_pixels = dirty_image.convert("L").tobytes()

    assert dirty_pixels != input_pixels
