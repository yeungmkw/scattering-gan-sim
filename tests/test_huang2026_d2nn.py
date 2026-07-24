"""Physics and differentiation checks for the Huang et al. 2026 profile."""

from __future__ import annotations

import math

import pytest
import torch

from d2nn import (
    AngularSpectrumPropagator,
    CoherentOpticsConfig,
    CorrelatedHeightPhaseDiffuser,
    DetectorResponse,
    Huang2026DiffuserConfig,
    Huang2026IncoherentDONN,
    Huang2026MultiWavelengthDONN,
    Huang2026ThreeLayerDONN,
    Huang2026VisibleOpticsConfig,
    MisalignmentTransform,
    SLMPhaseResponse,
    ThinLensOperator,
    VisibleDirectPropagationOperator,
)


def _field(shape: tuple[int, int], *, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    generator = torch.Generator().manual_seed(91)
    real = torch.rand((1, *shape), generator=generator, dtype=real_dtype)
    imaginary = 0.1 * torch.rand(
        (1, *shape),
        generator=generator,
        dtype=real_dtype,
    )
    return torch.complex(real, imaginary).to(dtype=dtype)


def _config(
    shape: tuple[int, int] = (12, 10),
    *,
    dtype: torch.dtype = torch.complex64,
) -> Huang2026VisibleOpticsConfig:
    del dtype
    return Huang2026VisibleOpticsConfig(
        field_shape=shape,
        wavelength=660e-9,
        pixel_size=8e-6,
        object_to_diffuser_distance=0.8e-3,
        diffuser_to_first_layer_distance=0.9e-3,
        layer_distances=(1.0e-3, 1.1e-3),
        last_layer_to_detector_distance=1.2e-3,
    )


def test_huang_geometry_defaults_and_explicit_typo_profile() -> None:
    nominal = Huang2026VisibleOpticsConfig(field_shape=(8, 8))
    typo = Huang2026VisibleOpticsConfig(
        field_shape=(8, 8),
        geometry_profile="supplement_typo_sensitivity",
    )

    assert nominal.segment_distances == pytest.approx(
        (0.0295, 0.0295, 0.0295, 0.0295, 0.0712)
    )
    assert nominal.total_optical_path == pytest.approx(0.1892)
    assert 4 * nominal.lens_focal_length == pytest.approx(nominal.total_optical_path)
    assert typo.segment_distances == pytest.approx(
        (0.00295, 0.00295, 0.00295, 0.00295, 0.0071)
    )
    assert typo.total_optical_path == pytest.approx(0.0189)
    assert 4 * typo.lens_focal_length != pytest.approx(typo.total_optical_path)


def test_diffuser_prefilter_statistics_correlation_control_and_phase_mapping() -> None:
    shape = (128, 128)
    base = Huang2026DiffuserConfig(
        field_shape=shape,
        pixel_size=8e-6,
        height_mean=63e-6,
        height_std=14e-6,
        correlation_length=4 * 8e-6,
    )
    diffuser = CorrelatedHeightPhaseDiffuser(base)
    seeds = list(range(8))
    white = diffuser.sample_unsmoothed_height(seeds, dtype=torch.float64)
    repeat = diffuser.sample_unsmoothed_height(seeds, dtype=torch.float64)

    assert torch.equal(white, repeat)
    assert float(white.mean()) == pytest.approx(63e-6, abs=0.2e-6)
    assert float(white.std(unbiased=True)) == pytest.approx(14e-6, rel=0.02)

    short = diffuser.sample_height(seeds, dtype=torch.float64)
    long = diffuser.sample_height(
        seeds,
        correlation_length=12 * 8e-6,
        dtype=torch.float64,
    )

    def lag_correlation(values: torch.Tensor, lag: int) -> float:
        centered = values - values.mean(dim=(-2, -1), keepdim=True)
        numerator = (
            centered[..., :, :-lag] * centered[..., :, lag:]
        ).mean()
        return float(numerator / centered.square().mean())

    assert float(short.mean()) == pytest.approx(63e-6, abs=0.3e-6)
    assert lag_correlation(short, 4) < 0.30
    assert lag_correlation(long, 4) > 0.50
    height = torch.tensor([[1.25e-6]], dtype=torch.float64)
    phase = diffuser.phase_from_height(height, 660e-9)
    expected = 2 * math.pi * 0.52 * height / 660e-9
    assert torch.allclose(phase, expected, atol=0.0, rtol=1e-14)


@pytest.mark.parametrize("complex_dtype", [torch.complex64, torch.complex128])
def test_huang_model_batch_dtype_intensity_and_all_layer_gradients(
    complex_dtype: torch.dtype,
) -> None:
    config = _config(dtype=complex_dtype)
    model = Huang2026ThreeLayerDONN(
        config,
        phase_initialization="uniform_0_to_2pi",
        phase_seed=7,
    )
    field = _field(config.field_shape, dtype=complex_dtype).expand(2, -1, -1)
    height = torch.zeros(
        (2, *config.field_shape),
        dtype=field.real.dtype,
    )
    output = model(field, height)
    weight = torch.linspace(
        0.1,
        1.0,
        math.prod(config.field_shape),
        dtype=output.dtype,
    ).reshape(config.field_shape)
    loss = (output * weight).mean()
    loss.backward()

    assert output.shape == (2, *config.field_shape)
    assert output.dtype == field.real.dtype
    assert torch.isfinite(output).all()
    assert torch.all(output >= 0)
    assert model.phase.grad is not None
    per_layer = model.phase.grad.flatten(start_dim=1).norm(dim=1)
    assert torch.isfinite(per_layer).all()
    assert torch.all(per_layer > 0)


def test_three_layer_path_uses_all_five_segments_in_order() -> None:
    config = _config((4, 4))
    model = Huang2026ThreeLayerDONN(config, phase_initialization="zero")
    calls: list[int] = []

    class Marker:
        def __init__(self, index: int) -> None:
            self.index = index

        def propagate(self, value: torch.Tensor) -> torch.Tensor:
            calls.append(self.index)
            return value + complex(self.index + 1, 0)

    model.path._nominal_propagators = tuple(Marker(index) for index in range(5))
    field = torch.ones((1, 4, 4), dtype=torch.complex64)
    output_field = model.forward_field(field, torch.zeros((1, 4, 4)))

    assert calls == [0, 1, 2, 3, 4]
    assert torch.allclose(output_field, torch.full_like(output_field, 16 + 0j))


def test_angular_spectrum_matches_manual_small_grid_reference() -> None:
    config = CoherentOpticsConfig(
        field_shape=(5, 6),
        wavelength=0.75e-3,
        pixel_size=0.3e-3,
        propagation_distance=2e-3,
        pad_factor=1,
    )
    field = _field(config.field_shape, dtype=torch.complex128)[0]
    actual = AngularSpectrumPropagator(config).propagate(field)

    fy = torch.fft.fftfreq(5, d=config.pixel_size, dtype=torch.float64)
    fx = torch.fft.fftfreq(6, d=config.pixel_size, dtype=torch.float64)
    grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
    wave_number = 2 * math.pi / config.wavelength
    kz = torch.sqrt(
        (
            wave_number**2
            - (2 * math.pi * grid_x).square()
            - (2 * math.pi * grid_y).square()
        ).to(torch.complex128)
    )
    expected = torch.fft.ifft2(
        torch.fft.fft2(field)
        * torch.exp(1j * config.propagation_distance * kz)
    )
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_incoherent_streaming_chunk_matches_full_values_and_gradients() -> None:
    config = _config((8, 8))
    full_model = Huang2026IncoherentDONN(
        Huang2026ThreeLayerDONN(
            config,
            phase_initialization="uniform_0_to_2pi",
            phase_seed=12,
        )
    )
    streamed_model = Huang2026IncoherentDONN(
        Huang2026ThreeLayerDONN(config, phase_initialization="zero")
    )
    streamed_model.load_state_dict(full_model.state_dict())
    field = _field(config.field_shape)
    height = torch.zeros((1, *config.field_shape))
    generator = torch.Generator().manual_seed(13)
    screens = torch.complex(
        torch.randn((1, 5, *config.field_shape), generator=generator),
        torch.randn((1, 5, *config.field_shape), generator=generator),
    ) / math.sqrt(2)
    weight = torch.linspace(0.2, 1.0, 64).reshape(8, 8)

    full = full_model(field, height, screens, chunk_size=None)
    (full * weight).mean().backward()
    streamed = streamed_model.forward_from_screen_generator(
        field,
        height,
        num_realizations=5,
        chunk_size=2,
        screen_generator=lambda start, count: screens[:, start : start + count],
        checkpoint_chunks=False,
    )
    (streamed * weight).mean().backward()

    assert torch.allclose(streamed, full, atol=2e-6, rtol=2e-5)
    assert full_model.coherent_model.phase.grad is not None
    assert streamed_model.coherent_model.phase.grad is not None
    assert torch.allclose(
        streamed_model.coherent_model.phase.grad,
        full_model.coherent_model.phase.grad,
        atol=3e-6,
        rtol=3e-5,
    )


def test_multiwavelength_channels_share_trainable_phase_and_are_nonnegative() -> None:
    config = _config((8, 8))
    model = Huang2026MultiWavelengthDONN(
        config,
        wavelengths=(491e-9, 532e-9, 660e-9),
        phase_initialization="uniform_0_to_2pi",
        phase_seed=21,
    )
    output = model(_field(config.field_shape), torch.zeros((1, 8, 8)))
    target = torch.rand((1, 3, 8, 8))
    torch.nn.functional.mse_loss(output, target).backward()

    assert output.shape == (1, 3, 8, 8)
    assert torch.isfinite(output).all()
    assert torch.all(output >= 0)
    assert model.phase.grad is not None
    assert torch.all(model.phase.grad.flatten(start_dim=1).norm(dim=1) > 0)


def test_s18_inverse_sinc_boundaries_recovery_and_monotonicity() -> None:
    amplitude = torch.linspace(0.0, 1.0, 257, dtype=torch.float64)
    inverse = SLMPhaseResponse.inverse_sinc(amplitude)
    modulation = 1.0 + inverse / math.pi
    recovered = SLMPhaseResponse.first_order_amplitude(modulation)
    hologram = SLMPhaseResponse.phase_only_hologram(
        amplitude,
        return_complex=True,
    )

    assert float(inverse[0]) == pytest.approx(-math.pi)
    assert float(inverse[-1]) == pytest.approx(0.0)
    assert torch.allclose(recovered, amplitude, atol=1e-12, rtol=1e-12)
    assert torch.all(torch.diff(recovered) >= -1e-12)
    assert torch.allclose(hologram.abs(), torch.ones_like(amplitude))


def test_slm_full_cycle_quantization_has_requested_distinct_phase_states() -> None:
    response = SLMPhaseResponse(phase_quantization_levels=2)
    commands = torch.tensor([0.1, 2.0])
    quantized = response(commands, 660e-9)
    transmissions = torch.exp(1j * quantized)

    assert torch.unique(quantized).numel() == 2
    assert not torch.allclose(transmissions[0], transmissions[1])
    with pytest.raises(ValueError, match=r"cover normalized drive \[0,1\]"):
        SLMPhaseResponse(
            lut={660e-9: ((0.0, 255.0), (0.0, 2 * math.pi))}
        )


def test_zero_misalignment_is_exact_identity_and_nonzero_shift_is_distinct() -> None:
    config = _config((8, 8))
    model = Huang2026ThreeLayerDONN(
        config,
        phase_initialization="uniform_0_to_2pi",
        phase_seed=31,
    )
    field = _field(config.field_shape)
    height = torch.zeros((1, 8, 8))
    ideal = model(field, height)
    identity = model(field, height, misalignment=MisalignmentTransform())
    shifted = model(
        field,
        height,
        misalignment=MisalignmentTransform(
            layer_shifts=((0, 0), (1, 0), (0, 0))
        ),
    )

    assert torch.equal(identity, ideal)
    assert not torch.allclose(shifted, ideal)


def test_direct_lens_and_donn_are_distinct_operator_routes() -> None:
    config = Huang2026VisibleOpticsConfig(field_shape=(8, 8))
    direct = VisibleDirectPropagationOperator(config)
    lens = ThinLensOperator(config)
    donn = Huang2026ThreeLayerDONN(config, phase_initialization="zero")
    field = _field(config.field_shape)
    height = torch.zeros((1, 8, 8))

    direct_output = direct(field, height)
    lens_output = lens(field, height)
    donn_output = donn(field, height)

    assert type(direct) is VisibleDirectPropagationOperator
    assert type(lens) is ThinLensOperator
    assert type(donn) is Huang2026ThreeLayerDONN
    assert not torch.allclose(direct_output, lens_output)
    assert not torch.allclose(lens_output, donn_output)


def test_detector_ideal_regression_and_configured_nonidealities() -> None:
    intensity = torch.linspace(0.0, 2.0, 64).reshape(1, 8, 8)
    assert torch.equal(DetectorResponse()(intensity), intensity)
    response = DetectorResponse(
        read_noise_std=0.01,
        gain=2.0,
        saturation=1.0,
        transmission=0.8,
        seed=5,
    )
    output = response(intensity)
    assert torch.isfinite(output).all()
    assert torch.all(output >= 0)
    assert torch.all(output <= 1.0)

    first = DetectorResponse(read_noise_std=0.01, seed=17)
    first_output = first(torch.ones((1, 8, 8)))
    state = {
        name: value.detach().clone()
        for name, value in first.state_dict().items()
    }
    second_output = first(torch.ones((1, 8, 8)))
    resumed = DetectorResponse(read_noise_std=0.01, seed=17)
    resumed.load_state_dict(state)
    assert not torch.equal(first_output, second_output)
    assert torch.equal(resumed(torch.ones((1, 8, 8))), second_output)
