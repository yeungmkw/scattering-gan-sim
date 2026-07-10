"""Core checks for coherent optical propagation and D2NN artifacts."""

from pathlib import Path

from PIL import Image
import torch

from d2nn import (
    AngularSpectrumPropagator,
    CoherentOpticsConfig,
    SingleLayerD2NN,
    apply_amplitude_particles,
    apply_phase_screen,
    field_intensity,
    field_phase,
    image_to_complex_field,
    make_amplitude_particles,
    make_random_phase_screen,
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
