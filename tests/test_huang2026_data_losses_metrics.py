"""Huang 2026 visible-light data, loss, and metric contracts."""

from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import TensorDataset

from coherent_data import (
    HUANG2026_BLIND_TEST_SPLIT,
    HUANG2026_TRAIN_SPLIT,
    Huang2026CoherenceSampler,
    Huang2026OnlineDiffuserSampler,
    Huang2026VisibleDataset,
    prepare_huang2026_visible_amplitude,
)
from losses import (
    huang2026_incoherent_mse,
    huang2026_intensity_mse,
    huang2026_multiwavelength_mse,
)
from metrics import (
    huang2026_dataset_statistics,
    huang2026_grouped_statistics,
    huang2026_pcc_per_image,
    huang2026_pcc_statistics,
)


def test_huang2026_preprocess_is_exact_28_to_320_to_400_amplitude() -> None:
    image = torch.ones(2, 1, 28, 28)

    amplitude = prepare_huang2026_visible_amplitude(image)

    assert amplitude.shape == (2, 1, 400, 400)
    assert amplitude.dtype == torch.float32
    assert torch.equal(amplitude[:, :, 40:360, 40:360], torch.ones(2, 1, 320, 320))
    assert torch.count_nonzero(amplitude[:, :, :40]) == 0
    assert torch.count_nonzero(amplitude[:, :, 360:]) == 0
    assert torch.count_nonzero(amplitude[:, :, :, :40]) == 0
    assert torch.count_nonzero(amplitude[:, :, :, 360:]) == 0


def test_huang2026_preprocess_small_shape_preserves_dtype_and_gradient() -> None:
    image = torch.arange(16, dtype=torch.float64).reshape(1, 1, 4, 4)
    image.requires_grad_()

    amplitude = prepare_huang2026_visible_amplitude(
        image,
        input_shape=(4, 4),
        resized_shape=(7, 8),
        canvas_shape=(10, 12),
    )
    amplitude.square().sum().backward()

    assert amplitude.shape == (1, 1, 10, 12)
    assert amplitude.dtype == torch.float64
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert float(image.grad.abs().sum()) > 0.0
    assert torch.count_nonzero(amplitude[..., :1, :]) == 0
    assert torch.count_nonzero(amplitude[..., -2:, :]) == 0
    with pytest.raises(ValueError, match="does not match input_shape"):
        prepare_huang2026_visible_amplitude(
            torch.ones(1, 1, 5, 4),
            input_shape=(4, 4),
            resized_shape=(7, 8),
            canvas_shape=(10, 12),
        )


def test_huang2026_diffuser_seed_schedule_repeats_updates_and_isolates_splits() -> None:
    train = Huang2026OnlineDiffuserSampler(
        split="train",
        base_seed=17,
        correlation_length_pixels=5.5,
        wavelengths=(491e-9, 532e-9, 660e-9),
    )
    blind = Huang2026OnlineDiffuserSampler(
        split="blind-test",
        base_seed=17,
        correlation_length_pixels=5.5,
        wavelengths=(491e-9, 532e-9, 660e-9),
    )

    train_seeds = {
        train.seed_for(iteration=iteration, object_id=object_id)
        for iteration in range(3)
        for object_id in range(4)
    }
    blind_seeds = {
        blind.seed_for(iteration=iteration, object_id=object_id)
        for iteration in range(3)
        for object_id in range(4)
    }

    assert train.split == HUANG2026_TRAIN_SPLIT
    assert blind.split == HUANG2026_BLIND_TEST_SPLIT
    assert train_seeds.isdisjoint(blind_seeds)
    assert all(seed < 2**62 for seed in train_seeds)
    assert all(2**62 <= seed < 2**63 for seed in blind_seeds)
    assert train.seed_for(iteration=2, object_id=3) == train.seed_for(
        iteration=2,
        object_id=3,
    )
    assert train.seed_for(iteration=2, object_id=3) != train.seed_for(
        iteration=3,
        object_id=3,
    )
    metadata = train.metadata(iteration=2, object_id=3)
    assert metadata["correlation_length_pixels"] == 5.5
    assert metadata["wavelengths_m"] == [491e-9, 532e-9, 660e-9]


def test_huang2026_online_diffuser_factory_tensor_keeps_gradient_chain() -> None:
    scale = torch.tensor(2.0, requires_grad=True)

    def factory(**_kwargs) -> torch.Tensor:
        return scale * torch.ones(5, 6)

    sampler = Huang2026OnlineDiffuserSampler(
        split="train",
        base_seed=3,
        correlation_length_pixels=4.0,
        diffuser_factory=factory,
    )

    record = sampler.sample(iteration=1, object_id=8)
    height = record["diffuser_height_m"]
    height.sum().backward()

    assert height.shape == (5, 6)
    assert height.requires_grad
    assert scale.grad == pytest.approx(torch.tensor(30.0))


def test_huang2026_coherence_sampler_matches_seed_power_and_coherence_length() -> None:
    sampler = Huang2026CoherenceSampler(
        (32, 32),
        split="train",
        base_seed=23,
        coherence_length_pixels=4.0,
    )

    screen_a = sampler.sample(iteration=2, object_id=7, realization=3)
    screen_b = sampler.sample(iteration=2, object_id=7, realization=3)
    screen_c = sampler.sample(iteration=3, object_id=7, realization=3)
    ensemble = sampler.sample_ensemble(
        iteration=5,
        object_id=7,
        num_realizations=128,
    )
    mean_power = ensemble.abs().square().mean()
    lag_four_correlation = (
        ensemble * torch.roll(ensemble.conj(), shifts=-4, dims=-1)
    ).real.mean() / mean_power

    assert screen_a.dtype == torch.complex64
    assert torch.equal(screen_a, screen_b)
    assert not torch.equal(screen_a, screen_c)
    assert ensemble.shape == (128, 32, 32)
    assert float(mean_power) == pytest.approx(1.0, abs=0.08)
    assert float(lag_four_correlation) == pytest.approx(math.exp(-1.0), abs=0.09)


def test_huang2026_dataset_records_online_identity_and_wavelength_metadata() -> None:
    images = torch.stack(
        (
            torch.linspace(0.0, 1.0, 16).reshape(1, 4, 4),
            torch.ones(1, 4, 4) * 0.5,
        )
    ).requires_grad_()
    labels = torch.tensor([2, 5])
    dataset = Huang2026VisibleDataset(
        TensorDataset(images, labels),
        split="train",
        base_seed=11,
        correlation_length_pixels=3.0,
        input_shape=(4, 4),
        resized_shape=(8, 8),
        canvas_shape=(12, 12),
        illumination_mode="multi-wavelength",
        wavelengths=(491e-9, 532e-9, 660e-9),
        object_ids=(100, 101),
    )

    first = dataset.sample_at(0, iteration=0)
    repeated = dataset[(0, 0)]
    next_iteration = dataset.sample_at(0, iteration=1)

    assert first["amplitude"].shape == (1, 12, 12)
    assert torch.equal(first["target_intensity"], first["amplitude"].square())
    assert int(first["object_id"]) == 100
    assert int(first["iteration"]) == 0
    assert int(first["diffuser_seed"]) == int(repeated["diffuser_seed"])
    assert int(first["diffuser_seed"]) != int(next_iteration["diffuser_seed"])
    assert first["illumination_mode"] == "multiwavelength"
    assert first["wavelength"].tolist() == pytest.approx([491e-9, 532e-9, 660e-9])
    assert first["metadata"] == {
        "schema_version": "huang2026-visible-sample-v1",
        "split": "train",
        "object_id": 100,
        "iteration": 0,
        "diffuser_seed": int(first["diffuser_seed"]),
        "correlation_length_pixels": 3.0,
        "diffuser_correlation_length_pixels": 3.0,
        "illumination_mode": "multiwavelength",
        "wavelength": [491e-9, 532e-9, 660e-9],
        "wavelengths_m": [491e-9, 532e-9, 660e-9],
    }
    first["target_intensity"].sum().backward()
    assert images.grad is not None
    assert float(images.grad[0].abs().sum()) > 0.0
    assert float(images.grad[1].abs().sum()) == 0.0


def test_huang2026_incoherent_dataset_adds_replayable_coherence_metadata() -> None:
    dataset = Huang2026VisibleDataset(
        TensorDataset(torch.ones(1, 1, 4, 4), torch.tensor([1])),
        split="blind_test",
        correlation_length_pixels=2.5,
        input_shape=(4, 4),
        resized_shape=(6, 6),
        canvas_shape=(8, 8),
        illumination_mode="incoherent",
    )

    sample = dataset.sample_at(0, iteration=4)

    assert sample["metadata"]["split"] == "blind_test"
    assert sample["metadata"]["diffuser_correlation_length_pixels"] == 2.5
    assert sample["metadata"]["coherence_length_pixels"] == 4.0
    assert sample["diffuser_correlation_length_pixels"] == pytest.approx(
        torch.tensor(2.5)
    )
    assert int(sample["coherence_seed"]) >= 2**62
    assert sample["coherence_length_pixels"] == pytest.approx(torch.tensor(4.0))


def test_huang2026_coherent_intensity_mse_matches_manual_value_and_gradient() -> None:
    output = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    target = torch.tensor([[[[0.0, 2.0], [1.0, 4.0]]]])

    loss, components = huang2026_intensity_mse(
        output,
        target,
        return_components=True,
    )
    expected = ((output - target[:, 0]).square()).mean()
    loss.backward()

    assert loss == expected
    assert components == {"intensity_mse": loss, "total": loss}
    assert output.grad is not None
    assert torch.isfinite(output.grad).all()


def test_huang2026_pcc_accepts_dataset_singleton_channel_target() -> None:
    prediction = torch.rand(3, 8, 8)
    target = prediction[:, None].clone()

    values = huang2026_pcc_per_image(prediction, target)

    assert values.shape == (3,)
    assert torch.allclose(values, torch.ones_like(values), atol=1e-6)


def test_huang2026_incoherent_mse_averages_intensity_before_single_mse() -> None:
    ensemble = torch.tensor(
        [
            [
                [[1.0, 3.0], [2.0, 4.0]],
                [[3.0, 1.0], [6.0, 2.0]],
            ]
        ],
        requires_grad=True,
    )
    target = torch.tensor([[[2.0, 1.0], [3.0, 4.0]]])
    averaged = ensemble.mean(dim=1)
    expected = (averaged - target).square().mean()

    ensemble_loss = huang2026_incoherent_mse(ensemble, target)
    averaged_loss = huang2026_incoherent_mse(
        averaged,
        target,
        input_is_ensemble=False,
    )
    ensemble_loss.backward()

    assert ensemble_loss == expected
    assert averaged_loss == expected
    assert ensemble.grad is not None
    assert torch.isfinite(ensemble.grad).all()


def test_huang2026_multiwavelength_mse_sums_not_averages_terms() -> None:
    outputs = {
        491e-9: torch.tensor([[[1.0, 2.0]]], requires_grad=True),
        532e-9: torch.tensor([[[3.0, 1.0]]], requires_grad=True),
        660e-9: torch.tensor([[[0.0, 4.0]]], requires_grad=True),
    }
    target = torch.tensor([[[1.0, 1.0]]])

    loss, components = huang2026_multiwavelength_mse(
        outputs,
        target,
        return_components=True,
    )
    expected_terms = [
        (output - target).square().mean() for output in outputs.values()
    ]
    expected = torch.stack(expected_terms).sum()
    loss.backward()

    assert loss == expected
    assert components["total"] == expected
    assert len(components) == 4
    assert all(output.grad is not None for output in outputs.values())

    stacked = torch.stack(tuple(outputs.values()), dim=1).detach()
    assert huang2026_multiwavelength_mse(stacked, target) == expected.detach()


def test_huang2026_pcc_and_dataset_statistics_are_per_image() -> None:
    target = torch.tensor(
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[0.0, 1.0], [2.0, 3.0]],
        ]
    )
    prediction = torch.stack((target[0], -target[1]))

    values = huang2026_pcc_per_image(prediction, target)
    summary = huang2026_pcc_statistics(prediction, target)
    direct_summary = huang2026_dataset_statistics(values)

    assert values.tolist() == pytest.approx([1.0, -1.0])
    assert summary == direct_summary
    assert summary["count"] == 2
    assert summary["mean"] == pytest.approx(0.0)
    assert summary["sample_std"] == pytest.approx(math.sqrt(2.0))
    assert summary["standard_error"] == pytest.approx(1.0)
    assert summary["ci95_normal"] == pytest.approx([-1.96, 1.96])
    assert summary["minimum"] == pytest.approx(-1.0)
    assert summary["maximum"] == pytest.approx(1.0)


def test_huang2026_grouped_statistics_cover_all_required_conditions() -> None:
    values = torch.tensor([1.0, 3.0, 5.0, 7.0])

    grouped = huang2026_grouped_statistics(
        values,
        diffuser_seeds=[10, 10, 20, 20],
        correlation_lengths=[4.0, 4.0, 8.0, 8.0],
        wavelengths=[491e-9, 532e-9, 491e-9, 532e-9],
        illumination_modes=["coherent", "coherent", "incoherent", "incoherent"],
        misalignments=[(0, 0, 0.0), (0, 0, 0.0), (5, 0, 0.0), (5, 0, 0.0)],
    )

    assert grouped["dataset"]["mean"] == 4.0
    assert [row["statistics"]["mean"] for row in grouped["per_diffuser"]] == [
        2.0,
        6.0,
    ]
    assert [
        row["statistics"]["mean"] for row in grouped["per_correlation_length"]
    ] == [2.0, 6.0]
    assert [row["statistics"]["mean"] for row in grouped["per_wavelength"]] == [
        3.0,
        5.0,
    ]
    assert [
        row["illumination_mode"] for row in grouped["per_illumination_mode"]
    ] == ["coherent", "incoherent"]
    assert [row["misalignment"] for row in grouped["per_misalignment"]] == [
        (0, 0, 0.0),
        (5, 0, 0.0),
    ]
    with pytest.raises(ValueError, match="same length"):
        huang2026_grouped_statistics(values, wavelengths=[491e-9])
