import torch
from torch.nn import functional as F

from optics import (
    RandomPSFChannel,
    forward_sanity_metrics,
    make_random_psfs,
)


def test_random_psfs_are_normalized() -> None:
    psfs = make_random_psfs(num_diffusers=3, kernel_size=7, seed=11)

    assert psfs.shape == (3, 1, 7, 7)
    assert torch.all(psfs >= 0)
    assert torch.allclose(psfs.sum(dim=(-2, -1)), torch.ones(3, 1), atol=1e-6)


def test_random_psf_channel_is_shape_preserving_and_deterministic() -> None:
    psfs = make_random_psfs(num_diffusers=2, kernel_size=5, seed=3)
    channel = RandomPSFChannel(psfs, gaussian_noise_std=0.01, speckle_contrast=0.1)
    clean = torch.rand(4, 1, 16, 16)
    diffuser_ids = torch.tensor([0, 1, 0, 1])
    generator_a = torch.Generator(device="cpu").manual_seed(5)
    generator_b = torch.Generator(device="cpu").manual_seed(5)

    corrupted_a = channel(clean, diffuser_ids, generator=generator_a)
    corrupted_b = channel(clean, diffuser_ids, generator=generator_b)

    assert corrupted_a.shape == clean.shape
    assert torch.allclose(corrupted_a, corrupted_b)
    assert float(corrupted_a.min()) >= 0.0
    assert float(corrupted_a.max()) <= 1.0


def test_random_psf_channel_matches_per_sample_convolution_without_noise() -> None:
    psfs = make_random_psfs(num_diffusers=3, kernel_size=5, seed=13)
    channel = RandomPSFChannel(psfs, gaussian_noise_std=0.0, speckle_contrast=0.0, normalize_output=False)
    clean = torch.rand(3, 1, 16, 16)
    diffuser_ids = torch.tensor([0, 2, 1])
    padding = channel.kernel_size // 2

    actual = channel(clean, diffuser_ids, add_noise=False)
    expected = torch.cat(
        [
            F.conv2d(clean[index : index + 1], psfs[int(diffuser_id)].unsqueeze(0), padding=padding)
            for index, diffuser_id in enumerate(diffuser_ids)
        ],
        dim=0,
    )

    assert torch.allclose(actual, expected, atol=1e-6)


def test_forward_sanity_metrics_are_scalar_floats() -> None:
    clean = torch.rand(2, 1, 8, 8)
    corrupted = clean * 0.5

    metrics = forward_sanity_metrics(clean, corrupted)

    assert "corrupted_contrast" in metrics
    assert all(isinstance(value, float) for value in metrics.values())
