import torch
from torch.utils.data import TensorDataset

from data import PairedScatteringDataset
from optics import RandomPSFChannel, make_random_psfs


def test_paired_scattering_dataset_returns_reconstruction_contract() -> None:
    images = torch.linspace(0, 1, steps=4 * 16 * 16).reshape(4, 1, 16, 16)
    labels = torch.arange(4)
    base = TensorDataset(images, labels)
    channel = RandomPSFChannel(
        make_random_psfs(num_diffusers=3, kernel_size=5, seed=7),
        gaussian_noise_std=0.01,
        speckle_contrast=0.1,
    )
    dataset = PairedScatteringDataset(base, channel=channel, diffuser_ids=(0, 2), seed=9)

    sample = dataset[1]
    repeat = dataset[1]

    assert set(sample) == {"clean", "corrupted", "diffuser_id", "label"}
    assert sample["clean"].shape == (1, 16, 16)
    assert sample["corrupted"].shape == (1, 16, 16)
    assert int(sample["diffuser_id"]) in {0, 2}
    assert torch.allclose(sample["corrupted"], repeat["corrupted"])
