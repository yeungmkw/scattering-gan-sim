"""Paired clean/corrupted datasets for scattering reconstruction.

The dataset layer fixes the experiment contract for E0/E1: a clean object image
is passed through a deterministic diffuser split and a random-PSF corruption
channel, yielding ``(clean, corrupted, diffuser_id)`` samples for supervised
reconstruction and seen/unseen diffuser evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Subset

from optics import RandomPSFChannel, diffuser_split


@dataclass(frozen=True)
class PairedDatasetBundle:
    """Train, seen-eval, and unseen-eval paired datasets."""

    train: Dataset
    seen_eval: Dataset
    unseen_eval: Dataset | None
    train_diffuser_ids: tuple[int, ...]
    test_diffuser_ids: tuple[int, ...]


class PairedScatteringDataset(Dataset[dict[str, torch.Tensor]]):
    """Wrap an image dataset and generate scattering-corrupted observations."""

    def __init__(
        self,
        base_dataset: Dataset,
        *,
        channel: RandomPSFChannel,
        diffuser_ids: tuple[int, ...] | list[int],
        seed: int = 0,
    ) -> None:
        if not diffuser_ids:
            raise ValueError("diffuser_ids must not be empty")
        self.base_dataset = base_dataset
        self.channel = channel
        self.diffuser_ids = tuple(int(diffuser_id) for diffuser_id in diffuser_ids)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        base_item = self.base_dataset[index]
        image, label = _unpack_image_label(base_item)
        clean = _as_single_channel_float(image)
        diffuser_id = self._diffuser_id_for_index(index)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + 1009 * int(index) + 9176 * int(diffuser_id))
        corrupted = self.channel(
            clean.unsqueeze(0),
            torch.tensor([diffuser_id], dtype=torch.long),
            generator=generator,
        )[0]
        return {
            "clean": clean.detach(),
            "corrupted": corrupted.detach(),
            "diffuser_id": torch.tensor(diffuser_id, dtype=torch.long),
            "label": torch.tensor(int(label), dtype=torch.long),
        }

    def _diffuser_id_for_index(self, index: int) -> int:
        offset = (int(index) * 1103515245 + self.seed) % len(self.diffuser_ids)
        return self.diffuser_ids[offset]


def build_torchvision_dataset(
    *,
    name: str,
    root: str | Path,
    train: bool,
    image_size: int,
    download: bool,
) -> Dataset:
    """Build a torchvision MNIST-family dataset with grayscale tensor output."""

    if image_size <= 0:
        raise ValueError("image_size must be positive")
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    normalized_name = name.strip().lower().replace("-", "_")
    if normalized_name == "mnist":
        return datasets.MNIST(root=str(root), train=train, transform=transform, download=download)
    if normalized_name in {"fashion_mnist", "fashionmnist"}:
        return datasets.FashionMNIST(root=str(root), train=train, transform=transform, download=download)
    raise ValueError("dataset must be 'MNIST' or 'Fashion-MNIST'")


def build_paired_datasets(
    *,
    dataset_name: str,
    root: str | Path,
    image_size: int,
    channel: RandomPSFChannel,
    num_train_diffusers: int,
    num_test_diffusers: int,
    seed: int,
    download: bool = False,
    limit_train: int | None = None,
    limit_eval: int | None = None,
) -> PairedDatasetBundle:
    """Build train, seen-eval, and unseen-eval paired datasets."""

    split = diffuser_split(num_train_diffusers, num_test_diffusers)
    train_base = build_torchvision_dataset(
        name=dataset_name,
        root=root,
        train=True,
        image_size=image_size,
        download=download,
    )
    eval_base = build_torchvision_dataset(
        name=dataset_name,
        root=root,
        train=False,
        image_size=image_size,
        download=download,
    )
    if limit_train is not None:
        train_base = _limited_subset(train_base, limit_train)
    if limit_eval is not None:
        eval_base = _limited_subset(eval_base, limit_eval)

    return PairedDatasetBundle(
        train=PairedScatteringDataset(
            train_base,
            channel=channel,
            diffuser_ids=split.train_ids,
            seed=seed,
        ),
        seen_eval=PairedScatteringDataset(
            eval_base,
            channel=channel,
            diffuser_ids=split.train_ids,
            seed=seed + 1,
        ),
        unseen_eval=(
            PairedScatteringDataset(
                eval_base,
                channel=channel,
                diffuser_ids=split.test_ids,
                seed=seed + 2,
            )
            if split.test_ids
            else None
        ),
        train_diffuser_ids=split.train_ids,
        test_diffuser_ids=split.test_ids,
    )


def _limited_subset(dataset: Dataset, limit: int) -> Subset:
    if limit <= 0:
        raise ValueError("dataset limits must be positive")
    return Subset(dataset, range(min(limit, len(dataset))))


def _unpack_image_label(base_item: Any) -> tuple[Any, int]:
    if isinstance(base_item, dict):
        return base_item["image"], int(base_item.get("label", 0))
    if isinstance(base_item, tuple) and len(base_item) >= 2:
        return base_item[0], int(base_item[1])
    return base_item, 0


def _as_single_channel_float(image: Any) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        from torchvision.transforms.functional import to_tensor

        image = to_tensor(image)
    image = image.to(dtype=torch.float32)
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.ndim != 3:
        raise ValueError("image must have shape (channels, height, width)")
    if image.shape[0] != 1:
        image = image.mean(dim=0, keepdim=True)
    return image.clamp(0.0, 1.0)
