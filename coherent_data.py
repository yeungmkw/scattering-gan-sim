"""Coherent optical paired samples for DNN reconstruction runs.

This file bridges the single-image D2NN inspection path to a minimal paired dataset:
``clean`` image targets are encoded as complex fields, corrupted by a phase
screen or amplitude particles, propagated to a dirty observation, passed
through a fixed single-layer D2NN, and exposed as tensors that a reconstructor
can consume. It is intentionally small and deterministic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset

from d2nn import (
    AngularSpectrumPropagator,
    CoherentOpticsConfig,
    SingleLayerD2NN,
    apply_amplitude_particles,
    apply_phase_screen,
    amplitude_to_complex_field,
    field_intensity,
    field_phase,
    image_to_complex_field,
    make_amplitude_particles,
    make_random_phase_screen,
)
from data import build_torchvision_dataset


class CoherentD2NNDataset(Dataset[dict[str, torch.Tensor]]):
    """Wrap a grayscale image dataset and emit coherent optical observations."""

    def __init__(
        self,
        base_dataset: Dataset,
        *,
        corruption: str = "phase",
        seed: int = 42,
        diffuser_ids: tuple[int, ...] | list[int] = (0,),
        d2nn_seed: int | None = None,
        optics_config: CoherentOpticsConfig | None = None,
    ) -> None:
        if corruption not in {"phase", "particles"}:
            raise ValueError("corruption must be 'phase' or 'particles'")
        if not diffuser_ids:
            raise ValueError("diffuser_ids must not be empty")
        self.base_dataset = base_dataset
        self.corruption = corruption
        self.seed = int(seed)
        self.diffuser_ids = tuple(int(diffuser_id) for diffuser_id in diffuser_ids)
        self.d2nn_seed = int(seed + 7919 if d2nn_seed is None else d2nn_seed)
        self.optics_config = optics_config
        self._simulators: dict[CoherentOpticsConfig, CoherentObservationSimulator] = {}

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        base_item = self.base_dataset[index]
        image, label = _unpack_image_label(base_item)
        clean = _as_single_channel_float(image)
        diffuser_id = self._diffuser_id_for_index(index)
        simulator = self._simulator_for_clean(clean)
        sample = simulator.simulate(
            clean,
            corruption=self.corruption,
            seed=self.seed + int(diffuser_id) * 1009,
            d2nn_seed=self.d2nn_seed,
        )
        sample["diffuser_id"] = torch.tensor(diffuser_id, dtype=torch.long)
        sample["label"] = torch.tensor(int(label), dtype=torch.long)
        return sample

    def _diffuser_id_for_index(self, index: int) -> int:
        offset = (int(index) * 1103515245 + self.seed) % len(self.diffuser_ids)
        return self.diffuser_ids[offset]

    def _simulator_for_clean(self, clean: torch.Tensor) -> "CoherentObservationSimulator":
        config = self.optics_config or CoherentOpticsConfig(field_shape=tuple(clean.shape[-2:]))
        if tuple(clean.shape[-2:]) != tuple(config.field_shape):
            raise ValueError("clean image shape must match optics_config.field_shape")
        simulator = self._simulators.get(config)
        if simulator is None:
            simulator = CoherentObservationSimulator(config)
            self._simulators[config] = simulator
        return simulator


class MaterializedCoherentDataset(Dataset[dict[str, torch.Tensor]]):
    """In-memory tensor copy of coherent samples for repeated GPU training."""

    def __init__(self, tensors: dict[str, torch.Tensor], *, copy: bool = True) -> None:
        if not tensors:
            raise ValueError("tensors must not be empty")
        lengths = {int(value.shape[0]) for value in tensors.values()}
        if len(lengths) != 1:
            raise ValueError("all tensors must have the same leading length")
        self.tensors = {
            name: value.detach().clone() if copy else value.detach()
            for name, value in tensors.items()
        }
        self.length = lengths.pop()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {name: value[index] for name, value in self.tensors.items()}


def materialize_coherent_dataset(dataset: Dataset[dict[str, torch.Tensor]]) -> MaterializedCoherentDataset:
    """Precompute coherent observations once so epochs do not recompute optics."""

    if len(dataset) == 0:
        raise ValueError("dataset must contain at least one sample")
    first = dataset[0]
    keys = tuple(first.keys())
    length = len(dataset)
    tensors = {
        key: torch.empty((length, *value.shape), dtype=value.dtype, device=value.device)
        for key, value in first.items()
    }
    for key, value in first.items():
        tensors[key][0].copy_(value)

    for index in range(1, length):
        sample = dataset[index]
        if set(sample.keys()) != set(keys):
            raise ValueError("dataset samples must expose stable keys")
        for key in keys:
            tensors[key][index].copy_(sample[key])
    return MaterializedCoherentDataset(tensors, copy=False)


class CoherentObservationSimulator:
    """Reusable coherent forward model for repeated samples with fixed optics."""

    def __init__(self, optics_config: CoherentOpticsConfig) -> None:
        self.config = optics_config
        self.propagator = AngularSpectrumPropagator(optics_config)
        self._d2nn_layers: dict[int, SingleLayerD2NN] = {}
        self._phase_screens: dict[int, torch.Tensor] = {}
        self._particle_masks: dict[int, torch.Tensor] = {}

    def simulate(
        self,
        clean: torch.Tensor,
        *,
        corruption: str,
        seed: int,
        d2nn_seed: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return clean target plus dirty/D2NN intensity observations."""

        if corruption not in {"phase", "particles"}:
            raise ValueError("corruption must be 'phase' or 'particles'")
        clean = _as_single_channel_float(clean)
        field = image_to_complex_field(clean)
        if tuple(field.shape[-2:]) != tuple(self.config.field_shape):
            raise ValueError("clean image shape must match optics_config.field_shape")

        corruption_seed = int(seed) + 1
        if corruption == "phase":
            scattered_field = apply_phase_screen(field, self._phase_screen(corruption_seed))
        else:
            scattered_field = apply_amplitude_particles(field, self._particle_mask(corruption_seed))

        dirty_field = self.propagator.propagate(scattered_field)
        d2nn = self._d2nn_layer(int(seed) + 2 if d2nn_seed is None else int(d2nn_seed))
        output_field = d2nn(dirty_field)
        dirty_intensity = _normalize_image(field_intensity(dirty_field).unsqueeze(1))[0]
        d2nn_intensity = _normalize_image(field_intensity(output_field).unsqueeze(1))[0]
        dirty_phase = _phase_to_unit_range(field_phase(dirty_field))[0].unsqueeze(0)
        return {
            "clean": clean.detach(),
            "dirty_intensity": dirty_intensity.detach(),
            "dirty_phase": dirty_phase.detach(),
            "d2nn_intensity": d2nn_intensity.detach(),
        }

    def _phase_screen(self, seed: int) -> torch.Tensor:
        phase_screen = self._phase_screens.get(seed)
        if phase_screen is None:
            phase_screen = make_random_phase_screen(self.config.field_shape, seed=seed)
            self._phase_screens[seed] = phase_screen
        return phase_screen

    def _particle_mask(self, seed: int) -> torch.Tensor:
        particle_mask = self._particle_masks.get(seed)
        if particle_mask is None:
            particle_mask = make_amplitude_particles(self.config.field_shape, seed=seed)
            self._particle_masks[seed] = particle_mask
        return particle_mask

    def _d2nn_layer(self, seed: int) -> SingleLayerD2NN:
        d2nn_layer = self._d2nn_layers.get(seed)
        if d2nn_layer is None:
            d2nn_layer = SingleLayerD2NN(self.config, seed=seed, trainable=False)
            self._d2nn_layers[seed] = d2nn_layer
        return d2nn_layer


def simulate_coherent_observation(
    clean: torch.Tensor,
    *,
    corruption: str,
    seed: int,
    d2nn_seed: int | None = None,
    optics_config: CoherentOpticsConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Return clean target plus dirty/D2NN intensity observations."""

    if corruption not in {"phase", "particles"}:
        raise ValueError("corruption must be 'phase' or 'particles'")
    clean = _as_single_channel_float(clean)
    config = optics_config or CoherentOpticsConfig(field_shape=tuple(clean.shape[-2:]))
    if tuple(clean.shape[-2:]) != tuple(config.field_shape):
        raise ValueError("clean image shape must match optics_config.field_shape")
    return CoherentObservationSimulator(config).simulate(
        clean,
        corruption=corruption,
        seed=seed,
        d2nn_seed=d2nn_seed,
    )


def prepare_luo2022_amplitude(
    image: torch.Tensor,
    *,
    resized_shape: tuple[int, int],
    canvas_shape: tuple[int, int],
) -> torch.Tensor:
    """Resize and center-pad MNIST as the amplitude input in paper equation (6)."""

    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError("image must have shape (B, 1, H, W) or (1, H, W)")
    resized_height, resized_width = resized_shape
    canvas_height, canvas_width = canvas_shape
    if min(resized_height, resized_width, canvas_height, canvas_width) <= 0:
        raise ValueError("resized_shape and canvas_shape values must be positive")
    if resized_height > canvas_height or resized_width > canvas_width:
        raise ValueError("resized_shape must fit inside canvas_shape")
    resized = F.interpolate(
        image.to(dtype=torch.float32),
        size=resized_shape,
        mode="bilinear",
        align_corners=False,
    )
    top = (canvas_height - resized_height) // 2
    left = (canvas_width - resized_width) // 2
    canvas = resized.new_zeros((resized.shape[0], 1, canvas_height, canvas_width))
    canvas[..., top : top + resized_height, left : left + resized_width] = resized
    return canvas


def prepare_luo2022_field(
    image: torch.Tensor,
    *,
    resized_shape: tuple[int, int],
    canvas_shape: tuple[int, int],
) -> torch.Tensor:
    """Return the zero-phase amplitude field used by the Luo 2022 R0 path."""

    amplitude = prepare_luo2022_amplitude(
        image,
        resized_shape=resized_shape,
        canvas_shape=canvas_shape,
    )
    return amplitude_to_complex_field(amplitude)


def build_coherent_mnist_datasets(
    *,
    root: str | Path = "data",
    image_size: int = 64,
    corruption: str = "phase",
    seed: int = 42,
    download: bool = False,
    limit_train: int = 8,
    limit_eval: int = 4,
    train_diffuser_ids: tuple[int, ...] | list[int] = (0,),
    eval_diffuser_ids: tuple[int, ...] | list[int] = (0,),
) -> tuple[CoherentD2NNDataset, CoherentD2NNDataset]:
    """Build tiny deterministic train/eval datasets for coherent runs."""

    train_base = build_torchvision_dataset(
        name="MNIST",
        root=root,
        train=True,
        image_size=image_size,
        download=download,
    )
    eval_base = build_torchvision_dataset(
        name="MNIST",
        root=root,
        train=False,
        image_size=image_size,
        download=download,
    )
    train_base = Subset(train_base, range(min(limit_train, len(train_base))))
    eval_base = Subset(eval_base, range(min(limit_eval, len(eval_base))))
    return (
        CoherentD2NNDataset(train_base, corruption=corruption, seed=seed, diffuser_ids=train_diffuser_ids),
        CoherentD2NNDataset(
            eval_base,
            corruption=corruption,
            seed=seed,
            diffuser_ids=eval_diffuser_ids,
            d2nn_seed=seed + 7919,
        ),
    )


def _normalize_image(image: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    low = image.amin(dim=(-2, -1), keepdim=True)
    high = image.amax(dim=(-2, -1), keepdim=True)
    return (image - low) / (high - low).clamp_min(eps)


def _phase_to_unit_range(phase: torch.Tensor) -> torch.Tensor:
    return ((phase + torch.pi) / (2 * torch.pi)).clamp(0.0, 1.0)


def _unpack_image_label(base_item: Any) -> tuple[Any, int]:
    if isinstance(base_item, tuple) and len(base_item) >= 2:
        return base_item[0], int(base_item[1])
    if isinstance(base_item, dict):
        return base_item["image"], int(base_item.get("label", 0))
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
