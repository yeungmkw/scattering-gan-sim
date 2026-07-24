"""Coherent optical paired samples for DNN reconstruction runs.

This file bridges the single-image D2NN inspection path to a minimal paired dataset:
``clean`` image targets are encoded as complex fields, corrupted by a phase
screen or amplitude particles, propagated to a dirty observation, passed
through a fixed single-layer D2NN, and exposed as tensors that a reconstructor
can consume. It is intentionally small and deterministic.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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


LUO2022_ASSIGNMENT_SCHEMA = "luo2022-fixed-depth-assignment-v1"
LUO2022_INTENSITY_CACHE_SCHEMA = "luo2022-intensity-cache-v1"
HUANG2026_VISIBLE_SAMPLE_SCHEMA = "huang2026-visible-sample-v1"
HUANG2026_TRAIN_SPLIT = "train"
HUANG2026_BLIND_TEST_SPLIT = "blind_test"
_ASSIGNMENT_FIELDS = (
    "object_id",
    "label",
    "diffuser_id",
    "training_epoch",
    "within_epoch_index",
    "row_id",
)


def canonical_sha256(value: Any) -> str:
    """Return SHA256 of a stable, whitespace-free canonical JSON encoding.

    Cache and assignment fingerprints use this helper instead of Python's
    process-randomized ``hash`` or a platform-dependent tensor serialization.
    The accepted value must therefore be JSON-compatible and contain no NaN or
    infinity.
    """

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def build_luo2022_fixed_depth_assignment(
    labels: Sequence[int] | torch.Tensor,
    *,
    num_diffusers: int,
    diffusers_per_epoch: int = 20,
    seed: int = 42,
    first_training_epoch: int = 1,
    object_ids: Sequence[int] | torch.Tensor | None = None,
    object_id_offset: int = 0,
) -> dict[str, Any]:
    """Assign each object once to a balanced Luo 2022 training diffuser.

    Objects are ordered by a domain-separated SHA256 rank and then distributed
    round-robin over the diffuser IDs.  This gives every diffuser either
    ``floor(N / D)`` or ``ceil(N / D)`` objects, remains stable across Python
    processes, and deliberately does not reuse the legacy LCG index rule.
    Rows are returned in diffuser-major order so cache generation can process
    all objects assigned to one diffuser together. ``object_id`` preserves a
    source-dataset identity (including validation IDs 50000--59999), while
    ``row_id`` is the compact split-local cache position after that ordering.
    """

    diffuser_count = int(num_diffusers)
    epoch_width = int(diffusers_per_epoch)
    start_epoch = int(first_training_epoch)
    if diffuser_count <= 0:
        raise ValueError("num_diffusers must be positive")
    if epoch_width <= 0:
        raise ValueError("diffusers_per_epoch must be positive")
    if start_epoch <= 0:
        raise ValueError("first_training_epoch must be positive")

    if isinstance(labels, torch.Tensor):
        if labels.ndim != 1:
            raise ValueError("labels tensor must be one-dimensional")
        object_labels = [int(value) for value in labels.detach().cpu().tolist()]
    else:
        object_labels = [int(value) for value in labels]
    if not object_labels:
        raise ValueError("labels must contain at least one object")

    id_offset = int(object_id_offset)
    if object_ids is not None and id_offset != 0:
        raise ValueError("object_ids and a nonzero object_id_offset are mutually exclusive")
    if object_ids is None:
        if id_offset < 0:
            raise ValueError("object_id_offset must be non-negative")
        source_object_ids = list(range(id_offset, id_offset + len(object_labels)))
        canonical_offset: int | None = id_offset
    elif isinstance(object_ids, torch.Tensor):
        if object_ids.ndim != 1:
            raise ValueError("object_ids tensor must be one-dimensional")
        source_object_ids = [int(value) for value in object_ids.detach().cpu().tolist()]
        canonical_offset = _contiguous_id_offset(source_object_ids)
    else:
        source_object_ids = [int(value) for value in object_ids]
        canonical_offset = _contiguous_id_offset(source_object_ids)
    if len(source_object_ids) != len(object_labels):
        raise ValueError("object_ids and labels must have the same length")
    if any(object_id < 0 for object_id in source_object_ids):
        raise ValueError("object_ids must be non-negative")
    if len(set(source_object_ids)) != len(source_object_ids):
        raise ValueError("object_ids must be unique")

    seed_value = int(seed)
    ranked_row_ids = sorted(
        range(len(object_labels)),
        key=lambda row_id: (
            hashlib.sha256(
                (
                    "luo2022-fixed-depth-assignment-v1\0"
                    f"{seed_value}\0{source_object_ids[row_id]}"
                ).encode("utf-8")
            ).digest(),
            source_object_ids[row_id],
        ),
    )
    object_to_diffuser = [0] * len(object_labels)
    for rank, row_id in enumerate(ranked_row_ids):
        object_to_diffuser[row_id] = rank % diffuser_count

    assigned_rows = []
    for source_row_id, (object_id, label) in enumerate(
        zip(source_object_ids, object_labels, strict=True)
    ):
        diffuser_id = object_to_diffuser[source_row_id]
        assigned_rows.append(
            {
                "object_id": object_id,
                "label": label,
                "diffuser_id": diffuser_id,
                "training_epoch": start_epoch + diffuser_id // epoch_width,
                "within_epoch_index": diffuser_id % epoch_width,
            }
        )
    assigned_rows.sort(key=lambda row: (row["diffuser_id"], row["object_id"]))
    rows = [
        {**row, "row_id": row_id}
        for row_id, row in enumerate(assigned_rows)
    ]
    metadata = {
        "schema_version": LUO2022_ASSIGNMENT_SCHEMA,
        "assignment_method": "sha256_rank_round_robin_diffuser_major_v1",
        "object_count": len(rows),
        "diffuser_count": diffuser_count,
        "diffusers_per_epoch": epoch_width,
        "training_epoch_count": math.ceil(diffuser_count / epoch_width),
        "first_training_epoch": start_epoch,
        "seed": seed_value,
        "object_id_offset": canonical_offset,
        "object_id_min": min(source_object_ids),
        "object_id_max": max(source_object_ids),
        "object_ids_sha": canonical_sha256(source_object_ids),
    }
    root_sha = canonical_sha256({"metadata": metadata, "rows": rows})
    return {"metadata": metadata, "rows": rows, "root_sha": root_sha}


class Luo2022IntensityCacheWriter:
    """Recoverable writer for fixed-depth raw float32 intensity shards.

    The cache is intentionally a storage primitive: orchestration supplies the
    operator and R0 fingerprints as well as assignment rows.  Every append is
    atomically committed through a manifest whose own root fingerprint covers
    all shard and scaling metadata.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        operator_id: str,
        assignment_sha: str,
        r0_fingerprint: str,
        split: str = "train",
        expected_shape: Sequence[int] | None = None,
        expected_rows: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        shape = _validate_sample_shape(expected_shape) if expected_shape is not None else None
        if expected_rows is not None and int(expected_rows) <= 0:
            raise ValueError("expected_rows must be positive when provided")
        requested = {
            "operator_id": str(operator_id),
            "assignment_sha": str(assignment_sha),
            "r0_fingerprint": str(r0_fingerprint),
            "split": str(split),
            "shape": list(shape) if shape is not None else None,
            "expected_rows": int(expected_rows) if expected_rows is not None else None,
        }
        if not requested["operator_id"]:
            raise ValueError("operator_id must not be empty")
        if not requested["assignment_sha"]:
            raise ValueError("assignment_sha must not be empty")
        if not requested["r0_fingerprint"]:
            raise ValueError("r0_fingerprint must not be empty")
        if not requested["split"]:
            raise ValueError("split must not be empty")

        self.root.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            self.manifest = _load_and_verify_cache_manifest(
                self.root,
                require_complete=False,
            )
            for key, expected in requested.items():
                actual = self.manifest.get(key)
                if expected is not None and actual != expected:
                    raise ValueError(
                        f"cache manifest {key} does not match requested value"
                    )
        else:
            unexpected = [path for path in self.root.iterdir() if not path.name.startswith(".")]
            if unexpected:
                raise ValueError("cache directory contains files but no manifest")
            self.manifest = {
                "schema_version": LUO2022_INTENSITY_CACHE_SCHEMA,
                "status": "building",
                **requested,
                "dtype": "float32",
                "row_count": 0,
                "shards": [],
                "scale": None,
            }
            _commit_cache_manifest(self.root, self.manifest)

    def append_shard(
        self,
        tensors: torch.Tensor | Mapping[str, torch.Tensor],
        records: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Atomically append, recover, or recognize an identical raw shard."""

        intensity = _extract_raw_intensity_tensor(tensors)
        normalized_records = [_validate_assignment_record(record) for record in records]
        if intensity.ndim < 2:
            raise ValueError("intensity shard must have shape (N, ...sample_shape)")
        if intensity.shape[0] != len(normalized_records):
            raise ValueError("intensity shard and records must have the same row count")
        if intensity.shape[0] == 0:
            raise ValueError("intensity shard must contain at least one row")
        sample_shape = tuple(int(value) for value in intensity.shape[1:])
        manifest_shape = self.manifest.get("shape")
        if manifest_shape is not None and tuple(manifest_shape) != sample_shape:
            raise ValueError("intensity sample shape does not match cache manifest")
        if not torch.isfinite(intensity).all():
            raise ValueError("raw intensity must contain only finite values")
        if bool((intensity < 0).any()):
            raise ValueError("raw intensity must be non-negative")

        raw_bytes = intensity.numpy().tobytes(order="C")
        records_bytes = _canonical_json_bytes(normalized_records)
        intensity_sha = hashlib.sha256(raw_bytes).hexdigest()
        records_sha = hashlib.sha256(records_bytes).hexdigest()
        content_sha = canonical_sha256(
            {
                "dtype": "float32",
                "shape": list(sample_shape),
                "row_count": len(normalized_records),
                "intensity_sha256": intensity_sha,
                "records_sha256": records_sha,
            }
        )

        for existing in self.manifest["shards"]:
            if existing["content_sha256"] == content_sha:
                return dict(existing)
        if self.manifest["status"] == "complete":
            raise RuntimeError("cannot append a new shard to a finalized cache")

        start_row = int(self.manifest["row_count"])
        for offset, record in enumerate(normalized_records):
            if record["row_id"] != start_row + offset:
                raise ValueError("cache records must have contiguous row_id values")
        expected_rows = self.manifest.get("expected_rows")
        if expected_rows is not None and start_row + len(normalized_records) > expected_rows:
            raise ValueError("appended shard would exceed expected_rows")

        shard_index = len(self.manifest["shards"])
        intensity_name = f"shard-{shard_index:06d}.f32"
        records_name = f"shard-{shard_index:06d}.records.json"
        intensity_path = self.root / intensity_name
        records_path = self.root / records_name
        _write_or_verify_recoverable_file(intensity_path, raw_bytes, intensity_sha)
        _write_or_verify_recoverable_file(records_path, records_bytes, records_sha)

        shard = {
            "index": shard_index,
            "start_row": start_row,
            "row_count": len(normalized_records),
            "shape": list(sample_shape),
            "dtype": "float32",
            "intensity_file": intensity_name,
            "records_file": records_name,
            "intensity_sha256": intensity_sha,
            "records_sha256": records_sha,
            "content_sha256": content_sha,
            "sha256": canonical_sha256(
                {
                    "index": shard_index,
                    "start_row": start_row,
                    "content_sha256": content_sha,
                }
            ),
            "max_value": float(intensity.max().item()),
        }
        self.manifest["shape"] = list(sample_shape)
        self.manifest["row_count"] = start_row + len(normalized_records)
        self.manifest["shards"].append(shard)
        _commit_cache_manifest(self.root, self.manifest)
        return dict(shard)

    def finalize(
        self,
        scale_method: str = "global_dataset_max",
        *,
        frozen_scale: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Seal with a local training scale or an explicit frozen train scale."""

        if scale_method != "global_dataset_max":
            raise ValueError("scale_method must be 'global_dataset_max'")
        normalized_frozen_scale = (
            _validate_frozen_cache_scale(
                frozen_scale,
                operator_id=self.manifest["operator_id"],
                scale_method=scale_method,
            )
            if frozen_scale is not None
            else None
        )
        if self.manifest["status"] == "complete":
            scale = self.manifest.get("scale") or {}
            if scale.get("method") != scale_method:
                raise ValueError("cache was finalized with a different scale method")
            if normalized_frozen_scale is not None and scale != normalized_frozen_scale:
                raise ValueError("cache was finalized with different frozen scale metadata")
            return _json_copy(self.manifest)
        if not self.manifest["shards"]:
            raise ValueError("cannot finalize an empty cache")
        expected_rows = self.manifest.get("expected_rows")
        if expected_rows is not None and self.manifest["row_count"] != expected_rows:
            raise ValueError("cache row_count does not match expected_rows")

        if normalized_frozen_scale is None:
            if self.manifest["split"] != "train":
                raise ValueError("non-training cache requires an explicit frozen_scale")
            observed_max = max(float(shard["max_value"]) for shard in self.manifest["shards"])
            self.manifest["scale"] = {
                "method": scale_method,
                "value": observed_max if observed_max > 0.0 else 1.0,
                "observed_max": observed_max,
                "fitted_split": "train",
                "scope": "dataset_scalar",
                "applied_at": "dataset_read",
                "reuse_mode": "local_training_fit",
            }
        else:
            if self.manifest["split"] == "train":
                raise ValueError("training cache must fit its own scale")
            self.manifest["scale"] = normalized_frozen_scale
        self.manifest["status"] = "complete"
        _commit_cache_manifest(self.root, self.manifest)
        self.manifest = _load_and_verify_cache_manifest(self.root, require_complete=True)
        return _json_copy(self.manifest)


def verify_luo2022_intensity_cache(root: str | Path) -> dict[str, Any]:
    """Verify the complete manifest and all raw/record shard hashes."""

    return _json_copy(_load_and_verify_cache_manifest(Path(root), require_complete=True))


def make_luo2022_frozen_scale(training_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Attach portable training-cache provenance to a reusable scalar scale."""

    if training_manifest.get("schema_version") != LUO2022_INTENSITY_CACHE_SCHEMA:
        raise ValueError("training manifest has an unsupported cache schema")
    if training_manifest.get("status") != "complete" or training_manifest.get("split") != "train":
        raise ValueError("frozen scale source must be a complete training cache")
    root_fingerprint = training_manifest.get("root_fingerprint")
    operator_id = training_manifest.get("operator_id")
    if not isinstance(root_fingerprint, str) or not root_fingerprint:
        raise ValueError("training manifest is missing its root fingerprint")
    if root_fingerprint != _cache_root_fingerprint(training_manifest):
        raise ValueError("training manifest root fingerprint mismatch")
    if not isinstance(operator_id, str) or not operator_id:
        raise ValueError("training manifest is missing its operator_id")
    scale = dict(training_manifest.get("scale") or {})
    scale["reuse_mode"] = "frozen_training_statistic"
    scale["provenance"] = {
        "source": "training_cache",
        "source_cache_root_fingerprint": root_fingerprint,
        "operator_id": operator_id,
    }
    return _validate_frozen_cache_scale(
        scale,
        operator_id=operator_id,
        scale_method="global_dataset_max",
    )


class Luo2022CachedIntensityDataset(Dataset[dict[str, torch.Tensor]]):
    """Read a sealed raw-intensity cache with its train-only scalar scaling."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.manifest = _load_and_verify_cache_manifest(self.root, require_complete=True)
        scale = self.manifest["scale"]
        if scale.get("fitted_split") != "train" or scale.get("scope") != "dataset_scalar":
            raise ValueError("cache scale must be a train-fitted dataset scalar")
        self.scale = float(scale["value"])
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("cache scale value must be finite and positive")

        self._rows: list[tuple[int, int, dict[str, int]]] = []
        self._shard_tensors: dict[int, torch.Tensor] = {}
        for shard_index, shard in enumerate(self.manifest["shards"]):
            records = json.loads((self.root / shard["records_file"]).read_text("utf-8"))
            for local_index, record in enumerate(records):
                self._rows.append((shard_index, local_index, _validate_assignment_record(record)))
        self.object_id_min = min(record["object_id"] for _shard, _local, record in self._rows)
        self.object_id_max = max(record["object_id"] for _shard, _local, record in self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index, local_index, record = self._rows[index]
        raw_shard = self._mapped_shard(shard_index)
        input_intensity = raw_shard[local_index].div(self.scale)
        return {
            "input_intensity": input_intensity,
            **{
                field: torch.tensor(record[field], dtype=torch.long)
                for field in _ASSIGNMENT_FIELDS
            },
        }

    def _mapped_shard(self, shard_index: int) -> torch.Tensor:
        tensor = self._shard_tensors.get(shard_index)
        if tensor is None:
            shard = self.manifest["shards"][shard_index]
            sample_shape = tuple(int(value) for value in shard["shape"])
            element_count = int(shard["row_count"]) * math.prod(sample_shape)
            tensor = torch.from_file(
                str(self.root / shard["intensity_file"]),
                shared=False,
                size=element_count,
                dtype=torch.float32,
            ).view(int(shard["row_count"]), *sample_shape)
            self._shard_tensors[shard_index] = tensor
        return tensor


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


def prepare_huang2026_visible_amplitude(
    image: torch.Tensor,
    *,
    input_shape: tuple[int, int] = (28, 28),
    resized_shape: tuple[int, int] = (320, 320),
    canvas_shape: tuple[int, int] = (400, 400),
) -> torch.Tensor:
    """Prepare the visible-light input amplitude from Supporting Note S3.

    The paper path is exactly MNIST ``28 x 28`` -> bilinear ``320 x 320`` ->
    centered zero padding on a ``400 x 400`` canvas.  The three shapes are
    parameters only so reduced-scale validation can exercise the identical
    operation.  Values are optical-field amplitudes: this function performs
    no square root, min-max normalization, clipping, or detachment.
    """

    expected_input = _validate_huang2026_shape(input_shape, name="input_shape")
    resized_size = _validate_huang2026_shape(resized_shape, name="resized_shape")
    canvas_size = _validate_huang2026_shape(canvas_shape, name="canvas_shape")
    if resized_size[0] > canvas_size[0] or resized_size[1] > canvas_size[1]:
        raise ValueError("resized_shape must fit inside canvas_shape")

    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 3:
        if image.shape[0] != 1:
            raise ValueError("three-dimensional image must have shape (1, H, W)")
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError("image must have shape (H, W), (1, H, W), or (B, 1, H, W)")
    if tuple(image.shape[-2:]) != expected_input:
        raise ValueError(
            f"image spatial shape {tuple(image.shape[-2:])} does not match input_shape "
            f"{expected_input}"
        )
    if not torch.is_floating_point(image):
        image = image.to(dtype=torch.float32)

    resized = F.interpolate(
        image,
        size=resized_size,
        mode="bilinear",
        align_corners=False,
    )
    pad_height = canvas_size[0] - resized_size[0]
    pad_width = canvas_size[1] - resized_size[1]
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left
    return F.pad(resized, (left, right, top, bottom), mode="constant", value=0.0)


class Huang2026OnlineDiffuserSampler:
    """Deterministic online diffuser schedule with split-separated seed space.

    Seeds are functions of ``(split, base_seed, iteration, object_id)``.
    Training seeds occupy ``[0, 2**62)`` and blind-test seeds occupy
    ``[2**62, 2**63)``, making the two namespaces disjoint by construction.
    An optional ``diffuser_factory`` is called online and its tensor is
    returned without detaching it; otherwise this class is a lightweight seed
    and metadata scheduler for :class:`d2nn.CorrelatedHeightPhaseDiffuser`.
    """

    def __init__(
        self,
        *,
        split: str,
        base_seed: int = 0,
        correlation_length_pixels: float,
        wavelengths: float | Sequence[float] = (660e-9,),
        diffuser_factory: Callable[..., torch.Tensor | Mapping[str, Any]] | None = None,
    ) -> None:
        self.split = _normalize_huang2026_split(split)
        self.base_seed = _validate_huang2026_nonnegative_int(base_seed, name="base_seed")
        self.correlation_length_pixels = _validate_huang2026_positive_float(
            correlation_length_pixels,
            name="correlation_length_pixels",
        )
        self.wavelengths = _normalize_huang2026_wavelengths(wavelengths)
        self.diffuser_factory = diffuser_factory

    def seed_for(self, *, iteration: int, object_id: int) -> int:
        """Return the reproducible diffuser seed for one online sample."""

        return _huang2026_domain_seed(
            domain="diffuser",
            split=self.split,
            base_seed=self.base_seed,
            iteration=iteration,
            object_id=object_id,
            realization=0,
        )

    def metadata(self, *, iteration: int, object_id: int) -> dict[str, Any]:
        """Return portable metadata for one diffuser realization."""

        iteration_value = _validate_huang2026_nonnegative_int(iteration, name="iteration")
        object_value = _validate_huang2026_nonnegative_int(object_id, name="object_id")
        return {
            "split": self.split,
            "iteration": iteration_value,
            "object_id": object_value,
            "diffuser_seed": self.seed_for(
                iteration=iteration_value,
                object_id=object_value,
            ),
            "correlation_length_pixels": self.correlation_length_pixels,
            "wavelengths_m": list(self.wavelengths),
        }

    def sample(
        self,
        *,
        iteration: int,
        object_id: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, Any]:
        """Generate one diffuser online, or return its seed record.

        The injected factory protocol is keyword-only:
        ``seed``, ``correlation_length_pixels``, ``wavelengths``, ``split``,
        and optionally ``device``/``dtype``.  A tensor result is exposed as
        ``diffuser_height_m`` so one physical diffuser can be converted at
        every wavelength without resampling it.  A mapping result is merged
        after checking that it does not overwrite provenance fields.
        """

        record = self.metadata(iteration=iteration, object_id=object_id)
        if self.diffuser_factory is None:
            return record
        factory_kwargs: dict[str, Any] = {
            "seed": record["diffuser_seed"],
            "correlation_length_pixels": self.correlation_length_pixels,
            "wavelengths": self.wavelengths,
            "split": self.split,
        }
        if device is not None:
            factory_kwargs["device"] = torch.device(device)
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        generated = self.diffuser_factory(**factory_kwargs)
        if isinstance(generated, torch.Tensor):
            record["diffuser_height_m"] = generated
            return record
        if not isinstance(generated, Mapping):
            raise TypeError("diffuser_factory must return a tensor or mapping")
        overlap = set(record).intersection(generated)
        if overlap:
            raise ValueError(
                "diffuser_factory must not overwrite sampler metadata fields: "
                f"{sorted(overlap)}"
            )
        record.update(generated)
        return record


class Huang2026CoherenceSampler:
    """Gaussian-Schell circular-complex screen sampler for Supporting Note S1.

    The frequency-domain filter is the square root of the Gaussian power
    spectrum in equation (S8), with spatial coherence length
    ``coherence_length_pixels``.  The discrete spectrum is normalized to unit
    *expected* mean power; individual realizations are not rescaled, preserving
    the circular-Gaussian distribution.  Screens are created from a CPU
    generator for device-independent seed replay and then moved to the
    requested device.
    """

    def __init__(
        self,
        field_shape: tuple[int, int],
        *,
        split: str,
        base_seed: int = 0,
        coherence_length_pixels: float = 4.0,
    ) -> None:
        self.field_shape = _validate_huang2026_shape(field_shape, name="field_shape")
        self.split = _normalize_huang2026_split(split)
        self.base_seed = _validate_huang2026_nonnegative_int(base_seed, name="base_seed")
        self.coherence_length_pixels = _validate_huang2026_positive_float(
            coherence_length_pixels,
            name="coherence_length_pixels",
        )

    def seed_for(self, *, iteration: int, object_id: int, realization: int) -> int:
        """Return one reproducible complex-screen seed."""

        return _huang2026_domain_seed(
            domain="coherence",
            split=self.split,
            base_seed=self.base_seed,
            iteration=iteration,
            object_id=object_id,
            realization=realization,
        )

    def metadata(
        self,
        *,
        iteration: int,
        object_id: int,
        realization: int,
    ) -> dict[str, Any]:
        return {
            "split": self.split,
            "iteration": _validate_huang2026_nonnegative_int(
                iteration,
                name="iteration",
            ),
            "object_id": _validate_huang2026_nonnegative_int(
                object_id,
                name="object_id",
            ),
            "realization": _validate_huang2026_nonnegative_int(
                realization,
                name="realization",
            ),
            "coherence_seed": self.seed_for(
                iteration=iteration,
                object_id=object_id,
                realization=realization,
            ),
            "coherence_length_pixels": self.coherence_length_pixels,
        }

    def sample(
        self,
        *,
        iteration: int,
        object_id: int,
        realization: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.complex64,
    ) -> torch.Tensor:
        """Return one unit-power Gaussian-Schell complex random screen."""

        if dtype not in {torch.complex64, torch.complex128}:
            raise TypeError("dtype must be torch.complex64 or torch.complex128")
        seed = self.seed_for(
            iteration=iteration,
            object_id=object_id,
            realization=realization,
        )
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        real_noise = torch.randn(
            self.field_shape,
            generator=generator,
            dtype=real_dtype,
        )
        imaginary_noise = torch.randn(
            self.field_shape,
            generator=generator,
            dtype=real_dtype,
        )
        circular_noise = torch.complex(real_noise, imaginary_noise) / math.sqrt(2.0)

        frequency_y = torch.fft.fftfreq(self.field_shape[0], dtype=real_dtype)
        frequency_x = torch.fft.fftfreq(self.field_shape[1], dtype=real_dtype)
        grid_y, grid_x = torch.meshgrid(frequency_y, frequency_x, indexing="ij")
        spectral_amplitude = torch.exp(
            -0.5
            * (math.pi * self.coherence_length_pixels) ** 2
            * (grid_x.square() + grid_y.square())
        )
        spectral_amplitude = spectral_amplitude / spectral_amplitude.square().mean().sqrt()
        screen = torch.fft.ifft2(
            circular_noise * spectral_amplitude.to(dtype=dtype),
            norm="ortho",
        )
        return screen.to(device=torch.device(device), dtype=dtype)

    def sample_ensemble(
        self,
        *,
        iteration: int,
        object_id: int,
        num_realizations: int,
        start_realization: int = 0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.complex64,
    ) -> torch.Tensor:
        """Return ``Nr`` screens; callers may invoke this in bounded chunks."""

        count = _validate_huang2026_positive_int(
            num_realizations,
            name="num_realizations",
        )
        start = _validate_huang2026_nonnegative_int(
            start_realization,
            name="start_realization",
        )
        return torch.stack(
            [
                self.sample(
                    iteration=iteration,
                    object_id=object_id,
                    realization=start + offset,
                    device=device,
                    dtype=dtype,
                )
                for offset in range(count)
            ],
            dim=0,
        )


class Huang2026VisibleDataset(Dataset[dict[str, Any]]):
    """Online MNIST amplitude records for the Huang et al. 2026 baseline.

    The dataset does not materialize detector outputs.  It emits an amplitude,
    its clean intensity target, an online diffuser seed (and optional factory
    tensor), and wavelength-aware provenance.  ``sample_at`` makes the training
    iteration explicit; ``__getitem__`` also accepts ``(iteration, index)`` so
    a DataLoader sampler can carry iteration identity without mutable worker
    state.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        *,
        split: str,
        correlation_length_pixels: float,
        base_seed: int = 0,
        input_shape: tuple[int, int] = (28, 28),
        resized_shape: tuple[int, int] = (320, 320),
        canvas_shape: tuple[int, int] = (400, 400),
        illumination_mode: str = "coherent",
        wavelengths: float | Sequence[float] = (660e-9,),
        diffuser_sampler: Huang2026OnlineDiffuserSampler | None = None,
        coherence_sampler: Huang2026CoherenceSampler | None = None,
        object_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.split = _normalize_huang2026_split(split)
        self.base_seed = _validate_huang2026_nonnegative_int(base_seed, name="base_seed")
        self.input_shape = _validate_huang2026_shape(input_shape, name="input_shape")
        self.resized_shape = _validate_huang2026_shape(
            resized_shape,
            name="resized_shape",
        )
        self.canvas_shape = _validate_huang2026_shape(
            canvas_shape,
            name="canvas_shape",
        )
        self.illumination_mode = _normalize_huang2026_illumination_mode(
            illumination_mode
        )
        self.wavelengths = _normalize_huang2026_wavelengths(wavelengths)
        self.correlation_length_pixels = _validate_huang2026_positive_float(
            correlation_length_pixels,
            name="correlation_length_pixels",
        )
        self.diffuser_sampler = diffuser_sampler or Huang2026OnlineDiffuserSampler(
            split=self.split,
            base_seed=self.base_seed,
            correlation_length_pixels=self.correlation_length_pixels,
            wavelengths=self.wavelengths,
        )
        if self.diffuser_sampler.split != self.split:
            raise ValueError("diffuser_sampler split must match dataset split")
        if self.diffuser_sampler.correlation_length_pixels != self.correlation_length_pixels:
            raise ValueError(
                "diffuser_sampler correlation length must match dataset configuration"
            )
        if self.diffuser_sampler.wavelengths != self.wavelengths:
            raise ValueError("diffuser_sampler wavelengths must match dataset wavelengths")

        if coherence_sampler is None and self.illumination_mode == "incoherent":
            coherence_sampler = Huang2026CoherenceSampler(
                self.canvas_shape,
                split=self.split,
                base_seed=self.base_seed,
                coherence_length_pixels=4.0,
            )
        if coherence_sampler is not None:
            if coherence_sampler.split != self.split:
                raise ValueError("coherence_sampler split must match dataset split")
            if coherence_sampler.field_shape != self.canvas_shape:
                raise ValueError(
                    "coherence_sampler field_shape must match dataset canvas_shape"
                )
        self.coherence_sampler = coherence_sampler
        self.iteration = 0

        if object_ids is None:
            self.object_ids: tuple[int, ...] | None = None
        else:
            if isinstance(object_ids, torch.Tensor):
                if object_ids.ndim != 1:
                    raise ValueError("object_ids tensor must be one-dimensional")
                normalized_ids = tuple(
                    int(value) for value in object_ids.detach().cpu().tolist()
                )
            else:
                normalized_ids = tuple(int(value) for value in object_ids)
            if len(normalized_ids) != len(base_dataset):
                raise ValueError("object_ids must have the same length as base_dataset")
            if any(value < 0 for value in normalized_ids):
                raise ValueError("object_ids must be non-negative")
            if len(set(normalized_ids)) != len(normalized_ids):
                raise ValueError("object_ids must be unique")
            self.object_ids = normalized_ids

    def __len__(self) -> int:
        return len(self.base_dataset)

    def set_iteration(self, iteration: int) -> None:
        """Set the explicit iteration used by integer indexing."""

        self.iteration = _validate_huang2026_nonnegative_int(
            iteration,
            name="iteration",
        )

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        if isinstance(index, tuple):
            if len(index) != 2:
                raise ValueError("tuple index must be (iteration, object_index)")
            iteration, object_index = index
            return self.sample_at(object_index, iteration=iteration)
        return self.sample_at(index, iteration=self.iteration)

    def sample_at(self, index: int, *, iteration: int) -> dict[str, Any]:
        """Return one online record for an explicit iteration and object."""

        index_value = int(index)
        if index_value < 0:
            index_value += len(self)
        if index_value < 0 or index_value >= len(self):
            raise IndexError(index)
        iteration_value = _validate_huang2026_nonnegative_int(
            iteration,
            name="iteration",
        )
        base_item = self.base_dataset[index_value]
        image, label = _unpack_image_label(base_item)
        amplitude = prepare_huang2026_visible_amplitude(
            image,
            input_shape=self.input_shape,
            resized_shape=self.resized_shape,
            canvas_shape=self.canvas_shape,
        )[0]
        object_id = self._object_id(index_value, base_item)
        diffuser_record = self.diffuser_sampler.sample(
            iteration=iteration_value,
            object_id=object_id,
            device=amplitude.device,
            dtype=amplitude.dtype,
        )
        diffuser_seed = int(diffuser_record["diffuser_seed"])
        wavelength_tensor = torch.tensor(
            self.wavelengths,
            dtype=torch.float64,
            device=amplitude.device,
        )
        metadata: dict[str, Any] = {
            "schema_version": HUANG2026_VISIBLE_SAMPLE_SCHEMA,
            "split": self.split,
            "object_id": object_id,
            "iteration": iteration_value,
            "diffuser_seed": diffuser_seed,
            "correlation_length_pixels": self.correlation_length_pixels,
            "diffuser_correlation_length_pixels": self.correlation_length_pixels,
            "illumination_mode": self.illumination_mode,
            "wavelength": (
                self.wavelengths[0]
                if len(self.wavelengths) == 1
                else list(self.wavelengths)
            ),
            "wavelengths_m": list(self.wavelengths),
        }
        sample: dict[str, Any] = {
            "amplitude": amplitude,
            "target_intensity": amplitude.square(),
            "label": torch.tensor(int(label), dtype=torch.long),
            "object_id": torch.tensor(object_id, dtype=torch.long),
            "iteration": torch.tensor(iteration_value, dtype=torch.long),
            "diffuser_seed": torch.tensor(diffuser_seed, dtype=torch.long),
            "correlation_length_pixels": torch.tensor(
                self.correlation_length_pixels,
                dtype=torch.float64,
            ),
            "diffuser_correlation_length_pixels": torch.tensor(
                self.correlation_length_pixels,
                dtype=torch.float64,
            ),
            "illumination_mode": self.illumination_mode,
            "wavelength": (
                wavelength_tensor[0] if len(self.wavelengths) == 1 else wavelength_tensor
            ),
            "wavelengths_m": wavelength_tensor,
            "metadata": metadata,
        }
        sampler_metadata_fields = {
            "split",
            "iteration",
            "object_id",
            "diffuser_seed",
            "correlation_length_pixels",
            "wavelengths_m",
        }
        sample.update(
            {
                key: value
                for key, value in diffuser_record.items()
                if key not in sampler_metadata_fields
            }
        )
        if self.coherence_sampler is not None:
            coherence_metadata = self.coherence_sampler.metadata(
                iteration=iteration_value,
                object_id=object_id,
                realization=0,
            )
            sample["coherence_seed"] = torch.tensor(
                coherence_metadata["coherence_seed"],
                dtype=torch.long,
            )
            sample["coherence_length_pixels"] = torch.tensor(
                self.coherence_sampler.coherence_length_pixels,
                dtype=torch.float64,
            )
            metadata.update(
                {
                    "coherence_seed": coherence_metadata["coherence_seed"],
                    "coherence_length_pixels": (
                        self.coherence_sampler.coherence_length_pixels
                    ),
                }
            )
        return sample

    def _object_id(self, index: int, base_item: Any) -> int:
        if self.object_ids is not None:
            return self.object_ids[index]
        if isinstance(base_item, Mapping) and "object_id" in base_item:
            value = base_item["object_id"]
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError("base item object_id tensor must be scalar")
                value = value.item()
            return _validate_huang2026_nonnegative_int(value, name="object_id")
        return index


def _normalize_huang2026_split(split: str) -> str:
    normalized = str(split).strip().lower().replace("-", "_")
    aliases = {
        "train": HUANG2026_TRAIN_SPLIT,
        "training": HUANG2026_TRAIN_SPLIT,
        "blind": HUANG2026_BLIND_TEST_SPLIT,
        "test": HUANG2026_BLIND_TEST_SPLIT,
        "blind_test": HUANG2026_BLIND_TEST_SPLIT,
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError("split must be 'train' or 'blind_test'") from error


def _normalize_huang2026_illumination_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "")
    aliases = {
        "coherent": "coherent",
        "incoherent": "incoherent",
        "multiwavelength": "multiwavelength",
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError(
            "illumination_mode must be coherent, incoherent, or multiwavelength"
        ) from error


def _normalize_huang2026_wavelengths(
    wavelengths: float | Sequence[float],
) -> tuple[float, ...]:
    if isinstance(wavelengths, (int, float)):
        values = (float(wavelengths),)
    else:
        values = tuple(float(value) for value in wavelengths)
    if not values:
        raise ValueError("wavelengths must not be empty")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("wavelengths must contain finite positive values")
    if len(set(values)) != len(values):
        raise ValueError("wavelengths must not contain duplicates")
    return values


def _validate_huang2026_shape(
    shape: Sequence[int],
    *,
    name: str,
) -> tuple[int, int]:
    values = tuple(shape)
    if (
        len(values) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in values)
        or any(value <= 0 for value in values)
    ):
        raise ValueError(f"{name} must contain two positive integers")
    return int(values[0]), int(values[1])


def _validate_huang2026_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a non-negative integer") from error
    if normalized != value or normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _validate_huang2026_positive_int(value: Any, *, name: str) -> int:
    normalized = _validate_huang2026_nonnegative_int(value, name=name)
    if normalized == 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _validate_huang2026_positive_float(value: Any, *, name: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite positive value") from error
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be a finite positive value")
    return normalized


def _huang2026_domain_seed(
    *,
    domain: str,
    split: str,
    base_seed: int,
    iteration: int,
    object_id: int,
    realization: int,
) -> int:
    split_value = _normalize_huang2026_split(split)
    seed_value = _validate_huang2026_nonnegative_int(base_seed, name="base_seed")
    iteration_value = _validate_huang2026_nonnegative_int(
        iteration,
        name="iteration",
    )
    object_value = _validate_huang2026_nonnegative_int(object_id, name="object_id")
    realization_value = _validate_huang2026_nonnegative_int(
        realization,
        name="realization",
    )
    digest = hashlib.sha256(
        (
            f"huang2026-visible/{domain}/v1\0{seed_value}\0"
            f"{iteration_value}\0{object_value}\0{realization_value}"
        ).encode("utf-8")
    ).digest()
    payload = int.from_bytes(digest[:8], byteorder="big") & ((1 << 62) - 1)
    namespace_bit = 0 if split_value == HUANG2026_TRAIN_SPLIT else 1 << 62
    return namespace_bit | payload


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


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_json_bytes(value).decode("utf-8"))


def _validate_sample_shape(shape: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in shape)
    if not values or any(value <= 0 for value in values):
        raise ValueError("sample shape must contain positive dimensions")
    return values


def _contiguous_id_offset(object_ids: Sequence[int]) -> int | None:
    if not object_ids:
        return None
    offset = int(object_ids[0])
    if all(int(object_id) == offset + index for index, object_id in enumerate(object_ids)):
        return offset
    return None


def _validate_assignment_record(record: Mapping[str, Any]) -> dict[str, int]:
    missing = [field for field in _ASSIGNMENT_FIELDS if field not in record]
    if missing:
        raise ValueError(f"assignment record is missing fields: {', '.join(missing)}")
    normalized = {field: int(record[field]) for field in _ASSIGNMENT_FIELDS}
    for field in (
        "object_id",
        "diffuser_id",
        "training_epoch",
        "within_epoch_index",
        "row_id",
    ):
        if normalized[field] < 0:
            raise ValueError(f"assignment record {field} must be non-negative")
    return normalized


def _extract_raw_intensity_tensor(
    tensors: torch.Tensor | Mapping[str, torch.Tensor],
) -> torch.Tensor:
    if isinstance(tensors, Mapping):
        available = [key for key in ("raw_intensity", "intensity") if key in tensors]
        if len(available) != 1:
            raise ValueError(
                "tensor mapping must contain exactly one of 'raw_intensity' or 'intensity'"
            )
        tensors = tensors[available[0]]
    if not isinstance(tensors, torch.Tensor):
        raise TypeError("tensors must be a torch.Tensor or tensor mapping")
    if tensors.is_complex():
        raise ValueError("raw intensity must be real-valued")
    return tensors.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"cache file is missing or unreadable: {path.name}") from error
    return digest.hexdigest()


def _write_or_verify_recoverable_file(
    path: Path,
    content: bytes,
    expected_sha: str,
) -> None:
    if path.exists():
        if _sha256_file(path) != expected_sha:
            raise ValueError(f"existing recovery file failed integrity check: {path.name}")
        return
    _atomic_write_bytes(path, content)
    if _sha256_file(path) != expected_sha:
        raise RuntimeError(f"atomic cache write failed integrity check: {path.name}")


def _cache_root_fingerprint(manifest: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "root_fingerprint"}
    return canonical_sha256(payload)


def _validate_frozen_cache_scale(
    scale: Mapping[str, Any],
    *,
    operator_id: str,
    scale_method: str,
) -> dict[str, Any]:
    if scale.get("method") != scale_method:
        raise ValueError("frozen scale method does not match the requested method")
    if scale.get("fitted_split") != "train" or scale.get("scope") != "dataset_scalar":
        raise ValueError("frozen scale must be a train-fitted dataset scalar")
    value = scale.get("value")
    observed_max = scale.get("observed_max")
    if not isinstance(value, (int, float)) or not math.isfinite(value) or float(value) <= 0.0:
        raise ValueError("frozen scale value must be finite and positive")
    if (
        not isinstance(observed_max, (int, float))
        or not math.isfinite(observed_max)
        or float(observed_max) < 0.0
    ):
        raise ValueError("frozen scale observed_max must be finite and non-negative")
    provenance = scale.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("frozen scale requires training-cache provenance")
    source_fingerprint = provenance.get("source_cache_root_fingerprint")
    if provenance.get("source") != "training_cache" or not isinstance(
        source_fingerprint, str
    ) or not source_fingerprint:
        raise ValueError("frozen scale provenance requires a training cache root fingerprint")
    if provenance.get("operator_id") != operator_id:
        raise ValueError("frozen scale operator_id does not match the cache operator")
    return {
        "method": scale_method,
        "value": float(value),
        "observed_max": float(observed_max),
        "fitted_split": "train",
        "scope": "dataset_scalar",
        "applied_at": "dataset_read",
        "reuse_mode": "frozen_training_statistic",
        "provenance": {
            "source": "training_cache",
            "source_cache_root_fingerprint": source_fingerprint,
            "operator_id": operator_id,
        },
    }


def _commit_cache_manifest(root: Path, manifest: dict[str, Any]) -> None:
    manifest["root_fingerprint"] = _cache_root_fingerprint(manifest)
    _atomic_write_bytes(root / "manifest.json", _canonical_json_bytes(manifest))


def _safe_cache_child(root: Path, name: Any) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("cache manifest contains an unsafe shard filename")
    return root / name


def _load_and_verify_cache_manifest(
    root: Path,
    *,
    require_complete: bool,
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text("utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError("cache manifest is missing or invalid") from error
    if not isinstance(manifest, dict):
        raise ValueError("cache manifest must be a JSON object")
    if manifest.get("schema_version") != LUO2022_INTENSITY_CACHE_SCHEMA:
        raise ValueError("unsupported cache manifest schema")
    if manifest.get("root_fingerprint") != _cache_root_fingerprint(manifest):
        raise ValueError("cache manifest root fingerprint mismatch")
    if manifest.get("status") not in {"building", "complete"}:
        raise ValueError("cache manifest has an invalid status")
    if require_complete and manifest["status"] != "complete":
        raise ValueError("cache is not finalized")
    if manifest.get("dtype") != "float32":
        raise ValueError("cache manifest dtype must be float32")
    for identity_field in ("operator_id", "assignment_sha", "r0_fingerprint"):
        if not isinstance(manifest.get(identity_field), str) or not manifest[identity_field]:
            raise ValueError(f"cache manifest {identity_field} must be a non-empty string")
    if not isinstance(manifest.get("split"), str) or not manifest["split"]:
        raise ValueError("cache manifest split must be a non-empty string")
    if not isinstance(manifest.get("shards"), list):
        raise ValueError("cache manifest shards must be a list")

    manifest_shape = manifest.get("shape")
    if manifest_shape is not None:
        manifest_shape = list(_validate_sample_shape(manifest_shape))
        if manifest["shape"] != manifest_shape:
            raise ValueError("cache manifest shape is not canonical")
    expected_start = 0
    for expected_index, shard in enumerate(manifest["shards"]):
        if not isinstance(shard, dict):
            raise ValueError("cache shard metadata must be an object")
        if shard.get("index") != expected_index or shard.get("start_row") != expected_start:
            raise ValueError("cache shard indices or row offsets are not contiguous")
        row_count = shard.get("row_count")
        if not isinstance(row_count, int) or row_count <= 0:
            raise ValueError("cache shard row_count must be positive")
        if shard.get("dtype") != "float32" or shard.get("shape") != manifest_shape:
            raise ValueError("cache shard shape or dtype does not match manifest")
        intensity_path = _safe_cache_child(root, shard.get("intensity_file"))
        records_path = _safe_cache_child(root, shard.get("records_file"))
        if _sha256_file(intensity_path) != shard.get("intensity_sha256"):
            raise ValueError(f"cache intensity shard integrity check failed: {intensity_path.name}")
        if _sha256_file(records_path) != shard.get("records_sha256"):
            raise ValueError(f"cache records shard integrity check failed: {records_path.name}")
        expected_bytes = row_count * math.prod(manifest_shape or ()) * 4
        if intensity_path.stat().st_size != expected_bytes:
            raise ValueError("cache intensity shard byte size does not match manifest")
        try:
            records = json.loads(records_path.read_text("utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("cache records shard is invalid JSON") from error
        if not isinstance(records, list) or len(records) != row_count:
            raise ValueError("cache records shard row count does not match manifest")
        for local_index, record in enumerate(records):
            normalized = _validate_assignment_record(record)
            if normalized != record:
                raise ValueError("cache assignment record is not canonical")
            if normalized["row_id"] != expected_start + local_index:
                raise ValueError("cache assignment row_id values are not contiguous")
        content_sha = canonical_sha256(
            {
                "dtype": "float32",
                "shape": manifest_shape,
                "row_count": row_count,
                "intensity_sha256": shard["intensity_sha256"],
                "records_sha256": shard["records_sha256"],
            }
        )
        if shard.get("content_sha256") != content_sha:
            raise ValueError("cache shard content fingerprint mismatch")
        if shard.get("sha256") != canonical_sha256(
            {
                "index": expected_index,
                "start_row": expected_start,
                "content_sha256": content_sha,
            }
        ):
            raise ValueError("cache shard fingerprint mismatch")
        expected_start += row_count
    if manifest.get("row_count") != expected_start:
        raise ValueError("cache manifest row_count does not match its shards")
    expected_rows = manifest.get("expected_rows")
    if expected_rows is not None and (
        not isinstance(expected_rows, int) or expected_rows <= 0
    ):
        raise ValueError("cache manifest expected_rows must be positive")

    if manifest["status"] == "complete":
        scale = manifest.get("scale")
        if not isinstance(scale, dict):
            raise ValueError("finalized cache is missing scale metadata")
        if scale.get("method") != "global_dataset_max":
            raise ValueError("unsupported cache scale method")
        if scale.get("fitted_split") != "train" or scale.get("scope") != "dataset_scalar":
            raise ValueError("cache scale is not a train-fitted dataset scalar")
        scale_value = scale.get("value")
        if not isinstance(scale_value, (int, float)) or not math.isfinite(scale_value):
            raise ValueError("cache scale value must be finite")
        if float(scale_value) <= 0.0:
            raise ValueError("cache scale value must be positive")
        if manifest["split"] == "train":
            observed_max = max(float(shard["max_value"]) for shard in manifest["shards"])
            expected_scale = observed_max if observed_max > 0.0 else 1.0
            if scale.get("reuse_mode") != "local_training_fit":
                raise ValueError("training cache must contain a locally fitted scale")
            if float(scale.get("observed_max", math.nan)) != observed_max:
                raise ValueError("training cache observed_max does not match its shards")
            if float(scale_value) != expected_scale:
                raise ValueError("training cache scale value does not match its shards")
            if "provenance" in scale:
                raise ValueError("locally fitted training scale must not contain reuse provenance")
        else:
            normalized_scale = _validate_frozen_cache_scale(
                scale,
                operator_id=manifest["operator_id"],
                scale_method="global_dataset_max",
            )
            if scale != normalized_scale:
                raise ValueError("frozen cache scale metadata is not canonical")
    elif manifest.get("scale") is not None:
        raise ValueError("building cache must not contain scale metadata")
    return manifest
