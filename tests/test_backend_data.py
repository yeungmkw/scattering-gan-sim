import json
from collections import Counter

import pytest
import torch

from coherent_data import (
    LUO2022_INTENSITY_CACHE_SCHEMA,
    Luo2022CachedIntensityDataset,
    Luo2022IntensityCacheWriter,
    build_luo2022_fixed_depth_assignment,
    canonical_sha256,
    make_luo2022_frozen_scale,
    verify_luo2022_intensity_cache,
)


def test_fixed_depth_assignment_is_balanced_at_paper_scale() -> None:
    assignment = build_luo2022_fixed_depth_assignment(
        torch.arange(50_000) % 10,
        num_diffusers=2_000,
        diffusers_per_epoch=20,
        seed=2022,
    )

    rows = assignment["rows"]
    counts = Counter(row["diffuser_id"] for row in rows)
    assert len(rows) == 50_000
    assert len(counts) == 2_000
    assert set(counts.values()) == {25}
    assert assignment["metadata"]["training_epoch_count"] == 100
    assert assignment["root_sha"] == canonical_sha256(
        {"metadata": assignment["metadata"], "rows": rows}
    )
    assert set(rows[0]) == {
        "object_id",
        "label",
        "diffuser_id",
        "training_epoch",
        "within_epoch_index",
        "row_id",
    }
    assert [row["row_id"] for row in rows] == list(range(50_000))
    assert [row["diffuser_id"] for row in rows] == sorted(
        row["diffuser_id"] for row in rows
    )
    for row in rows:
        assert row["training_epoch"] == 1 + row["diffuser_id"] // 20
        assert row["within_epoch_index"] == row["diffuser_id"] % 20
        assert row["label"] == row["object_id"] % 10


def test_fixed_depth_assignment_handles_remainders_and_seed_determinism() -> None:
    labels = [index % 3 for index in range(17)]
    first = build_luo2022_fixed_depth_assignment(
        labels,
        num_diffusers=5,
        diffusers_per_epoch=2,
        seed=7,
    )
    repeat = build_luo2022_fixed_depth_assignment(
        labels,
        num_diffusers=5,
        diffusers_per_epoch=2,
        seed=7,
    )
    changed = build_luo2022_fixed_depth_assignment(
        labels,
        num_diffusers=5,
        diffusers_per_epoch=2,
        seed=8,
    )

    counts = Counter(row["diffuser_id"] for row in first["rows"])
    assert max(counts.values()) - min(counts.values()) <= 1
    assert first == repeat
    assert first["root_sha"] != changed["root_sha"]
    assert {
        row["object_id"]: row["diffuser_id"] for row in first["rows"]
    } != {
        row["object_id"]: row["diffuser_id"] for row in changed["rows"]
    }
    assert canonical_sha256({"b": [2, 3], "a": 1}) == canonical_sha256(
        {"a": 1, "b": [2, 3]}
    )


def test_fixed_depth_assignment_preserves_validation_source_object_ids() -> None:
    labels = [3, 1, 4, 1, 5, 9]
    validation = build_luo2022_fixed_depth_assignment(
        labels,
        num_diffusers=4,
        diffusers_per_epoch=2,
        seed=13,
        object_id_offset=50_000,
    )
    explicit = build_luo2022_fixed_depth_assignment(
        labels,
        num_diffusers=4,
        diffusers_per_epoch=2,
        seed=13,
        object_ids=list(range(50_000, 50_006)),
    )

    assert sorted(row["object_id"] for row in validation["rows"]) == list(
        range(50_000, 50_006)
    )
    assert [row["row_id"] for row in validation["rows"]] == list(range(6))
    assert validation["metadata"]["object_id_min"] == 50_000
    assert validation["metadata"]["object_id_max"] == 50_005
    assert validation["metadata"]["object_ids_sha"] == canonical_sha256(
        list(range(50_000, 50_006))
    )
    assert validation["metadata"]["object_id_offset"] == 50_000
    assert explicit["metadata"]["object_id_offset"] == 50_000
    assert explicit["rows"] == validation["rows"]
    assert explicit["root_sha"] == validation["root_sha"]

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_luo2022_fixed_depth_assignment(
            labels,
            num_diffusers=4,
            object_ids=list(range(6)),
            object_id_offset=50_000,
        )


def test_cache_write_recover_finalize_and_read_scalar_scaled_data(tmp_path) -> None:
    assignment = build_luo2022_fixed_depth_assignment(
        [9, 8, 7, 6, 5],
        num_diffusers=3,
        diffusers_per_epoch=2,
        seed=11,
    )
    cache_root = tmp_path / "cache"
    writer = Luo2022IntensityCacheWriter(
        cache_root,
        operator_id="fixed-depth-test-operator",
        assignment_sha=assignment["root_sha"],
        r0_fingerprint=canonical_sha256({"profile": "R0-test"}),
        expected_shape=(1, 2, 2),
        expected_rows=5,
    )
    first_raw = torch.tensor([1.0, 2.0], dtype=torch.float64).view(2, 1, 1, 1).expand(-1, 1, 2, 2)
    first_shard = writer.append_shard(first_raw, assignment["rows"][:2])

    recovered = Luo2022IntensityCacheWriter(
        cache_root,
        operator_id="fixed-depth-test-operator",
        assignment_sha=assignment["root_sha"],
        r0_fingerprint=canonical_sha256({"profile": "R0-test"}),
        expected_shape=(1, 2, 2),
        expected_rows=5,
    )
    repeated_shard = recovered.append_shard(
        {"raw_intensity": first_raw},
        assignment["rows"][:2],
    )
    assert repeated_shard == first_shard
    assert recovered.manifest["row_count"] == 2
    assert len(recovered.manifest["shards"]) == 1

    second_raw = torch.tensor([4.0, 5.0, 10.0]).view(3, 1, 1, 1).expand(-1, 1, 2, 2)
    recovered.append_shard(second_raw, assignment["rows"][2:])
    manifest = recovered.finalize()
    repeated_manifest = recovered.finalize()

    assert manifest == repeated_manifest
    assert manifest["schema_version"] == LUO2022_INTENSITY_CACHE_SCHEMA
    assert manifest["status"] == "complete"
    assert manifest["operator_id"] == "fixed-depth-test-operator"
    assert manifest["assignment_sha"] == assignment["root_sha"]
    assert manifest["shape"] == [1, 2, 2]
    assert manifest["dtype"] == "float32"
    assert manifest["scale"] == {
        "method": "global_dataset_max",
        "value": 10.0,
        "observed_max": 10.0,
        "fitted_split": "train",
        "scope": "dataset_scalar",
        "applied_at": "dataset_read",
        "reuse_mode": "local_training_fit",
    }
    assert all("sha256" in shard for shard in manifest["shards"])
    assert manifest == verify_luo2022_intensity_cache(cache_root)

    dataset = Luo2022CachedIntensityDataset(cache_root)
    assert len(dataset) == 5
    assert set(dataset[0]) == {
        "input_intensity",
        "object_id",
        "label",
        "diffuser_id",
        "training_epoch",
        "within_epoch_index",
        "row_id",
    }
    assert dataset[0]["input_intensity"].shape == (1, 2, 2)
    assert dataset[0]["input_intensity"].dtype == torch.float32
    assert torch.allclose(dataset[0]["input_intensity"], torch.full((1, 2, 2), 0.1))
    assert torch.allclose(dataset[1]["input_intensity"], torch.full((1, 2, 2), 0.2))
    assert torch.allclose(dataset[4]["input_intensity"], torch.ones(1, 2, 2))
    assert int(dataset[3]["object_id"]) == assignment["rows"][3]["object_id"]
    assert int(dataset[3]["label"]) == assignment["rows"][3]["label"]


def test_validation_cache_requires_and_reuses_frozen_training_scale(tmp_path) -> None:
    train_assignment = build_luo2022_fixed_depth_assignment(
        [0, 1], num_diffusers=2, seed=5
    )
    train_root = tmp_path / "train-cache"
    train_writer = Luo2022IntensityCacheWriter(
        train_root,
        operator_id="shared-operator",
        assignment_sha=train_assignment["root_sha"],
        r0_fingerprint="r0-test",
        split="train",
    )
    train_writer.append_shard(
        torch.tensor([2.0, 10.0]).view(2, 1, 1, 1).expand(-1, 1, 2, 2),
        train_assignment["rows"],
    )
    train_manifest = train_writer.finalize()
    frozen_scale = make_luo2022_frozen_scale(train_manifest)

    validation_assignment = build_luo2022_fixed_depth_assignment(
        [2, 3], num_diffusers=2, seed=6, object_id_offset=50_000
    )
    validation_root = tmp_path / "validation-cache"
    validation_writer = Luo2022IntensityCacheWriter(
        validation_root,
        operator_id="shared-operator",
        assignment_sha=validation_assignment["root_sha"],
        r0_fingerprint="r0-test",
        split="validation",
    )
    validation_writer.append_shard(
        torch.tensor([5.0, 20.0]).view(2, 1, 1, 1).expand(-1, 1, 2, 2),
        validation_assignment["rows"],
    )
    with pytest.raises(ValueError, match="requires an explicit frozen_scale"):
        validation_writer.finalize()
    validation_manifest = validation_writer.finalize(frozen_scale=frozen_scale)

    assert validation_manifest["scale"]["value"] == 10.0
    assert validation_manifest["scale"]["fitted_split"] == "train"
    assert validation_manifest["scale"]["reuse_mode"] == "frozen_training_statistic"
    assert validation_manifest["scale"]["provenance"] == {
        "source": "training_cache",
        "source_cache_root_fingerprint": train_manifest["root_fingerprint"],
        "operator_id": "shared-operator",
    }
    validation_dataset = Luo2022CachedIntensityDataset(validation_root)
    assert torch.allclose(validation_dataset[0]["input_intensity"], torch.full((1, 2, 2), 0.5))
    assert torch.allclose(validation_dataset[1]["input_intensity"], torch.full((1, 2, 2), 2.0))

    wrong_operator_scale = {
        **frozen_scale,
        "provenance": {**frozen_scale["provenance"], "operator_id": "other"},
    }
    with pytest.raises(ValueError, match="operator_id"):
        validation_writer.finalize(frozen_scale=wrong_operator_scale)


def test_cache_rejects_intensity_shard_tampering(tmp_path) -> None:
    assignment = build_luo2022_fixed_depth_assignment(
        [0, 1], num_diffusers=2, seed=3
    )
    cache_root = tmp_path / "cache"
    writer = Luo2022IntensityCacheWriter(
        cache_root,
        operator_id="operator",
        assignment_sha=assignment["root_sha"],
        r0_fingerprint="r0-test",
    )
    writer.append_shard(torch.ones(2, 1, 2, 2), assignment["rows"])
    manifest = writer.finalize()

    shard_path = cache_root / manifest["shards"][0]["intensity_file"]
    damaged = bytearray(shard_path.read_bytes())
    damaged[0] ^= 0x01
    shard_path.write_bytes(damaged)

    with pytest.raises(ValueError, match="integrity check failed"):
        verify_luo2022_intensity_cache(cache_root)
    with pytest.raises(ValueError, match="integrity check failed"):
        Luo2022CachedIntensityDataset(cache_root)


def test_cache_rejects_manifest_tampering_and_shape_mismatch(tmp_path) -> None:
    assignment = build_luo2022_fixed_depth_assignment(
        [0, 1], num_diffusers=2, seed=4
    )
    cache_root = tmp_path / "cache"
    writer = Luo2022IntensityCacheWriter(
        cache_root,
        operator_id="operator",
        assignment_sha=assignment["root_sha"],
        r0_fingerprint="r0-test",
        expected_shape=(1, 2, 2),
    )
    with pytest.raises(ValueError, match="sample shape"):
        writer.append_shard(torch.ones(2, 1, 3, 3), assignment["rows"])
    writer.append_shard(torch.ones(2, 1, 2, 2), assignment["rows"])
    writer.finalize()

    manifest_path = cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["operator_id"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="root fingerprint mismatch"):
        verify_luo2022_intensity_cache(cache_root)
