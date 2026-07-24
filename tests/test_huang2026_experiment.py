"""CLI, configuration, routing, and checkpoint tests for the Huang profile."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
import shutil

import pytest
import torch
from torch.utils.data import Dataset

import experiment
from coherent_data import (
    Huang2026CoherenceSampler,
    Huang2026VisibleDataset,
)
from d2nn import (
    Luo2022FourLayerD2NN,
    Luo2022OpticsConfig,
)


class _TinyMNIST(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, count: int, *, offset: int = 0) -> None:
        self.count = count
        self.offset = offset

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        generator = torch.Generator().manual_seed(self.offset + index)
        image = torch.rand((1, 28, 28), generator=generator)
        image[:, :4, :] = 0.0
        image[:, -4:, :] = 0.0
        return image, index % 10


def _tiny_datasets(
    runtime_config: dict[str, object],
) -> tuple[Huang2026VisibleDataset, Huang2026VisibleDataset]:
    field_shape = tuple(int(value) for value in runtime_config["field_shape"])
    resized_shape = tuple(int(value) for value in runtime_config["resized_shape"])
    wavelengths = tuple(
        float(value) * 1e-9 for value in runtime_config["wavelengths_nm"]
    )
    mode = str(runtime_config["mode"])
    seed = int(runtime_config["seed"])
    correlation = float(runtime_config["diffuser"]["correlation_length_pixels"])
    coherence_length = float(
        runtime_config["coherence"]["coherence_length_pixels"]
    )

    def build(split: str, count: int, offset: int) -> Huang2026VisibleDataset:
        coherence = (
            Huang2026CoherenceSampler(
                field_shape,
                split=split,
                base_seed=seed,
                coherence_length_pixels=coherence_length,
            )
            if mode == "incoherent"
            else None
        )
        return Huang2026VisibleDataset(
            _TinyMNIST(count, offset=offset),
            split=split,
            correlation_length_pixels=correlation,
            base_seed=seed,
            resized_shape=resized_shape,
            canvas_shape=field_shape,
            illumination_mode=mode,
            wavelengths=wavelengths,
            coherence_sampler=coherence,
        )

    training = runtime_config["training"]
    return (
        build("train", int(training["train_limit"]), 100),
        build("blind_test", int(training["eval_limit"]), 10_000),
    )


def _args(
    output_dir: Path,
    *,
    action: str,
    mode: str = "coherent",
    extra: list[str] | None = None,
) -> object:
    values = [
        "d2nn",
        "--profile",
        "huang2026_visible",
        "--mode",
        mode,
        "--action",
        action,
        "--small-run",
        "--device",
        "cpu",
        "--grid-size",
        "8",
        "--input-size",
        "6",
        "--train-limit",
        "4",
        "--eval-limit",
        "2",
        "--batch-size",
        "2",
        "--output-dir",
        str(output_dir),
    ]
    values.extend(extra or [])
    return experiment.build_parser().parse_args(values)


def test_public_contracts_use_only_four_evidence_categories_and_inherit() -> None:
    allowed = {
        "paper_confirmed",
        "paper_inferred",
        "project_choice",
        "suspected_paper_typo",
    }
    for mode, path in experiment.DEFAULT_HUANG2026_CONFIGS.items():
        contract = experiment.load_huang2026_contract(path)
        assert contract["mode"] == mode

        def walk(node: object) -> None:
            if not isinstance(node, dict):
                return
            for key, value in node.items():
                if key.endswith("_evidence"):
                    assert value in allowed
                else:
                    walk(value)

        walk(contract)
    incoherent = experiment.load_huang2026_contract(
        experiment.DEFAULT_HUANG2026_CONFIGS["incoherent"]
    )
    assert incoherent["grid"]["shape"] == [400, 400]
    assert incoherent["coherence"]["nr_training"] == 20


def test_mode_and_explicit_config_mismatch_is_rejected_before_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "mismatch"
    args = _args(
        output,
        action="inspect",
        extra=[
            "--config-path",
            str(experiment.DEFAULT_HUANG2026_CONFIGS["incoherent"]),
        ],
    )
    with pytest.raises(ValueError, match="does not match"):
        experiment.dispatch(args)
    assert not output.exists()


def test_actual_inspect_cli_route_writes_s18_and_optical_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiment, "_build_huang2026_datasets", _tiny_datasets)
    output = tmp_path / "inspect"
    result = experiment.dispatch(_args(output, action="inspect"))

    assert result["profile_id"] == "huang2026_visible"
    assert result["slm_encoding"]["monotonic_first_order_amplitude"] is True
    for relative in (
        "config.json",
        "contract.json",
        "source_config.json",
        "manifest.json",
        "metrics.json",
        "slm_encoding.json",
        "sample_records.jsonl",
        "samples/inspection.png",
        "samples/slm_s18_phase_only_hologram.png",
    ):
        assert (output / relative).is_file()
    portable_contract = experiment.load_huang2026_contract(
        output / "contract.json"
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert experiment.canonical_sha256(portable_contract) == manifest[
        "source_contract_sha256"
    ]


def test_actual_multiwavelength_train_writes_first_wavelength_sample_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiment, "_build_huang2026_datasets", _tiny_datasets)
    output = tmp_path / "multiwavelength"
    result = experiment.dispatch(
        _args(
            output,
            action="train",
            mode="multiwavelength",
            extra=["--epochs", "1", "--checkpoint-interval", "1"],
        )
    )

    assert result["metrics"]["sample_panel_condition"] == {
        "mode": "multiwavelength",
        "nr": 1,
        "displayed_wavelength_nm": 491.0,
        "multiwavelength_display_policy": "first configured wavelength",
    }
    assert (output / "samples" / "ideal_evaluation.png").is_file()


def test_direct_and_lens_control_is_lazy_and_uses_explicit_operator_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiment, "_build_huang2026_datasets", _tiny_datasets)
    output = tmp_path / "controls"
    result = experiment.dispatch(
        _args(
            output,
            action="control",
            extra=["--control", "direct", "lens"],
        )
    )

    assert result["operator_bindings"] == {
        "direct": "VisibleDirectPropagationOperator",
        "lens": "ThinLensOperator",
    }
    assert result["nominal_total_path_matched"] is True
    assert (output / "samples" / "controls.png").is_file()
    assert (output / "control_sample_records.jsonl").is_file()
    assert not (output / "checkpoints" / "latest.pt").exists()


def test_evaluation_diffuser_assignment_is_batch_size_invariant(
    tmp_path: Path,
) -> None:
    contract = experiment.load_huang2026_contract(
        experiment.DEFAULT_HUANG2026_CONFIGS["coherent"]
    )
    args = _args(tmp_path / "unused", action="inspect")
    runtime = experiment.build_huang2026_runtime_config(
        contract,
        args,
        config_path=experiment.DEFAULT_HUANG2026_CONFIGS["coherent"],
        device=torch.device("cpu"),
    )
    runtime["training"]["eval_limit"] = 4
    _train, evaluation = _tiny_datasets(runtime)
    model = experiment._build_huang2026_model(runtime)
    diffuser = experiment._huang2026_diffuser(runtime)
    detector = experiment._huang2026_detector_response(runtime)
    results = []
    for batch_size in (1, 2, 4):
        binding = copy.deepcopy(runtime)
        binding["training"]["batch_size"] = batch_size
        results.append(
            experiment.evaluate_huang2026_model(
                model,
                evaluation,
                binding,
                device=torch.device("cpu"),
                diffuser=diffuser,
                detector=detector,
            )
        )

    expected_seeds = [
        row["diffuser_seed"] for row in results[0]["pcc"]["per_diffuser"]
    ]
    for result in results[1:]:
        assert [
            row["diffuser_seed"] for row in result["pcc"]["per_diffuser"]
        ] == expected_seeds
        assert result["pcc"]["dataset"]["mean"] == pytest.approx(
            results[0]["pcc"]["dataset"]["mean"],
            abs=1e-7,
        )


def test_checkpoint_resume_replays_epoch_boundary_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiment, "_build_huang2026_datasets", _tiny_datasets)
    common = [
        "--epochs",
        "2",
        "--checkpoint-interval",
        "1",
        "--lr",
        "0.01",
    ]
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    baseline_result = experiment.dispatch(
        _args(uninterrupted, action="train", extra=common)
    )
    independent_evaluation = tmp_path / "independent-evaluation"
    evaluation_result = experiment.dispatch(
        _args(
            independent_evaluation,
            action="evaluate",
            extra=[
                "--batch-size",
                "1",
                "--run-dir",
                str(uninterrupted),
            ],
        )
    )
    assert evaluation_result["metrics"]["object_count"] == 2
    evaluation_manifest = json.loads(
        (independent_evaluation / "manifest.json").read_text()
    )
    assert evaluation_manifest["checkpoint_binding"]["global_step"] == 4

    original_save = experiment.save_huang2026_checkpoint
    calls = 0

    def fault_after_last_batch(*args: object, **kwargs: object) -> object:
        nonlocal calls
        payload = original_save(*args, **kwargs)
        calls += 1
        if calls == 2:
            raise RuntimeError("fault after final batch checkpoint")
        return payload

    monkeypatch.setattr(
        experiment,
        "save_huang2026_checkpoint",
        fault_after_last_batch,
    )
    with pytest.raises(RuntimeError, match="fault after final batch"):
        experiment.dispatch(_args(resumed, action="train", extra=common))
    monkeypatch.setattr(experiment, "save_huang2026_checkpoint", original_save)
    resumed_result = experiment.dispatch(
        _args(resumed, action="train", extra=[*common, "--resume"])
    )

    baseline_checkpoint = torch.load(
        uninterrupted / "checkpoints" / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    resumed_checkpoint = torch.load(
        resumed / "checkpoints" / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert baseline_checkpoint["global_step"] == resumed_checkpoint["global_step"]
    assert baseline_checkpoint["history"] == resumed_checkpoint["history"]
    for name, value in baseline_checkpoint["model_state"].items():
        assert torch.equal(value, resumed_checkpoint["model_state"][name])
    assert (uninterrupted / "history.json").read_bytes() == (
        resumed / "history.json"
    ).read_bytes()
    assert (uninterrupted / "history.jsonl").read_bytes() == (
        resumed / "history.jsonl"
    ).read_bytes()
    assert (uninterrupted / "sample_records.jsonl").read_bytes() == (
        resumed / "sample_records.jsonl"
    ).read_bytes()
    baseline_manifest = json.loads(
        (uninterrupted / "manifest.json").read_text()
    )
    resumed_manifest = json.loads((resumed / "manifest.json").read_text())
    assert baseline_manifest["artifacts"]["history_journal"] == "history.jsonl"
    assert baseline_manifest["paper_equations"]["backpropagation"] == [
        "S12",
        "S15",
    ]
    assert (
        baseline_manifest["checkpoint_binding"].pop("file_sha256")
        != ""
    )
    assert resumed_manifest["checkpoint_binding"].pop("file_sha256") != ""
    assert baseline_manifest == resumed_manifest
    assert baseline_result["training"] == resumed_result["training"]

    completed_manifest = (resumed / "manifest.json").read_bytes()
    completed_history = (resumed / "history.json").read_bytes()
    committed_records = (resumed / "sample_records.jsonl").read_bytes()
    with (resumed / "sample_records.jsonl").open("ab") as handle:
        handle.write(b'{"uncommitted_crash_tail":')
    experiment.dispatch(
        _args(resumed, action="train", extra=[*common, "--resume"])
    )
    assert (resumed / "manifest.json").read_bytes() == completed_manifest
    assert (resumed / "history.json").read_bytes() == completed_history
    assert (resumed / "sample_records.jsonl").read_bytes() == committed_records

    with pytest.raises(ValueError, match="resume contract mismatch"):
        experiment.dispatch(
            _args(
                resumed,
                action="train",
                extra=[
                    "--epochs",
                    "2",
                    "--checkpoint-interval",
                    "1",
                    "--lr",
                    "0.02",
                    "--resume",
                ],
            )
        )


def test_checkpoint_and_journals_reject_tampering_or_truncation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiment, "_build_huang2026_datasets", _tiny_datasets)
    source = tmp_path / "source"
    common = ["--epochs", "1", "--checkpoint-interval", "1"]
    experiment.dispatch(_args(source, action="train", extra=common))

    checkpoint_tamper = tmp_path / "checkpoint-tamper"
    shutil.copytree(source, checkpoint_tamper)
    checkpoint_path = checkpoint_tamper / "checkpoints" / "latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["global_step"] += 1
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="integrity validation failed"):
        experiment.dispatch(
            _args(
                checkpoint_tamper,
                action="train",
                extra=[*common, "--resume"],
            )
        )

    records_missing = tmp_path / "records-missing"
    shutil.copytree(source, records_missing)
    (records_missing / "sample_records.jsonl").unlink()
    with pytest.raises(ValueError, match="missing or truncated"):
        experiment.dispatch(
            _args(records_missing, action="train", extra=[*common, "--resume"])
        )

    records_duplicate = tmp_path / "records-duplicate"
    shutil.copytree(source, records_duplicate)
    records_path = records_duplicate / "sample_records.jsonl"
    lines = records_path.read_bytes().splitlines(keepends=True)
    assert len(lines) >= 2
    lines[1] = lines[0]
    records_path.write_bytes(b"".join(lines))
    with pytest.raises(ValueError, match="sample-record"):
        experiment.dispatch(
            _args(records_duplicate, action="train", extra=[*common, "--resume"])
        )

    history_truncated = tmp_path / "history-truncated"
    shutil.copytree(source, history_truncated)
    history_path = history_truncated / "history.jsonl"
    history_path.write_bytes(history_path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="history journal"):
        experiment.dispatch(
            _args(history_truncated, action="train", extra=[*common, "--resume"])
        )


def test_frozen_luo_public_contracts_remain_byte_and_source_identical() -> None:
    config_hash = hashlib.sha256(
        Path("configs/luo2022_r0.json").read_bytes()
    ).hexdigest()
    optics_hash = hashlib.sha256(
        inspect.getsource(Luo2022OpticsConfig).encode()
    ).hexdigest()
    model_hash = hashlib.sha256(
        inspect.getsource(Luo2022FourLayerD2NN).encode()
    ).hexdigest()

    assert config_hash == "1ac74bd4f3358626f19b6248fc5c103ab6cf48bdf130f6ff20c63a4222bd2f53"
    assert optics_hash == "f7bbf5aef3e431bc949378b8687bb5acf185051af87dc9c2bdd75a9d7f0ee27c"
    assert model_hash == "33d90aebc13f1bd66bdb4be7c95da3ef72ad2739d9ab83e2a9aa6629183c2065"
