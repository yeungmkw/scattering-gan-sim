import torch
from torch.utils.data import TensorDataset

from coherent_data import (
    CoherentD2NNDataset,
    MaterializedCoherentDataset,
    materialize_coherent_dataset,
    prepare_luo2022_amplitude,
    prepare_luo2022_field,
    simulate_coherent_observation,
)


def test_simulate_coherent_observation_contract_and_determinism() -> None:
    clean = torch.rand(1, 16, 16)

    sample_a = simulate_coherent_observation(clean, corruption="phase", seed=5)
    sample_b = simulate_coherent_observation(clean, corruption="phase", seed=5)

    assert set(sample_a) == {"clean", "dirty_intensity", "dirty_phase", "d2nn_intensity"}
    for key in sample_a:
        assert sample_a[key].shape == (1, 16, 16)
        assert sample_a[key].dtype == torch.float32
        assert torch.allclose(sample_a[key], sample_b[key])
    assert float(sample_a["dirty_intensity"].min()) >= 0.0
    assert float(sample_a["dirty_intensity"].max()) <= 1.0
    assert float(sample_a["d2nn_intensity"].min()) >= 0.0
    assert float(sample_a["d2nn_intensity"].max()) <= 1.0


def test_simulate_coherent_observation_supports_particles() -> None:
    clean = torch.rand(1, 16, 16)

    sample = simulate_coherent_observation(clean, corruption="particles", seed=9)

    assert sample["dirty_phase"].shape == (1, 16, 16)
    assert sample["d2nn_intensity"].shape == (1, 16, 16)
    assert not torch.allclose(sample["d2nn_intensity"], clean)


def test_coherent_d2nn_dataset_wraps_tensor_dataset() -> None:
    images = torch.rand(3, 1, 16, 16)
    labels = torch.arange(3)
    dataset = CoherentD2NNDataset(TensorDataset(images, labels), corruption="phase", seed=4)

    sample = dataset[1]
    repeat = dataset[1]

    assert set(sample) == {"clean", "dirty_intensity", "dirty_phase", "d2nn_intensity", "diffuser_id", "label"}
    assert int(sample["label"]) == 1
    assert int(sample["diffuser_id"]) == 0
    assert sample["clean"].shape == (1, 16, 16)
    assert torch.allclose(sample["d2nn_intensity"], repeat["d2nn_intensity"])
    assert len(dataset._simulators) == 1
    simulator = next(iter(dataset._simulators.values()))
    assert len(simulator._d2nn_layers) == 1
    assert len(simulator._phase_screens) == 1


def test_coherent_d2nn_dataset_uses_explicit_diffuser_ids() -> None:
    images = torch.rand(4, 1, 16, 16)
    labels = torch.arange(4)
    dataset = CoherentD2NNDataset(TensorDataset(images, labels), corruption="phase", seed=4, diffuser_ids=(0, 2))

    diffuser_ids = {int(dataset[index]["diffuser_id"]) for index in range(len(dataset))}

    assert diffuser_ids == {0, 2}


def test_materialize_coherent_dataset_preserves_sample_values() -> None:
    images = torch.rand(3, 1, 16, 16)
    labels = torch.arange(3)
    dataset = CoherentD2NNDataset(TensorDataset(images, labels), corruption="particles", seed=8)
    expected = [dataset[index] for index in range(len(dataset))]

    materialized = materialize_coherent_dataset(dataset)

    assert len(materialized) == len(dataset)
    for index, expected_sample in enumerate(expected):
        actual_sample = materialized[index]
        assert set(actual_sample) == set(expected_sample)
        for key, expected_value in expected_sample.items():
            if torch.is_floating_point(expected_value):
                assert torch.allclose(actual_sample[key], expected_value)
            else:
                assert torch.equal(actual_sample[key], expected_value)


def test_materialized_coherent_dataset_constructor_copies_by_default() -> None:
    source = {"clean": torch.ones(2, 1, 4, 4)}
    materialized = MaterializedCoherentDataset(source)

    source["clean"][0].zero_()

    assert torch.all(materialized[0]["clean"] == 1)


def test_prepare_luo2022_input_resizes_and_center_pads_amplitude() -> None:
    image = torch.ones(2, 1, 28, 28)

    amplitude = prepare_luo2022_amplitude(
        image,
        resized_shape=(32, 32),
        canvas_shape=(48, 48),
    )
    field = prepare_luo2022_field(
        image,
        resized_shape=(32, 32),
        canvas_shape=(48, 48),
    )

    assert amplitude.shape == (2, 1, 48, 48)
    assert field.shape == (2, 48, 48)
    assert torch.all(amplitude[:, :, 8:40, 8:40] == 1)
    assert torch.all(amplitude[:, :, :8] == 0)
    assert torch.allclose(field.real, amplitude[:, 0])
