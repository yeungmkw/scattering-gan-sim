import pytest
import torch

from metrics import reconstruction_metrics


def test_psnr_is_invariant_to_batch_partition() -> None:
    target = torch.zeros(4, 1, 2, 2)
    prediction = torch.zeros_like(target)
    prediction[0] = 0.1
    prediction[1] = 0.2
    prediction[2] = 0.6
    prediction[3] = 0.9

    all_at_once = reconstruction_metrics(prediction, target)["psnr"]
    in_two_batches = sum(
        reconstruction_metrics(prediction[index : index + 2], target[index : index + 2])["psnr"] * 2
        for index in range(0, 4, 2)
    ) / 4
    one_at_a_time = sum(
        reconstruction_metrics(prediction[index : index + 1], target[index : index + 1])["psnr"]
        for index in range(4)
    ) / 4

    assert all_at_once == pytest.approx(in_two_batches)
    assert all_at_once == pytest.approx(one_at_a_time)
