import math

import pytest
import torch

from metrics import (
    digit_group_statistics,
    pair_level_tail_statistics,
    paired_diffuser_delta_statistics,
    per_image_reconstruction_metrics,
    psnr_per_image,
    scalar_summary,
    two_level_diffuser_summary,
)


def test_per_image_metrics_match_hand_calculated_psnr_and_pearson() -> None:
    target = torch.tensor(
        [
            [[[0.0, 0.25], [0.5, 1.0]]],
            [[[0.0, 0.25], [0.5, 1.0]]],
        ],
        dtype=torch.float64,
    )
    prediction = target.clone()
    prediction[0] = prediction[0] + 0.1
    prediction[1, 0, 0, 0] = -10.0

    metrics = per_image_reconstruction_metrics(
        prediction,
        target,
        ssim_window_size=3,
    )

    assert set(metrics) == {
        "psnr",
        "ssim",
        "pearson_full_canvas",
        "pearson_target_support",
    }
    assert all(value.shape == (2,) for value in metrics.values())
    assert metrics["psnr"][0].item() == pytest.approx(20.0)
    assert metrics["pearson_full_canvas"][0].item() == pytest.approx(1.0)
    assert metrics["pearson_target_support"].tolist() == pytest.approx([1.0, 1.0])
    assert metrics["pearson_full_canvas"][1].item() < 1.0

    expected_psnr = torch.tensor(
        [20.0, 10.0 * math.log10(1.0 / 0.04)],
        dtype=torch.float64,
    )
    assert torch.allclose(
        psnr_per_image(
            torch.tensor([[[[0.1, 0.1]]], [[[0.2, 0.2]]]], dtype=torch.float64),
            torch.zeros(2, 1, 1, 2, dtype=torch.float64),
        ),
        expected_psnr,
    )


def test_scalar_summary_uses_sample_sd_se_and_normal_ci() -> None:
    summary = scalar_summary([1.0, 2.0, 3.0, 4.0])
    expected_sd = math.sqrt(5.0 / 3.0)
    expected_se = expected_sd / 2.0

    assert summary["count"] == 4
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["sample_std"] == pytest.approx(expected_sd)
    assert summary["standard_error"] == pytest.approx(expected_se)
    assert summary["ci95_normal"] == pytest.approx(
        [2.5 - 1.96 * expected_se, 2.5 + 1.96 * expected_se]
    )
    assert summary["minimum"] == 1.0
    assert summary["maximum"] == 4.0


def test_scalar_summary_single_sample_has_no_invented_uncertainty() -> None:
    summary = scalar_summary(torch.tensor([7.5]))

    assert summary == {
        "count": 1,
        "mean": 7.5,
        "sample_std": None,
        "standard_error": None,
        "ci95_normal": None,
        "minimum": 7.5,
        "maximum": 7.5,
    }
    with pytest.raises(ValueError, match="empty"):
        scalar_summary([])


def test_two_level_summary_uses_diffuser_means_or_no_diffuser_objects() -> None:
    diffuser_summary = two_level_diffuser_summary(
        [1.0, 3.0, 10.0, 14.0],
        diffuser_ids=["d0", "d0", "d1", "d1"],
    )

    assert diffuser_summary["aggregation_unit"] == "diffuser"
    assert diffuser_summary["pair_count"] == 4
    assert diffuser_summary["unit_count"] == 2
    assert [row["diffuser_id"] for row in diffuser_summary["per_diffuser"]] == ["d0", "d1"]
    assert [row["mean"] for row in diffuser_summary["per_diffuser"]] == [2.0, 12.0]
    assert diffuser_summary["statistics"]["mean"] == 7.0
    assert diffuser_summary["statistics"]["sample_std"] == pytest.approx(math.sqrt(50.0))

    no_diffuser_summary = two_level_diffuser_summary([1.0, 3.0], diffuser_ids=None)
    assert no_diffuser_summary["aggregation_unit"] == "object"
    assert no_diffuser_summary["unit_count"] == 2
    assert no_diffuser_summary["per_diffuser"] == []
    assert no_diffuser_summary["statistics"]["mean"] == 2.0
    assert no_diffuser_summary["statistics"]["sample_std"] == pytest.approx(math.sqrt(2.0))


def test_pair_tail_is_cvar5_and_linear_fifth_percentile_with_stable_ties() -> None:
    summary = pair_level_tail_statistics(range(21))

    assert summary["pair_count"] == 21
    assert summary["bottom_count"] == 2
    assert summary["cvar5_mean"] == 0.5
    assert summary["percentile_5"] == 1.0
    assert summary["bottom_indices"] == [0, 1]

    tied = pair_level_tail_statistics([0.0, 0.0, 0.0, *([1.0] * 18)])
    assert tied["bottom_indices"] == [0, 1]


def test_digit_groups_cover_zero_through_nine_and_retain_two_levels() -> None:
    groups = digit_group_statistics(
        [1.0, 3.0, 5.0, 9.0],
        [0, 0, 1, 1],
        diffuser_ids=["d0", "d1", "d0", "d1"],
    )

    assert list(groups) == [str(digit) for digit in range(10)]
    assert groups["0"]["aggregation_unit"] == "diffuser"
    assert groups["0"]["statistics"]["mean"] == 2.0
    assert groups["1"]["statistics"]["mean"] == 7.0
    assert groups["2"]["pair_count"] == 0
    assert groups["2"]["statistics"]["count"] == 0
    assert groups["2"]["statistics"]["ci95_normal"] is None


def test_paired_delta_matches_diffuser_ids_and_summarizes_comparison_minus_reference() -> None:
    summary = paired_diffuser_delta_statistics(
        {"d1": 2.0, "d0": 1.0},
        {"d0": 4.0, "d1": 3.0},
    )

    assert summary["delta_definition"] == "comparison_minus_reference"
    assert summary["diffuser_ids"] == ["d1", "d0"]
    assert summary["deltas"] == [1.0, 3.0]
    assert summary["statistics"]["mean"] == 2.0
    assert summary["statistics"]["sample_std"] == pytest.approx(math.sqrt(2.0))

    with pytest.raises(ValueError, match="do not match"):
        paired_diffuser_delta_statistics({"d0": 1.0}, {"d1": 2.0})
