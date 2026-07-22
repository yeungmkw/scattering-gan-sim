import torch

from experiment import adversarial_loss
from losses import (
    ReconstructionLossWeights,
    luo2022_d2nn_energy_breakdown_per_pair,
    luo2022_d2nn_components_per_pair,
    luo2022_d2nn_loss,
    masked_pearson_per_image,
    pearson_per_image,
    reconstruction_loss,
)
from patchgan import PatchDiscriminator
from unet import UNetReconstructor


def test_unet_reconstructor_and_loss_shapes() -> None:
    model = UNetReconstructor(base_channels=4)
    corrupted = torch.rand(2, 1, 16, 16)
    clean = torch.rand(2, 1, 16, 16)

    prediction = model(corrupted)
    loss, components = reconstruction_loss(
        prediction,
        clean,
        ReconstructionLossWeights(l1=1.0, negative_pearson=0.1, ssim=0.1),
    )

    assert prediction.shape == clean.shape
    assert loss.ndim == 0
    assert {"l1", "negative_pearson", "ssim", "total"}.issubset(components)


def test_patch_discriminator_outputs_patch_logits() -> None:
    discriminator = PatchDiscriminator(base_channels=4)
    corrupted = torch.rand(2, 1, 32, 32)
    candidate = torch.rand(2, 1, 32, 32)

    logits = discriminator(corrupted, candidate)

    assert logits.shape[0] == 2
    assert logits.shape[1] == 1
    assert logits.ndim == 4


def test_adversarial_loss_is_scalar() -> None:
    logits = torch.zeros(2, 1, 4, 4)

    loss = adversarial_loss(logits, target_is_real=True)

    assert loss.ndim == 0
    assert loss > 0


def test_luo2022_loss_uses_raw_intensity_and_updates_every_pair() -> None:
    target = torch.zeros(2, 1, 8, 8)
    target[:, :, 2:6, 2:6] = 1.0
    output = (target[:, None, 0] * 0.8 + 0.1).expand(2, 3, 8, 8).clone()
    output.requires_grad_()

    loss, components = luo2022_d2nn_loss(output, target)
    loss.backward()

    per_pair = luo2022_d2nn_components_per_pair(output.detach(), target)
    assert loss.ndim == 0
    assert set(components) == {"total", "negative_pearson", "energy", "pearson"}
    assert all(value.shape == (2, 3) for value in per_pair.values())
    assert all(
        torch.allclose(components[name], per_pair[name].mean(), atol=1e-7)
        for name in components
    )
    assert torch.allclose(components["pearson"], torch.tensor(1.0), atol=1e-6)
    assert output.grad is not None
    assert torch.all(output.grad.flatten(start_dim=2).abs().sum(dim=2) > 0)


def test_masked_pearson_matches_full_canvas_and_ignores_pixels_outside_roi() -> None:
    target = torch.tensor(
        [
            [
                [0.0, 0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6, 0.7],
                [0.8, 0.9, 1.0, 0.2],
                [0.3, 0.4, 0.5, 0.6],
            ]
        ]
    )
    prediction = target.clone()
    full_mask = torch.ones_like(target, dtype=torch.bool)
    center_mask = torch.zeros_like(target, dtype=torch.bool)
    center_mask[..., 1:3, 1:3] = True
    corrupted_outside = prediction.clone()
    corrupted_outside[~center_mask] = -10.0

    assert torch.allclose(
        masked_pearson_per_image(prediction, target, full_mask),
        pearson_per_image(prediction, target),
    )
    assert torch.allclose(
        masked_pearson_per_image(corrupted_outside, target, center_mask),
        torch.ones(1),
        atol=1e-6,
    )


def test_luo2022_energy_breakdown_preserves_frozen_loss_and_scales_linearly() -> None:
    target = torch.zeros(2, 1, 8, 8)
    target[:, :, 2:6, 2:6] = 1.0
    output = torch.rand(2, 3, 8, 8)

    components = luo2022_d2nn_components_per_pair(output, target)
    energy = luo2022_d2nn_energy_breakdown_per_pair(output, target)
    scaled_components = luo2022_d2nn_components_per_pair(10.0 * output, target)
    support = (target[:, 0] > 0).to(dtype=output.dtype)[:, None].expand_as(output)
    support_pixels = support.sum(dim=(-2, -1)).clamp_min(1e-8)
    frozen_legacy_energy = (
        1.0 * ((1.0 - support) * output).sum(dim=(-2, -1))
        - 0.5 * (support * output).sum(dim=(-2, -1))
    ) / support_pixels

    assert torch.allclose(energy["energy"], components["energy"])
    assert torch.equal(energy["energy"], frozen_legacy_energy)
    assert torch.allclose(
        energy["inside_per_support_pixel"],
        (target[:, 0, 2:6, 2:6].new_zeros(2, 3) + 0.0)
        + (
            output[:, :, 2:6, 2:6].sum(dim=(-2, -1))
            / energy["support_pixels"]
        ),
    )
    assert torch.allclose(scaled_components["pearson"], components["pearson"], atol=1e-6)
    assert torch.allclose(scaled_components["energy"], 10.0 * components["energy"])
