import torch

from experiment import adversarial_loss
from losses import ReconstructionLossWeights, reconstruction_loss
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
