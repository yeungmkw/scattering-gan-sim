# System Overview

## Purpose

The current project is a reusable exploratory simulation system for imaging
through a scattering-like optical channel. It is not a generic image-to-image
GAN demo. The core question is whether a neural reconstructor can recover a
clean target from a polluted coherent intensity observation, and whether a
PatchGAN refinement stage improves reconstruction without hiding fidelity loss.

## Current Pipeline

```text
clean MNIST target
  -> zero-phase coherent field encoding
  -> phase screen or amplitude-particle corruption
  -> free-space propagation
  -> fixed single-layer D2NN intensity readout
  -> U-Net reconstruction
  -> optional PatchGAN refinement
  -> metrics, figures, checkpoints, manifest
```

The current validated GPU run uses the `phase` corruption path. The same code
also supports `particles` for small checks and experiment variants.

## Main Files

| File | Role |
|---|---|
| `experiment.py` | Main CLI and experiment orchestration: `d2nn`, `unet`, `gan`, `compare`, `full`. |
| `d2nn.py` | Coherent field conversion, phase screen, particle mask, angular-spectrum propagation, and single-layer D2NN. |
| `coherent_data.py` | Deterministic paired coherent samples with clean target, dirty intensity, dirty phase, D2NN intensity, and diffuser id. |
| `unet.py` | U-Net reconstructor `G(y) -> x_hat`. |
| `patchgan.py` | Conditional discriminator for `condition + candidate` reconstruction pairs. |
| `losses.py` | L1, negative Pearson, SSIM-like, and Fourier loss terms. |
| `metrics.py` | L1, MSE, PSNR, SSIM, and Pearson evaluation metrics. |
| `runtime.py` | Device selection, seeding, JSON output, and run-directory preparation. |

## Standard Commands

Inspect the optical path:

```bash
uv run python -m experiment d2nn \
  --output-dir outputs/d2nn_inspection \
  --download \
  --corruption phase
```

Run the full U-Net and GAN comparison:

```bash
uv run python -m experiment full \
  --output-dir outputs/gpu_phase_comparison \
  --download \
  --corruption phase \
  --device cuda \
  --unet-epochs 20 \
  --gan-epochs 10 \
  --batch-size 32 \
  --train-limit 2048 \
  --eval-limit 256 \
  --base-channels 16
```

## Claim Boundary

The current system proves that the coherent simulation and neural
reconstruction pipeline runs end-to-end and can train a useful inverse model.
It does not prove hardware validity. The omitted effects remain PSF
calibration, hardware alignment, detector calibration, fabrication constraints,
and any optical implementation of the GAN.
