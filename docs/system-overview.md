# System Overview

## Document Scope

This document originally described the legacy digital-reconstruction
prototype. That path remains reusable, but it is no longer the only validated
system. A separate paper-aligned `luo2022_r0` profile now implements the
four-layer trainable diffractive baseline without U-Net or GAN.

## Purpose

The current project is a reusable exploratory simulation system for imaging
through a scattering-like optical channel. It is not a generic image-to-image
GAN demo. The core question is whether a neural reconstructor can recover a
clean target from a polluted coherent intensity observation, and whether a
PatchGAN refinement stage improves reconstruction without hiding fidelity loss.

## Legacy Digital-Reconstruction Pipeline

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

The historical validated GPU run uses the `phase` corruption path. The same code
also supports `particles` for small checks and experiment variants.

## Paper-Aligned R0 Status

The terahertz four-layer R0 run is sealed as a
`reproduction-inspired result`:

- the paper-aligned correlated diffuser, propagation geometry, four
  trainable phase layers, and published loss are implemented;
- the full 100-epoch `n=20` baseline completed;
- all 2,000 training diffusers, 20 unseen diffusers, and the no-diffuser
  control were evaluated on 10,000 test objects per diffuser;
- target-support PCC is the primary architecture-comparison diagnostic, while
  full-canvas PCC is retained for historical regression;
- the known/new memory-effect ordering and the trained-versus-direct control
  gap are present;
- the remaining depth-trend gate requires independent 2-layer and 5-layer
  runs.

Exact equality with every paper figure is not the goal. The project requires
the key populations, controls, trends, and provenance needed to draw and
audit a compact reproduction-style figure set.

## Main Files

| File | Role |
|---|---|
| `experiment.py` | Main CLI and experiment orchestration: `d2nn`, `unet`, `gan`, `compare`, `full`. |
| `d2nn.py` | Coherent field conversion, legacy phase/particle paths, Rayleigh-Sommerfeld propagation, correlated diffusers, and trainable D2NN layers. |
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

The project now has two distinct evidence levels: an exploratory
U-Net/PatchGAN pipeline and a sealed four-layer terahertz R0 baseline. Neither
proves hardware validity. Omitted effects still include hardware alignment,
detector calibration, fabrication constraints, and any optical implementation
of the GAN. Unpublished author details also prevent calling R0 an exact
numerical reproduction.
