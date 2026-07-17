# Research Roadmap

## Phase 0: Provenance Contract

Completed on July 17, 2026:

1. Experiment classes now follow one rule: fixed phase/U-Net is `E0`,
   multi- or unseen-diffuser phase runs are `E1`, GAN adds `E2`, and particle
   corruption is `E3`.
2. Every new training run writes a schema-v1 `config.json` before training and
   references it from `manifest.json`.
3. U-Net and GAN use the same configurable reconstruction-loss weights.
4. Metrics remain in a flat `metrics.json`, while the manifest and config
   record the aggregation protocol, including mean per-image PSNR.
5. The historical GPU phase comparison metadata and documentation were
   synchronized without changing its checkpoints or metric files.

## Immediate Next Steps

The immediate task is a **terahertz reproduction baseline** for
*Computational imaging without a computer*. The baseline must match the
paper's optical architecture closely enough that later modifications can be
evaluated against it.

The published parameters and frozen implementation choices are encoded in
[`configs/luo2022_r0.json`](../configs/luo2022_r0.json). Code changes must cite
the corresponding paper equation where applicable. Internal equation audits
and reproduction working notes remain local rather than being published with
the repository.

1. Preserve the current independent uniform phase-screen path as the named
   control profile `iid_phase_v1`.
2. Add a `luo2022_phase_v1` profile inspired by *Computational imaging without
   a computer*: correlated random height maps, wavelength/material-dependent
   phase modulation, and explicit propagation geometry.
3. Align the order of object modulation, free-space propagation, diffuser
   modulation, diffractive layer, and detector sampling. Preserve raw detector
   intensity separately from the normalized neural-network input.
4. Implement the paper's trainable four-layer phase-only D2NN without a U-Net
   or GAN in the reconstruction path.
5. Add deterministic diffuser identities and the paper-aligned training and
   testing protocol before comparing neural architectures.
6. Validate the existing angular-spectrum propagator against a
   Rayleigh-Sommerfeld reference on representative sampled fields, including
   aliasing, padding, and energy checks.
7. Run small-resolution checks first, followed by a declared full-resolution
   terahertz reproduction run. Compare its correlation, image quality, optical
   efficiency, and diffuser-generalization results with the paper.
8. Only after the reproduction gap is measured should the project add U-Net,
   PatchGAN, reduced D2NN depth, or other optimizations.

## Reproduction Boundary

The paper's equations and simulation procedure are implementation inputs, but
its reported results are not treated as a reusable pretrained system.

- The project will first reproduce the paper's **forward-physics contract**:
  diffuser statistics, phase conversion, propagation distances, detector
  intensity, and diffuser sampling protocol.
- The reference neural architecture is the paper's four-layer diffractive
  neural network (D2NN), trained numerically but performing inference through
  optical propagation. It is not a digital U-Net.
- The current package structure and experiment CLI remain in place. A
  paper-aligned profile is added to the existing system rather than creating a
  separate root-level reproduction project.
- The current fixed single-layer D2NN plus U-Net path is neither the paper
  baseline nor sufficient evidence of an improvement. It remains an
  exploratory engineering path until the four-layer reference is available.
- Until the four-layer all-optical D2NN is implemented and trained under a
  sufficiently matched protocol, results must be called paper-aligned or
  reproduction-inspired, not a paper reproduction.
- Reduced spatial grids may be used for tests and small runs. Claim-level
  comparisons require a declared full-resolution profile and matching geometry.
- Visible-light work begins only after the terahertz reproduction and
  controlled optimization comparisons. It is the bridge to the available
  laboratory hardware, not a shortcut around the terahertz reference.

## Controlled Optimization Ladder

Every optimization must change one declared factor while keeping the
terahertz forward model, dataset split, diffuser set, training budget, seeds,
and evaluation metrics fixed.

| Level | System | Purpose |
|---|---|---|
| R0 | paper-aligned four-layer D2NN only | reproduce the reference result |
| R1 | R0 plus supervised U-Net refinement | measure the marginal effect of the digital decoder |
| R2 | same R1 generator plus PatchGAN loss | measure the marginal effect of adversarial training |
| R3 | reduced-depth D2NN variants | test optical-layer efficiency under the same protocol |
| R4 | other losses or architecture changes, one at a time | attribute each claimed improvement |

R2 must be compared with R1, not only with R0. Otherwise a gain cannot be
attributed specifically to the GAN because the U-Net and adversarial objective
would have changed at the same time.

## Visible-Light Translation

Visible-light simulation and optimization begin after the R0-R4 terahertz
comparison is stable.

1. First reproduce the terahertz result and quantify the gap from the paper.
2. Establish which optimization provides a real gain under the same terahertz
   conditions.
3. Scale the validated system to visible wavelengths in an ideal,
   wavelength-normalized simulation.
4. Add laboratory constraints: available source, spatial light modulator or
   fabricated phase-mask properties, material dispersion, detector pixel size,
   feature size, propagation distance, alignment error, and sensor noise.
5. Re-optimize the optical geometry and network for visible-light hardware
   rather than assuming that changing only the wavelength preserves accuracy.

## Experiment Matrix

The `R0`-`R4` levels above define the causal comparison sequence. The
`E0`-`E4` identifiers below classify experiment families and do not imply
execution order; the paper reproduction `R0` belongs to the trainable-D2NN
family `E4`.

| ID | Goal | Forward model | Neural model | Required evidence |
|---|---|---|---|---|
| E0 | Prove inverse training works | fixed coherent phase screen | U-Net | loss curve, eval metrics, sample grid |
| E1 | Test diffuser generalization | train diffusers vs held-out diffusers | U-Net | separate seen/unseen metrics |
| E2 | Test adversarial refinement | same coherent forward model | U-Net + PatchGAN | U-Net vs GAN metrics and images |
| E3 | Stress optical corruption | phase screen and particle mask variants | U-Net, optional GAN | per-corruption metrics |
| E4 | Explore optical front-end | trainable D2NN or hybrid decoder | U-Net/GAN-assisted | ablation showing optical stage effect |

## Paper-Motivated Hypotheses

- Memory-effect and PSF literature motivates controlled forward models and
  strict statement of what physics is included or omitted.
- Deep speckle reconstruction papers motivate supervised reconstruction from
  corrupted intensity observations and held-out scattering conditions.
- Adversarial/YGAN-style reconstruction motivates PatchGAN only after the
  supervised reconstructor is stable.
- Diffractive optical network papers motivate a later trainable D2NN or hybrid
  optical/digital front-end, not an immediate hardware claim.

## Claim Discipline

Use these labels consistently:

| Label | Meaning |
|---|---|
| small run | Code path runs and produces expected artifacts. |
| exploratory result | A real training run with useful but incomplete evidence. |
| reproduction-inspired result | Experiment matrix and controls are close to a paper mechanism. |
| claim candidate | Multiple seeds, splits, ablations, and failure cases support a narrow statement. |

The current GPU phase comparison is an exploratory result.
