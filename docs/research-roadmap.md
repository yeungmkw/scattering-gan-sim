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

## Current Status and Next Decision

Updated July 24, 2026:

The terahertz four-layer `n=20` R0 baseline is complete and sealed. The
published parameters and frozen implementation choices are encoded in
[`configs/luo2022_r0.json`](../configs/luo2022_r0.json). The completed evidence
contains the key Figure-5 populations: all 2,000 training diffusers, the
epochs 1-99 subset, the final-10-epoch subset, the final-epoch 20, 20 unseen
diffusers, and a no-diffuser control, each evaluated on 10,000 test objects.
Direct/no-D2NN controls and target-support diagnostics are also available.

This is enough to establish reproduction capability and to draw a compact
reproduction-style figure set. The project does not require every figure or
pixel value in the paper to be recreated. Exact author-level numerical
equivalence remains impossible because several implementation and metric
details are unpublished.

The execution decision is:

1. **Freeze R0 and its diffuser model.** Do not tune unpublished parameters
   merely to chase digitized paper values, and do not repeat completed
   post-hoc evaluations without a new, declared evidence gap.
2. **Run the fixed-four-layer backend ablation next.** B0 measures the same
   lightweight supervised U-Net without the D2NN; R1 adds supervised U-Net
   refinement after frozen R0; R2 adds PatchGAN training after the same R1
   warmup. This isolates the optical-front-end and adversarial contributions
   before another optical model is trained.
3. **Keep R1 and R2 causally matched.** They share a 20-epoch supervised
   warmup and then branch for 10 continued epochs: supervised-only for R1 and
   adversarial for R2. B0 trains for 30 supervised epochs with the same U-Net
   capacity, batch size, seed, and reconstruction objective.
4. **Defer the depth scan.** Independent two-layer and five-layer models remain
   useful for a later depth trend, but are not the current gate. No reduced-depth
   claim is permitted from the fixed-four-layer study.
5. **Defer visible-light optimization until the backend result and later depth
   evidence are stable.** Visible-light work then starts from measured hardware
   geometry, quantization, alignment, efficiency, and detector constraints.

The public, executable source of truth for this stage is
[`configs/luo2022_fixed4_backend.json`](../configs/luo2022_fixed4_backend.json).
Its status is `exploratory fixed-depth backend ablation`.

The `n=1`, `n=10`, and `n=15` paper curves require separate models. They are
optional later controls, not blockers for closing R0 or starting the
depth/backend program.

### Key Data and Figure Readiness

| Evidence or figure family | Current status | Decision |
|---|---|---|
| Published equations, geometry, loss, and `n=20` protocol | complete | frozen as the R0 contract |
| 100-epoch training history and final checkpoint provenance | complete | no repeat training |
| Figure-5-style `n=20` populations and error bars | complete | numerical panels can be drawn now |
| Figure-6-style known/new/no-diffuser comparison | complete for the `n=20` slice | sufficient for the compact reproduction figure set |
| Direct/no-D2NN control | complete | retain beside the trained network result |
| Full-canvas, center, and target-support PCC | complete | target support is primary; full canvas is regression |
| Example outputs and phase-map panels | generation path exists | regenerate read-only from the frozen checkpoint when a final figure layout is chosen |
| Fixed-four-layer backend ablation | contract defined for B0, R1, and R2 | current experiment |
| Figure-7-style depth trend | 4-layer complete; 2-layer and 5-layer missing | deferred until the backend ablation is complete |
| `n=1`, `n=10`, `n=15` memory curves | missing independent models | optional later, not an R0 blocker |
| Hardware, resolution-target, pruning, and lens panels | not part of the current numerical scope | do not block the next research stage |

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
- The fixed single-layer D2NN plus U-Net path is neither the paper baseline
  nor sufficient evidence of an improvement. It remains an exploratory
  engineering path and must be compared against the now-available four-layer
  reference before supporting an optimization claim.
- The four-layer all-optical D2NN is now implemented and trained. Because the
  author implementation, exact ROI, initialization, and source arrays remain
  unpublished, the sealed result remains `reproduction-inspired`, not an
  exact numerical reproduction.
- Reduced spatial grids may be used for tests and small runs. Claim-level
  comparisons require a declared full-resolution profile and matching geometry.
- Visible-light work begins only after the terahertz reproduction and
  controlled optimization comparisons. It is the bridge to the available
  laboratory hardware, not a shortcut around the terahertz reference.

## Fixed-Four-Layer Backend Ablation

The current comparison keeps the R0 four-layer optical model frozen. Cached
operator outputs are raw `float32` detector intensities. Scaling is fitted on
the training split only, using a separate global dataset maximum for each
operator, and the frozen statistic is reused for validation and evaluation.
The object-diffuser assignment is a declared project choice and is shared
across comparable variants.

| Level | System | Schedule | Purpose |
|---|---|---|---|
| R0 | sealed paper-aligned four-layer D2NN only | frozen; no retraining | provide the optical reference |
| B0 | direct/no-D2NN operator plus supervised U-Net | 30 supervised epochs | bound what the lightweight digital backend can recover without trained diffractive layers |
| R1 | frozen R0 plus supervised U-Net | 20 warmup + 10 supervised epochs | measure the marginal effect of supervised digital refinement |
| R2 | frozen R0 plus the same U-Net and PatchGAN | same 20-epoch warmup + 10 adversarial epochs | measure the marginal adversarial effect relative to R1 |

All three digital variants use `base_channels=4`, batch size 32, seed 0, and
unit-weight L1 reconstruction. The generator uses Adam with learning rate
`0.002` and betas `(0.9, 0.999)`; R2's discriminator uses Adam with learning
rate `0.0002` and betas `(0.5, 0.999)`, with adversarial weight `0.01`.
Known, seed-disjoint unseen, and no-diffuser conditions are reported
separately with fixed example object IDs.

R2 must be compared with R1, not only with R0. B0 must remain beside them so
the optical contribution cannot be silently assigned to the digital model.
This single-seed, fixed-depth study is an exploratory result; it cannot support
depth-efficiency, hardware, multiple-scattering, or broad GAN-superiority
claims.

## Visible-Light Translation

Visible-light simulation and optimization begin after the depth/backend
terahertz comparison is stable.

1. Use the sealed R0, fixed-four-layer backend result, and later depth trend as
   the terahertz references.
2. Establish which backend contribution provides a real gain under the same
   terahertz conditions before varying optical depth.
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
