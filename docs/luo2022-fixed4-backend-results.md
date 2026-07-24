# Fixed-Four-Layer Backend Ablation

## Status

This result is frozen as an
`exploratory fixed-depth backend ablation`.

- public result tag: `luo2022-fixed4-backend-v1`;
- executable contract:
  [`configs/luo2022_fixed4_backend.json`](../configs/luo2022_fixed4_backend.json);
- formal training source: `1f5b05c`;
- enhanced evaluator source: `44c6fb8`;
- optical reference: the unchanged `luo2022-r0-v3` four-layer R0;
- dataset: deterministic 50,000-object training, 10,000-object validation,
  and 10,000-object evaluation splits from MNIST;
- diffuser protocol: the frozen synthetic correlated thin-phase model;
- optical depth: fixed at four layers for R0, R1, and R2;
- training replicates: one (seed 0);
- joint optical-digital optimization: none.

The final evaluation contains 1,640,000 unique object-condition-model rows.
Every known or unseen diffuser is evaluated on all 10,000 test objects, the
per-object table reproduces the compact metric aggregates within numerical
tolerance, and all reported values are finite. The R0 checkpoint, runtime
configuration, model state, and phase tensors have identical before/after
fingerprints.

This study measures digital-backend effects behind a fixed optical model. It
does not establish a depth trend, hardware performance, multiple-scattering
validity, or a general advantage for adversarial reconstruction.

## Controlled Comparison

| Variant | Optical operator | Digital backend | Training schedule | Purpose |
|---|---|---|---|---|
| R0 | sealed four-layer D2NN | none | frozen; no retraining | pure optical reference |
| B0 | direct, no D2NN | supervised U-Net | 30 epochs | no-D2NN digital-backend control |
| R1 | sealed four-layer D2NN | supervised U-Net | 20-epoch shared warm-up + 10 supervised epochs | supervised digital refinement |
| R2 | sealed four-layer D2NN | same U-Net + PatchGAN | same warm-up + 10 adversarial epochs | adversarial marginal effect |

B0, R1, and R2 use the same U-Net capacity, unit-weight L1 reconstruction
term, object order, diffuser assignment, batch size, and generator-update
budget. R1 and R2 start from the identical epoch-20 generator and optimizer
state and receive 15,630 continuation updates each. B0 receives 46,890
supervised updates, matching the total generator-update budget of the two
30-epoch optical-input variants. R2 additionally uses an adversarial term with
weight 0.01.

The detector inputs are raw `float32` intensities divided by an operator-specific
global maximum fitted on the training split only. The direct and four-layer
operators therefore have separately fitted scaling constants. Consequently,
R1 minus B0 is an end-to-end operator-path difference, not a pure optical causal
effect independent of preprocessing. There is no per-image normalization and
no clipping. The R0 checkpoint is unchanged; the table below is a read-only
unified recomputation using the four-layer operator's training-fitted scale.
Its PCC values regress to the sealed R0 result, while its PSNR and SSIM belong
to this unified backend-evaluation protocol.

## Evaluation Protocol

All variants are evaluated on the same 10,000 MNIST test objects under three
conditions:

- 20 final-epoch known diffusers;
- 20 seed-disjoint unseen diffusers;
- one no-diffuser control.

Target-support PCC is computed separately for each test object on pixels where
the clean target is greater than zero. The mask is used only for evaluation,
not as a model input. Full-canvas PCC is retained
as an exact regression metric; PSNR, SSIM, and the mean of the worst 5% of
all object-diffuser-pair target-support PCC values are complementary measures.
For known and unseen
conditions, standard deviations and confidence intervals use the diffuser as
the aggregation unit. For the single no-diffuser condition, they use the
object as the aggregation unit. Confidence intervals are normal-approximation
descriptive intervals, `mean +/- 1.96 * standard error`; they do not represent
multi-seed uncertainty.

## Primary Results

### Final-Epoch Known Diffusers

| Variant | Target PCC mean | SD | 95% CI | PSNR | SSIM | Worst 5% PCC |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 0.750835 | 0.014167 | [0.744626, 0.757044] | 15.336 | 0.8764 | 0.540848 |
| B0 | 0.639226 | 0.018216 | [0.631242, 0.647209] | 12.458 | 0.1894 | 0.346057 |
| R1 | 0.901730 | 0.006801 | [0.898750, 0.904711] | 24.189 | 0.8873 | 0.779788 |
| R2 | 0.882590 | 0.005867 | [0.880019, 0.885162] | 24.515 | 0.9103 | 0.724172 |

### Seed-Disjoint Unseen Diffusers

| Variant | Target PCC mean | SD | 95% CI | PSNR | SSIM | Worst 5% PCC |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 0.728476 | 0.015484 | [0.721690, 0.735262] | 15.289 | 0.8747 | 0.516795 |
| B0 | 0.637944 | 0.015657 | [0.631082, 0.644806] | 12.458 | 0.1893 | 0.344114 |
| R1 | 0.891183 | 0.011327 | [0.886219, 0.896148] | 23.729 | 0.8814 | 0.757190 |
| R2 | 0.873739 | 0.008236 | [0.870130, 0.877349] | 24.241 | 0.9081 | 0.704969 |

### No-Diffuser Control

| Variant | Target PCC mean | Object SD | 95% CI | PSNR | SSIM | Worst 5% PCC |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 0.804843 | 0.074508 | [0.803383, 0.806304] | 15.502 | 0.8819 | 0.605522 |
| B0 | 0.678514 | 0.110545 | [0.676347, 0.680681] | 12.519 | 0.1937 | 0.420111 |
| R1 | 0.927057 | 0.033143 | [0.926407, 0.927706] | 25.278 | 0.8998 | 0.828988 |
| R2 | 0.899756 | 0.047425 | [0.898826, 0.900685] | 25.099 | 0.9150 | 0.767623 |

The full-canvas PCC regression values are:

| Condition | R0 | B0 | R1 | R2 |
|---|---:|---:|---:|---:|
| Final known | 0.900042 | 0.688414 | 0.956167 | 0.948522 |
| Unseen | 0.890324 | 0.688100 | 0.951495 | 0.945003 |
| No diffuser | 0.922755 | 0.704153 | 0.966809 | 0.955839 |

The complete compact aggregate is available as
[`luo2022_fixed4_backend_summary.json`](assets/luo2022_fixed4_backend_summary.json).
Digit-stratified target-support PCC, full-canvas PCC, PSNR, and SSIM are
available for all 120 condition-model-digit groups in
[`luo2022_fixed4_backend_per_digit.csv`](assets/luo2022_fixed4_backend_per_digit.csv).

## Three Causal Questions

### 1. Supervised Digital Reconstruction: R1 Minus R0

Adding the supervised U-Net behind the frozen optical system increases
target-support PCC by:

| Condition | Mean delta | 95% CI |
|---|---:|---:|
| Final known | +0.150895 | [+0.146384, +0.155406] |
| Unseen | +0.162708 | [+0.158637, +0.166778] |
| No diffuser | +0.122213 | [+0.121202, +0.123224] |

This is a system-level digital-refinement gain. It does not imply that the
optical-only R0 and the digital system have the same deployment cost.
The known and unseen intervals use matched-diffuser deltas; the no-diffuser
interval uses paired-object deltas.

### 2. End-to-End Optical-Path Difference: R1 Minus B0

With the same lightweight U-Net architecture and update budget, the frozen
four-layer path exceeds the direct path in target-support PCC by:

| Condition | Mean delta | 95% CI |
|---|---:|---:|
| Final known | +0.262504 | [+0.255199, +0.269810] |
| Unseen | +0.253239 | [+0.244979, +0.261500] |
| No diffuser | +0.248543 | [+0.246774, +0.250312] |

This comparison supports a useful end-to-end four-layer-plus-U-Net path under
the declared protocol. It must not be described as an isolated optical causal
effect because each operator uses its own training-fitted intensity scale.
B0 also peaks earlier on the validation set and then degrades by epoch 30;
the table retains the protocol-fixed final checkpoint rather than selecting a
post-hoc early-stopped digital upper bound.
The known and unseen intervals use matched-diffuser deltas; the no-diffuser
interval uses paired-object deltas.

### 3. Adversarial Marginal Effect: R2 Minus R1

R2 and R1 are the strictest matched pair. PatchGAN produces a metric trade-off:

| Condition | Target PCC delta (95% CI) | PSNR delta | SSIM delta |
|---|---:|---:|---:|
| Final known | -0.019140 [-0.022065, -0.016215] | +0.326 | +0.02298 |
| Unseen | -0.017444 [-0.020583, -0.014305] | +0.512 | +0.02675 |
| No diffuser | -0.027301 [-0.027766, -0.026835] | -0.180 | +0.01520 |

The worst-5% target PCC also decreases by approximately 0.056, 0.052, and
0.061 for known, unseen, and no-diffuser conditions. The discriminator remains
numerically stable, but becomes strong late in training. Under the primary PCC
objective, GAN does not provide a positive marginal gain. Its SSIM and partial
PSNR improvements form a structural-similarity-versus-correlation trade-off,
not an overall accuracy improvement. The confidence intervals use the same
matched-diffuser or paired-object units as the other causal deltas.

## Model and Budget Cost

| Variant | Optical layers | Inference digital parameters | Training-only parameters | Mean digital inference / object | Digital generator updates in this ablation |
|---|---:|---:|---:|---:|---:|
| R0 | 4 | 0 | 0 | 0 ms | 0 in this study |
| B0 | 0 | 7,557 | 0 | 0.424 ms | 46,890 |
| R1 | 4 | 7,557 | 0 | 0.424 ms | 46,890 total |
| R2 | 4 | 7,557 | 2,885 discriminator | 0.431 ms | 46,890 generator total |

The R2 discriminator is discarded for inference. Therefore 10,442 is a
training-time total, not the R2 inference parameter count. Measured digital
inference time is hardware-dependent and is retained in the evaluator's cost
figure and compact aggregate as a descriptive quantity rather than a universal
speed claim. This table excludes the prior R0 optical-training budget, optical
propagation latency, phase-neuron count, fabrication cost, and hardware cost;
R0's zero digital updates do not mean zero total system cost.

## Figures and Fixed Examples

- [Metric comparison](assets/luo2022_fixed4_backend_metrics.png)
- [Inference cost](assets/luo2022_fixed4_backend_cost.png)
- [Matched training curves](assets/luo2022_fixed4_backend_training_curves.png)
- [Final-known fixed examples](assets/luo2022_fixed4_backend_samples_known.png)
- [Unseen fixed examples](assets/luo2022_fixed4_backend_samples_unseen.png)
- [No-diffuser fixed examples](assets/luo2022_fixed4_backend_samples_no_diffuser.png)

The sample object IDs were fixed before evaluation and cover digits 0 through
9. They were not selected according to GAN performance. Optical panels show
operator-specific train-scaled inputs with fixed `[0, 1]` display limits, not
raw radiometric intensity.

## Integrity and Claim Boundary

The final evaluator checks the shared warm-up checkpoint, generator and
optimizer branch state, object/diffuser order, update budgets, cache identity,
R0 identity, per-object coverage, and read-only optical phase tensors. The
public repository retains compact tables and figures; raw caches, checkpoints,
per-object rows, machine logs, and host-specific execution records remain
excluded.

The supported statement is narrow:

> Under one deterministic MNIST split, one seed, the frozen four-layer
> thin-phase protocol, and operator-specific train-only global scaling, a small
> supervised U-Net substantially improves the four-layer output and the
> four-layer-plus-U-Net path exceeds the direct-plus-U-Net path. Adding the
> matched PatchGAN trades lower PCC and worst-tail fidelity for higher SSIM and
> partial PSNR gains.

The result does not support general GAN superiority, a reduced optical-depth
claim, multi-seed uncertainty, broad out-of-distribution generalization,
visible-light or hardware performance, or volumetric/multiple-scattering
validity.

## Next Decision Gate

The pure-optical 2/4/5-layer terahertz depth trend is scientifically useful but
deferred: the missing depths require independent from-scratch training, and
the current evidence gain does not justify that GPU cost. The source paper's
depth values may be used as explicitly external reference data. Local depth
training is required only before making a project-specific reduced-depth or
depth-by-backend Pareto claim.

R1, not R2, remains the default PCC-priority digital backend candidate because
the adversarial branch does not improve the primary PCC objective. No depth,
multi-seed, visible-light, or hardware experiment is part of this frozen
result, and none is launched automatically from this decision.
