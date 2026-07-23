# Luo 2022 R0 Result Summary

## Status

The four-layer terahertz R0 baseline is sealed as a
`reproduction-inspired result`.

- contract freeze: `2026-07-19.3`;
- training source commit: `cdf165c`;
- public freeze tag: `luo2022-r0-v3`;
- optical profile: correlated thin phase diffuser followed by four trainable
  phase-only diffractive layers;
- digital backend: none;
- adversarial loss: none.

The result establishes an executable and auditable reference for subsequent
depth, hybrid optical-digital, robustness, and visible-light experiments. It
does not claim exact numerical reproduction of every paper figure.

## Frozen Protocol

| Item | Value |
|---|---:|
| Frequency | 400 GHz |
| Wavelength | 0.75 mm |
| Grid | 240 x 240 |
| Phase layers | 4 |
| Training objects | 50,000 MNIST images |
| Test objects | 10,000 MNIST images |
| Training epochs | 100 |
| Diffusers per epoch | 20 |
| Unique training diffusers | 2,000 |
| Unseen evaluation diffusers | 20 |

The object split, diffuser seed namespaces, loss, learning-rate schedule,
geometry, and aggregation protocol are frozen in
[`configs/luo2022_r0.json`](../configs/luo2022_r0.json).

## Primary Results

Target-support PCC is the primary, post-hoc project diagnostic for subsequent
architecture comparisons. Full-canvas PCC remains a historical regression
metric because the paper does not publish the exact spatial evaluation
region.

| Population | Diffusers | Target-support mean PCC | Sample standard deviation |
|---|---:|---:|---:|
| All training diffusers | 2,000 | 0.735138 | 0.017659 |
| Epochs 1-99 training diffusers | 1,980 | 0.734979 | 0.017623 |
| Final 10 epochs | 200 | 0.736429 | 0.018103 |
| Final-epoch known diffusers | 20 | 0.750835 | 0.014167 |
| New unseen diffusers | 20 | 0.728476 | 0.015484 |
| No diffuser | 1 | 0.804843 | n/a |

The corresponding memory/generalization diagnostics are:

- final known minus epochs 1-99: `0.015856`;
- final known minus new unseen: `0.022359`;
- epochs 1-99 minus new unseen: `0.006503`.

Full-canvas final-known and new-diffuser PCC are `0.900042` and `0.890324`,
respectively. These values are retained for exact evaluator regression and
must not be compared with an unpublished paper ROI as if the spatial domains
were identical.

## Optical Control

The project-defined direct post-diffuser, no-D2NN control uses the same object
and diffuser populations without trainable phase layers. It is a finite-window
single-propagation control and is not claimed to reproduce unpublished
implementation details of the paper's supplementary control.

| Population | Target-support mean PCC |
|---|---:|
| Final-epoch known diffusers | 0.500484 |
| New unseen diffusers | 0.493678 |
| No diffuser | 0.565890 |

The trained four-layer gain over the direct control is approximately `0.250`
for final-known diffusers and `0.235` for unseen diffusers.

## Evidence Integrity

The local evidence ledger contains 2,021 unique per-diffuser rows:

- 2,000 training diffusers covering all 100 epochs;
- 20 unseen diffusers;
- one no-diffuser control;
- 10,000 test objects per row.

The full-canvas recomputation matches the frozen evaluator exactly, the
checkpoint remained read-only during post-hoc evaluation, and all retained
evidence checksums pass. Generated rows and checkpoints are intentionally
excluded from the public repository.

## Acceptance and Claim Boundary

The project-defined post-training absolute-level, optical-control,
memory-structure, and integrity gates passed. The depth-trend gate remains
deferred; the next controlled study keeps all four optical layers frozen.

The result supports the following narrow statement:

> Under the frozen synthetic thin-phase diffuser protocol, the four-layer
> diffractive model reconstructs both training-distribution and seed-disjoint
> unseen diffusers and provides a substantial target-support PCC gain over
> direct propagation.

It does not establish volumetric or multiple-scattering validity, hardware
performance, exact author ROI equivalence, multi-seed uncertainty, or an
advantage over all digital reconstruction methods.

## Next Research Gate

The next experiment is the fixed-four-layer B0/R1/R2 backend ablation defined
in
[`configs/luo2022_fixed4_backend.json`](../configs/luo2022_fixed4_backend.json).
The four-layer result and every number in this document remain unchanged.

- B0 trains the lightweight supervised U-Net on the direct/no-D2NN operator.
- R1 trains the same U-Net after frozen R0.
- R2 branches from R1's identical 20-epoch supervised warmup and adds 10
  adversarial epochs; R1 receives a matched 10-epoch supervised continuation.

This is an exploratory fixed-depth backend ablation. It can measure digital
and adversarial marginal effects under this protocol, but it cannot establish
an optical depth trend. Independent two-layer and five-layer training is
deferred and remains necessary before any reduced-depth claim.
