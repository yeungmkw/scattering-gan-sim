# Huang 2026 Visible-Light Numerical Baseline

This profile is a numerical, reproduction-inspired implementation of:

> L. Huang et al., “Optical Computational Imaging Through Unknown Random
> Diffusers in Visible Spectrum,” *Laser & Photonics Reviews* 20, e01168
> (2026), DOI 10.1002/lpor.202501168.

It is not a calibrated reproduction of the authors’ hardware. The profile
implements the paper’s thin phase diffuser, scalar angular-spectrum
propagation, three phase-only diffractive layers, coherent and
Gaussian–Schell ensemble illumination, three-wavelength propagation, optical
controls, and configurable numerical error models. Volumetric multiple
scattering, polarization, a measured SLM lookup table, and hardware
calibration are outside the baseline.

## Public contracts

The mode-specific contracts are:

- `configs/huang2026_visible_coherent.json`
- `configs/huang2026_visible_incoherent.json`
- `configs/huang2026_visible_multiwavelength.json`

The incoherent and multi-wavelength files inherit the shared coherent
contract and override only their mode-specific fields. Every configured value
has an adjacent label from exactly four evidence categories:

- `paper_confirmed`: explicit in the main article or its supporting notes.
- `paper_inferred`: constrained by published equations, geometry, or field
  conventions, but not separately stated.
- `project_choice`: an implementation, numerical, or execution decision that
  is not published.
- `suspected_paper_typo`: a conflicting published value retained only in an
  explicitly selected sensitivity profile.

These labels describe parameter provenance, not result quality. In
particular, `paper_confirmed` does not turn a numerical run into a hardware
reproduction.

## Equation-to-code map

| Paper mechanism | Equation | Implementation |
|---|---:|---|
| Smoothed random-height diffuser and phase delay | Main (1) | `Huang2026DiffuserConfig`, `CorrelatedHeightPhaseDiffuser` |
| Diffuser autocorrelation model | Main (2) | correlation-length parameter and diffuser tests |
| Object propagation and diffuser multiplication | Main (3) | `Huang2026ThreeLayerDONN.forward_field` |
| Angular-spectrum transfer function | Main (4) | reused `AngularSpectrumPropagator`, one instance per distance and wavelength |
| Phase-only layer transmittance | Main (5) | trainable phase command used in `exp(j phi)` |
| Cascaded layer propagation | Main (6), Supporting (S9) | `Huang2026ThreeLayerDONN` |
| Raw detector intensity | Main (7), Supporting (S10) | DONN and control operators return `abs(field)**2` |
| Gaussian–Schell coherence definition | Main (8), Supporting (S1) | `Huang2026CoherenceSampler` |
| Incoherent ensemble intensity | Main (9), Supporting (S13) | `Huang2026IncoherentDONN` |
| Complex-screen power spectrum | Main (10), Supporting (S2)–(S8) | `Huang2026CoherenceSampler` |
| Detector-intensity MSE | Main (11), Supporting (S11) | `huang2026_intensity_mse` |
| Coherent phase-gradient backpropagation | Supporting (S12) | `train_huang2026_model` uses autograd through the complex field, intensity MSE, and every trainable phase layer before the Adam update |
| IC-DONN MSE after intensity averaging | Supporting (S14)–(S15) | `huang2026_incoherent_mse` |
| Per-image Pearson correlation | Main (12) | `huang2026_pcc_per_image` and grouped summaries |
| Thin-lens control | Supporting (S16)–(S17) | `ThinLensOperator` |
| Phase-only SLM input hologram | Supporting (S18) | `SLMPhaseResponse.phase_only_hologram` |
| Axial and radial alignment | Supporting (S19), Note S7 | `MisalignmentTransform` |
| Multi-wavelength intensity and loss | Supporting (S22)–(S24) | `Huang2026MultiWavelengthDONN`, `huang2026_multiwavelength_mse` |
| Wavelength-dependent SLM response | Supporting (S25)–(S27) | `SLMPhaseResponse` |

The main-text rendering of the Gaussian–Schell coherence factor in Equation
(8) conflicts with the separation-dependent form in Supporting Equation
(S1). The implementation follows the complete derivation in (S1), (S7), and
(S8): coherence depends on the spatial separation and the complex screen is
sampled from the stated Gaussian power spectrum.

## Confirmed parameters and explicit boundaries

The default optical canvas is 400×400 with an 8 µm pitch. MNIST 28×28 images
are bilinearly resized to 320×320 and centered with zero padding on the
400×400 canvas. The resized image is the input field amplitude. The clean
target intensity is therefore its squared amplitude; this is a
`paper_inferred` convention and is never replaced by per-image min–max
normalization.

In main Equation (1), the independent prefilter random field \(W\)—not the
postfilter correlated height map—has Gaussian mean 63 µm and standard
deviation 14 µm. The implementation samples \(W\) with those statistics,
applies the published Gaussian convolution, and uses the result as diffuser
height. It deliberately does not re-center or re-scale the smoothed map to
63/14 µm, because such a postfilter standard-deviation normalization is not
published. The phase delay then uses glass index 1.52, index contrast 0.52,
and the wavelength-dependent mapping in Equation (1). Training diffusers are
generated online; training and blind-test seeds occupy separate deterministic
namespaces.

The nominal geometry is:

```text
object -> diffuser       29.5 mm   paper_inferred
diffuser -> layer 1      29.5 mm   paper_inferred
layer 1 -> layer 2       29.5 mm   paper_confirmed
layer 2 -> layer 3       29.5 mm   paper_confirmed
layer 3 -> detector      71.2 mm   paper_confirmed
total                   189.2 mm
```

Only this default total equals the reported lens-control path,
`4 × 47.3 mm = 189.2 mm`.
Supporting Note S7 instead states 2.95 mm and 7.1 mm. Those values are a
factor of ten shorter and produce an 18.9 mm DONN/direct path. The lens keeps
the published 47.3 mm focal length and therefore remains a 189.2 mm 4f path;
the typo-sensitivity controls are intentionally reported as not path matched.
The conflicting distances are available only through:

```bash
--geometry-profile supplement_typo_sensitivity
```

They are never averaged with, or silently substituted for, the default
geometry.

Supporting Equation (S17) reports a circular-pupil radius equal to half the
diagonal of the rectangular field. That circle contains the complete sampled
grid, including its corners. `ThinLensOperator` implements the published
outer-circle radius; it does not introduce an unreported inscribed aperture.

## Coherent, incoherent, and multi-wavelength modes

The coherent model propagates one coherent realization through the complete
optical path and returns raw detector intensity. The IC-DONN multiplies each
input by independent complex
Gaussian–Schell screens, performs an independent coherent propagation for
each screen, and averages detector intensities. The paper values are
`Nr=20` for training and `Nr=2000` for blind testing. Execution-only chunking
accumulates the exact ensemble sum before dividing by `Nr`; it does not
change the mathematical loss.

The multi-wavelength model propagates 491, 532, and 660 nm channels with
separate angular-spectrum transfer functions and sums their individual MSE
losses as Supporting Equation (S24) specifies. The paper shows measured SLM
phase-response curves but does not publish their numeric lookup table. The
default continuous wavelength-dependent response is therefore a
`project_choice`, not measured data. Numeric drive/phase pairs can be supplied
through a configuration LUT.

## Input encoding and explicit error models

The `inspect` action exercises the phase-only input-field encoding from
Supporting Equation (S18). For normalized target amplitude \(A\) and phase
\(\phi\), it evaluates the inverse-sinc branch on \([-\pi,0]\), forms
\(M=1+\operatorname{sinc}^{-1}(A)/\pi\),
\(F=\phi-\pi M\), and \(T=\exp(jMF)\), then verifies that the simulated
first-order amplitude is monotonic and recovers \(A\). It saves the encoded
phase hologram, boundary values, and numerical recovery error. This simulates
the phase-only hologram and first diffraction order; carrier-order spatial
filtering and a calibrated SLM lookup table remain omitted.

The ideal baseline leaves phase quantization, phase clamping, SLM spatial
smoothing, abrupt phase-jump errors, detector shot noise, detector read
noise, gain error, saturation, and transmission loss disabled. Their
configurable switches or values are `project_choice` error models for
sensitivity studies; they are not inferred hardware calibration.

The public misalignment action uses a zero-filled boundary and the following
project-selected stress grid:

| Perturbation | Configured grid | Applied location |
|---|---|---|
| Lateral x shift | `0, +5, -5, +10, -10` pixels | zero-based phase-layer index 1 |
| Axial offset | `0, +0.5, -0.5, +1.0, -1.0` mm | zero-based propagation-segment index 2 |

The paper-reported experimental ranges, 4–12 pixels laterally and
0.05–0.1 mm axially, are recorded separately as `paper_confirmed` context;
they are not silently substituted for the configured stress grid. The
zero-shift/zero-offset condition must reproduce the ideal path exactly, and
each nonzero condition reports its PCC change relative to that ideal.

## CLI

Commands below include `--download` whenever MNIST may need to be fetched in
a clean clone. Inspect the coherent profile without training and audit the
Supporting Equation (S18) encoding:

```bash
uv run python -m experiment d2nn \
  --profile huang2026_visible \
  --mode coherent \
  --action inspect \
  --download \
  --output-dir outputs/huang2026_visible_inspect
```

Run an explicitly reduced coherent training:

```bash
uv run python -m experiment d2nn \
  --profile huang2026_visible \
  --mode coherent \
  --action train \
  --execution-label small \
  --device cpu \
  --download \
  --output-dir outputs/huang2026_visible_coherent_small
```

Run reduced IC-DONN and multi-wavelength smoke training:

```bash
uv run python -m experiment d2nn \
  --profile huang2026_visible \
  --mode incoherent \
  --action train \
  --execution-label small \
  --nr 4 \
  --diffuser-chunk-size 2 \
  --download \
  --output-dir outputs/huang2026_visible_incoherent_small

uv run python -m experiment d2nn \
  --profile huang2026_visible \
  --mode multiwavelength \
  --action train \
  --execution-label small \
  --download \
  --output-dir outputs/huang2026_visible_multiwavelength_small
```

Resume an interrupted run with the exact same binding:

```bash
uv run python -m experiment d2nn \
  --profile huang2026_visible \
  --mode coherent \
  --action train \
  --execution-label small \
  --download \
  --output-dir outputs/huang2026_visible_coherent_small \
  --resume
```

Changing an optical, optimizer, sampling, object-order, or training-scale
field causes strict resume validation to fail. `checkpoints/latest.pt` stores
the next epoch and batch, optimizer state, deterministic global iteration,
compact epoch accumulators, detector state, random-number-generator states,
and rolling-hash commits for the append-only history and sample-record
journals. Checkpoints are written at the configured update interval and at
epoch boundaries. Resume verifies the payload integrity hash, deterministic
cursor, model/optimizer state, and both committed journal prefixes; only an
uncommitted crash-tail is truncated.

Evaluate, compare controls, and assess misalignment without retraining. Keep
the completed training run read-only with `--run-dir`, and give each action
its own fresh `--output-dir`:

```bash
uv run python -m experiment d2nn \
  --profile huang2026_visible \
  --mode coherent \
  --action evaluate \
  --execution-label small \
  --download \
  --run-dir outputs/huang2026_visible_coherent_small \
  --output-dir outputs/huang2026_visible_coherent_blind_evaluation

uv run python -m experiment d2nn \
  --profile huang2026_visible \
  --mode coherent \
  --action control \
  --execution-label small \
  --control direct lens donn \
  --download \
  --run-dir outputs/huang2026_visible_coherent_small \
  --output-dir outputs/huang2026_visible_coherent_controls

uv run python -m experiment d2nn \
  --profile huang2026_visible \
  --mode coherent \
  --action misalignment \
  --execution-label small \
  --download \
  --run-dir outputs/huang2026_visible_coherent_small \
  --output-dir outputs/huang2026_visible_coherent_misalignment
```

The three controls use the same object, diffuser, sampling grid, and detector
response, but are implemented by independent operators. Under
`paper_default`, they also share the 189.2 mm object-to-detector path. Under
`supplement_typo_sensitivity`, the DONN/direct path is 18.9 mm while the
published 4f lens path remains 189.2 mm, so the action records
`nominal_total_path_matched: false`. The nominal misalignment transform is an
exact identity.

Assess the full-size numerical resource path:

```bash
uv run python -m experiment d2nn \
  --profile huang2026_visible \
  --mode coherent \
  --action assess \
  --execution-label full \
  --device cpu \
  --output-dir outputs/huang2026_visible_resource_assessment
```

Although the command binds `--mode coherent`, it intentionally runs two
synthetic 400×400 forward/backward probes: one coherent three-layer DONN step
and one streamed IC-DONN step with `Nr=20`, chunk size 4, and activation
checkpointing. It records timing, finite nonzero phase-gradient norms,
observed process/device memory, and bounded tensor-size estimates without
materializing all 20 screens at once. This is not a 60,000-object training
run and cannot establish convergence.

## Interpreting results

Training convergence and blind-diffuser generalization are separate
questions:

- A lower final-epoch mean training objective than the first-epoch mean
  (or, for a one-epoch smoke run, a lower final update than the first update),
  together with finite nonzero gradients and updates for all three phase
  layers, is the small-run convergence check. Fixed training-object and blind
  probes are also recorded as observations, but neither is substituted for
  this stochastic online-diffuser training criterion.
- Blind generalization must be measured by `--action evaluate`, which uses the
  disjoint blind-test diffuser seed namespace and reports per-image PCC and
  grouped statistics. A small run remains a small numerical run even when its
  blind PCC improves.
- Control and misalignment actions diagnose optical-path choices and
  sensitivity. They do not upgrade either a convergence result or a
  generalization result into a hardware claim.

## Run artifacts

Fresh action directories are self-contained. If an action is intentionally
written into its training directory instead, it preserves the training
manifest and writes `<action>_manifest.json`; the independent commands above
write `manifest.json` in each action directory.

| Artifact | Produced by | Meaning |
|---|---|---|
| `config.json` | every action | resolved runtime and invocation binding |
| `contract.json` | every action | fully resolved inherited public contract |
| `source_config.json` | every action | selected mode-specific source contract |
| `manifest.json` | fresh action directory | action, equation, physics, status, and artifact manifest |
| `evaluate_manifest.json`, `control_manifest.json`, `misalignment_manifest.json` | action sharing a training directory | action manifest that does not replace the training manifest |
| `history.json` | training | compact first/last loss, epoch summaries, and committed-journal metadata |
| `history.jsonl` | training | append-only per-update and epoch-summary loss journal |
| `metrics.json` | train, evaluate, inspect, assess | training/evaluation summaries or action-specific scalar metrics |
| `sample_records.jsonl` | train, inspect | per-object diffuser, coherence, wavelength, and iteration metadata |
| `ideal_evaluation_sample_records.jsonl` | evaluate and training final evaluation | blind-evaluation sample metadata |
| `control_metrics.json`, `control_sample_records.jsonl` | control | direct/lens/DONN bindings, path-match flag, metrics, and sample metadata |
| `misalignment_metrics.json`, `*_evaluation_sample_records.jsonl` | misalignment | configured grid, ideal-relative PCC changes, and condition metadata |
| `slm_encoding.json` | inspect | Equation (S18) inverse-sinc boundaries, monotonicity, and recovery error |
| `resource_assessment.json` | assess | coherent and IC `Nr=20`/chunk-4 resource probes |
| `checkpoints/latest.pt` | training | atomic strict-resume model, optimizer, detector, cursor, compact history state, journal commits, and RNG state |
| `samples/` | all numerical actions | target, corrupted input, reconstruction/control, error, hologram, or resource figures |

Reported results must retain one of these labels: `small run`,
`exploratory result`, `reproduction-inspired result`, or `claim candidate`.
The reduced acceptance runs in this repository are small numerical runs.
Neither paper-aligned parameters nor successful unit tests imply calibrated
physical validity.

Training and checkpoint-consuming action manifests bind the exact checkpoint
payload integrity digest, model fingerprint, global step, and file SHA-256.
External run directories are recorded only by the portable logical name
`checkpoints/latest.pt` plus `source_kind`; host-specific absolute paths are
not written into public artifacts.
