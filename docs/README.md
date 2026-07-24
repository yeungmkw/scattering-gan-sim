# Project Notes

This folder contains human-facing summaries for the scattering GAN simulation
prototype. The code is intentionally small, but the research story needs to be
explicit because the current result is an exploratory simulation result, not a
physical hardware claim.

Recommended reading order:

1. `system-overview.md`: what the current pipeline does and how files map to
   the optical/DNN/GAN stages.
2. `research-roadmap.md`: completed decisions and the evidence required before
   making stronger
   claims.
3. `luo2022-r0-results.md`: the sealed public R0 protocol, key results,
   controls, integrity boundary, and completed follow-on boundary.
4. `luo2022-fixed4-backend-results.md`: the completed R0/B0/R1/R2 fixed-depth
   comparison, matched GAN branch, costs, and claim boundary.
5. `huang2026-visible-baseline.md`: the visible-light coherent, incoherent,
   multi-wavelength, control, Supporting Equation (S18) SLM input-encoding,
   resource, and misalignment numerical contracts. It defines the four public
   evidence labels and keeps training convergence separate from blind-diffuser
   generalization.

The executable Luo 2022 R0 source of truth is
[`configs/luo2022_r0.json`](../configs/luo2022_r0.json). The completed
fixed-four-layer B0/R1/R2 backend result is defined by
[`configs/luo2022_fixed4_backend.json`](../configs/luo2022_fixed4_backend.json).
It keeps R0 frozen, uses portable raw-intensity cache metadata, and records the
train-only scaling, schedules, optimizers, fixed evaluation examples, and
claim boundary needed to reproduce the comparison.

The companion notebook is a Chinese walkthrough of the simulation principle:

```text
notebooks/scattering_gan_summary.ipynb
```

It explains how the polluted optical signal is generated, how that signal is
fed through DNN reconstruction, and how GAN refinement is attached after the
baseline. Its result cells expect a locally generated `outputs/` run; the
public repository instead keeps the representative figures under `docs/assets/`.

## Archive

Superseded but still useful public material is kept under
[`archive/`](archive/). The legacy fixed-phase U-Net/PatchGAN result and its
figures remain reproducible there, but they are no longer part of the main
reading path.

Detailed reproduction contracts, evidence ledgers, acceptance notes, and
research surveys are retained locally under the Git-ignored
`archive/internal/` directory. They remain available for audit and future
research decisions without cluttering the public documentation surface.
