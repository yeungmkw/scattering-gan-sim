# Project Notes

This folder contains human-facing summaries for the scattering GAN simulation
prototype. The code is intentionally small, but the research story needs to be
explicit because the current result is an exploratory simulation result, not a
physical hardware claim.

Recommended reading order:

1. `system-overview.md`: what the current pipeline does and how files map to
   the optical/DNN/GAN stages.
2. `gpu-phase-comparison.md`: the completed GPU run comparing coherent U-Net
   and U-Net+PatchGAN.
3. `research-roadmap.md`: what should be added next before making stronger
   claims.

The companion notebook is a Chinese walkthrough of the simulation principle:

```text
notebooks/scattering_gan_summary.ipynb
```

It explains how the polluted optical signal is generated, how that signal is
fed through DNN reconstruction, and how GAN refinement is attached after the
baseline. Its result cells expect a locally generated `outputs/` run; the
public repository instead keeps the representative figures under `docs/assets/`.
