# GPU Phase Comparison

## Run Summary

This run compares the coherent U-Net baseline against a U-Net generator refined
with a conditional PatchGAN discriminator.

| Field | Value |
|---|---|
| Experiment class | E0 + E2: fixed diffuser baseline plus GAN refinement |
| Device | CUDA, RTX 4060 Laptop GPU |
| Dataset | MNIST |
| Corruption | fixed coherent phase screen |
| D2NN seed | `7961` |
| Train samples | `2048` |
| Eval samples | `256` |
| Batch size | `32` |
| U-Net epochs | `20` |
| GAN epochs | `10` |
| Base channels | `16` |
| GAN generator init | U-Net checkpoint from the same run |

## Final Metrics

| Metric | U-Net | U-Net + PatchGAN | Direction | GAN Outcome |
|---|---:|---:|---|---|
| L1 | 0.0549016 | 0.0519791 | lower is better | better |
| MSE | 0.0231906 | 0.0223656 | lower is better | better |
| PSNR (per-image mean) | 16.3830 | 16.5356 | higher is better | better |
| SSIM | 0.805810 | 0.815968 | higher is better | better |
| Pearson | 0.835173 | 0.825300 | higher is better | worse |

## Interpretation

The U-Net baseline trained successfully: training L1 dropped from about
`0.305` to `0.0298`, and eval SSIM rose from about `0.069` to `0.806`.

The PatchGAN refinement improved L1, MSE, PSNR, and SSIM by a small amount, but
Pearson correlation decreased. That means the GAN stage is not automatically a
scientific win. It may smooth or sharpen the reconstruction in ways that help
some perceptual metrics while slightly reducing correlation fidelity.

The table is synchronized with the canonical saved `metrics.json` values.
PSNR is the mean of per-image PSNR values, and all dataset metrics use
sample-count-weighted aggregation. On July 17, 2026, the older PSNR values in
this note were corrected because they did not match the saved metrics and
history artifacts; the model checkpoints and metric files were not changed.

## Artifacts

| Artifact | Path |
|---|---|
| Metrics plot | [gpu_phase_comparison_metrics.png](assets/gpu_phase_comparison_metrics.png) |
| Sample comparison | [gpu_phase_comparison_samples.png](assets/gpu_phase_comparison_samples.png) |
| Checkpoints and raw run files | Kept out of Git; regenerate with `python -m experiment full`. |

The ignored raw run directory has backfilled schema-v1 manifests and
`config.json` snapshots. They are marked as migrated legacy metadata because
the original runtime/Git provenance was not recorded.

## Status

Status label: exploratory result.

This result is useful as a baseline for future experiments, but it should not
be written as a claim candidate until it is repeated across seeds, diffuser
splits, and more optical corruption variants.
