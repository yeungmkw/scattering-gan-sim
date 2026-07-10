# Research Roadmap

## Immediate Next Steps

1. Repeat the current phase-screen experiment with multiple seeds.
2. Add explicit seen/unseen diffuser splits for coherent phase screens.
3. Compare phase-screen and particle-mask corruption under the same training
   budget.
4. Add a non-GAN U-Net ablation with different loss terms: L1, negative
   Pearson, SSIM-like, and Fourier consistency.
5. Tune the adversarial loss weight after the U-Net baseline is stable.

## Experiment Matrix

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
