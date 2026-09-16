# Fusion demonstration

## What this shows

The four public datasets are different people. Omics and CGM share 22 real
patients through the crosswalk (19 with an SSPG label); wearable and retinal
imaging share no patients with anyone. A real all-four-modalities matrix does
not exist in public data, so this demo builds a **virtual cohort** of synthetic
patients that carry all four modalities at once and runs the real fusion
pipeline on it end to end.

## Truth is external to the model

- virtual cohort: a hidden latent risk `z`, drawn before any feature exists and
  never shown to the model. Every modality is generated from `z` through noisy,
  nonlinear maps, so no single modality reveals it and the model must fuse.
- real 22: measured SSPG (used elsewhere in the project to score the real
  linked cohort). Here the real omics block only calibrates the coupling.

## The three controls (why this is not circular)

1. **best single modality** — fusion must beat the strongest modality alone.
2. **fusion on scrambled** — break the shared-`z` coupling by permuting each
   modality's rows independently; the fusion advantage must vanish.
3. **additive baseline** — a standardized hand-sum of modality scores, the
   baseline to beat. Beating it shows captured cross-modal interaction, not
   addition. It is never used as a training label.

## Calibration

- calibration used real omics block: **True**
- note: omics loading calibrated to 19 real linked patients
- calibrated loadings: `{'omics': 0.5475, 'cgm': 0.7, 'wearable': 0.5, 'imaging': 0.6}`

Wearable keeps its default loading (disjoint patient ids, no linked block).
**Imaging is a synthetic placeholder block** until the retinal CNN is trained
on AWS and its embeddings are wired in via `imaging_bridge` — at which point
the imaging loading will be anchored to the real CNN instead of the default.

## Results

| model                         |          R2 |    Spearman |       RMSE |
|:------------------------------|------------:|------------:|-----------:|
| fusion (all modalities)       |  0.839578   |   0.937542  |   0.407665 |
| best single (omics)           |  0.719316   | nan         | nan        |
| fusion on scrambled (control) |  0.00371977 |   0.0691679 |   1.01592  |
| additive baseline             | -3.27378    |   0.0987655 |   2.10415  |

### Verdict

- `fusion_beats_best_single`: **True**
- `scramble_advantage_vanishes`: **True**
- `fusion_beats_additive`: **True**

## Files

- `reports/fusion_results.csv` — the comparison table above
- `reports/fusion_verdict.json` — the verdict dict
- `reports/fusion_comparison.png` — grouped R2 bar chart
