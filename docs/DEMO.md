# HURDLE demo UI

An interactive Streamlit demonstration of the HURDLE multimodal
diabetes-risk fusion pipeline.

## Honesty disclosure (read first)

**This is a portfolio demonstration, not a validated clinical tool.**

The pipeline fuses four modalities — omics, continuous glucose (CGM),
wearable, and retinal imaging — into a single insulin-resistance /
diabetes-risk score anchored on SSPG (steady-state plasma glucose). But the
four public datasets that back these modalities are **different people**:
omics and CGM share 22 real patients via a crosswalk; wearable and retinal
imaging share no patients with anyone. A real "all four modalities per
patient" matrix does not exist in public data.

To demonstrate the fusion end to end, the app builds a **virtual cohort**:
synthetic patients who carry all four modalities at once, generated from a
hidden latent risk factor `z` drawn *before* any feature exists. Every
modality's features are produced from `z` through noisy, nonlinear maps, so no
single modality reveals `z` and the model must genuinely fuse them to recover
it. The imaging modality's coupling strength is **calibrated from the real
RetinaMNIST CNN embeddings** (`models/imaging_embeddings.npz`), so the imaging
block is anchored to a real trained network even though the per-patient join
is virtual.

The app never diagnoses, screens, or advises any individual. The honesty
banner at the top of the page states this in plain language.

## How to run

From the repo root, with the `hurdle` environment active:

```bash
PYTHONPATH=src streamlit run app/streamlit_app.py
```

Streamlit opens the app in your browser (default http://localhost:8501).

To run the smoke tests:

```bash
PYTHONPATH=src pytest tests/test_streamlit_app.py -q
```

## What each panel shows

1. **Honesty banner (top).** A prominent `st.warning` stating that this is a
   portfolio demonstration on a virtual cohort of disjoint public datasets
   coupled through a latent risk factor, and is not a validated clinical tool.

2. **Patient inputs (synthetic).** Four modality panels laid out in two rows
   of two columns:
   - *Omics* — HbA1c, fasting glucose, triglycerides (sliders).
   - *CGM* — mean glucose, glucose CV, time-in-range (sliders).
   - *Wearable* — resting heart rate, daily steps, sleep hours (sliders).
   - *Imaging* — diabetic retinopathy grade 0–4 (selectbox).

   Each control is an interpretable clinical analogue. Sliders are normalized
   to a signed risk signal centered at their mid value (with the direction
   flipped for protective features such as time-in-range, steps, and sleep),
   then averaged per modality and mapped into the fusion feature space. Nothing
   is read from a real record.

3. **Predict risk (button).** On click the app:
   - maps the inputs into a single synthetic patient row in the virtual-cohort
     feature space;
   - runs the deployable late-fusion predictor — one RandomForest base model
     per modality (matching `fusion.py`'s estimator config) feeding a RidgeCV
     meta-model over the base scores;
   - renders **(a)** the fused risk score (a latent-`z` analogue), **(b)** a
     risk tier — Low / Moderate / High by SSPG-analogue tertile of the cohort
     fused score, and **(c)** a horizontal bar chart of per-modality
     contribution.

   The contribution bars weight each modality's base score by its
   **leave-one-out ablation drop**, so omics visibly dominates and imaging is
   the weakest contributor — exactly what the ablation report shows.

4. **Why trust this? (collapsible).** Negative-control evidence that fusion
   does real cross-modal work. The comparison table is computed **live** by
   `hurdle.fusion.fusion.compare_controls`, which runs the real
   fuse / scramble / additive-baseline pipeline on the calibrated virtual
   cohort (target = hidden latent risk) — it is not hardcoded. Representative
   values, consistent with `reports/modality_ablation.csv` and the reference
   ablation run:
   - Fusion (all 4 modalities) **R² ≈ 0.85**
   - Fusion on scrambled cohort (control) **R² ≈ 0.01** — breaking the
     shared-latent coupling collapses performance.
   - Additive hand-sum baseline **R² ≈ −3.5** — a naive standardized sum fails
     badly, so the meta-model captures interaction the hand-sum cannot.
   - Standalone R² per modality (from the ablation report): omics 0.782,
     cgm 0.619, wearable 0.574, imaging 0.307.

   If `compare_controls` cannot run for any reason, the panel falls back to the
   reference R² values (0.850 / 0.013 / −3.55) with a visible info notice.

5. **Footer caption.** States the fusion architecture and that the surfaced
   numbers come from `reports/modality_ablation.csv`.

## Real numbers surfaced

All performance numbers are read at runtime from
`reports/modality_ablation.csv` (leave-one-out drops and standalone R² per
modality, plus the full 4-modality fusion R²). If that file is missing, the app
falls back to a mirrored copy of the same constants and shows an `st.info`
notice. The scrambled-control and additive-baseline R² values shown in the
"Why trust this?" panel are computed live by `compare_controls`, with the
task-provided reference numbers (0.850 / 0.013 / −3.55) as a labeled fallback.

## Error handling

All compute (predictor build and per-patient prediction) is wrapped in
`try/except` with `st.error`, so a failure surfaces a visible message rather
than a silent crash or a blank page.
