# Data Card: Multimodal Diabetes Risk Predictor

Documents every raw data source: provenance, subject counts, structure, variables,
units, missingness, licence, and known quirks. Values marked ✅ were verified by
direct inspection of the downloaded files; values marked ⏳ are pending extraction.

All four sources originate from Stanford's Snyder-lab iPOP programme (or its DR-cohort
analogue for imaging), so the disease theme — type-2-diabetes risk / insulin
resistance — is shared, but the **cohorts are not identical** (see the linkage note
at the end).

---

## 1. Omics + clinical labs — Zhou et al. 2019 ✅

| Field | Value |
|---|---|
| Source | Zhou, Sailani et al. 2019, *Nature* 569:663–671 |
| DOI | 10.1038/s41586-019-1236-x |
| Files | `zhou2019_supp_tables.xlsx` (19.3 MB), `zhou2019_within_subject_assoc.xlsx` (24.5 MB), `zhou2019_between_subject_assoc.xlsx` (4.4 MB) |
| Subjects | 106 adults, at risk of type-2 diabetes (risk-enriched, IR/IS) |
| Duration | ~4 years longitudinal, quarterly healthy visits + illness/stress events |
| Modalities | Transcriptome, metabolome, proteome, cytokines, clinical labs, gut + nasal microbiome |
| License | CC0 (public domain) |

**Structure (main workbook, 40 sheets).** The load-bearing sheets:
- `S1_Subjects` — master table, 106 rows. Columns: `SubjectID` (e.g. `ZOZOW1T`), `consented`, `IRIS` (IR / IS / Unknown), `SSPG` (steady-state plasma glucose, mg/dL — the continuous insulin-resistance measure), `FPG`, `Class` (Diabetic/Prediabetic/Control/Crossover), `Gender`, `Ethnicity`, `Adj.age`, `BMI`, visit counts, `Days_Span`.
- `S4_HealthyIQR` — analyte × subject matrix, ~12,380 analytes × ~94 subjects (healthy-visit IQRs).
- `S8_SSPGassoc.` — 86 analytes × 73 subject columns (SSPG-associated).
- `S9_ISIRassoc.` — 153 analytes × 72 subject columns; `IRIS` label embedded as row 0 (0/1).
- `S7_TimeAssoc.`, and the two association workbooks (S30 within-subject, S31 between-subject correlation tables) are reference results, not training matrices.

**Labels available:** `SSPG` (continuous, primary regression target), `IRIS` (binary IR/IS — 35 IR / 31 IS / 40 Unknown across the full 106), `Class` (clinical status).

**Subject-ID note (load-bearing for fusion):** the analyte-matrix column headers embed *two* IDs, e.g. `69-001/ZOZOW1T` — an iPOP site-code (`69-XXX` / `70-XXXX`) and the `Z-code`. The crosswalk between them is saved as `subject_crosswalk_omics.csv` (89 iPOP subjects) and is what links omics to CGM.

**Missingness / quirks:** SSPG missing for ~40 subjects; proteomics has known batch effects; only 66 subjects have a definite IR/IS label. High-dimensional, low-n — regularise hard.

---

## 2. Continuous glucose monitoring (CGM) — Hall et al. 2018 ✅

| Field | Value |
|---|---|
| Source | Hall, Perelman et al. 2018, *PLoS Biology* 16(7):e2005143 |
| DOI | 10.1371/journal.pbio.2005143 (PMID 30040822, PMC6057684) |
| File | `hall2018_cgm_S1_Data` (ASCII, tab-delimited; distributed gzipped as supp file s010) |
| Subjects | 57 adults without prior diabetes diagnosis |
| License | CC-BY 4.0 |

**Structure.** One row per glucose reading, 4 columns:
- `DisplayTime` — local timestamp (use this for analysis)
- `GlucoseValue` — blood glucose, **mg/dL**
- `subjectId` — e.g. `1636-69-001`
- `InternalTime` — device internal clock (can drift from DisplayTime)

**Sampling rate:** one reading every **5 minutes** (~288/day) while worn.

**Study design:** free-living CGM + responses to standardized test meals; defines "glucotypes" (low/moderate/severe glucose variability) via spectral clustering. Same metabolic panel (SSPG, HbA1c, TG, HDL) as the omics cohort.

**Subject-ID families (verified):** the 57 split into `1636-69-XXX` (18), `1636-70-XXXX` (5), and `2133-XXX` (34). The `69`/`70` families are iPOP subjects; `2133` is a separate study with no omics link.

**Missingness / quirks:** gaps during sensor changes; per-subject recording length varies (typically ~2 weeks); the two timestamp columns can diverge.

---

## 3. Retinal fundus imaging — RetinaMNIST (MedMNIST v2) ✅

| Field | Value |
|---|---|
| Source | RetinaMNIST, MedMNIST v2 (Yang et al. 2023); derived from the DeepDRiD DR challenge |
| Repository | Zenodo record 10519652 |
| Files | `retinamnist.npz` (28×28, 3.3 MB), `retinamnist_224.npz` (224×224, 128 MB) |
| Images | 1,600 colour fundus photographs |
| License | CC-BY 4.0 |

**Structure.** Six numpy arrays: `{train,val,test}_images` + `{train,val,test}_labels`.
- Splits: **1,080 train / 120 val / 400 test**
- Image arrays: `(N, H, W, 3)` uint8, 0–255 (H=W=28 or 224)
- Labels: `(N, 1)` uint8, values **0–4** = diabetic-retinopathy grade (0 none → 4 proliferative)

**Class imbalance (verified, train split):** grade 0 = 486, 1 = 128, 2 = 206, 3 = 194, 4 = 66. Grade 0 ≈ 45%, grade 4 ≈ 6% → use class-weighted loss and report macro-AUC / macro-F1 / confusion matrix, not raw accuracy.

**Cohort note:** separate DR cohort — **no metabolic label and no patient-level link** to iPOP. Used as a matched-disease imaging modality (see linkage note).

**Resolution caveat:** at 28×28 fine lesions (microaneurysms) are washed out — use 28px only to smoke-test the training loop; train the reported CNN on 224px.

---

## 4. Wearable biosensors — Li et al. 2017 ⏳ (partial — bytes still downloading)

| Field | Value |
|---|---|
| Source | Li, Dunn et al. 2017, *PLoS Biology* 15(1):e2001402 |
| DOI | 10.1371/journal.pbio.2001402 (PMID 28081144) |
| File | `Stanford_Wearables_data.tar` (~20.4 GB; tar of 43 `SubjectN_rawdata.zip`) |
| Subjects | 43 adults, ages 35–70, preference for T2D risk |
| Device | Basis smartwatch (+ Scanadu, iHealth, Masimo, Withings, RadTarge), up to 11 months (avg 152 days) |
| License | CC0 |

**Signals (from the paper — ✅ known):** heart rate, skin temperature, blood oxygen (SpO2), steps, sleep, walking/biking/running minutes, calories, accelerometer/activity, weight, gamma/X-ray radiation exposure. >250,000 measurements/day across all sensors.

**⏳ TO CONFIRM ON EXTRACTION** (fill once `Subject1_rawdata.zip` is unzipped):
- exact per-sensor **sampling rates** (HR, skin-temp, SpO2 cadences differ)
- exact **file layout inside each subject zip** (which CSVs, column names, units)
- **timestamp format** and timezone handling
- per-subject **recording length** and missingness pattern

**Cohort note:** 43 subjects use plain `SubjectN` IDs — **disjoint from the omics `Z-code` scheme**, so there is *no* patient-level link to the omics/CGM cohort. Only 20 of 43 had SSPG testing. Treated as a matched-cohort modality in the fusion (see below).

---

## Cross-modal linkage (measured, not assumed)

See `cgm_omics_overlap_summary.json` for the full result.

| Modality pair | Linkage | n |
|---|---|---|
| **CGM ↔ omics** | ✅ real same-patient (matched via iPOP site-codes `69-XXX`/`70-XXXX`) | **22** (19 with SSPG; 10 IS / 9 IR) |
| Wearable ↔ omics | ✗ none (disjoint `SubjectN` vs `Z-code` IDs) | 0 |
| Retina ↔ any | ✗ none (separate DR cohort, no metabolic label) | 0 |

**Consequence for modelling (see PLAYBOOK §C0):** the four modalities are fused on a
principled **virtual cohort** whose cross-modal dependencies are calibrated to the real
associations measured on the 22 co-observed CGM↔omics patients, validated with a scramble
negative-control, and anchored by the real (small-n) CGM+omics fusion result. The pipeline
is architecturally ready for a fully-linked cohort — only the loader changes.
