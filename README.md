# HURDLE — Multimodal Diabetes-Risk Pipeline

An end-to-end machine-learning pipeline that integrates **four biomedical data modalities** —
multi-omics + clinical labs, wearable biosensors, continuous glucose monitoring (CGM), and retinal
fundus imaging — into a single insulin-resistance / type-2-diabetes risk model, with the supporting
engineering (relational store, cloud infrastructure-as-code, CI, tests) built to a production register.

Built as a portfolio project for a Biomedical & Health AI Scientist role.

## Honest framing (read this first)

The four public datasets are **different patients**. No public dataset links all four modalities per
person. Where cohorts overlap they are used as real anchors (omics ∩ CGM = 22 shared patients via a
site-code crosswalk); where they do not, the fusion layer is demonstrated on a **virtual cohort** —
synthetic patients whose modalities are coupled through a hidden latent risk factor, calibrated to the
real overlap. This is stated explicitly wherever a fusion result appears. The pipeline is
cohort-agnostic: given a real single-cohort multimodal dataset, the same code runs unchanged.

Single-modality results (e.g. omics → SSPG regression, R²≈0.50 / Pearson r≈0.71, XGBoost under
leave-one-out CV on 59 real patients × 86 analytes) are computed on **real data** and labelled as such.
Exact values are hyperparameter-dependent; see `scripts/run_omics_baseline.py` to reproduce.

## What's inside

| Area | Location |
|---|---|
| ML base (LOO/group CV, metrics, transform ladder) + model wrappers | `src/hurdle/ml/` |
| Nested-consensus feature selection (filter/wrapper/embedded/explain) | `src/hurdle/feature_selection/` |
| Per-modality feature builders (omics, wearable, CGM, S4 high-dim) | `src/hurdle/features/` |
| Retinal CNN (transfer learning: RETFound / ResNet-50 / DINOv2) | `src/hurdle/imaging/` |
| Fusion + virtual-cohort layer | `src/hurdle/fusion/` |
| Statistics (bootstrap CIs, permutation tests, calibration) | `src/hurdle/stats/` |
| Privacy-preserving ML (differential privacy, federated averaging) | `src/hurdle/privacy/` |
| Relational store (SQLite → BigQuery) | `src/hurdle/db/` |
| Model serving (FastAPI /predict) | `src/hurdle/serving/` |
| Cloud IaC (AWS + GCP, Terraform) | `infra/` |
| Analysis / report scripts | `scripts/` |
| Tests (pytest) | `tests/` |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest -q          # run the test suite
ruff check src tests scripts
```

Data is not committed (see `.gitignore`). To fetch the public sources, see `scripts/download_data.sh`
and `docs/DATA_CARD.md`.

## Key scripts

```bash
python scripts/build_db.py             # build the SQLite store from cleaned interim data
python scripts/run_omics_baseline.py   # omics -> SSPG leave-one-out baseline
python scripts/run_consensus_panel.py  # nested-consensus biomarker panel on real omics
python scripts/run_fusion_demo.py      # fusion + controls on the virtual cohort
python scripts/train_imaging.py --backbone resnet50 --epochs 15 --res 224   # retinal CNN
python scripts/make_showcase_figures.py  # regenerate the figure suite
```

## Documentation

- `reports/PROJECT_REPORT.md` — full project writeup (methods, results, person-spec mapping)
- `docs/DATA_CARD.md` — dataset provenance, licences, sizes
- `docs/SQL_LAYER.md` — schema and query design
- `docs/CLOUD_SETUP.md` — AWS/GCP provisioning walkthrough
- `docs/SERVING.md` — Docker → ECR → deploy path

## Data sources

| Modality | Source | Licence |
|---|---|---|
| Multi-omics + clinical | Zhou et al. 2019, *Nature* (iPOP) | CC0 |
| Wearable biosensors | Li et al. 2017, *PLoS Biol* (iPOP) | CC0 |
| CGM | Hall et al. 2018, *PLoS Biol* | CC-BY |
| Retinal imaging | RetinaMNIST (MedMNIST v2) | CC-BY 4.0 |
