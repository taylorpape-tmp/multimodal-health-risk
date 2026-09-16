# Running this project

Two ways to run it, depending on how much you want to reproduce.

## 1. The interactive demo (one command, no setup)

You need Docker installed. From the repo root:

```bash
docker compose up
```

Then open http://localhost:8501 in a browser. First run builds the image
(a few minutes); after that it starts in seconds. Stop with Ctrl-C.

The demo has two tabs:

- **Real result (omics to SSPG)** runs on the real cleaned omics matrix that
  ships inside the image (59 patients, 85 analytes). It computes the
  leave-one-out prediction of steady-state plasma glucose live, with a
  label-permutation null test and an accumulated-local-effects curve. Nothing
  synthetic.
- **Fusion demo (virtual cohort)** is a slider playground on a synthetic cohort.
  No real patient carries all four modalities, so this tab exists to show the
  fusion machinery, not a clinical result. It is labelled as such in the app.

## 2. The test suite (verify the code runs)

You need Python 3.11+. From the repo root:

```bash
pip install -e ".[dev]"
pytest
```

The tests use small synthetic fixtures, so they need no external data and run
anywhere. They cover the model base class, each modality's feature builder, the
fusion controls, the privacy and stats layers, and the serving API.

## 3. The full pipeline (optional, needs the public data)

The raw datasets are large and are not committed. To fetch them from their
public sources:

```bash
bash scripts/download_data.sh          # all sources
bash scripts/download_data.sh omics    # or one at a time
```

Then the scripts under `scripts/` reproduce each result (for example
`python scripts/run_omics_baseline.py` for the omics to SSPG baseline). See the
script headers for what each one does.

## Layout

```
src/hurdle/    the package: features, models, fusion, privacy, stats, serving
app/           the Streamlit demo
scripts/       one script per result / data step
tests/         pytest suite
infra/         Terraform for the AWS/GCP deployment
data/cleaning/ the cleaning scripts that produce data/interim/
```
