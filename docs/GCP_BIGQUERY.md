# GCP BigQuery — cloud SQL store

This loads the project's relational store (the same tables that live locally in
`data/hurdle.db`) into **Google BigQuery**, demonstrating the GCP half of the
"cloud systems (AWS and GCP)" competency end to end. AWS holds object storage +
compute infra (S3/ECR/EC2 launch template via Terraform); GCP BigQuery holds the
queryable relational/analytical store — the same ingest data, two clouds, on
purpose.

## Why user-identity auth (not a service-account key)

The GCP organization enforces `iam.disableServiceAccountKeyCreation` (Google's
"Secure by Default" policy), so downloadable JSON keys are blocked — and that is
the more secure posture anyway. We authenticate with **Application Default
Credentials (ADC)** using your own GCP identity instead. No static key file is
created or stored.

## One-time setup (in your terminal)

```bash
#1. install the gcloud CLI if you don't have it
#   https://cloud.google.com/sdk/docs/install  (macOS: brew install --cask google-cloud-sdk)

#2. install the python client into the hurdle env
conda activate hurdle   # or: source your venv
pip install google-cloud-bigquery pandas-gbq

#3. authenticate with your own identity (opens a browser)
gcloud auth application-default login

#4. point the SDK at your project
gcloud config set project hurdle-diabetes
```

## Run the load

```bash
cd ~/Desktop/resume/HURDLE
python scripts/load_bigquery.py --project hurdle-diabetes --location EU
```

(`--location EU` keeps data in the EU; use `US` if you prefer. Pick one and keep
it — a dataset's location is fixed at creation.)

Expected output:

```
dataset ready: hurdle-diabetes.hurdle_dataset (EU)
  loaded subjects: 60 rows, 9 cols -> hurdle-diabetes.hurdle_dataset.subjects
  loaded omics_features: 14135 rows, 5 cols -> hurdle-diabetes.hurdle_dataset.omics_features
proof query (top 5 SSPG patients with analyte counts):
  ZXXXXXX  IRIS=IR  SSPG=...  analytes=...
  ...
```

## Example BigQuery SQL (run in the BigQuery Studio console)

The payoff of the SQL layer — one query assembles a modeling view instead of
stitching DataFrames by hand:

```sql
-- 1. insulin-resistant vs insulin-sensitive counts
SELECT iris, COUNT(*) AS n, ROUND(AVG(sspg), 1) AS mean_sspg
FROM `hurdle-diabetes.hurdle_dataset.subjects`
WHERE iris IN ('IR', 'IS')
GROUP BY iris;

-- 2. wide omics matrix for the S8 SSPG panel (subjects x analytes), one query
SELECT subject_id, analyte, value
FROM `hurdle-diabetes.hurdle_dataset.omics_features`
WHERE panel = 'S8_sspg';

-- 3. top analytes present across the most patients
SELECT analyte, COUNT(DISTINCT subject_id) AS n_patients
FROM `hurdle-diabetes.hurdle_dataset.omics_features`
GROUP BY analyte
ORDER BY n_patients DESC
LIMIT 10;
```

## Cost

The store is ~1.4 MB and 14k rows. BigQuery's free tier is 10 GB storage + 1 TB
queries/month, so this load and these queries cost **£0**. (Your account also has
free-trial credit.)

## What this demonstrates for the person spec

- **Cloud systems (AWS and GCP)** — Essential. AWS provisioned via Terraform
  (S3/ECR/EC2); GCP BigQuery loaded here with real data and real queries.
- **Database technologies / SQL** — Essential. The same normalized schema
  (subjects + long-format omics_features, joined by `subject_id`) runs on both
  SQLite (local) and BigQuery (cloud) — the store is portable by design.
