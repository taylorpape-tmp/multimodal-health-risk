"""Load the local HURDLE SQLite store into a real BigQuery dataset.

This is the cloud-SQL demonstration for the project: the same relational store
that lives locally in data/hurdle.db is loaded into Google BigQuery, proving the
GCP half of the "cloud systems (AWS and GCP)" competency end to end.

Auth: uses Application Default Credentials (ADC). Run
    gcloud auth application-default login
once in your terminal first, so no service-account key file is needed (key
creation is blocked by the org's Secure-by-Default policy, and ADC with your own
identity is the more secure pattern anyway).

Why a dedicated loader instead of pointing ingest.get_engine at a bigquery:// URL:
the local ingest uses SQLite-only SQL (OR IGNORE, autoincrement PKs) that BigQuery
does not support. Reading each table into a DataFrame and using the BigQuery
client's load-from-dataframe path is the standard, dialect-clean way to load a
small store, so that is what this script does.

Usage:
    python scripts/load_bigquery.py --project hurdle-diabetes
    python scripts/load_bigquery.py --project hurdle-diabetes --dataset hurdle_dataset --location EU
"""
import argparse
import sqlite3
import sys

import pandas as pd

LOCAL_DB = "data/hurdle.db"
#only load tables that actually hold data; skip empty placeholders
TABLES = ["subjects", "omics_features"]


def load(project, dataset, location):
    try:
        from google.cloud import bigquery
    except ImportError:
        sys.exit("google-cloud-bigquery not installed. Run: pip install google-cloud-bigquery")

    client = bigquery.Client(project=project)
    ds_id = f"{project}.{dataset}"

    #create the dataset if it does not exist (idempotent)
    ds = bigquery.Dataset(ds_id)
    ds.location = location
    client.create_dataset(ds, exists_ok=True)
    print(f"dataset ready: {ds_id} ({location})")

    con = sqlite3.connect(LOCAL_DB)
    for table in TABLES:
        df = pd.read_sql_query(f"SELECT * FROM {table}", con)
        if df.empty:
            print(f"  skip {table}: 0 rows")
            continue
        table_id = f"{ds_id}.{table}"
        job = client.load_table_from_dataframe(
            df, table_id,
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
        )
        job.result()
        loaded = client.get_table(table_id)
        print(f"  loaded {table}: {loaded.num_rows} rows, {len(loaded.schema)} cols -> {table_id}")
    con.close()

    #proof query: join subjects to their omics feature counts, top SSPG patients
    q = f"""
        SELECT s.subject_id, s.iris, s.sspg, COUNT(o.analyte) AS n_analytes
        FROM `{ds_id}.subjects` s
        LEFT JOIN `{ds_id}.omics_features` o USING (subject_id)
        WHERE s.sspg IS NOT NULL
        GROUP BY s.subject_id, s.iris, s.sspg
        ORDER BY s.sspg DESC
        LIMIT 5
    """
    print("\nproof query (top 5 SSPG patients with analyte counts):")
    for row in client.query(q).result():
        print(f"  {row.subject_id}  IRIS={row.iris}  SSPG={row.sspg}  analytes={row.n_analytes}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="GCP project id, e.g. hurdle-diabetes")
    ap.add_argument("--dataset", default="hurdle_dataset")
    ap.add_argument("--location", default="US", help="BigQuery location: US or EU")
    a = ap.parse_args()
    load(a.project, a.dataset, a.location)
