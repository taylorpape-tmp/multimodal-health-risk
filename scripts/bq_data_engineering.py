"""BigQuery data-engineering layer for the HURDLE store.

Demonstrates warehouse data-engineering technique (not just a load):
  - ELT layering: raw -> staging -> analytics datasets
  - partitioned + clustered analytics tables (query pruning)
  - analytical views: wide PIVOT matrix + multimodal patient view
  - window-function / CTE analytics
  - SQL data-quality checks (null, range, referential integrity)

Run after `python scripts/load_bigquery.py` has created the raw tables, or
standalone (it loads raw from SQLite itself). Auth = Application Default
Credentials (gcloud auth application-default login). Cost on this ~1.4 MB
store is within the BigQuery free tier.

Usage:
  python scripts/bq_data_engineering.py --project hurdle-diabetes --location EU
"""
import argparse
import sqlite3

import pandas as pd

DB = "data/hurdle.db"
RAW_TABLES = ["subjects", "omics_features", "cgm_features", "subject_crosswalk"]


def _read(table):
    con = sqlite3.connect(DB)
    df = pd.read_sql_query(f"SELECT * FROM {table}", con)
    con.close()
    return df


def run(project, location):
    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    raw = f"{project}.hurdle_raw"
    stg = f"{project}.hurdle_staging"
    ana = f"{project}.hurdle_analytics"
    for ds in (raw, stg, ana):
        d = bigquery.Dataset(ds)
        d.location = location
        client.create_dataset(d, exists_ok=True)
        print(f"dataset ready: {ds}")

    #ELT step 1 (extract+load): raw tables straight from SQLite
    for t in RAW_TABLES:
        df = _read(t)
        job = client.load_table_from_dataframe(
            df, f"{raw}.{t}",
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
        )
        job.result()
        print(f"raw.{t}: {job.output_rows} rows")

    #ELT step 2 (transform): staging = typed, cleaned, deduped
    ddl_staging = f"""
    CREATE OR REPLACE TABLE `{stg}.subjects` AS
      SELECT DISTINCT
        subject_id,
        UPPER(CAST(iris AS STRING)) AS iris_class,
        SAFE_CAST(sspg AS FLOAT64) AS sspg,
        SAFE_CAST(bmi AS FLOAT64) AS bmi,
        SAFE_CAST(age AS FLOAT64) AS age
      FROM `{raw}.subjects`
      WHERE subject_id IS NOT NULL;
    """
    client.query(ddl_staging).result()
    print("staging.subjects built")

    #ELT step 3 (load analytics): partitioned + clustered omics table
    #range-partition by a hashed bucket, cluster by panel+subject for pruning
    ddl_omics = f"""
    CREATE OR REPLACE TABLE `{ana}.omics_features`
      PARTITION BY RANGE_BUCKET(part_bucket, GENERATE_ARRAY(0, 10, 1))
      CLUSTER BY panel, subject_id AS
      SELECT
        subject_id, panel, analyte,
        SAFE_CAST(value AS FLOAT64) AS value,
        MOD(ABS(FARM_FINGERPRINT(subject_id)), 10) AS part_bucket
      FROM `{raw}.omics_features`
      WHERE value IS NOT NULL;
    """
    client.query(ddl_omics).result()
    print("analytics.omics_features built (partitioned + clustered)")

    #analytics view 1: wide subjects x analytes matrix via PIVOT (S8 SSPG panel)
    top = client.query(f"""
      SELECT analyte FROM `{ana}.omics_features`
      WHERE panel = 'S8_sspg'
      GROUP BY analyte ORDER BY COUNT(DISTINCT subject_id) DESC LIMIT 8
    """).result()
    analyte_list = ", ".join(f"'{r.analyte}'" for r in top)
    if analyte_list:
        ddl_pivot = f"""
        CREATE OR REPLACE VIEW `{ana}.omics_wide_s8` AS
        SELECT * FROM (
          SELECT subject_id, analyte, value
          FROM `{ana}.omics_features` WHERE panel = 'S8_sspg'
        ) PIVOT (AVG(value) FOR analyte IN ({analyte_list}));
        """
        client.query(ddl_pivot).result()
        print("analytics.omics_wide_s8 view built (PIVOT long->wide)")

    #analytics view 2: multimodal patient view (subjects + omics agg + cgm agg via crosswalk)
    ddl_mm = f"""
    CREATE OR REPLACE VIEW `{ana}.patient_multimodal` AS
    WITH omics_agg AS (
      SELECT subject_id,
             COUNT(DISTINCT analyte) AS n_analytes,
             AVG(value) AS omics_mean
      FROM `{ana}.omics_features` GROUP BY subject_id
    ),
    cgm_agg AS (
      SELECT x.omics_subject_id AS subject_id,
             MAX(IF(c.feature='mean_glucose', c.value, NULL)) AS cgm_mean_glucose,
             MAX(IF(c.feature='tir', c.value, NULL)) AS cgm_time_in_range
      FROM `{raw}.subject_crosswalk` x
      JOIN `{raw}.cgm_features` c ON c.subject_id = x.cgm_subject_id
      GROUP BY x.omics_subject_id
    )
    SELECT s.subject_id, s.iris_class, s.sspg, s.bmi, s.age,
           o.n_analytes, o.omics_mean,
           g.cgm_mean_glucose, g.cgm_time_in_range,
           g.cgm_mean_glucose IS NOT NULL AS has_cgm
    FROM `{stg}.subjects` s
    LEFT JOIN omics_agg o USING (subject_id)
    LEFT JOIN cgm_agg g USING (subject_id);
    """
    client.query(ddl_mm).result()
    print("analytics.patient_multimodal view built (multimodal join)")

    #window-function analytics: rank patients by SSPG within IRIS class
    print("\nwindow-function query (top SSPG per IRIS class):")
    q_win = f"""
      SELECT subject_id, iris_class, sspg,
             RANK() OVER (PARTITION BY iris_class ORDER BY sspg DESC) AS rk
      FROM `{stg}.subjects` WHERE sspg IS NOT NULL QUALIFY rk <= 2
      ORDER BY iris_class, rk
    """
    for r in client.query(q_win).result():
        print(f"  {r.iris_class}  rank {r.rk}: {r.subject_id} SSPG={r.sspg}")

    #data-quality checks (SQL assertions)
    print("\ndata-quality checks:")
    dq = f"""
      SELECT
        (SELECT COUNTIF(sspg IS NULL) FROM `{stg}.subjects`) AS null_sspg,
        (SELECT COUNTIF(value < 0) FROM `{ana}.omics_features`) AS negative_values,
        (SELECT COUNT(*) FROM `{raw}.subject_crosswalk` x
           LEFT JOIN `{stg}.subjects` s ON s.subject_id = x.omics_subject_id
           WHERE s.subject_id IS NULL) AS orphan_crosswalk_rows
    """
    r = list(client.query(dq).result())[0]
    print(f"  subjects with null SSPG: {r.null_sspg}")
    print(f"  negative omics values: {r.negative_values}")
    print(f"  orphan crosswalk rows (referential integrity): {r.orphan_crosswalk_rows}")
    print("\ndata-engineering layer complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--location", default="EU")
    a = ap.parse_args()
    run(a.project, a.location)
