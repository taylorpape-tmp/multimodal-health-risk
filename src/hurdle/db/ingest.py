"""Build the engine and ingest cleaned data into the store.

get_engine() returns a SQLite engine by default; pass a BigQuery URL (or set
HURDLE_DB_URL) to point the exact same ingest/query code at GCP.
"""
import os

import pandas as pd
from sqlalchemy import create_engine, insert

from . import schema


def get_engine(url=None, echo=False):
    #default to a local sqlite file under the repo's data/ dir; env var or arg
    #can point at BigQuery ('bigquery://project/dataset') for the cloud demo
    url = url or os.environ.get("HURDLE_DB_URL", "sqlite:///data/hurdle.db")
    return create_engine(url, echo=echo)


def _long_from_wide(df, id_name, feat_name):
    #turn a subjects-x-features wide frame into (subject, feature, value) rows
    long = df.reset_index().melt(id_vars=df.index.name or "index",
                                 var_name=feat_name, value_name="value")
    long = long.rename(columns={df.index.name or "index": id_name})
    return long.dropna(subset=["value"])


def ingest_omics(engine, interim_dir="data/interim"):
    #S8 (SSPG) and S9 (IRIS) cleaned matrices -> subjects + omics_features (long)
    panels = {"S8_sspg": ("omics_S8_sspg_clean.csv", "SSPG"),
              "S9_isir": ("omics_S9_isir_clean.csv", "IRIS")}
    subj_rows, omics_rows = {}, []
    for panel, (fname, label) in panels.items():
        path = os.path.join(interim_dir, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0)
        analyte_cols = [c for c in df.columns if c != label]
        for sid, row in df.iterrows():
            subj_rows.setdefault(sid, {})
            if label == "SSPG":
                subj_rows[sid]["sspg"] = float(row[label])
            else:
                subj_rows[sid]["iris"] = str(row[label])
            for a in analyte_cols:
                v = row[a]
                if pd.notna(v):
                    omics_rows.append({"subject_id": sid, "panel": panel,
                                       "analyte": a, "value": float(v)})
    subjects = [{"subject_id": sid, **vals} for sid, vals in subj_rows.items()]
    with engine.begin() as conn:
        if subjects:
            #insert subjects that carry omics; OR IGNORE lets a subject appearing
            #in both panels (S8 and S9) insert once without a duplicate-key error
            for s in subjects:
                conn.execute(insert(schema.subjects).prefix_with("OR IGNORE"), s)
        if omics_rows:
            conn.execute(insert(schema.omics_features), omics_rows)
    return {"subjects": len(subjects), "omics_cells": len(omics_rows)}


def ingest_feature_matrix(engine, table, df, id_name="subject_id"):
    #generic loader for wearable/cgm subjects-x-features frames into a long table
    long = _long_from_wide(df, id_name, "feature")
    rows = long.to_dict("records")
    with engine.begin() as conn:
        if rows:
            conn.execute(insert(table), rows)
    return {"rows": len(rows)}


def ingest_crosswalk(engine, df):
    #df columns: omics_subject_id, cgm_subject_id, site_code, wearable_subject_id
    keep = ["omics_subject_id", "cgm_subject_id", "site_code", "wearable_subject_id"]
    rows = [{k: (r.get(k) if pd.notna(r.get(k)) else None) for k in keep}
            for r in df.to_dict("records")]
    with engine.begin() as conn:
        if rows:
            conn.execute(insert(schema.subject_crosswalk), rows)
    return {"rows": len(rows)}
