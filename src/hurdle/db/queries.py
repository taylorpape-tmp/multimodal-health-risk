"""Query helpers — the JOINs that turn the normalized store back into modeling
matrices. This is the payoff of the SQL layer: one query assembles the feature
matrix instead of stitching DataFrames by hand.
"""
import pandas as pd
from sqlalchemy import text


def omics_matrix(engine, panel):
    #pivot the long omics_features back to a subjects-x-analytes matrix for one
    #panel, joined to its label from subjects (SSPG for S8, IRIS for S9)
    sql = text("""
        SELECT o.subject_id, o.analyte, o.value,
               s.sspg AS sspg, s.iris AS iris
        FROM omics_features o
        JOIN subjects s ON s.subject_id = o.subject_id
        WHERE o.panel = :panel
    """)
    df = pd.read_sql(sql, engine, params={"panel": panel})
    if df.empty:
        return df
    label_col = "sspg" if panel == "S8_sspg" else "iris"
    labels = df.groupby("subject_id")[label_col].first()
    wide = df.pivot(index="subject_id", columns="analyte", values="value")
    wide.insert(0, label_col, labels)
    return wide


def overlap_cohort(engine):
    #the honest real-overlap cohort: subjects that have BOTH omics and CGM,
    #joined through the crosswalk. this is the real-N validation anchor.
    sql = text("""
        SELECT x.omics_subject_id, x.cgm_subject_id, x.site_code,
               s.iris, s.sspg
        FROM subject_crosswalk x
        JOIN subjects s ON s.subject_id = x.omics_subject_id
        WHERE x.cgm_subject_id IS NOT NULL
    """)
    return pd.read_sql(sql, engine)


def feature_wide(engine, table_name, subject_ids=None):
    #pivot a long feature table (wearable_features / cgm_features) to wide
    q = f"SELECT subject_id, feature, value FROM {table_name}"
    df = pd.read_sql(text(q), engine)
    if df.empty:
        return df
    if subject_ids is not None:
        df = df[df["subject_id"].isin(list(subject_ids))]
    return df.pivot(index="subject_id", columns="feature", values="value")


def subject_count(engine):
    with engine.connect() as conn:
        return conn.execute(text("SELECT COUNT(*) FROM subjects")).scalar()
