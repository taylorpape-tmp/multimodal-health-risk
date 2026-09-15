"""Tests for the SQL layer: schema, ingest, and the JOIN/pivot queries.

Runs entirely on an in-memory SQLite engine so it is fast and touches no disk.
"""
import pandas as pd
import pytest
from sqlalchemy import create_engine

from hurdle.db import ingest, queries, schema


@pytest.fixture
def engine():
    #shared in-memory sqlite for the duration of one test
    eng = create_engine("sqlite:///:memory:")
    schema.create_all(eng)
    return eng


@pytest.fixture
def small_omics(tmp_path):
    #write tiny S8/S9 cleaned csvs in the on-disk format the ingest expects
    s9 = pd.DataFrame(
        {"IRIS": ["IR", "IS", "IR"], "ALKP": [1.0, 2.0, 3.0], "ALT": [4.0, 5.0, 6.0]},
        index=["ZAAA111", "ZBBB222", "ZCCC333"])
    s9.index.name = "SubjectID"
    s8 = pd.DataFrame(
        {"SSPG": [120.0, 80.0], "HDL": [40.0, 55.0], "TGL": [1.5, 0.9]},
        index=["ZAAA111", "ZDDD444"])
    s8.index.name = "SubjectID"
    s9.to_csv(tmp_path / "omics_S9_isir_clean.csv")
    s8.to_csv(tmp_path / "omics_S8_sspg_clean.csv")
    return str(tmp_path)


def test_schema_creates_all_tables(engine):
    names = set(schema.metadata.tables.keys())
    assert {"subjects", "omics_features", "wearable_features", "cgm_features",
            "subject_crosswalk", "predictions", "metrics"} <= names
    #tables actually exist in the db
    from sqlalchemy import inspect
    assert set(inspect(engine).get_table_names()) >= {"subjects", "omics_features"}


def test_ingest_omics_populates_subjects_and_features(engine, small_omics):
    stats = ingest.ingest_omics(engine, interim_dir=small_omics)
    #4 distinct subjects across the two panels (ZAAA111 shared)
    assert queries.subject_count(engine) == 4
    #S9: 3 subjects x 2 analytes = 6 cells; S8: 2 x 2 = 4 -> 10 total
    assert stats["omics_cells"] == 10


def test_omics_matrix_pivots_back_to_wide(engine, small_omics):
    ingest.ingest_omics(engine, interim_dir=small_omics)
    m = queries.omics_matrix(engine, "S9_isir")
    #3 subjects, label col + 2 analytes
    assert m.shape == (3, 3)
    assert "iris" in m.columns and "ALKP" in m.columns and "ALT" in m.columns
    assert m.loc["ZAAA111", "ALKP"] == 1.0
    assert m.loc["ZBBB222", "iris"] == "IS"


def test_sspg_label_attached_to_s8(engine, small_omics):
    ingest.ingest_omics(engine, interim_dir=small_omics)
    m = queries.omics_matrix(engine, "S8_sspg")
    assert "sspg" in m.columns
    assert m.loc["ZAAA111", "sspg"] == 120.0


def test_crosswalk_overlap_join(engine, small_omics):
    ingest.ingest_omics(engine, interim_dir=small_omics)
    cw = pd.DataFrame([
        {"omics_subject_id": "ZAAA111", "cgm_subject_id": "1636-69-001",
         "site_code": "69-001", "wearable_subject_id": None},
        {"omics_subject_id": "ZBBB222", "cgm_subject_id": None,
         "site_code": None, "wearable_subject_id": None},
    ])
    ingest.ingest_crosswalk(engine, cw)
    overlap = queries.overlap_cohort(engine)
    #only ZAAA111 has a CGM id -> exactly one overlap row, label joined
    assert len(overlap) == 1
    assert overlap.iloc[0]["omics_subject_id"] == "ZAAA111"
    assert overlap.iloc[0]["iris"] == "IR"


def test_feature_matrix_roundtrip(engine):
    #a wearable subjects-x-features frame -> long table -> wide query
    df = pd.DataFrame(
        {"rhr": [60.0, 55.0], "mvpa_min": [30.0, 45.0]},
        index=["Subject1", "Subject2"])
    df.index.name = "subject_id"
    ingest.ingest_feature_matrix(engine, schema.wearable_features, df)
    wide = queries.feature_wide(engine, "wearable_features")
    assert wide.shape == (2, 2)
    assert wide.loc["Subject1", "rhr"] == 60.0
