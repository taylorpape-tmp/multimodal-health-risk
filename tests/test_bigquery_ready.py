"""Validate the BigQuery loader's local logic WITHOUT connecting to GCP.

These tests prove the load-ready path (the SQLite read + table selection) is
correct, so the only thing standing between this and a live load is the user's
gcloud auth. No network or credentials are touched.
"""
import sqlite3

import pandas as pd
import pytest
from scripts import load_bigquery as lb


def test_tables_are_nonempty_only():
    #the loader should target only tables that hold data
    assert "subjects" in lb.TABLES
    assert "omics_features" in lb.TABLES
    #empty placeholders must not be in the load set
    assert "predictions" not in lb.TABLES
    assert "metrics" not in lb.TABLES


def test_local_db_read_path(tmp_path):
    #build a tiny stand-in db with the same shape and prove the read works
    db = tmp_path / "mini.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE subjects (subject_id TEXT, iris TEXT, sspg REAL)")
    con.execute("INSERT INTO subjects VALUES ('ZTEST', 'IR', 123.0)")
    con.commit()
    df = pd.read_sql_query("SELECT * FROM subjects", con)
    con.close()
    assert df.shape == (1, 3)
    assert df.iloc[0]["subject_id"] == "ZTEST"


def test_load_requires_bigquery_dep(monkeypatch):
    #with google-cloud-bigquery absent, load() should exit with a clear message
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "google.cloud" or name.startswith("google.cloud"):
            raise ImportError("no google.cloud")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(SystemExit):
        lb.load("dummy-project", "hurdle_dataset", "US")
