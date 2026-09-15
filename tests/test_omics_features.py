"""Tests for the omics feature builder (S9 IRIS classification, S8 SSPG regression).

Fast: a tiny synthetic interim dir is written to a tmp_path so nothing touches
real data on disk. One test does read the real interim files if present, but is
skipped when they are absent so the suite stays green on a clean checkout.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hurdle.features.omics import (
    IRIS_LABEL,
    SSPG_LABEL,
    build_feature_matrix,
    load_omics,
)

REAL_INTERIM = Path("data/interim")


def _write_synthetic(interim_dir, with_ratio=True, iris_nan_row=True):
    #tiny stand-in for the real cleaned files: patient rows, label column,
    #a handful of analyte columns. TGL+HDL present only when with_ratio.
    rng = np.random.default_rng(0)
    n = 8
    sid = [f"S{i:02d}" for i in range(n)]
    common = {
        "SubjectID": sid,
        "EGFR": rng.normal(90, 10, n),
        "WBC": rng.normal(6, 1, n),
    }
    if with_ratio:
        common["TGL"] = rng.uniform(50, 200, n)
        common["HDL"] = rng.uniform(35, 70, n)

    s9 = pd.DataFrame({**common, "IRIS": [0, 1, 0, 1, 0, 1, 0, 1]})
    if iris_nan_row:
        #one patient with a missing IRIS label, mirroring the real S9 file
        s9.loc[n - 1, "IRIS"] = np.nan
    s9.to_csv(Path(interim_dir) / "omics_S9_isir_clean.csv", index=False)

    s8 = pd.DataFrame({**common, "SSPG": rng.uniform(40, 280, n)})
    s8.to_csv(Path(interim_dir) / "omics_S8_sspg_clean.csv", index=False)


def test_load_omics_shapes_and_labels(tmp_path):
    _write_synthetic(tmp_path)
    s9, s8 = load_omics(tmp_path)
    #patient index preserved as SubjectID
    assert s9.index.name == "SubjectID" and s8.index.name == "SubjectID"
    #label columns present
    assert IRIS_LABEL in s9.columns and SSPG_LABEL in s8.columns
    #S9 keeps all 8 rows at load time (label filtering happens in the matrix step)
    assert len(s9) == 8 and len(s8) == 8


def test_build_matrix_iris_classification_contract(tmp_path):
    _write_synthetic(tmp_path)
    X, y, cols = build_feature_matrix(tmp_path, target="IRIS")
    #the NaN-label patient is dropped, not imputed or invented
    assert len(X) == 7 and len(y) == 7
    #contract: X columns exactly equal returned feature_cols, label not among them
    assert list(X.columns) == cols
    assert IRIS_LABEL not in cols
    #binary integer target, no NaN anywhere
    assert set(np.unique(y)) <= {0, 1}
    assert not X.isna().any().any()
    assert not pd.isna(y).any()
    #index alignment between X and y
    assert list(X.index) == list(y.index)


def test_build_matrix_sspg_regression_contract(tmp_path):
    _write_synthetic(tmp_path)
    X, y, cols = build_feature_matrix(tmp_path, target="SSPG")
    assert len(X) == 8 and len(y) == 8
    assert list(X.columns) == cols
    assert SSPG_LABEL not in cols
    #continuous target
    assert np.issubdtype(y.dtype, np.floating)
    assert not X.isna().any().any()
    assert not pd.isna(y).any()


def test_ratio_feature_present_when_inputs_exist(tmp_path):
    _write_synthetic(tmp_path, with_ratio=True)
    X, _, cols = build_feature_matrix(tmp_path, target="SSPG", add_ratios=True)
    assert "TG_HDL_ratio" in cols
    #value equals TGL/HDL computed from the raw columns
    expected = X["TGL"] / X["HDL"]
    assert np.allclose(X["TG_HDL_ratio"].values, expected.values)
    assert not X["TG_HDL_ratio"].isna().any()


def test_ratio_feature_absent_when_inputs_missing(tmp_path):
    _write_synthetic(tmp_path, with_ratio=False)
    X, _, cols = build_feature_matrix(tmp_path, target="SSPG", add_ratios=True)
    #no TGL/HDL columns -> the ratio must not be fabricated
    assert "TG_HDL_ratio" not in cols
    assert "TG_HDL_ratio" not in X.columns


def test_bad_target_raises(tmp_path):
    _write_synthetic(tmp_path)
    with pytest.raises(ValueError):
        build_feature_matrix(tmp_path, target="NOPE")


@pytest.mark.skipif(
    not (REAL_INTERIM / "omics_S9_isir_clean.csv").exists(),
    reason="real interim files not present",
)
def test_real_files_load_expected_shapes():
    #grounds the contract against the actual cleaned files when they are on disk
    s9, s8 = load_omics(REAL_INTERIM)
    assert s9.shape == (60, 153)   #152 features + IRIS label
    assert s8.shape == (59, 86)    #85 features + SSPG label
    X, y, cols = build_feature_matrix(REAL_INTERIM, target="IRIS", add_ratios=False)
    #one IRIS label is missing in the real S9 file -> that patient is dropped
    assert len(X) == 59 and not X.isna().any().any()
    assert set(np.unique(y)) <= {0, 1}
