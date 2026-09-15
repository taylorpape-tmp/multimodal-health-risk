"""Tests for the wide-omics (S4) feature-engineering pipeline, on a small
synthetic analytes-x-patients matrix so they run fast."""
import numpy as np
import pandas as pd
import pytest

from hurdle.features import omics_highdim as hd


@pytest.fixture
def fake_s4():
    #12 patients, build an S4-shaped sheet: analytes as rows, patient cols named
    #'NN-NNN/Zcode', plus summary cols the pipeline must ignore
    rng = np.random.default_rng(0)
    n_analytes, n_pat = 40, 12
    pcols = [f"69-{i:03d}/Z{i:05d}" for i in range(n_pat)]
    data = rng.normal(size=(n_analytes, n_pat))
    df = pd.DataFrame(data, columns=pcols)
    df.insert(0, "Unnamed: 0", [f"A{i}" for i in range(n_analytes)])
    df["All"] = 0.0
    df["Type"] = "metabolite"
    return df


def test_orient_transposes_to_patients_x_analytes(fake_s4):
    X, sites, z = hd.orient(fake_s4)
    assert X.shape == (12, 40)                 #patients x analytes
    assert sites[0] == "69-000" and z[0] == "Z00000"
    assert list(X.index)[0] == "Z00000"


def test_prevalence_filter_drops_mostly_missing():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [np.nan, np.nan, np.nan, 1.0]})
    out = hd.prevalence_filter(X, max_missing=0.5)
    assert "a" in out.columns and "b" not in out.columns


def test_variance_filter_drops_constant():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "const": [5.0, 5.0, 5.0]})
    out = hd.variance_filter(X)
    assert "a" in out.columns and "const" not in out.columns


def test_impute_uses_provided_medians_no_leak():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    med = pd.Series({"a": 10.0})               #train-fold median passed in
    out, used = hd.impute(X, medians=med)
    assert out["a"].tolist() == [1.0, 10.0, 3.0]
    assert used["a"] == 10.0


def test_correlation_prune_collapses_redundant():
    base = np.arange(20, dtype=float)
    X = pd.DataFrame({"a": base, "a_copy": base, "b": base[::-1]})
    out, dropped = hd.correlation_prune(X, threshold=0.95)
    #a and a_copy are identical -> one dropped; b is anti-correlated with a (|r|=1)
    assert "a_copy" in dropped or "b" in dropped
    assert out.shape[1] < 3


def test_pca_embed_shape_and_variance(fake_s4):
    X, _, _ = hd.orient(fake_s4)
    X, _ = hd.impute(X)
    pcs, evr = hd.pca_embed(X, n_components=5)
    assert pcs.shape == (12, 5)
    assert 0 < evr.sum() <= 1.0 + 1e-9


def test_build_highdim_matrix_trace(fake_s4):
    out = hd.build_highdim_matrix(fake_s4, n_pca=5)
    t = out["trace"]
    #the reduction is monotone non-increasing through the filter stages
    assert t["start_analytes"] == 40
    assert t["after_prevalence"] >= t["after_variance"] >= t["after_correlation_prune"]
    assert out["pca"].shape == (12, t["n_pca_components"])
    assert 0 < t["pca_variance_explained"] <= 1.0 + 1e-9
