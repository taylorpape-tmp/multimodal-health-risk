"""Tests for CGM glucose-variability features against traces with known metrics."""
import numpy as np
import pandas as pd
import pytest

from hurdle.features.cgm import (
    FEATURE_ORDER,
    build_cgm_matrix,
    extract_cgm_features,
)


def _trace(values, subject="s1", start="2014-02-03 00:00:00", freq_min=5):
    #long-format frame like the Hall feed: DisplayTime, GlucoseValue, subjectId
    t = pd.date_range(start, periods=len(values), freq=f"{freq_min}min")
    return pd.DataFrame({
        "DisplayTime": t.astype(str),
        "GlucoseValue": [str(v) for v in values],
        "subjectId": subject,
    })


@pytest.fixture
def flat_100():
    #constant 100 mg/dL, in range -> SD 0, CV 0, TIR 1
    return _trace([100] * 288)


@pytest.fixture
def square_wave():
    #alternating 80/160 every step over >24 h so the 24 h-lag MODD is computable
    return _trace(([80, 160] * 300))


def test_flat_series_zero_variability(flat_100):
    f = extract_cgm_features(flat_100)
    assert f["mean_glucose"] == pytest.approx(100.0)
    assert f["sd_glucose"] == pytest.approx(0.0, abs=1e-9)
    assert f["cv_glucose"] == pytest.approx(0.0, abs=1e-9)
    assert f["iqr_glucose"] == pytest.approx(0.0, abs=1e-9)
    assert f["mage"] == pytest.approx(0.0, abs=1e-9)
    assert f["tir"] == pytest.approx(1.0)
    assert f["tar"] == pytest.approx(0.0)
    assert f["tbr"] == pytest.approx(0.0)


def test_time_in_range_partition_sums_to_one(square_wave):
    f = extract_cgm_features(square_wave)
    assert f["tir"] + f["tar"] + f["tbr"] == pytest.approx(1.0)


def test_known_tir_partition():
    #100 in-range, 40 lows (<70), 60 highs (>180) -> exact fractions
    vals = [100] * 100 + [50] * 40 + [200] * 60
    f = extract_cgm_features(_trace(vals))
    n = len(vals)
    assert f["tbr"] == pytest.approx(40 / n)
    assert f["tar"] == pytest.approx(60 / n)
    assert f["tir"] == pytest.approx(100 / n)


def test_oscillation_has_positive_mage(square_wave):
    #80/160 swings are ~40 above the trace SD, so MAGE must be positive and ~80
    f = extract_cgm_features(square_wave)
    assert f["mage"] > 0
    assert f["mage"] == pytest.approx(80.0, rel=1e-6)


def test_cv_matches_definition():
    #CV = SD/mean*100 on a non-trivial series
    vals = [90, 110, 120, 80, 100, 130, 70, 100]
    f = extract_cgm_features(_trace(vals))
    g = np.array(vals, dtype=float)
    assert f["cv_glucose"] == pytest.approx(g.std(ddof=1) / g.mean() * 100)


def test_gmi_matches_definition(flat_100):
    #GMI = 3.31 + 0.02392*mean
    f = extract_cgm_features(flat_100)
    assert f["gmi"] == pytest.approx(3.31 + 0.02392 * 100.0)


def test_j_index_matches_definition():
    vals = [90, 110, 120, 80, 100, 130, 70, 100]
    f = extract_cgm_features(_trace(vals))
    g = np.array(vals, dtype=float)
    assert f["j_index"] == pytest.approx(0.001 * (g.mean() + g.std(ddof=1)) ** 2)


def test_risk_indices_directional():
    #a low-heavy trace has LBGI > HBGI; a high-heavy trace the reverse
    low = extract_cgm_features(_trace([55] * 100))
    high = extract_cgm_features(_trace([250] * 100))
    assert low["lbgi"] > low["hbgi"]
    assert high["hbgi"] > high["lbgi"]


def test_conga_modd_finite_on_long_trace():
    #two days of 5-min data: CONGA (1 h lag) and MODD (24 h lag) are computable
    rng = np.random.default_rng(0)
    vals = np.clip(rng.normal(120, 20, size=576), 40, 300).round().astype(int)
    f = extract_cgm_features(_trace(vals))
    assert np.isfinite(f["conga1"])
    assert np.isfinite(f["modd"])
    assert f["modd"] >= 0


def test_no_nan_on_valid_trace(square_wave):
    f = extract_cgm_features(square_wave)
    assert all(np.isfinite(v) for v in f.values())


def test_non_numeric_sentinels_are_dropped():
    #Hall data carries 'Low' sentinels; they must not poison the mean
    vals = [100] * 10 + ["Low"] * 3
    f = extract_cgm_features(_trace(vals))
    assert f["mean_glucose"] == pytest.approx(100.0)
    assert np.isfinite(f["sd_glucose"])


def test_build_matrix_contract():
    #two subjects -> 2 rows, exact feature columns, subject-id index, no NaN in
    #the fully-observed metrics
    df = pd.concat([
        _trace([100] * 288, subject="1636-69-001"),
        _trace(([80, 160] * 144), subject="1636-69-026"),
    ], ignore_index=True)
    path = None
    #round-trip through a temp file so we exercise the reader too
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as fh:
        df.to_csv(fh, sep="\t", index=False)
        path = fh.name
    mat = build_cgm_matrix(path)
    assert list(mat.columns) == FEATURE_ORDER
    assert mat.shape == (2, len(FEATURE_ORDER))
    assert mat.index.tolist() == ["1636-69-001", "1636-69-026"]
    assert mat.index.name == "subjectId"
    #flat subject is fully in range; oscillating subject has positive MAGE
    assert mat.loc["1636-69-001", "tir"] == pytest.approx(1.0)
    assert mat.loc["1636-69-026", "mage"] > 0


def test_missing_columns_raise(tmp_path):
    bad = tmp_path / "bad.tsv"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(bad, sep="\t", index=False)
    with pytest.raises(ValueError):
        build_cgm_matrix(str(bad))
