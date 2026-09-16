"""Tests for the time-series foundation-model embedding path.

The real model (AutonLab/MOMENT-1-small) is loaded ONCE per session and reused;
if torch/momentfm or the weights are unavailable the whole module skips rather
than faking an embedding. All series are tiny synthetic signals built in-process.
"""
import numpy as np
import pandas as pd
import pytest

from hurdle.features.ts_foundation import (
    EMBED_DIM,
    SEQ_LEN,
    _resample_to_length,
    build_ts_embeddings,
    embed_series,
    load_ts_model,
)

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def model():
    #load the real foundation model once; skip the module if it cannot be loaded
    #(offline, missing deps) so we never fabricate embeddings to make tests pass
    pytest.importorskip("torch")
    pytest.importorskip("momentfm")
    try:
        return load_ts_model()
    except Exception as exc:                        #noqa: BLE001
        pytest.skip(f"foundation model unavailable: {exc}")


def _flat(n=600):
    return np.full(n, 70.0)


def _oscillating(n=600, period=50):
    t = np.arange(n)
    return 70.0 + 20.0 * np.sin(2 * np.pi * t / period)


def test_resample_length_and_edges():
    #downsample, upsample, and degenerate inputs all yield exactly SEQ_LEN
    assert len(_resample_to_length(np.arange(5000), SEQ_LEN)) == SEQ_LEN
    assert len(_resample_to_length(np.arange(30), SEQ_LEN)) == SEQ_LEN
    assert np.all(_resample_to_length([], SEQ_LEN) == 0.0)
    assert np.all(_resample_to_length([42.0], SEQ_LEN) == 42.0)


def test_resample_drops_nonfinite():
    #NaN/inf are dropped before resampling; a constant survivor stays constant
    x = [5.0, np.nan, 5.0, np.inf, 5.0]
    out = _resample_to_length(x, SEQ_LEN)
    assert np.isfinite(out).all()
    assert out == pytest.approx(5.0)


def test_embedding_fixed_length_and_finite(model):
    emb = embed_series(model, _oscillating())
    assert emb.shape == (EMBED_DIM,)
    assert np.isfinite(emb).all()


def test_embedding_deterministic(model):
    #same input -> identical embedding (eval + no_grad)
    x = _oscillating()
    a = embed_series(model, x)
    b = embed_series(model, x)
    assert np.array_equal(a, b)


def test_embedding_differs_for_different_inputs(model):
    #a flat series and an oscillating one must land far apart in embedding space
    flat = embed_series(model, _flat())
    osc = embed_series(model, _oscillating())
    assert not np.allclose(flat, osc)
    dist = np.linalg.norm(flat - osc)
    ref = np.linalg.norm(osc)
    assert dist > 1e-3 * ref


def test_embedding_robust_to_length(model):
    #short and long versions of the same shape still embed to a fixed length
    short = embed_series(model, _oscillating(n=40))
    long = embed_series(model, _oscillating(n=6000))
    assert short.shape == long.shape == (EMBED_DIM,)


def test_build_ts_embeddings_shapes(model, tmp_path):
    #two synthetic wearable subjects + a tiny CGM feed -> one row each, EMBED_DIM
    #columns, correct indices, and parquet files written to disk
    idx = pd.date_range("2015-01-01", periods=3 * 1440, freq="1min", tz="UTC")
    h = idx.hour + idx.minute / 60.0
    for stem, phase in (("Basis_101", 0.0), ("Basis_102", 6.0)):
        hr = 65 + 10 * np.cos(2 * np.pi * (h - phase) / 24.0)
        df = pd.DataFrame(
            {"hr": hr, "accel_magnitude": 50.0, "skin_temp": 90.0}, index=idx
        )
        df.to_parquet(tmp_path / f"{stem}.parquet")

    #tab-separated CGM feed matching the Hall column contract, two subjects
    t = pd.date_range("2014-02-03", periods=600, freq="5min")
    cgm_rows = []
    for sid, base in (("1636-69-001", 100.0), ("1636-69-002", 140.0)):
        vals = base + 30 * np.sin(np.arange(600) / 20.0)
        cgm_rows.append(pd.DataFrame({
            "DisplayTime": t.astype(str),
            "GlucoseValue": [f"{v:.0f}" for v in vals],
            "subjectId": sid,
        }))
    cgm_path = tmp_path / "cgm_feed"
    pd.concat(cgm_rows).to_csv(cgm_path, sep="\t", index=False)

    wear_out = tmp_path / "wear_emb.parquet"
    cgm_out = tmp_path / "cgm_emb.parquet"
    wear_df, cgm_df = build_ts_embeddings(
        tmp_path, cgm_path, model=model, wear_out=wear_out, cgm_out=cgm_out
    )

    assert wear_df.shape == (2, EMBED_DIM)
    assert list(wear_df.index) == ["Basis_101", "Basis_102"]
    assert cgm_df.shape == (2, EMBED_DIM)
    assert list(cgm_df.index) == ["1636-69-001", "1636-69-002"]
    assert np.isfinite(wear_df.values).all()
    assert np.isfinite(cgm_df.values).all()
    assert wear_out.exists() and cgm_out.exists()
    #the two wearable subjects have shifted circadian phase -> distinct embeddings
    assert not np.allclose(wear_df.iloc[0].values, wear_df.iloc[1].values)
