"""Tests for the SHAP feature-attribution layer.

The synthetic fixture builds a matrix where the target is a KNOWN function of a
few features and one column is pure, target-independent noise. A correct
attribution must rank the true signal features at the top by mean|SHAP| and push
the injected noise column toward the bottom. A tiny real-data test grounds the
contract against the actual S8 matrix + consensus panel when both are on disk.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hurdle.explain.shap_attribution import (
    compare_rankings,
    fit_and_explain,
    signal_concentration,
)

REAL_INTERIM = Path("data/interim")
REAL_CONSENSUS = Path("reports/consensus_panel_sspg.csv")


def _synthetic(n=400, seed=0):
    #target is driven by sig_a (strong) and sig_b (moderate); noise_pure is drawn
    #independently of y and must attribute ~nothing. distractor_* are real inputs
    #with no effect. n is large enough that the tree separates signal from noise.
    rng = np.random.default_rng(seed)
    sig_a = rng.normal(size=n)
    sig_b = rng.normal(size=n)
    noise_pure = rng.normal(size=n)
    distractor_1 = rng.normal(size=n)
    distractor_2 = rng.normal(size=n)
    y = 5.0 * sig_a + 2.0 * sig_b + rng.normal(scale=0.1, size=n)
    X = pd.DataFrame(
        {
            "distractor_1": distractor_1,
            "sig_a": sig_a,
            "noise_pure": noise_pure,
            "distractor_2": distractor_2,
            "sig_b": sig_b,
        }
    )
    return X, pd.Series(y, name="target")


def test_known_signal_feature_ranks_top_by_shap():
    X, y = _synthetic()
    out = fit_and_explain(X, y, model="xgboost")
    shap_rank = out["shap_ranking"]
    #the strongest driver must be #1 by mean|SHAP|
    assert shap_rank.iloc[0]["analyte"] == "sig_a"
    #both true signal features rank in the top 2
    top2 = set(shap_rank.head(2)["analyte"])
    assert top2 == {"sig_a", "sig_b"}


def test_pure_noise_feature_ranks_near_bottom():
    X, y = _synthetic()
    out = fit_and_explain(X, y, model="xgboost")
    shap_rank = out["shap_ranking"].set_index("analyte")
    n_feat = len(shap_rank)
    #the target-independent column ranks in the bottom half...
    assert shap_rank.loc["noise_pure", "rank_shap"] > n_feat / 2
    #...and its mean|SHAP| is a small fraction of the top feature's
    top_val = shap_rank["mean_abs_shap"].max()
    assert shap_rank.loc["noise_pure", "mean_abs_shap"] < 0.1 * top_val


def test_gain_ranking_agrees_with_shap_on_top_signal():
    X, y = _synthetic()
    out = fit_and_explain(X, y, model="xgboost")
    gain_rank = out["gain_ranking"]
    #native gain, an independent measure, also puts sig_a first
    assert gain_rank.iloc[0]["analyte"] == "sig_a"
    assert set(gain_rank.head(2)["analyte"]) == {"sig_a", "sig_b"}


def test_shap_values_shape_and_outputs():
    X, y = _synthetic(n=120)
    out = fit_and_explain(X, y, model="xgboost")
    assert out["shap_values"].shape == (120, X.shape[1])
    assert out["feature_names"] == list(X.columns)
    #both rankings cover every feature exactly once
    assert set(out["shap_ranking"]["analyte"]) == set(X.columns)
    assert set(out["gain_ranking"]["analyte"]) == set(X.columns)
    assert list(out["shap_ranking"]["rank_shap"]) == list(range(1, X.shape[1] + 1))


def test_unsupported_model_raises():
    X, y = _synthetic(n=50)
    with pytest.raises(ValueError):
        fit_and_explain(X, y, model="randomforest")


def test_signal_concentration_counts_few_features():
    X, y = _synthetic()
    out = fit_and_explain(X, y, model="xgboost")
    conc = signal_concentration(out["shap_ranking"], thresholds=(0.80, 0.95))
    #two features generate the signal -> ~2 features should reach 80% of the mass
    assert conc["n_features_for_fraction"][0.80] <= 2
    assert conc["cumulative_fraction"][-1] == pytest.approx(1.0)
    assert conc["n_features_total"] == X.shape[1]


def test_compare_rankings_joins_three_methods(tmp_path):
    X, y = _synthetic()
    out = fit_and_explain(X, y, model="xgboost")
    #a stand-in consensus file: sig_a/sig_b frequent, noise absent entirely
    con = pd.DataFrame(
        {
            "analyte": ["sig_a", "sig_b", "distractor_1"],
            "vote_count": [59, 50, 3],
            "selection_frequency": [1.0, 0.85, 0.05],
        }
    )
    con_csv = tmp_path / "consensus.csv"
    con.to_csv(con_csv, index=False)

    table = compare_rankings(out["shap_ranking"], out["gain_ranking"], con_csv, top_n=5)
    assert list(table.columns) == [
        "analyte",
        "mean_abs_shap",
        "native_gain",
        "consensus_freq",
        "rank_shap",
        "rank_gain",
    ]
    #top row is sig_a and it carries the max consensus frequency
    assert table.iloc[0]["analyte"] == "sig_a"
    assert table.loc[table["analyte"] == "sig_a", "consensus_freq"].iloc[0] == 1.0
    #an analyte missing from the consensus file is NaN, never a fabricated 0
    noise_row = table[table["analyte"] == "noise_pure"]
    if not noise_row.empty:
        assert pd.isna(noise_row["consensus_freq"].iloc[0])


@pytest.mark.skipif(
    not (REAL_INTERIM / "omics_S8_sspg_clean.csv").exists() or not REAL_CONSENSUS.exists(),
    reason="real S8 matrix or consensus panel not present",
)
def test_real_s8_attribution_agrees_with_consensus():
    from hurdle.features.omics import build_feature_matrix

    X, y, _ = build_feature_matrix(REAL_INTERIM, target="SSPG", add_ratios=True)
    out = fit_and_explain(X, y, model="xgboost")
    table = compare_rankings(out["shap_ranking"], out["gain_ranking"], REAL_CONSENSUS, top_n=20)
    #every top-SHAP analyte is a real column from the matrix
    assert set(table["analyte"]).issubset(set(X.columns))
    #consensus's most-selected analytes should surface high in the SHAP ranking:
    #at least half of the consensus top-5 appear in the SHAP top-10
    consensus = pd.read_csv(REAL_CONSENSUS).sort_values("selection_frequency", ascending=False)
    con_top5 = set(consensus.head(5)["analyte"])
    shap_top10 = set(out["shap_ranking"].head(10)["analyte"])
    assert len(con_top5 & shap_top10) >= 3
