"""Tests for the four-family nested consensus selector."""
import numpy as np
import pandas as pd
import pytest

from hurdle.feature_selection.nested_consensus import (
    ConsensusConfig,
    embedded_vote,
    explain_vote,
    filter_vote,
    run_nested_consensus,
    wrapper_vote,
)


@pytest.fixture
def pool_frame():
    #f0,f1 informative; f2,f3,f4 noise. 30 rows.
    rng = np.random.default_rng(0)
    n = 30
    X = rng.normal(size=(n, 5))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 0.05 * rng.normal(size=n)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    return df


FEATS = [f"f{i}" for i in range(5)]


def test_each_family_returns_subset_of_pool(pool_frame):
    cfg = ConsensusConfig()
    y = pool_frame["target"].values
    D = pool_frame[FEATS]
    tr = list(range(len(y)))
    for fam in (filter_vote, wrapper_vote, embedded_vote, explain_vote):
        chosen = fam(tr, D, y, FEATS, cfg)
        assert isinstance(chosen, set)
        assert chosen <= set(FEATS), f"{fam.__name__} returned features outside the pool"


def test_families_pick_up_informative_features(pool_frame):
    #on a strong linear signal, the informative features should be voted by
    #at least one family (sanity that the selectors actually select signal)
    cfg = ConsensusConfig()
    y = pool_frame["target"].values
    D = pool_frame[FEATS]
    tr = list(range(len(y)))
    all_votes = set()
    for fam in (filter_vote, wrapper_vote, embedded_vote, explain_vote):
        all_votes |= fam(tr, D, y, FEATS, cfg)
    assert "f0" in all_votes and "f1" in all_votes


def test_explain_topk_respected(pool_frame):
    cfg = ConsensusConfig()
    cfg.explain_topk = 2
    y = pool_frame["target"].values
    chosen = explain_vote(list(range(len(y))), pool_frame[FEATS], y, FEATS, cfg)
    assert len(chosen) == 2


def test_nested_consensus_runs_and_returns_valid_metrics(pool_frame):
    cfg = ConsensusConfig()
    summary, freq = run_nested_consensus(pool_frame, FEATS, "target", cfg)
    #summary: one row per final model, R2 finite and <= 1
    assert set(summary["final_model"]) == set(cfg.final_models)
    assert summary["external_R2"].notna().all()
    assert (summary["external_R2"] <= 1.0).all()
    #freq: every feature accounted for, frequencies in [0,1]
    assert set(freq["feature"]) == set(FEATS)
    assert (freq["frequency"] >= 0).all() and (freq["frequency"] <= 1).all()


def test_informative_features_frozen_most_often(pool_frame):
    #the two real features should enter the frozen set more often than pure noise
    cfg = ConsensusConfig()
    _, freq = run_nested_consensus(pool_frame, FEATS, "target", cfg)
    freq = freq.set_index("feature")["in_frozen_set"]
    assert freq["f0"] > freq["f4"]
    assert freq["f1"] > freq["f4"]
