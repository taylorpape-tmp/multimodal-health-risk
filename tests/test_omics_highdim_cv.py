"""Tests for the leak-safe cross-validated S4 reduction+model wrapper.

Small synthetic wide matrix (few patients, many analytes) so the suite stays
fast. The three load-bearing guarantees are asserted directly:
  1. no fold's reduction is fit on the held-out patient (leakage guard),
  2. the pooled out-of-fold prediction vector has length n,
  3. on a signal-bearing set the leaky reduction is >= the honest one (the
     optimism gap points the right way).
"""
import numpy as np
import pandas as pd
import pytest

from hurdle.features import omics_highdim_cv as cv


def _wide(n_pat=16, n_analytes=60, signal=True, seed=0):
    #patients x analytes with a handful of informative analytes; y is a linear
    #combination of them plus noise so the reduction has something to recover
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_pat, n_analytes))
    idx = [f"Z{i:05d}" for i in range(n_pat)]
    cols = [f"A{j}" for j in range(n_analytes)]
    df = pd.DataFrame(X, index=idx, columns=cols)
    if signal:
        y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 1.5 * X[:, 2] + 0.3 * rng.normal(size=n_pat)
    else:
        y = rng.normal(size=n_pat)
    return df, y


def test_oof_length_equals_n():
    X, y = _wide()
    res = cv.highdim_cv_score(X, y, n_pca=5, model="ridge")
    assert res["oof_pred"].shape == (len(y),)
    assert res["n"] == len(y)
    assert not np.isnan(res["oof_pred"]).any()          #every patient got an OOF pred


def test_y_length_mismatch_raises():
    X, y = _wide()
    with pytest.raises(ValueError):
        cv.highdim_cv_score(X, y[:-1], n_pca=5)


def test_no_fold_fits_on_held_out_row(monkeypatch):
    #spy on fit_reduction: record the index labels of every training matrix it
    #sees, then assert the fold's held-out patient never appears among them
    X, y = _wide()
    seen_train_labels = []
    real = cv.fit_reduction

    def spy(X_train, **kw):
        seen_train_labels.append(set(X_train.index))
        return real(X_train, **kw)

    monkeypatch.setattr(cv, "fit_reduction", spy)
    res = cv.highdim_cv_score(X, y, n_pca=5, model="ridge")   #honest -> per-fold fit

    labels = list(X.index)
    assert len(seen_train_labels) == len(res["fold_trace"])   #one fit per fold
    for tr_labels, fold in zip(seen_train_labels, res["fold_trace"]):
        held = {labels[i] for i in fold["test_idx"]}
        assert held.isdisjoint(tr_labels)                     #test row absent from fit
        assert len(tr_labels) == len(labels) - len(held)      #exactly the rest


def _low_rank(n_pat, n_analytes, k_latent, seed, noise=1.5):
    #analytes are noisy readouts of k latent factors; y depends on the factors,
    #so a reduction has real low-rank signal to recover
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n_pat, k_latent))
    load = rng.normal(size=(k_latent, n_analytes))
    X = factors @ load + noise * rng.normal(size=(n_pat, n_analytes))
    y = 3.0 * factors[:, 0] - 2.0 * factors[:, 1] + 0.3 * rng.normal(size=n_pat)
    df = pd.DataFrame(X, index=[f"Z{i:05d}" for i in range(n_pat)],
                      columns=[f"A{j}" for j in range(n_analytes)])
    return df, y


def test_leaky_ge_honest_on_signal_set():
    #the optimism gap points the honest way: the leaky (all-rows) reduction
    #scores >= the honest (per-fold) one on a signal-bearing set.
    #NOTE: under leave-one-OUT the unsupervised-PCA leak is O(1/n), the held-out
    #point barely moves the axes, so the gap is near zero and sign-noisy. The
    #leak only bites materially when a whole CHUNK is held out (the leaky PCA is
    #then fit including many test rows). We therefore hold out large groups, and
    #average over replicate datasets so the DIRECTION is asserted, not one draw.
    n_pat, n_groups = 24, 2
    groups = np.tile(np.arange(n_groups), n_pat // n_groups)[:n_pat]
    honest_r2, leaky_r2 = [], []
    for seed in range(5):
        X, y = _low_rank(n_pat=n_pat, n_analytes=150, k_latent=3, seed=seed)
        gap = cv.optimism_gap(X, y, groups=groups, n_pca=6, model="ridge")
        honest_r2.append(gap["honest"]["R2"])
        leaky_r2.append(gap["leaky"]["R2"])
        assert gap["gap_R2"] >= -1e-9                        #never inverts per draw
    assert np.mean(leaky_r2) >= np.mean(honest_r2)           #optimism, on average


def test_group_cv_holds_out_whole_group():
    #leave-one-group-out: two patients per group -> each fold holds out 2 rows
    X, y = _wide(n_pat=12)
    groups = np.repeat(np.arange(6), 2)
    res = cv.highdim_cv_score(X, y, groups=groups, n_pca=4, model="ridge")
    assert res["oof_pred"].shape == (12,)
    assert all(len(f["test_idx"]) == 2 for f in res["fold_trace"])


def test_reducer_trace_is_monotone_and_applies_to_new_rows():
    X, y = _wide()
    reducer = cv.fit_reduction(X.iloc[:12], n_pca=5)
    t = reducer.trace
    assert t["after_prevalence"] >= t["after_variance"] >= t["after_correlation_prune"]
    #the reducer learned on 12 rows applies cleanly to the 4 held-out rows
    pcs = reducer.apply(X.iloc[12:])
    assert pcs.shape == (4, t["n_pca_components"])
