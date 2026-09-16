"""Tests for the nested-CV rigor checks (stats/rigor.py).

All on deterministic synthetic data -- no real files touched. A signal-bearing set
(y linear in a few features) verifies the honest invariants: nesting does not score
ABOVE non-nesting beyond a small tolerance, and the real model clears the
predict-the-mean baseline. Null-feature sets verify the shuffled/random baselines
land near zero, which is what makes a positive real score meaningful.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from hurdle.stats import (
    dummy_mean_r2,
    nested_cv_r2,
    nonnested_cv_r2,
    random_features_r2,
    shuffled_features_r2,
)

GRID = {"model__n_estimators": [100, 300], "model__max_depth": [2, 3]}


def _make_estimator():
    return Pipeline([
        ("model", XGBRegressor(random_state=0, objective="reg:squarederror",
                               learning_rate=0.1, subsample=0.8,
                               colsample_bytree=0.8, tree_method="hist")),
    ])


@pytest.fixture
def signal_xy():
    #40 rows, y strongly linear in f0,f1 (+3 noise features) -> learnable signal
    rng = np.random.default_rng(0)
    n = 40
    X = rng.normal(size=(n, 5))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.1 * rng.normal(size=n)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]), y


def test_nested_not_above_nonnested(signal_xy):
    #core anti-leakage invariant: honest nested R2 must not exceed the optimistic
    #non-nested R2 by more than a small tolerance (nested <= non_nested + tol)
    X, y = signal_xy
    nn, _, _ = nonnested_cv_r2(_make_estimator, GRID, X, y)
    ne, _ = nested_cv_r2(_make_estimator, GRID, X, y)
    assert ne <= nn + 0.05, f"nested {ne:.3f} exceeded non-nested {nn:.3f} + tol"


def test_real_model_beats_predict_mean(signal_xy):
    #on a signal-bearing set the tuned model must clear predict-the-training-mean
    X, y = signal_xy
    ne, _ = nested_cv_r2(_make_estimator, GRID, X, y)
    b_mean = dummy_mean_r2(X, y)
    assert ne > b_mean
    assert ne > 0.3, f"expected clear signal, got nested R2={ne:.3f}"


def test_predict_mean_r2_near_zero(signal_xy):
    #DummyRegressor(mean) on pooled LOO scores ~0 by construction (<= 0, near 0)
    X, y = signal_xy
    b_mean = dummy_mean_r2(X, y)
    assert b_mean < 0.05


def test_shuffled_and_random_baselines_near_zero(signal_xy):
    #decoupling X from y (shuffle) or replacing X with noise must collapse R2 to
    #~0 or below -- a real predictor cannot be built from a destroyed mapping
    X, y = signal_xy
    b_shuf = shuffled_features_r2(_make_estimator, GRID, X, y)
    b_rand = random_features_r2(_make_estimator, GRID, X, y)
    assert b_shuf < 0.15, f"shuffled baseline not near 0: {b_shuf:.3f}"
    assert b_rand < 0.15, f"random baseline not near 0: {b_rand:.3f}"


def test_optimism_gap_nonnegative_on_null():
    #on pure noise, non-nested tuning still cherry-picks a config on all the data,
    #so its R2 should be >= the nested R2 (the gap is the leakage nesting removes)
    rng = np.random.default_rng(3)
    n = 40
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    y = rng.normal(size=n)
    nn, _, _ = nonnested_cv_r2(_make_estimator, GRID, X, y)
    ne, _ = nested_cv_r2(_make_estimator, GRID, X, y)
    assert nn >= ne - 0.05
