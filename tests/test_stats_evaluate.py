"""tests for evaluate_model: real LOO via MLModel + CI + permutation p."""
import numpy as np
import pandas as pd
import pytest

from hurdle.ml.linear_models import RidgeModel
from hurdle.stats import evaluate_model


@pytest.fixture
def reg_xy():
    #clean linear signal so LOO R2 is high and permutation p is small
    rng = np.random.default_rng(0)
    n = 40
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    y = 2 * X["f0"].values - X["f1"].values + 0.1 * rng.normal(size=n)
    return X, y


def test_evaluate_model_regression_contract(reg_xy):
    X, y = reg_xy
    res = evaluate_model(RidgeModel, X, y, task="regression",
                         n_boot=300, n_perm=100, seed=0)
    assert res["name"] == "Ridge"
    assert res["n"] == len(y)
    assert len(res["preds"]) == len(y)
    for m in ("R2", "RMSE", "Pearson"):
        d = res[m]
        assert set(d) == {"point", "lo", "hi", "p"}
        assert d["lo"] <= d["point"] <= d["hi"]
        assert 0.0 < d["p"] <= 1.0


def test_evaluate_model_recovers_signal(reg_xy):
    X, y = reg_xy
    res = evaluate_model(RidgeModel, X, y, task="regression",
                         n_boot=200, n_perm=100, seed=0)
    #strong signal -> high R2 and significant permutation p
    assert res["R2"]["point"] > 0.8
    assert res["R2"]["p"] < 0.05


def test_evaluate_model_null_not_significant():
    #target independent of features -> R2 near/below 0 and p not significant
    rng = np.random.default_rng(1)
    n = 40
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    y = rng.normal(size=n)
    res = evaluate_model(RidgeModel, X, y, task="regression",
                         n_boot=200, n_perm=100, seed=0)
    assert res["R2"]["p"] > 0.05
