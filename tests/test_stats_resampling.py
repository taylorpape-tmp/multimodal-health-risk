"""tests for bootstrap_ci and permutation_test."""
import numpy as np
import pytest

from hurdle.stats import bootstrap_ci, permutation_test
from hurdle.stats.metrics import auroc, pearson, r2, rmse


@pytest.fixture
def paired():
    #y_pred is a noisy copy of y_true -> strong positive relationship
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=200)
    y_pred = y_true + 0.3 * rng.normal(size=200)
    return y_true, y_pred


def test_bootstrap_ci_brackets_point(paired):
    y_true, y_pred = paired
    for fn in (r2, rmse, pearson):
        point, lo, hi = bootstrap_ci(y_true, y_pred, fn, n_boot=500, seed=1)
        assert lo <= point <= hi
        assert hi > lo


def test_bootstrap_ci_deterministic(paired):
    y_true, y_pred = paired
    a = bootstrap_ci(y_true, y_pred, r2, n_boot=300, seed=7)
    b = bootstrap_ci(y_true, y_pred, r2, n_boot=300, seed=7)
    assert a == b


def test_bootstrap_ci_length_mismatch_raises():
    with pytest.raises(ValueError):
        bootstrap_ci(np.zeros(5), np.zeros(4), r2)


def test_bootstrap_ci_auroc_on_scores():
    #separable scores -> AUROC near 1, CI within [0,1]
    rng = np.random.default_rng(2)
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    s = np.r_[rng.normal(0, 1, 50), rng.normal(3, 1, 50)]
    point, lo, hi = bootstrap_ci(y, s, auroc, n_boot=400, seed=3)
    assert 0.0 <= lo <= point <= hi <= 1.0
    assert point > 0.8


def test_permutation_test_detects_real_signal():
    #predict_fn returns a POSITION-fixed feature correlated with the true labels
    #(as a real CV on informative features would): true labels score high,
    #permuted labels break the alignment and score at chance
    rng = np.random.default_rng(4)
    y = rng.normal(size=60)
    feature = y + 0.2 * rng.normal(size=60)

    def predict_fn(labels):
        return feature

    observed, p = permutation_test(y, predict_fn, r2, n_perm=200, seed=0,
                                   metric_name="R2")
    assert observed > 0.8
    assert p < 0.05


def test_permutation_test_null_is_not_significant():
    #predictions unrelated to labels -> p should be large
    rng = np.random.default_rng(5)
    y = rng.normal(size=60)
    fixed = rng.normal(size=60)

    def predict_fn(labels):
        return fixed

    _, p = permutation_test(y, predict_fn, r2, n_perm=200, seed=0, metric_name="R2")
    assert p > 0.2


def test_permutation_test_pvalue_in_unit_range():
    rng = np.random.default_rng(6)
    y = rng.normal(size=40)

    def predict_fn(labels):
        return labels + 0.1 * rng.normal(size=len(labels))

    _, p = permutation_test(y, predict_fn, r2, n_perm=100, seed=0, metric_name="R2")
    assert 0.0 < p <= 1.0
