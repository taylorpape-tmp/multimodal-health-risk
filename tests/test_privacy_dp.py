"""Tests for the differential-privacy primitives (Gaussian mechanism).

All synthetic and deterministic; verifies the sensitivity, clipping, noise
calibration, and composition claims made in the docstrings rather than trusting
them. One test reads the real omics matrix if present and is skipped otherwise.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from hurdle.privacy.dp import (
    analytic_gaussian_sigma,
    classic_gaussian_sigma,
    dp_gaussian_mean,
    dp_output_perturbed_logistic,
    dp_summary_table,
)

REAL_INTERIM = Path("data/interim")


def test_classic_sigma_is_linear_in_inverse_epsilon():
    #classic bound sigma = Delta*sqrt(2 ln(1.25/delta))/eps -> halving eps doubles sigma
    s1 = classic_gaussian_sigma(1.0, 0.5, 1e-5)
    s2 = classic_gaussian_sigma(1.0, 1.0, 1e-5)
    assert np.isclose(s1 / s2, 2.0)


def test_classic_variance_scales_as_inverse_epsilon_squared():
    #the documented guarantee: noise VARIANCE ~ 1/eps^2 exactly for the classic bound
    for eps in (0.1, 0.25, 0.5):
        r = classic_gaussian_sigma(1.0, eps, 1e-5) ** 2 / classic_gaussian_sigma(1.0, 2 * eps, 1e-5) ** 2
        assert np.isclose(r, 4.0)


def test_analytic_sigma_satisfies_privacy_curve():
    #analytic sigma must be the root of the exact Gaussian privacy curve (residual ~0)
    sens, delta = 1.0, 1e-5
    for eps in (0.1, 0.5, 1.0, 2.0, 5.0):
        sigma = analytic_gaussian_sigma(sens, eps, delta)
        a = sens / (2 * sigma) - eps * sigma / sens
        b = -sens / (2 * sigma) - eps * sigma / sens
        resid = norm.cdf(a) - np.exp(eps) * norm.cdf(b) - delta
        assert abs(resid) < 1e-9


def test_analytic_never_looser_than_classic_in_valid_regime():
    #for eps<=1 both are valid; the analytic mechanism is provably no worse (>= less noise)
    for eps in (0.1, 0.5, 1.0):
        assert analytic_gaussian_sigma(1.0, eps, 1e-5) <= classic_gaussian_sigma(1.0, eps, 1e-5) + 1e-9


def test_sigma_monotonic_decreasing_in_epsilon():
    prev = np.inf
    for eps in (0.1, 0.5, 1.0, 2.0, 5.0):
        s = analytic_gaussian_sigma(1.0, eps, 1e-5)
        assert s < prev
        prev = s


def test_infinite_epsilon_gives_zero_noise():
    assert analytic_gaussian_sigma(1.0, np.inf, 1e-5) == 0.0
    assert classic_gaussian_sigma(1.0, np.inf, 1e-5) == 0.0


def test_dp_gaussian_mean_infinite_epsilon_is_exact_clipped_mean():
    x = np.array([1.0, 2.0, 3.0, 100.0])
    val, sigma = dp_gaussian_mean(x, np.inf, 1e-5, bounds=(0.0, 10.0))
    assert sigma == 0.0
    #100 is clipped to 10 before averaging
    assert np.isclose(val, np.mean([1.0, 2.0, 3.0, 10.0]))


def test_dp_gaussian_mean_enforces_clipping():
    #an out-of-range record cannot move the release beyond what the clipped data allows
    x = np.array([5.0, 5.0, 5.0, 1e9])
    val_inf, _ = dp_gaussian_mean(x, np.inf, 1e-5, bounds=(0.0, 10.0))
    assert val_inf <= 10.0  #the 1e9 outlier is clipped to 10, not averaged raw


def test_dp_gaussian_mean_noise_variance_scales_with_epsilon():
    #empirical: draw many releases; variance of the noise ~ sigma^2 ~ 1/eps^2 (classic)
    x = np.zeros(50)  #true clipped mean is 0 so the release IS the noise
    bounds = (0.0, 1.0)
    def noise_var(eps, reps=4000):
        vals = [dp_gaussian_mean(x, eps, 1e-5, bounds, seed=i, calibration="classic")[0]
                for i in range(reps)]
        return np.var(vals)
    v_lo = noise_var(0.5)
    v_hi = noise_var(1.0)
    #halving eps quadruples variance; allow Monte-Carlo slack
    assert 3.0 < v_lo / v_hi < 5.0


def test_dp_gaussian_mean_is_seed_reproducible():
    x = np.linspace(0, 1, 30)
    a = dp_gaussian_mean(x, 1.0, 1e-5, (0, 1), seed=7)
    b = dp_gaussian_mean(x, 1.0, 1e-5, (0, 1), seed=7)
    assert a == b


def test_dp_gaussian_mean_bad_bounds_raise():
    with pytest.raises(ValueError):
        dp_gaussian_mean(np.array([1.0]), 1.0, 1e-5, bounds=(1.0, 0.0))


def test_dp_summary_table_splits_budget_by_basic_composition():
    rng = np.random.default_rng(0)
    import pandas as pd
    X = pd.DataFrame(rng.normal(size=(40, 5)), columns=list("abcde"))
    tbl = dp_summary_table(X, epsilon=1.0, delta=1e-5, seed=0)
    #basic composition: per-feature budget is epsilon/d, delta/d
    assert tbl.attrs["composition"] == "basic (sequential)"
    assert np.isclose(tbl.attrs["per_feature_epsilon"], 1.0 / 5)
    assert np.isclose(tbl.attrs["per_feature_delta"], 1e-5 / 5)
    assert list(tbl.index) == list("abcde")
    assert (tbl["sigma"] > 0).all()


def test_dp_summary_table_error_shrinks_with_more_budget():
    #more epsilon -> less noise -> smaller mean absolute error on average
    rng = np.random.default_rng(1)
    import pandas as pd
    X = pd.DataFrame(rng.normal(size=(60, 8)))
    lo = dp_summary_table(X, epsilon=0.2, delta=1e-5, seed=3)["abs_error"].mean()
    hi = dp_summary_table(X, epsilon=5.0, delta=1e-5, seed=3)["abs_error"].mean()
    assert hi < lo


def test_dp_summary_table_infinite_epsilon_recovers_true_means():
    import pandas as pd
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(30, 4)))
    tbl = dp_summary_table(X, epsilon=np.inf, delta=1e-5, seed=0)
    assert np.allclose(tbl["dp_mean"].values, tbl["true_mean"].values)
    assert np.allclose(tbl["abs_error"].values, 0.0)


def test_output_perturbed_logistic_sensitivity_bound_holds():
    #empirically confirm ||w' - w|| <= 2/(n*lam) under one-record replacement
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    n, d, lam = 60, 8, 0.2
    X = rng.normal(size=(n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(int)

    def wstar(Xa, ya):
        clf = LogisticRegression(C=1.0 / (n * lam), fit_intercept=False,
                                 solver="lbfgs", max_iter=5000)
        clf.fit(Xa, ya)
        return clf.coef_.ravel()

    w0 = wstar(X, y)
    bound = 2.0 / (n * lam)
    worst = 0.0
    for j in range(n):
        Xr, yr = X.copy(), y.copy()
        v = rng.normal(size=d)
        Xr[j] = v / np.linalg.norm(v)
        yr[j] = 1 - yr[j]
        worst = max(worst, np.linalg.norm(wstar(Xr, yr) - w0))
    assert worst <= bound + 1e-9


def test_output_perturbed_logistic_infinite_epsilon_is_nonprivate():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    y = (X[:, 0] > 0).astype(int)
    w_inf, sigma = dp_output_perturbed_logistic(X, y, np.inf, 1e-5, lam=0.1)
    assert sigma == 0.0
    assert w_inf.shape == (5,)


def test_output_perturbed_logistic_noise_grows_as_epsilon_shrinks():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 6))
    y = (X[:, 0] > 0).astype(int)
    _, s_lo = dp_output_perturbed_logistic(X, y, 0.1, 1e-5, lam=0.1, seed=0)
    _, s_hi = dp_output_perturbed_logistic(X, y, 5.0, 1e-5, lam=0.1, seed=0)
    assert s_lo > s_hi > 0


def test_output_perturbed_logistic_rejects_nonbinary_y():
    X = np.zeros((5, 3))
    with pytest.raises(ValueError):
        dp_output_perturbed_logistic(X, np.array([0, 1, 2, 0, 1]), 1.0, 1e-5)


@pytest.mark.skipif(
    not (REAL_INTERIM / "omics_S8_sspg_clean.csv").exists(),
    reason="real interim files not present",
)
def test_dp_summary_on_real_omics_runs():
    from hurdle.features.omics import build_feature_matrix
    X, _, _ = build_feature_matrix(REAL_INTERIM, target="SSPG", add_ratios=False)
    tbl = dp_summary_table(X, epsilon=1.0, delta=1e-5, seed=0)
    assert len(tbl) == X.shape[1]
    assert (tbl["sigma"] > 0).all()
    assert not tbl["dp_mean"].isna().any()
