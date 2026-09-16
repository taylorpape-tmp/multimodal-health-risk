"""Differential privacy via the Gaussian mechanism, numpy/scipy only.

The Gaussian mechanism releases f(D)+N(0,sigma^2 I) where sigma is calibrated to
the L2 sensitivity of f and the target (epsilon,delta). Two calibrations are
provided:

  classic_gaussian_sigma  sigma = Delta * sqrt(2 ln(1.25/delta)) / epsilon
                          the Dwork & Roth (2014) closed form. Valid only for
                          epsilon <= 1; noise variance is exactly proportional to
                          1/epsilon^2 (used for the variance-scaling guarantee).

  analytic_gaussian_sigma the Balle & Wang (2018) analytic mechanism: the smallest
                          sigma satisfying the exact Gaussian privacy curve
                            delta = Phi(Delta/2sigma - eps*sigma/Delta)
                                    - e^eps * Phi(-Delta/2sigma - eps*sigma/Delta)
                          solved by bisection. Valid for ALL epsilon>0, so it is
                          the default for releases whose sweep includes epsilon>1.

epsilon=inf means "no privacy": sigma=0, the mechanism returns f(D) exactly.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def classic_gaussian_sigma(sensitivity, epsilon, delta):
    """Noise scale from the classic bound sigma = Delta*sqrt(2 ln(1.25/delta))/eps.

    Delta is the L2 sensitivity. The closed form is only a valid (eps,delta)-DP
    guarantee for epsilon in (0,1]; for larger epsilon it stays finite but is no
    longer tight, so analytic_gaussian_sigma is preferred there. Because sigma is
    linear in 1/epsilon, the noise variance scales exactly as 1/epsilon^2.
    """
    if epsilon == np.inf:
        return 0.0
    if epsilon <= 0 or delta <= 0 or delta >= 1:
        raise ValueError("require epsilon>0 and delta in (0,1)")
    if sensitivity < 0:
        raise ValueError("sensitivity must be non-negative")
    return sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon


def analytic_gaussian_sigma(sensitivity, epsilon, delta):
    """Smallest sigma satisfying the exact Gaussian privacy curve (Balle & Wang 2018).

    Solves delta(sigma)=0 for the privacy loss of adding N(0,sigma^2) to a query of
    L2 sensitivity Delta, where
        delta(sigma) = Phi(Delta/2sigma - eps*sigma/Delta)
                       - e^eps * Phi(-Delta/2sigma - eps*sigma/Delta).
    delta(sigma) is strictly decreasing in sigma, so a unique root exists and is
    found by bisection. Valid for all epsilon>0 (unlike the classic bound).
    """
    if epsilon == np.inf:
        return 0.0
    if epsilon <= 0 or delta <= 0 or delta >= 1:
        raise ValueError("require epsilon>0 and delta in (0,1)")
    if sensitivity < 0:
        raise ValueError("sensitivity must be non-negative")
    if sensitivity == 0:
        return 0.0

    def curve(sigma):
        a = sensitivity / (2.0 * sigma) - epsilon * sigma / sensitivity
        b = -sensitivity / (2.0 * sigma) - epsilon * sigma / sensitivity
        return norm.cdf(a) - np.exp(epsilon) * norm.cdf(b) - delta

    hi = sensitivity
    while curve(hi) > 0:  #curve decreasing in sigma; expand upper bracket until <0
        hi *= 2.0
    return brentq(curve, 1e-12, hi, xtol=1e-12)


def dp_gaussian_mean(x, epsilon, delta, bounds, seed=None, calibration="analytic"):
    """(epsilon,delta)-DP release of the mean of a bounded 1-D sample.

    Mechanism: clip each x_i to [lo,hi], compute the empirical mean of the clipped
    values, then add Gaussian noise N(0,sigma^2).

    Sensitivity: the mean is (1/n) sum of values each confined to a range of width
    R = hi-lo. Changing one record moves the sum by at most R and the mean by at
    most R/n, so the L2 sensitivity of the mean is Delta = R/n (bounded-difference
    / add-remove-one on a fixed-n query, the standard clipped-mean sensitivity).

    sigma is calibrated to (Delta,epsilon,delta) by the chosen calibration
    ('analytic', default, valid for all epsilon; or 'classic', 1/epsilon^2 variance
    but only tight for epsilon<=1). epsilon=inf returns the exact clipped mean.

    Returns (dp_mean, sigma). The noise is drawn from the given seed for
    reproducibility; sigma is the released noise scale, not itself private
    (it depends only on n, bounds, epsilon, delta).
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x must be non-empty")
    lo, hi = bounds
    if hi <= lo:
        raise ValueError("bounds must satisfy hi>lo")
    n = x.size
    clipped = np.clip(x, lo, hi)
    mean = float(clipped.mean())
    sensitivity = (hi - lo) / n
    sigma = _sigma(calibration, sensitivity, epsilon, delta)
    if sigma == 0.0:
        return mean, 0.0
    rng = np.random.default_rng(seed)
    return mean + float(rng.normal(0.0, sigma)), sigma


def dp_summary_table(X, epsilon, delta, bounds=None, seed=None, calibration="analytic"):
    """Per-feature (epsilon,delta)-DP means for a patient-by-feature matrix.

    Releases one DP mean per column. The total privacy budget (epsilon,delta) is
    split across the d features by BASIC (sequential) composition: each per-feature
    release gets (epsilon/d, delta/d), so the d releases compose to (epsilon,delta)
    overall. This is the conservative composition; advanced composition could give
    a tighter budget but is not claimed here.

    bounds: (d,2) array of per-feature [lo,hi] clip ranges, or None to use each
    column's observed [min,max]. Using observed min/max is itself weakly data
    dependent and is offered only for the demo; a truly private deployment must fix
    public bounds in advance. The returned frame records which was used.

    Returns a DataFrame indexed by feature with columns
    [true_mean, dp_mean, abs_error, sigma, lo, hi].
    """
    import pandas as pd

    cols = list(X.columns) if hasattr(X, "columns") else [f"f{i}" for i in range(X.shape[1])]
    A = np.asarray(X, dtype=float)
    n, d = A.shape
    if d == 0:
        raise ValueError("X must have at least one feature")
    if bounds is None:
        bnds = np.column_stack([A.min(axis=0), A.max(axis=0)])
        bounds_source = "observed"
    else:
        bnds = np.asarray(bounds, dtype=float)
        if bnds.shape != (d, 2):
            raise ValueError(f"bounds must have shape ({d},2)")
        bounds_source = "supplied"
    #basic composition: divide the budget evenly across the d feature releases
    eps_i = epsilon / d if epsilon != np.inf else np.inf
    delta_i = delta / d
    ss = np.random.SeedSequence(seed)
    seeds = ss.spawn(d)
    rows = []
    for j, name in enumerate(cols):
        lo, hi = float(bnds[j, 0]), float(bnds[j, 1])
        #degenerate constant column: zero-width range -> exact mean, no noise
        if hi <= lo:
            m = float(np.clip(A[:, j], lo, lo).mean() if hi == lo else A[:, j].mean())
            rows.append((name, float(A[:, j].mean()), m, abs(m - A[:, j].mean()), 0.0, lo, hi))
            continue
        true_mean = float(A[:, j].mean())
        dp_mean, sigma = dp_gaussian_mean(
            A[:, j], eps_i, delta_i, (lo, hi),
            seed=seeds[j], calibration=calibration,
        )
        rows.append((name, true_mean, dp_mean, abs(dp_mean - true_mean), sigma, lo, hi))
    out = pd.DataFrame(
        rows, columns=["feature", "true_mean", "dp_mean", "abs_error", "sigma", "lo", "hi"]
    ).set_index("feature")
    out.attrs["composition"] = "basic (sequential)"
    out.attrs["per_feature_epsilon"] = eps_i
    out.attrs["per_feature_delta"] = delta_i
    out.attrs["bounds_source"] = bounds_source
    return out


def dp_output_perturbed_logistic(X, y, epsilon, delta, lam=0.1, seed=None,
                                 calibration="analytic"):
    """(epsilon,delta)-DP logistic regression by output perturbation.

    Chaudhuri, Monteleoni & Sarwate (2011): train an L2-regularized logistic
    regression, then release w_dp = w* + N(0,sigma^2 I). With every feature row
    L2-normalized so ||x_i|| <= 1 and objective (1/n) sum loss + (lam/2)||w||^2,
    the minimizer w* has L2 sensitivity Delta = 2/(n*lam) to replacing one record
    (their Corollary 8; the logistic loss is 1-Lipschitz in the margin). sigma is
    calibrated to (Delta,epsilon,delta) by the Gaussian mechanism.

    We use output perturbation rather than DP-SGD deliberately: it needs no
    per-step gradient-clipping or Renyi/moments accountant, so the (epsilon,delta)
    guarantee is exact and auditable in ~30 lines of numpy/sklearn with no heavy
    dependency. Rows are L2-normalized inside this function to enforce the
    ||x_i||<=1 precondition the sensitivity bound requires.

    Returns (w_dp, sigma). epsilon=inf returns the non-private w* (sigma=0).
    Intercept is not fit (folding a bias into the norm bound would loosen it);
    center/scale features upstream if needed.
    """
    from sklearn.linear_model import LogisticRegression

    A = np.asarray(X, dtype=float)
    yv = np.asarray(y).ravel().astype(int)
    n, d = A.shape
    if n == 0 or d == 0:
        raise ValueError("X must be non-empty")
    if set(np.unique(yv)) - {0, 1}:
        raise ValueError("y must be binary 0/1")
    #enforce ||x_i|| <= 1, the precondition of the 2/(n*lam) sensitivity bound
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    An = A / norms
    #sklearn minimizes C*sum_loss + 0.5||w||^2; matching (1/n)sum + (lam/2)||w||^2
    #needs C = 1/(n*lam)
    clf = LogisticRegression(C=1.0 / (n * lam), fit_intercept=False,
                             solver="lbfgs", max_iter=5000)
    clf.fit(An, yv)
    w = clf.coef_.ravel()
    sensitivity = 2.0 / (n * lam)
    sigma = _sigma(calibration, sensitivity, epsilon, delta)
    if sigma == 0.0:
        return w, 0.0
    rng = np.random.default_rng(seed)
    return w + rng.normal(0.0, sigma, size=d), sigma


def _sigma(calibration, sensitivity, epsilon, delta):
    if calibration == "analytic":
        return analytic_gaussian_sigma(sensitivity, epsilon, delta)
    if calibration == "classic":
        return classic_gaussian_sigma(sensitivity, epsilon, delta)
    raise ValueError("calibration must be 'analytic' or 'classic'")
