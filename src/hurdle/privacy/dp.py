"""Differential privacy via the Gaussian mechanism (numpy/scipy only).

Releases f(D)+N(0,sigma^2 I) with sigma calibrated to the L2 sensitivity of f and
the target (epsilon, delta), via either classic_gaussian_sigma (Dwork & Roth,
valid only for epsilon<=1) or analytic_gaussian_sigma (Balle & Wang 2018, the
default, valid for all epsilon>0). epsilon=inf means no privacy: sigma=0 and f(D)
is returned exactly.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def classic_gaussian_sigma(sensitivity, epsilon, delta):
    """Noise scale from the classic bound sigma = Delta*sqrt(2 ln(1.25/delta))/eps.

    Delta is the L2 sensitivity, and this is only a valid guarantee for epsilon in
    (0,1] (use analytic_gaussian_sigma above that). Since sigma is linear in
    1/epsilon, noise variance scales as 1/epsilon^2.
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

    Solves delta(sigma)=0 for adding N(0,sigma^2) to a query of L2 sensitivity Delta;
    delta(sigma) is strictly decreasing, so the unique root is found by bisection.
    Valid for all epsilon>0, unlike the classic bound.
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

    Clips each x_i to [lo,hi], takes the empirical mean (L2 sensitivity (hi-lo)/n),
    and adds Gaussian noise calibrated by the chosen calibration ('analytic' default,
    or 'classic'). Returns (dp_mean, sigma), where epsilon=inf gives the exact clipped
    mean and sigma is not itself private.
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

    Releases one DP mean per column, splitting the budget by basic composition so
    each of the d features gets (epsilon/d, delta/d); bounds is a (d,2) array of clip
    ranges, or None to use each column's observed min/max (data dependent, demo only).
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

    Following Chaudhuri, Monteleoni & Sarwate (2011), fits L2-regularized logistic
    regression on L2-normalized rows (giving sensitivity 2/(n*lam)) and releases
    w* + Gaussian noise. Returns (w_dp, sigma); epsilon=inf gives the non-private w*,
    and no intercept is fit (center/scale upstream if needed).
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
