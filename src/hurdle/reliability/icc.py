"""Intraclass correlation (ICC) for inter-modality reliability.

ICC is the standard psychometric measure of inter-rater reliability; here the k
'raters' are the k modalities of the fusion pipeline, each a noisy read of the same
person's latent risk, so a high ICC means the modalities corroborate one another.
Implements the two-way random, absolute-agreement family from Shrout & Fleiss (1979)
Case 2 (ICC(2,1) single measures, ICC(2,k) the k-modality mean) with F-based CIs from
McGraw & Wong (1996).
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class ICCResult:
    #one ICC point estimate plus its F-based confidence interval and the ANOVA
    #mean squares it was computed from (exposed so callers can audit the fit)
    icc_type: str          #"ICC(2,1)" or "ICC(2,k)"
    icc: float             #point estimate
    ci_lo: float           #lower confidence bound (nan if not computable)
    ci_hi: float           #upper confidence bound (nan if not computable)
    confidence: float      #nominal coverage, e.g. 0.95
    n_subjects: int        #n (rows)
    k_raters: int          #k (columns / modalities)
    f_value: float         #MSR / MSE
    ms_rows: float         #MSR
    ms_cols: float         #MSC
    ms_error: float        #MSE

    def as_row(self):
        #flat dict for writing to a results table
        return {
            "icc_type": self.icc_type, "icc": self.icc,
            "ci_lo": self.ci_lo, "ci_hi": self.ci_hi,
            "k_modalities": self.k_raters, "n": self.n_subjects,
        }


def _anova_mean_squares(ratings):
    #two-way ANOVA decomposition of an (n subjects x k raters) matrix into the
    #between-subjects (rows), between-raters (cols) and residual mean squares
    x = np.asarray(ratings, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"ratings must be 2-D (n_subjects, k_raters); got shape {x.shape}")
    n, k = x.shape
    if n < 2 or k < 2:
        raise ValueError(f"need at least 2 subjects and 2 raters; got n={n}, k={k}")
    if not np.isfinite(x).all():
        raise ValueError("ratings contain non-finite values; drop or impute before ICC")

    grand = x.mean()
    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)

    #sums of squares
    ss_total = ((x - grand) ** 2).sum()
    ss_rows = k * ((row_means - grand) ** 2).sum()
    ss_cols = n * ((col_means - grand) ** 2).sum()
    ss_error = ss_total - ss_rows - ss_cols

    df_rows = n - 1
    df_cols = k - 1
    df_error = (n - 1) * (k - 1)

    ms_rows = ss_rows / df_rows
    ms_cols = ss_cols / df_cols
    ms_error = ss_error / df_error
    return n, k, ms_rows, ms_cols, ms_error


def _ci_two_way_absolute(n, k, msr, msc, mse, r, average, confidence):
    #F-based CI for ICC(2,1) [average=False] or ICC(2,k) [average=True],
    #McGraw & Wong (1996) Table 7, cases ICC(A,1) / ICC(A,k)
    alpha = 1.0 - confidence
    if mse <= 0 or not np.isfinite(r) or abs(1.0 - r) < 1e-12:
        return float("nan"), float("nan")

    #Satterthwaite degrees of freedom for the error term (shared by both cases)
    a = (k * r) / (n * (1.0 - r))
    b = 1.0 + (k * r * (n - 1.0)) / (n * (1.0 - r))
    denom = (a * msc) ** 2 / (k - 1.0) + (b * mse) ** 2 / ((n - 1.0) * (k - 1.0))
    if denom <= 0:
        return float("nan"), float("nan")
    v = (a * msc + b * mse) ** 2 / denom

    f_lower = stats.f.ppf(1.0 - alpha / 2.0, n - 1, v)
    f_upper = stats.f.ppf(1.0 - alpha / 2.0, v, n - 1)

    #single-measure (,1) absolute-agreement bounds
    lo1_den = f_lower * (k * msc + (k * n - k - n) * mse) + n * msr
    hi1_den = k * msc + (k * n - k - n) * mse + n * f_upper * msr
    lo1 = n * (msr - f_lower * mse) / lo1_den if lo1_den != 0 else float("nan")
    hi1 = n * (f_upper * msr - mse) / hi1_den if hi1_den != 0 else float("nan")

    if not average:
        return lo1, hi1

    #average-measure (,k) bounds: Spearman-Brown step-up of the single bounds
    #(McGraw & Wong 1996, Table 7 note for ICC(A,k))
    def _step_up(x):
        d = 1.0 + (k - 1.0) * x
        return (k * x) / d if d != 0 else float("nan")
    return _step_up(lo1), _step_up(hi1)


def icc(ratings, icc_type="ICC(2,1)", confidence=0.95):
    """Intraclass correlation for an (n_subjects, k_raters) matrix whose columns are
    raters/modalities.

    "ICC(2,1)" is single-measure reliability (how much one modality agrees with
    another) and "ICC(2,k)" is the reliability of the k-modality mean, both two-way
    random and absolute agreement. Returns an ICCResult with the point estimate,
    F-based CI, and ANOVA mean squares.
    """
    valid = {"ICC(2,1)", "ICC(2,k)"}
    if icc_type not in valid:
        raise ValueError(f"icc_type must be one of {sorted(valid)}; got {icc_type!r}")

    n, k, msr, msc, mse = _anova_mean_squares(ratings)
    f_value = msr / mse if mse > 0 else float("inf")

    if icc_type == "ICC(2,1)":
        denom = msr + (k - 1) * mse + (k / n) * (msc - mse)
        point = (msr - mse) / denom if denom != 0 else float("nan")
        average = False
    else:
        denom = msr + (msc - mse) / n
        point = (msr - mse) / denom if denom != 0 else float("nan")
        average = True

    ci_lo, ci_hi = _ci_two_way_absolute(n, k, msr, msc, mse, point, average, confidence)
    return ICCResult(
        icc_type=icc_type, icc=float(point), ci_lo=float(ci_lo), ci_hi=float(ci_hi),
        confidence=confidence, n_subjects=n, k_raters=k, f_value=float(f_value),
        ms_rows=float(msr), ms_cols=float(msc), ms_error=float(mse),
    )
