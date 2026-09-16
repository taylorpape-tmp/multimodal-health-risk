"""Uncertainty and significance by resampling.

bootstrap_ci does a paired bootstrap over samples for a point metric.
permutation_test builds a label-permutation null that respects the CV structure:
the caller's predict_fn reruns the whole cross-validation on the permuted labels,
so the permutation happens outside the CV and cannot leak.
"""
import numpy as np

from .metrics import GREATER_IS_BETTER


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=2000, seed=0, alpha=0.05):
    """Paired-bootstrap CI for metric_fn evaluated on (y_true, y_pred).

    Resamples indices with replacement, keeping y_true/y_pred paired, and returns
    (point, lo, hi): the metric on the full sample plus the alpha/2 and 1-alpha/2
    percentiles of the bootstrap distribution. Resamples that make the metric
    undefined (one-class AUROC, constant Pearson) are dropped, not counted as zeros.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    point = float(metric_fn(y_true, y_pred))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            v = float(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            continue
        if np.isfinite(v):
            stats.append(v)
    if not stats:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def permutation_test(y_true, predict_fn, metric_fn, n_perm=1000, seed=0,
                     greater_is_better=None, metric_name=None):
    """Label-permutation p-value for 'is the model better than chance'.

    predict_fn(labels) must rerun the full cross-validation on the labels it gets, so
    each of n_perm shuffles re-tests the whole pipeline with no leakage and p is the
    add-one estimate (1 + count) / (1 + n_perm). greater_is_better picks the tail
    (True for scores like R2/AUROC, False for errors like RMSE); if None it is looked
    up from metric_name, else defaults to True.
    """
    if greater_is_better is None:
        greater_is_better = GREATER_IS_BETTER.get(metric_name, True)
    y_true = np.asarray(y_true)
    observed = float(metric_fn(y_true, predict_fn(y_true)))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y_true)
        v = float(metric_fn(y_perm, predict_fn(y_perm)))
        if not np.isfinite(v):
            continue
        if (v >= observed) if greater_is_better else (v <= observed):
            count += 1
    p = (1 + count) / (1 + n_perm)
    return observed, p
