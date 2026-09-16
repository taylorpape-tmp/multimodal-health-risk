"""calibration assessment for regression and classification.

regression: binned predicted-vs-observed means (a reliability curve for a
continuous target). classification: reliability-diagram data (predicted prob vs
observed frequency), the Brier score, and expected calibration error (ECE).
"""
import numpy as np


def calibration_curve_reg(y_true, y_pred, n_bins=10):
    """bin predictions into n_bins equal-width bins over the predicted range and
    return (pred_mean, true_mean, counts) for the non-empty bins.

    a well-calibrated regressor has pred_mean ~= true_mean in every bin (points
    fall on the identity line).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    lo, hi = y_pred.min(), y_pred.max()
    if hi == lo:
        hi = lo + 1e-9
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(y_pred, edges[1:-1]), 0, n_bins - 1)
    pred_mean, true_mean, counts = [], [], []
    for b in range(n_bins):
        m = idx == b
        c = int(m.sum())
        if c == 0:
            continue
        pred_mean.append(float(y_pred[m].mean()))
        true_mean.append(float(y_true[m].mean()))
        counts.append(c)
    return np.array(pred_mean), np.array(true_mean), np.array(counts)


def reliability_diagram(y_true, y_prob, n_bins=10):
    """bin predicted probabilities into n_bins equal-width bins over [0,1] and
    return (pred_mean, obs_freq, counts) for the non-empty bins.

    pred_mean is the mean predicted probability in the bin, obs_freq the observed
    fraction of positives. a calibrated classifier has pred_mean ~= obs_freq.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    pred_mean, obs_freq, counts = [], [], []
    for b in range(n_bins):
        m = idx == b
        c = int(m.sum())
        if c == 0:
            continue
        pred_mean.append(float(y_prob[m].mean()))
        obs_freq.append(float(y_true[m].mean()))
        counts.append(c)
    return np.array(pred_mean), np.array(obs_freq), np.array(counts)


def brier_score(y_true, y_prob):
    """mean squared error between predicted probability and binary outcome."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """ECE: count-weighted mean gap between predicted prob and observed frequency
    across equal-width probability bins.
    """
    pred_mean, obs_freq, counts = reliability_diagram(y_true, y_prob, n_bins=n_bins)
    if counts.sum() == 0:
        return float("nan")
    return float(np.sum(counts * np.abs(pred_mean - obs_freq)) / counts.sum())
