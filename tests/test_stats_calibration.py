"""tests for calibration curves, Brier score, and ECE."""
import numpy as np

from hurdle.stats import (
    brier_score,
    calibration_curve_reg,
    expected_calibration_error,
    reliability_diagram,
)


def test_calibration_curve_reg_perfect_is_on_identity():
    #perfect predictions -> pred_mean == true_mean per bin
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    pred_mean, true_mean, counts = calibration_curve_reg(y, y, n_bins=5)
    assert np.allclose(pred_mean, true_mean)
    assert counts.sum() == 200


def test_calibration_curve_reg_shapes_match():
    rng = np.random.default_rng(1)
    y = rng.normal(size=100)
    p = y + 0.5 * rng.normal(size=100)
    pm, tm, c = calibration_curve_reg(y, p, n_bins=10)
    assert len(pm) == len(tm) == len(c)
    assert c.sum() == 100


def test_reliability_diagram_perfect_calibration():
    #probs equal to observed frequency within each group -> curve on diagonal
    #perfect calibration: predicted prob equals observed frequency in each bin
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    prob = np.r_[np.full(50, 0.0), np.full(50, 1.0)]
    pm, obs, counts = reliability_diagram(y, prob, n_bins=10)
    assert np.allclose(pm, obs)
    assert counts.sum() == 100


def test_brier_score_bounds():
    y = np.array([0, 0, 1, 1])
    #perfect
    assert brier_score(y, np.array([0.0, 0.0, 1.0, 1.0])) == 0.0
    #worst-case confident-wrong -> 1.0
    assert brier_score(y, np.array([1.0, 1.0, 0.0, 0.0])) == 1.0


def test_ece_zero_for_perfect_calibration():
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    prob = np.r_[np.full(50, 0.0), np.full(50, 1.0)]
    assert expected_calibration_error(y, prob, n_bins=10) == 0.0


def test_ece_positive_for_miscalibrated():
    #always predicts 0.5 but half are positive in a confident split -> ECE > 0
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    prob = np.r_[np.full(50, 0.9), np.full(50, 0.1)]
    assert expected_calibration_error(y, prob, n_bins=10) > 0.5
