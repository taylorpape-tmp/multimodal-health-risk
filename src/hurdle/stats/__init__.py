"""statistics and evaluation layer: uncertainty (bootstrap CIs), significance
(label-permutation tests that respect the CV structure), and calibration.
"""
from .calibration import (
    brier_score,
    calibration_curve_reg,
    expected_calibration_error,
    reliability_diagram,
)
from .evaluate import evaluate_model
from .resampling import bootstrap_ci, permutation_test
from .rigor import (
    dummy_mean_r2,
    nested_cv_r2,
    nonnested_cv_r2,
    random_features_r2,
    shuffled_features_r2,
)

__all__ = [
    "bootstrap_ci",
    "permutation_test",
    "calibration_curve_reg",
    "reliability_diagram",
    "brier_score",
    "expected_calibration_error",
    "evaluate_model",
    "nested_cv_r2",
    "nonnested_cv_r2",
    "dummy_mean_r2",
    "shuffled_features_r2",
    "random_features_r2",
]
