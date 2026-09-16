"""Privacy-preserving ML primitives for the omics modality.

Differential privacy via the Gaussian mechanism (dp) and FedAvg with a
federated-training simulation (federated), both using only numpy/sklearn. The exact
sensitivity, noise calibration, and composition claims live in the function
docstrings.
"""
from .dp import (
    analytic_gaussian_sigma,
    classic_gaussian_sigma,
    dp_gaussian_mean,
    dp_output_perturbed_logistic,
    dp_summary_table,
)
from .federated import (
    central_logistic,
    federated_average,
    simulate_federated_training,
)

__all__ = [
    "analytic_gaussian_sigma",
    "classic_gaussian_sigma",
    "dp_gaussian_mean",
    "dp_output_perturbed_logistic",
    "dp_summary_table",
    "central_logistic",
    "federated_average",
    "simulate_federated_training",
]
