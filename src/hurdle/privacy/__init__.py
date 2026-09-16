"""Privacy-preserving ML primitives for the omics modality.

Two techniques the person-spec names as desirable, implemented with numpy/sklearn
only (no opacus/flower/tensorflow-federated), so `pip install -e .` is unchanged:

  dp        differential privacy via the Gaussian mechanism (bounded-mean release,
            per-feature DP summary tables, output-perturbed logistic regression)
  federated FedAvg weighted aggregation and a federated-training simulation that
            partitions the real patients across sites and never shares raw rows

Every claim about privacy (sensitivity, noise calibration, composition) is stated
explicitly in the function docstrings; a wrong DP guarantee is worse than none.
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
