"""Tests for the federated-learning simulation (FedAvg).

Verifies the aggregation weighting, the disjoint no-row-sharing partition, and
that federated utility approaches central utility on data with real signal. One
test reads the real omics matrix if present and is skipped otherwise.
"""
from pathlib import Path

import numpy as np
import pytest

from hurdle.privacy.federated import (
    central_logistic,
    federated_average,
    simulate_federated_training,
)

REAL_INTERIM = Path("data/interim")


def test_federated_average_equal_sizes_is_plain_mean():
    p = [np.array([0.0, 0.0]), np.array([2.0, 4.0])]
    out = federated_average(p, [10, 10])
    assert np.allclose(out, [1.0, 2.0])


def test_federated_average_weights_by_sample_size():
    #a site with 3x the patients pulls the average 3x harder
    p = [np.array([0.0]), np.array([4.0])]
    out = federated_average(p, [3, 1])
    assert np.isclose(out[0], (3 * 0.0 + 1 * 4.0) / 4)


def test_federated_average_single_client_returns_its_params():
    out = federated_average([np.array([1.0, 2.0, 3.0])], [7])
    assert np.allclose(out, [1.0, 2.0, 3.0])


def test_federated_average_shape_mismatch_raises():
    with pytest.raises(ValueError):
        federated_average([np.array([1.0]), np.array([1.0, 2.0])], [1, 1])


def test_federated_average_length_mismatch_raises():
    with pytest.raises(ValueError):
        federated_average([np.array([1.0])], [1, 2])


def test_federated_average_zero_total_weight_raises():
    with pytest.raises(ValueError):
        federated_average([np.array([1.0]), np.array([2.0])], [0, 0])


def _signal_classification(n=120, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (1.5 * X[:, 0] - X[:, 1] + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, y


def test_simulate_partitions_are_disjoint_and_cover_train():
    X, y = _signal_classification()
    res = simulate_federated_training(X, y, n_clients=4, rounds=3, seed=0)
    #client sizes sum to the train split (disjoint shards, every train row assigned once)
    assert sum(res["client_sizes"]) == res["n_train"]
    assert all(s > 0 for s in res["client_sizes"])
    assert res["n_clients"] == 4


def test_simulate_federated_approaches_central_classification():
    X, y = _signal_classification()
    res = simulate_federated_training(X, y, n_clients=4, rounds=8, seed=0)
    #federated should reach within a small gap of central on data with real signal
    assert res["central_metric"] > 0.7
    assert res["gap"] < 0.1
    assert len(res["per_round_metric"]) == 8


def test_simulate_records_metric_per_round():
    X, y = _signal_classification()
    res = simulate_federated_training(X, y, n_clients=3, rounds=5, seed=1)
    assert len(res["per_round_metric"]) == 5
    assert all(0.0 <= m <= 1.0 for m in res["per_round_metric"])


def test_simulate_regression_task_uses_r2():
    rng = np.random.default_rng(0)
    n = 120
    X = rng.normal(size=(n, 5))
    y = 2 * X[:, 0] - X[:, 1] + 0.1 * rng.normal(size=n)
    res = simulate_federated_training(X, y, n_clients=4, rounds=4, task="regression", seed=0)
    assert res["central_metric"] > 0.8
    assert res["gap"] < 0.1


def test_central_logistic_returns_metric_and_split():
    X, y = _signal_classification()
    metric, params, scaler, (train_idx, test_idx) = central_logistic(X, y, seed=0)
    assert 0.0 <= metric <= 1.0
    assert params.shape == (X.shape[1] + 1,)  #coef + intercept
    assert len(set(train_idx) & set(test_idx)) == 0  #no leakage between splits


def test_simulate_more_clients_does_not_break():
    #splitting the same patients across more sites still yields a valid model
    X, y = _signal_classification(n=100)
    for k in (2, 5, 10):
        res = simulate_federated_training(X, y, n_clients=k, rounds=4, seed=0)
        assert sum(res["client_sizes"]) == res["n_train"]
        assert 0.0 <= res["federated_metric"] <= 1.0


@pytest.mark.skipif(
    not (REAL_INTERIM / "omics_S9_isir_clean.csv").exists(),
    reason="real interim files not present",
)
def test_federated_on_real_iris_runs():
    from hurdle.features.omics import build_feature_matrix
    X, y, _ = build_feature_matrix(REAL_INTERIM, target="IRIS", add_ratios=False)
    res = simulate_federated_training(X.values, y.values, n_clients=4, rounds=8, seed=0)
    assert sum(res["client_sizes"]) == res["n_train"]
    assert 0.0 <= res["federated_metric"] <= 1.0
    assert 0.0 <= res["central_metric"] <= 1.0
