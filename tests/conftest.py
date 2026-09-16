"""Shared synthetic fixtures, deterministic and tiny so the suite is fast and
does not touch real data on disk."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def regression_frame():
    #40 samples, 5 informative features + target = 2*x0 - x1 + noise
    rng = np.random.default_rng(0)
    n = 40
    X = rng.normal(size=(n, 5))
    y = 2 * X[:, 0] - X[:, 1] + 0.1 * rng.normal(size=n)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    return df


@pytest.fixture
def classification_frame():
    #40 samples, binary target driven by a linear boundary on f0,f1
    rng = np.random.default_rng(1)
    n = 40
    X = rng.normal(size=(n, 5))
    logit = 1.5 * X[:, 0] - 1.0 * X[:, 1]
    y = (logit > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    return df


@pytest.fixture
def grouped_frame():
    #12 rows from 4 subjects (3 rows each) for subject-wise CV tests
    rng = np.random.default_rng(2)
    subjects = np.repeat(["s1", "s2", "s3", "s4"], 3)
    X = rng.normal(size=(12, 3))
    y = X[:, 0] + 0.1 * rng.normal(size=12)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
    df["target"] = y
    return df, subjects


@pytest.fixture
def skewed_frame():
    #right-skewed positive features (like metabolite abundances) for transform tests
    rng = np.random.default_rng(3)
    n = 50
    X = rng.lognormal(mean=0.0, sigma=1.5, size=(n, 4))
    y = np.log1p(X[:, 0]) + 0.1 * rng.normal(size=n)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y
    return df
