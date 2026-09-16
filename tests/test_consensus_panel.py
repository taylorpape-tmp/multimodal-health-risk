"""Test that nested consensus yields a non-empty, well-formed panel on signal.

Covers the contract the run_consensus_panel.py script relies on: on a synthetic
set that carries real linear signal, run_nested_consensus returns a non-empty
frozen selection and every selection frequency lies in [0, 1].
"""
import numpy as np
import pandas as pd

from hurdle.feature_selection.nested_consensus import (
    ConsensusConfig,
    run_nested_consensus,
)

FEATS = [f"f{i}" for i in range(6)]


def _signal_frame():
    #f0,f1,f2 informative; f3,f4,f5 pure noise. 35 rows, low noise so the
    #selectors have clear signal to agree on.
    rng = np.random.default_rng(7)
    n = 35
    X = rng.normal(size=(n, 6))
    y = 4 * X[:, 0] - 3 * X[:, 1] + 2 * X[:, 2] + 0.05 * rng.normal(size=n)
    df = pd.DataFrame(X, columns=FEATS)
    df["target"] = y
    return df


def test_consensus_returns_nonempty_selection_with_valid_frequencies():
    cfg = ConsensusConfig()
    _, freq = run_nested_consensus(_signal_frame(), FEATS, "target", cfg)
    #at least one analyte must enter the frozen set at least once
    assert (freq["in_frozen_set"] > 0).any(), "selection was empty on a signal set"
    #selection frequencies are proper fractions in [0, 1]
    assert (freq["frequency"] >= 0).all()
    assert (freq["frequency"] <= 1).all()
    #and every pooled feature is accounted for in the frequency table
    assert set(freq["feature"]) == set(FEATS)
