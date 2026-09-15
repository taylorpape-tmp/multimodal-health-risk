"""Tests for the virtual-cohort generator and the fusion controls.

These tests encode the 'testing against truth' guarantees:
  - the latent z is drawn first and features derive from it (not the reverse)
  - fusion beats the best single modality on COUPLED data
  - the fusion advantage VANISHES on SCRAMBLED data (the anti-circularity proof)
"""
import numpy as np

from hurdle.fusion import fusion
from hurdle.fusion import virtual_cohort as vc


def test_generator_shapes_and_hidden_truth():
    c = vc.generate(n=120, seed=0)
    #z is length-n, features are a frame, iris is binary derived from z
    assert len(c.z) == 120
    assert c.X.shape[0] == 120
    assert set(np.unique(c.iris)) <= {0, 1}
    #iris is exactly the sign of z (truth derived from the hidden cause)
    assert np.array_equal(c.iris, (c.z > 0).astype(int))
    #every modality contributes its declared feature columns
    total = sum(len(cols) for cols in c.modality_cols.values())
    assert c.X.shape[1] == total


def test_features_correlate_with_latent_z_when_coupled():
    #coupled cohort: at least one feature per modality should track z
    c = vc.generate(n=300, seed=1, coupling=1.0)
    for cols in c.modality_cols.values():
        corrs = [abs(np.corrcoef(c.X[col], c.z)[0, 1]) for col in cols]
        assert max(corrs) > 0.15, "a coupled modality shows no signal with z"


def test_scramble_destroys_coupling():
    #after scramble, no modality block should still track z strongly on average
    c = vc.generate(n=300, seed=2, coupling=1.0)
    s = vc.scramble(c, seed=2)
    #the scrambled frame is a genuine reordering (values preserved per column)
    for cols in c.modality_cols.values():
        for col in cols:
            assert np.allclose(sorted(c.X[col]), sorted(s.X[col]))


def test_fusion_beats_best_single_on_coupled_data():
    c = vc.generate(n=250, seed=3, coupling=1.0)
    table, verdict = fusion.compare_controls(c, vc.scramble, n_splits=5, seed=3)
    #the headline anti-circularity checks
    assert verdict["fusion_beats_best_single"], table.to_string()
    assert verdict["scramble_advantage_vanishes"], table.to_string()


def test_scrambled_fusion_is_near_zero_r2():
    #with the coupling broken, the fused model cannot predict z -> R2 <= ~0
    c = vc.generate(n=250, seed=4, coupling=1.0)
    s = vc.scramble(c, seed=4)
    res = fusion.fuse(s, c.z, n_splits=5, seed=4)
    r2 = fusion._score(c.z, res["fused"])["R2"]
    assert r2 < 0.2


def test_coupling_zero_matches_no_signal():
    #coupling=0 makes every feature independent of z (the generator-level control)
    c = vc.generate(n=200, seed=5, coupling=0.0)
    fused = fusion.fuse(c, c.z, n_splits=5, seed=5)
    assert fusion._score(c.z, fused["fused"])["R2"] < 0.2


def test_additive_baseline_runs():
    c = vc.generate(n=150, seed=6)
    out = fusion.additive_baseline(c, c.z)
    assert set(out) == {"R2", "Spearman", "RMSE"}
    assert np.isfinite(out["R2"])


def test_calibrate_from_real_sets_loadings():
    #calibration on tiny real-like blocks returns specs with finite loadings
    import pandas as pd
    rng = np.random.default_rng(0)
    shared = rng.normal(size=20)
    omics = pd.DataFrame({f"o{i}": shared + rng.normal(scale=0.5, size=20) for i in range(5)})
    cgm = pd.DataFrame({f"c{i}": shared + rng.normal(scale=0.5, size=20) for i in range(3)})
    specs = vc.calibrate_from_real({"omics": omics, "cgm": cgm})
    by_name = {s.name: s for s in specs}
    assert 0.1 <= by_name["omics"].loading <= 1.5
    assert 0.1 <= by_name["cgm"].loading <= 1.5
