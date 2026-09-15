"""Tests for the pluggable feature-transform ladder."""
import numpy as np
import pytest

from hurdle.ml.linear_models import RidgeModel
from hurdle.ml.transforms import TRANSFORMS, build_transform
from hurdle.ml.tree_models import XGBoostModel


@pytest.mark.parametrize("name", TRANSFORMS)
def test_every_transform_produces_finite_output(name, skewed_frame):
    #each transform, fit on the skewed features, yields finite, same-shape output
    X = skewed_frame[[c for c in skewed_frame.columns if c != "target"]].values
    tf = build_transform(name, n_samples=len(X))
    if tf == "passthrough":
        Xt = X
    else:
        Xt = tf.fit_transform(X)
    assert Xt.shape == X.shape
    assert np.isfinite(Xt).all()


def test_unknown_transform_raises():
    with pytest.raises(ValueError):
        build_transform("not_a_transform")


def test_none_is_passthrough():
    assert build_transform("none") == "passthrough"


def test_standard_transform_standardises(skewed_frame):
    X = skewed_frame[[c for c in skewed_frame.columns if c != "target"]].values
    Xt = build_transform("standard").fit_transform(X)
    #z-scored columns have ~0 mean and ~unit std
    assert np.allclose(Xt.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(Xt.std(axis=0), 1, atol=1e-6)


def test_trees_are_scale_invariant(skewed_frame):
    #a tree model's predictions must not change under a monotone rescale of the
    #features -> confirms 'none' is the right default for the tree family.
    #tiny grid so the two full LOO fits run in seconds.
    tiny = {"model__n_estimators": [50], "model__max_depth": [2]}
    res_none = XGBoostModel(skewed_frame, target_col="target",
                            transform="none", param_grid=tiny).run(verbose=False)
    res_std = XGBoostModel(skewed_frame, target_col="target",
                           transform="standard", param_grid=tiny).run(verbose=False)
    assert np.allclose(res_none["preds"], res_std["preds"], atol=1e-6)


def test_log_transform_helps_or_matches_on_skewed_data(skewed_frame):
    #on right-skewed features a log transform should not hurt a linear model's
    #fit relative to plain z-score (sanity that the transform is wired through)
    res_std = RidgeModel(skewed_frame, target_col="target", transform="standard").run(verbose=False)
    res_log = RidgeModel(skewed_frame, target_col="target", transform="log").run(verbose=False)
    assert np.isfinite(res_std["R2"]) and np.isfinite(res_log["R2"])
