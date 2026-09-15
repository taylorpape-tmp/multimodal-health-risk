"""Tests for the MLModel base contract and the model wrappers."""
import numpy as np
import pytest

from hurdle.ml.base import MLModel
from hurdle.ml.linear_models import ElasticNetModel, LinearRegressionModel, RidgeModel
from hurdle.ml.tree_models import LightGBMModel, RandomForestModel, XGBoostModel

REG_MODELS = [LinearRegressionModel, RidgeModel, ElasticNetModel,
              RandomForestModel, XGBoostModel, LightGBMModel]

#tiny 1-point grids so tuned models fit in seconds in tests; the production
#grids on the classes stay full. keys use the 'model__' pipeline-step prefix.
TINY_GRID = {
    "ElasticNet":    {"model__alpha": [0.1], "model__l1_ratio": [0.5]},
    "Random Forest": {"model__n_estimators": [50], "model__max_depth": [2]},
    "XGBoost":       {"model__n_estimators": [50], "model__max_depth": [2]},
    "LightGBM":      {"model__n_estimators": [50], "model__num_leaves": [7]},
}


def _make(cls, frame, **kw):
    #build a model with a tiny grid if it is a tuned model, else default
    grid = TINY_GRID.get(cls.name, "default")
    return cls(frame, target_col="target", param_grid=grid, **kw)


def test_base_is_abstract(regression_frame):
    #the base class must not be usable directly: _estimator raises
    m = MLModel(regression_frame, target_col="target")
    with pytest.raises(NotImplementedError):
        m._estimator()


@pytest.mark.parametrize("cls", REG_MODELS)
def test_regression_models_run_and_report_metrics(cls, regression_frame):
    res = _make(cls, regression_frame).run(verbose=False)
    #the contract: name, n, predictions, and the regression metric set
    assert res["name"] == cls.name
    assert res["n"] == len(regression_frame)
    assert len(res["preds"]) == len(regression_frame)
    for k in ("R2", "MAE", "RMSE", "Pearson", "Spearman"):
        assert k in res and np.isfinite(res[k])


@pytest.mark.parametrize("cls", [LinearRegressionModel, RidgeModel])
def test_linear_models_recover_strong_signal(cls, regression_frame):
    #target is a clean linear function of the features -> R2 should be high
    res = cls(regression_frame, target_col="target").run(verbose=False)
    assert res["R2"] > 0.8


def test_fold_meta_recorded_for_tuned_model(regression_frame):
    #Ridge records its chosen alpha per fold
    res = RidgeModel(regression_frame, target_col="target").run(verbose=False)
    assert "alpha" in res["fold_meta"]
    assert len(res["fold_meta"]["alpha"]) == len(regression_frame)


def test_default_transform_by_family(regression_frame):
    #trees default to no scaling (scale-invariant); linear models to z-score
    assert XGBoostModel(regression_frame, target_col="target").transform == "none"
    assert RidgeModel(regression_frame, target_col="target").transform == "standard"


def test_subject_wise_cv_holds_out_whole_groups(grouped_frame):
    #with groups given, each fold must hold out one entire subject
    frame, subjects = grouped_frame
    m = RidgeModel(frame, target_col="target", groups=subjects)
    test_group_sizes = []
    for _, test_idx in m._splitter():
        held = set(np.asarray(subjects)[test_idx])
        assert len(held) == 1                       #exactly one subject per fold
        test_group_sizes.append(len(test_idx))
    #4 subjects x 3 rows -> 4 folds, 3 rows each
    assert len(test_group_sizes) == 4
    assert all(s == 3 for s in test_group_sizes)


def test_classification_task_reports_classification_metrics(classification_frame):
    class LogRegModel(MLModel):
        name = "LogReg"
        task = "classification"
        family = "linear"

        def _estimator(self):
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000)

    res = LogRegModel(classification_frame, target_col="target").run(verbose=False)
    for k in ("Accuracy", "F1", "AUROC"):
        assert k in res
    assert 0.0 <= res["Accuracy"] <= 1.0
