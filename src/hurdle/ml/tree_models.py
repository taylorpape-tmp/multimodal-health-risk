"""Tree-ensemble model wrappers (RandomForest, XGBoost, LightGBM), ported from
dom_study. These are the trees-win-on-wide-small-n workhorses for omics.

family='tree' -> default transform is 'none' (trees are scale-invariant, so
scaling the features would be wasted work). param_grid keys use the 'model__'
pipeline-step prefix; _fold_meta reads the estimator from the 'model' step.
"""
import warnings

warnings.filterwarnings("ignore")   #silence lightgbm/sklearn feature-name warnings

from lightgbm import LGBMRegressor  #noqa: E402
from sklearn.ensemble import RandomForestRegressor  #noqa: E402
from xgboost import XGBRegressor  #noqa: E402

from .base import MLModel  #noqa: E402


class RandomForestModel(MLModel):
    name = 'Random Forest'
    task = 'regression'
    family = 'tree'
    verbose_folds = True
    param_grid = {
        'model__n_estimators':     [200, 500],
        'model__max_depth':        [2, 3, 4],
        'model__min_samples_leaf': [2, 3, 5],
        'model__max_features':     [1.0, 0.5],
    }

    def _estimator(self):
        return RandomForestRegressor(random_state=0)

    def _fold_meta(self, est):
        m = est.named_steps['model']
        return {
            'n_estimators':     m.n_estimators,
            'max_depth':        m.max_depth,
            'min_samples_leaf': m.min_samples_leaf,
            'max_features':     m.max_features,
        }


class XGBoostModel(MLModel):
    name = 'XGBoost'
    task = 'regression'
    family = 'tree'
    param_grid = {
        'model__n_estimators':     [100, 300],
        'model__max_depth':        [1, 2, 3],
        'model__learning_rate':    [0.01, 0.05, 0.1],
        'model__subsample':        [0.8],
        'model__colsample_bytree': [0.8],
    }

    def _estimator(self):
        return XGBRegressor(random_state=0, objective='reg:squarederror')

    def _fold_meta(self, est):
        m = est.named_steps['model']
        return {
            'n_estimators':  m.n_estimators,
            'max_depth':     m.max_depth,
            'learning_rate': m.learning_rate,
        }


class LightGBMModel(MLModel):
    name = 'LightGBM'
    task = 'regression'
    family = 'tree'
    verbose_folds = True
    param_grid = {
        'model__n_estimators':      [100, 300],
        'model__num_leaves':        [3, 7, 15],
        'model__learning_rate':     [0.01, 0.05, 0.1],
        'model__min_child_samples': [5, 10],
        'model__subsample':         [0.8],
        'model__colsample_bytree':  [0.8],
    }

    def _estimator(self):
        return LGBMRegressor(random_state=0, verbose=-1)

    def _fold_meta(self, est):
        m = est.named_steps['model']
        return {
            'n_estimators':  m.n_estimators,
            'num_leaves':    m.num_leaves,
            'learning_rate': m.learning_rate,
        }
