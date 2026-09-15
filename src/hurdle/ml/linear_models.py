"""Linear model wrappers (OLS, Ridge, ElasticNet), ported from dom_study.

Each returns a BARE estimator; the MLModel base wraps it with the chosen
transform (default 'standard' for the linear family). fold_meta reads the
estimator from the pipeline's 'model' step.
"""
import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression, RidgeCV

from .base import MLModel


class LinearRegressionModel(MLModel):
    name = 'OLS'
    task = 'regression'
    family = 'linear'

    def _estimator(self):
        return LinearRegression()


class RidgeModel(MLModel):
    #RidgeCV self-tunes alpha via internal CV, so no outer param_grid needed
    name = 'Ridge'
    task = 'regression'
    family = 'linear'
    ALPHAS = np.logspace(-3, 3, 25)

    def _estimator(self):
        return RidgeCV(alphas=self.ALPHAS)

    def _fold_meta(self, est):
        return {'alpha': est.named_steps['model'].alpha_}


class ElasticNetModel(MLModel):
    name = 'ElasticNet'
    task = 'regression'
    family = 'linear'
    param_grid = {
        'model__alpha':    np.logspace(-3, 1, 20),
        'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    }

    def _estimator(self):
        return ElasticNet(max_iter=10000)

    def _fold_meta(self, est):
        m = est.named_steps['model']
        return {'alpha': m.alpha, 'l1_ratio': m.l1_ratio}
