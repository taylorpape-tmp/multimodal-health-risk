"""Nested-CV rigor checks: show reported omics scores are not inflated by
hyperparameter-selection leakage and that the model beats trivial baselines.

Two constructions share one outer leave-one-out loop and differ only in whether the
tuning grid saw the held-out point: the non-nested (leaky) version tunes once on all
the data and is optimistic, while the nested (honest) version tunes inside each outer
fold, and the gap between their pooled-LOO R2 is the inflation nesting removes.
Trivial baselines (training-mean, row-shuffled features, random-noise features) run
on the same split, and a genuine model must beat all three.
"""
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut


def _as_array(X):
    #accept a DataFrame or ndarray; the fold loop indexes positionally
    return np.asarray(getattr(X, "values", X), dtype=float)


def _pooled_loo(fit_predict_fold, X, y):
    """Run outer LOO, calling fit_predict_fold(X_tr, y_tr, X_te) -> preds for the
    held-out point of each fold, and return the pooled out-of-fold predictions.
    """
    X = _as_array(X)
    y = np.asarray(y, dtype=float)
    preds = np.zeros(len(y), dtype=float)
    for tr, te in LeaveOneOut().split(X):
        preds[te] = fit_predict_fold(X[tr], y[tr], X[te])
    return preds


def _inner_cv(n_splits, seed):
    #shuffled K-fold for the inner tuning loop; deterministic via seed
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def nested_cv_r2(make_estimator, param_grid, X, y, inner_splits=5, seed=0):
    """Honest nested pooled-LOO R2: tune inside each outer fold on the training
    portion only. Returns (r2, pooled_preds).
    """
    inner = _inner_cv(inner_splits, seed)

    def fold(x_tr, y_tr, x_te):
        gs = GridSearchCV(make_estimator(), param_grid, cv=inner,
                          scoring="r2", n_jobs=1)
        gs.fit(x_tr, y_tr)
        return gs.best_estimator_.predict(x_te)

    preds = _pooled_loo(fold, X, y)
    return r2_score(np.asarray(y, dtype=float), preds), preds


def nonnested_cv_r2(make_estimator, param_grid, X, y, inner_splits=5, seed=0):
    """Optimistic non-nested pooled-LOO R2: pick ONE best config via inner CV on
    ALL the data (the leak), then run LOO with it frozen. Returns
    (r2, pooled_preds, best_params).
    """
    Xa = _as_array(X)
    ya = np.asarray(y, dtype=float)
    inner = _inner_cv(inner_splits, seed)
    gs = GridSearchCV(make_estimator(), param_grid, cv=inner,
                      scoring="r2", n_jobs=1)
    gs.fit(Xa, ya)
    best = dict(gs.best_params_)

    def fold(x_tr, y_tr, x_te):
        est = make_estimator().set_params(**best)
        est.fit(x_tr, y_tr)
        return est.predict(x_te)

    preds = _pooled_loo(fold, Xa, ya)
    return r2_score(ya, preds), preds, best


def dummy_mean_r2(X, y):
    """Pooled-LOO R2 of predicting the training-fold mean (DummyRegressor)."""
    def fold(x_tr, y_tr, x_te):
        return DummyRegressor(strategy="mean").fit(x_tr, y_tr).predict(x_te)

    preds = _pooled_loo(fold, X, y)
    return r2_score(np.asarray(y, dtype=float), preds)


def shuffled_features_r2(make_estimator, param_grid, X, y, inner_splits=5, seed=0):
    """Nested pooled-LOO R2 with the design matrix decoupled from y by permuting its
    rows, so feature distributions and correlations stay intact but the X->y mapping
    is destroyed. Expected near 0.
    """
    Xa = _as_array(X)
    rng = np.random.default_rng(seed)
    Xs = Xa[rng.permutation(len(y))]
    return nested_cv_r2(make_estimator, param_grid, Xs, y, inner_splits, seed)[0]


def random_features_r2(make_estimator, param_grid, X, y, inner_splits=5, seed=0):
    """Nested pooled-LOO R2 on random-noise features of the same shape as X.
    Expected near 0.
    """
    Xa = _as_array(X)
    rng = np.random.default_rng(seed + 1)
    Xr = rng.normal(size=Xa.shape)
    return nested_cv_r2(make_estimator, param_grid, Xr, y, inner_splits, seed)[0]
