"""Leak-safe cross-validated modeling of the wide omics matrix (S4_HealthyIQR).

build_highdim_matrix fits impute/PCA on all rows, a mild scoring leak since the
held-out patient shapes the medians and PC axes it is later projected onto; this
module closes that gap by refitting the whole reduction (prevalence, variance,
impute, correlation prune, standardize, PCA) on each fold's train rows in
leave-one-out (or leave-one-group-out) CV, then scoring the pooled out-of-fold
predictions once. leak=True reproduces the optimistic path (reduction fit once on
all rows, only the model refit per fold) so optimism_gap() can report both side by
side. s4 is the oriented patient x analyte matrix (NaNs allowed, kept so the
imputer can be fit per fold) from omics_highdim.orient()[0] and y is the aligned
target.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from sklearn.preprocessing import StandardScaler

from . import omics_highdim as hd

_RIDGE_ALPHAS = np.logspace(-3, 3, 25)


def _fast_medians(X):
    #column medians ignoring NaN, as a Series aligned to X.columns; identical to
    #X.median(axis=0) but numpy-fast (per-fold on ~12k analytes this matters)
    return pd.Series(np.nanmedian(X.values, axis=0), index=X.columns)


def _correlation_prune(X, threshold=0.95):
    """Greedy first-keep correlation prune, same output as
    omics_highdim.correlation_prune but faster (one vectorized corrcoef, numpy
    inner loop). Within a correlated cluster the first column is kept and the rest
    dropped.
    """
    corr = np.abs(np.nan_to_num(np.corrcoef(X.values, rowvar=False)))
    cols = list(X.columns)
    p = len(cols)
    alive = np.ones(p, dtype=bool)
    drop = []
    for i in range(p):
        if not alive[i]:
            continue
        hit = np.where((corr[i, i + 1:] > threshold) & alive[i + 1:])[0] + (i + 1)
        for j in hit:
            if alive[j]:
                alive[j] = False
                drop.append(cols[j])
    keep = [c for c, a in zip(cols, alive) if a]
    return X[keep], sorted(drop)


class Reducer:
    """A reduction fit on training rows and applied to new rows.

    Holds the state learned from train (surviving columns, medians, pruned columns,
    scaler, PCA). apply() replays those on any matrix without re-estimating
    anything, so a held-out patient never influences the transform it is scored
    through.
    """

    def __init__(self, keep_cols, medians, pca_cols, scaler, pca, trace):
        self.keep_cols = keep_cols
        self.medians = medians
        self.pca_cols = pca_cols
        self.scaler = scaler
        self.pca = pca
        self.trace = trace

    def apply(self, X):
        #select the train-chosen columns (reindex tolerates missing/extra cols),
        #impute with TRAIN medians, drop the train-pruned columns, then the
        #train-fit scaler + PCA -> compact PC features
        Xk = X.reindex(columns=self.keep_cols)
        Xi, _ = hd.impute(Xk, medians=self.medians)
        Xp = Xi[self.pca_cols]
        Z = self.scaler.transform(Xp.values)
        comps = self.pca.transform(Z)
        return pd.DataFrame(comps, index=X.index,
                            columns=[f"omics_pc{i}" for i in range(comps.shape[1])])


def fit_reduction(X_train, max_missing=0.5, corr_threshold=0.95, n_pca=20, seed=0):
    """Fit the full S4 reduction on train rows only and return a Reducer.

    Every statistic (kept columns, medians, pruned columns, scaling, PC axes) comes
    from X_train and nothing else.
    """
    Xp = hd.prevalence_filter(X_train, max_missing=max_missing)
    after_prevalence = Xp.shape[1]

    Xv = hd.variance_filter(Xp)
    after_variance = Xv.shape[1]
    keep_cols = list(Xv.columns)

    #medians from TRAIN only, then impute train before the unsupervised prune/PCA
    medians = _fast_medians(Xv)
    Xi, _ = hd.impute(Xv, medians=medians)
    Xc, dropped = _correlation_prune(Xi, threshold=corr_threshold)
    pca_cols = list(Xc.columns)
    after_corr = Xc.shape[1]

    scaler = StandardScaler().fit(Xc.values)
    Z = scaler.transform(Xc.values)
    k = min(n_pca, *Z.shape)
    pca = PCA(n_components=k, random_state=seed).fit(Z)

    trace = {
        "n_train": X_train.shape[0],
        "start_analytes": X_train.shape[1],
        "after_prevalence": after_prevalence,
        "after_variance": after_variance,
        "correlation_dropped": len(dropped),
        "after_correlation_prune": after_corr,
        "n_pca_components": k,
        "pca_variance_explained": float(np.sum(pca.explained_variance_ratio_)),
    }
    return Reducer(keep_cols, medians, pca_cols, scaler, pca, trace)


def _make_model(model):
    #ridge self-tunes alpha via internal CV; xgboost uses light, fixed defaults
    #so a per-fold refit over ~n folds stays fast on tiny-n omics
    if model == "ridge":
        return RidgeCV(alphas=_RIDGE_ALPHAS)
    if model == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=0, objective="reg:squarederror")
    raise ValueError(f"unknown model {model!r}; choose 'ridge' or 'xgboost'")


def _splits(n, groups):
    if groups is not None:
        g = np.asarray(groups)
        return list(LeaveOneGroupOut().split(np.arange(n), np.arange(n), g))
    return list(LeaveOneOut().split(np.arange(n)))


def _regression_metrics(y, pred):
    return {
        "R2": float(r2_score(y, pred)),
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "Pearson": float(pearsonr(y, pred)[0]),
        "Spearman": float(spearmanr(y, pred)[0]),
    }


def highdim_cv_score(s4, y, groups=None, n_pca=20, model="ridge",
                     max_missing=0.5, corr_threshold=0.95, seed=0, leak=False):
    """Cross-validated score of the S4 reduction plus a regression model.

    s4 is the oriented patient x analyte matrix (NaNs allowed) and y the aligned
    continuous target; groups gives leave-one-group-out, None gives leave-one-out.
    With leak=False (default, honest) the reduction is fit per fold on train rows,
    while leak=True fits it once on all rows and refits only the model per fold, and
    the return is a dict with the pooled OOF metrics, OOF predictions, the leak flag,
    and a per-fold trace.
    """
    X = s4 if isinstance(s4, pd.DataFrame) else pd.DataFrame(s4)
    y = np.asarray(y, dtype=float)
    n = X.shape[0]
    if len(y) != n:
        raise ValueError(f"y has {len(y)} rows but s4 has {n} patients")

    splits = _splits(n, groups)
    oof = np.full(n, np.nan)
    fold_trace = []

    #leaky path: one reduction on all rows, reused every fold
    global_reducer = None
    if leak:
        global_reducer = fit_reduction(X, max_missing=max_missing,
                                       corr_threshold=corr_threshold,
                                       n_pca=n_pca, seed=seed)
        pcs_all = global_reducer.apply(X)

    for f, (tr, te) in enumerate(splits):
        if leak:
            P_tr, P_te = pcs_all.iloc[tr].values, pcs_all.iloc[te].values
            trace = dict(global_reducer.trace)
        else:
            reducer = fit_reduction(X.iloc[tr], max_missing=max_missing,
                                    corr_threshold=corr_threshold,
                                    n_pca=n_pca, seed=seed)
            P_tr = reducer.apply(X.iloc[tr]).values
            P_te = reducer.apply(X.iloc[te]).values
            trace = dict(reducer.trace)

        est = _make_model(model)
        est.fit(P_tr, y[tr])
        oof[te] = est.predict(P_te)

        trace.update({"fold": f, "test_idx": [int(i) for i in te],
                      "train_idx": [int(i) for i in tr]})
        fold_trace.append(trace)

    out = {"model": model, "n": int(n), "n_pca": int(n_pca), "leak": bool(leak),
           "oof_pred": oof, "fold_trace": fold_trace}
    out.update(_regression_metrics(y, oof))
    return out


def optimism_gap(s4, y, groups=None, n_pca=20, model="ridge",
                 max_missing=0.5, corr_threshold=0.95, seed=0):
    """Run both the honest (per-fold) and leaky (all-rows) reductions.

    Returns {'honest', 'leaky', 'gap_R2', 'gap_Pearson'}; a positive gap means the
    leaky number is optimistic relative to the honest one.
    """
    honest = highdim_cv_score(s4, y, groups=groups, n_pca=n_pca, model=model,
                              max_missing=max_missing, corr_threshold=corr_threshold,
                              seed=seed, leak=False)
    leaky = highdim_cv_score(s4, y, groups=groups, n_pca=n_pca, model=model,
                             max_missing=max_missing, corr_threshold=corr_threshold,
                             seed=seed, leak=True)
    return {"honest": honest, "leaky": leaky,
            "gap_R2": leaky["R2"] - honest["R2"],
            "gap_Pearson": leaky["Pearson"] - honest["Pearson"]}
