"""Leak-safe cross-validated modeling of the wide omics matrix (S4_HealthyIQR).

omics_highdim.build_highdim_matrix reduces S4 (12k analytes -> 20 PCs) with the
impute/PCA steps fit on ALL rows. That is fine for EDA but is a mild leak for
scoring: the held-out patient influences the medians and the principal axes it
is then projected onto. This module closes that gap.

highdim_cv_score runs leave-one-out (or leave-one-group-out) CV where, INSIDE
each fold, the whole reduction is fit on TRAIN rows only:
  prevalence filter -> variance filter -> median impute -> correlation prune
  -> standardize -> PCA
The fitted reducer is then APPLIED to the held-out patient, a model is fit on
the train PCs and predicts the held-out PC vector. The pooled out-of-fold (OOF)
predictions are scored once. This is the honest number.

leak=True reproduces the optimistic path (reduction fit once on all rows, only
the model refit per fold) so the optimism gap can be reported honestly side by
side. optimism_gap() returns both.

Input contract: `s4` is the ORIENTED patient x analyte matrix (NaNs allowed,
one row per patient) as returned by omics_highdim.orient()[0]; `y` is the
aligned target vector (same order / index). Keeping NaNs in means the imputer
can be fit per fold instead of on pre-imputed values.
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
    """Greedy first-keep correlation prune, identical output to
    omics_highdim.correlation_prune but with a single vectorized corrcoef and a
    numpy inner loop (drops a per-fold O(p^2) Python double loop to O(p)).

    Within a highly-correlated cluster the first column (in column order) is kept
    and the rest dropped, matching the reference's keep-the-first semantics.
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
    """A reduction fit on a fixed set of training rows, applyable to new rows.

    Stores exactly the state learned from TRAIN: which analyte columns survive
    prevalence+variance filtering, the train medians used for imputation, which
    columns survive correlation pruning, and the fitted scaler + PCA. apply()
    replays those on any matrix without re-estimating anything, so a held-out
    patient never influences the transform it is scored through.
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
    """Fit the full S4 reduction on TRAIN rows only; return a Reducer.

    Every statistic (kept columns, medians, pruned columns, scaling, PC axes)
    is estimated from X_train and from nothing else.
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
    """Cross-validated score of the S4 reduction + a regression model.

    Parameters
    ----------
    s4 : DataFrame
        Oriented patient x analyte matrix (NaNs allowed), one row per patient.
    y : array-like
        Aligned continuous target (e.g. SSPG), same length / order as s4 rows.
    groups : array-like or None
        Per-row group id -> leave-one-group-out; None -> leave-one-out.
    n_pca, model, max_missing, corr_threshold, seed
        Reduction / model knobs.
    leak : bool
        False (default, HONEST): fit the reduction per fold on train rows only.
        True (LEAKY): fit the reduction once on ALL rows, refit only the model
        per fold -> reproduces build_highdim_matrix's mild leak for comparison.

    Returns
    -------
    dict with the pooled OOF metrics, the OOF prediction vector, the leak flag,
    and a per-fold trace (train size, surviving-analyte counts, test index).
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

    Returns {'honest': {...}, 'leaky': {...}, 'gap_R2', 'gap_Pearson'} where a
    positive gap means the leaky number is optimistic relative to the honest one.
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
