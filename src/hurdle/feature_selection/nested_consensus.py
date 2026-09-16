"""Nested four-family consensus feature selection, ported from dom_study.

Four selector families (filter by Pearson correlation, wrapper by forward LOO R^2
gain, embedded by nonzero LASSO/ElasticNet coefficient, explain by top-k
standardised OLS coefficients) vote on each feature, and a feature enters the frozen
set once enough families back it. The whole selection is repeated inside each outer
leave-one-out fold so the held-out point never informs selection, with target,
feature pool, and vote threshold all parameters.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


class ConsensusConfig:
    vote_threshold = 3          #families (of 4) required to enter the frozen set
    filter_alpha = 0.05
    wrap_min_gain = 0.01
    wrap_max_feat = 6
    explain_topk = 5
    alphas_en = np.logspace(-3, 1, 20)
    l1_ratios = (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
    ridge_alphas = np.logspace(-3, 3, 25)
    zero_tol = 1e-6
    final_models = ('Ridge', 'OLS', 'ElasticNet')
    inner_cv_splits = 5
    seed = 0


def _loo_r2(cols, idx, D, y):
    #closed-form leave-one-out R^2 of an OLS fit on the given rows/columns
    Xd = D.iloc[idx]
    yy = y[idx]
    X = np.column_stack([np.ones(len(Xd))] + [Xd[c].values.astype(float) for c in cols])
    inv = np.linalg.pinv(X.T @ X)
    h = np.clip(np.einsum('ij,jk,ik->i', X, inv, X), 0, 1 - 1e-12)
    resid = (yy - X @ (inv @ X.T @ yy)) / (1 - h)
    return 1 - (resid ** 2).sum() / ((yy - yy.mean()) ** 2).sum()


def filter_vote(tr, D, y, pool, cfg):
    return {c for c in pool
            if pearsonr(D.iloc[tr][c].values.astype(float), y[tr])[1] < cfg.filter_alpha}


def wrapper_vote(tr, D, y, pool, cfg):
    chosen, best = [], -np.inf
    while len(chosen) < cfg.wrap_max_feat:
        cand = [c for c in pool if c not in chosen]
        if not cand:
            break
        score, feat = max((_loo_r2(chosen + [c], tr, D, y), c) for c in cand)
        if score - best < cfg.wrap_min_gain:
            break
        chosen.append(feat)
        best = score
    return set(chosen)


def embedded_vote(tr, D, y, pool, cfg):
    X = D.iloc[tr][pool].values.astype(float)
    Xs = StandardScaler().fit_transform(X)
    inner = KFold(n_splits=cfg.inner_cv_splits, shuffle=True, random_state=cfg.seed)
    out = set()
    for kind in ('lasso', 'en'):
        if kind == 'lasso':
            m = LassoCV(alphas=cfg.alphas_en, cv=inner, max_iter=10000).fit(Xs, y[tr])
        else:
            m = ElasticNetCV(alphas=cfg.alphas_en, l1_ratio=list(cfg.l1_ratios),
                             cv=inner, max_iter=10000).fit(Xs, y[tr])
        out |= {f for f, c in zip(pool, m.coef_) if abs(c) > cfg.zero_tol}
    return out


def explain_vote(tr, D, y, pool, cfg):
    #standardised-coefficient ranking of an OLS fit on the training rows
    X = StandardScaler().fit_transform(D.iloc[tr][pool].values.astype(float))
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y[tr]
    order = np.argsort(-np.abs(beta))
    return set(np.array(pool)[order[:cfg.explain_topk]])


def _final_predict(kind, Xtr, ytr, Xte, cfg):
    sc = StandardScaler().fit(Xtr)
    Z, Zt = sc.transform(Xtr), sc.transform(Xte)
    if kind == 'Ridge':
        m = RidgeCV(alphas=cfg.ridge_alphas).fit(Z, ytr)
    elif kind == 'OLS':
        m = LinearRegression().fit(Z, ytr)
    else:
        inner = KFold(n_splits=cfg.inner_cv_splits, shuffle=True, random_state=cfg.seed)
        m = ElasticNetCV(alphas=cfg.alphas_en, l1_ratio=list(cfg.l1_ratios),
                         cv=inner, max_iter=10000).fit(Z, ytr)
    return m.predict(Zt)[0]


def run_nested_consensus(frame, feature_cols, target_col='target', cfg=None):
    """Nested leave-one-out consensus selection + final-model evaluation.

    Returns (summary_df, frozen_frequency_df). summary_df has one row per final
    model with its fully-external R^2/MAE/RMSE and the mean frozen-set size;
    frozen_frequency_df has, per feature, how many outer folds it entered.
    """
    cfg = cfg or ConsensusConfig()
    D = frame[feature_cols].reset_index(drop=True)
    y = frame[target_col].values.astype(float)
    feats = list(feature_cols)
    n = len(y)

    preds = {k: np.zeros(n) for k in cfg.final_models}
    frozen_counts = {f: 0 for f in feats}
    set_sizes = []

    for i in range(n):
        tr = [j for j in range(n) if j != i]
        votes = {f: 0 for f in feats}
        for fam in (filter_vote(tr, D, y, feats, cfg),
                    wrapper_vote(tr, D, y, feats, cfg),
                    embedded_vote(tr, D, y, feats, cfg),
                    explain_vote(tr, D, y, feats, cfg)):
            for f in fam:
                votes[f] += 1
        frozen = [f for f in feats if votes[f] >= cfg.vote_threshold] \
            or [max(votes, key=votes.get)]
        set_sizes.append(len(frozen))
        for f in frozen:
            frozen_counts[f] += 1
        Xtr = D.iloc[tr][frozen].values.astype(float)
        Xte = D.iloc[[i]][frozen].values.astype(float)
        for k in cfg.final_models:
            preds[k][i] = _final_predict(k, Xtr, y[tr], Xte, cfg)

    def metrics(p):
        r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        return r2, np.abs(y - p).mean(), np.sqrt(((y - p) ** 2).mean())

    summary = pd.DataFrame([
        {'final_model': k,
         'external_R2': round(metrics(preds[k])[0], 4),
         'MAE': round(metrics(preds[k])[1], 4),
         'RMSE': round(metrics(preds[k])[2], 4),
         'mean_frozen_set_size': round(float(np.mean(set_sizes)), 2),
         'n': n}
        for k in cfg.final_models])

    freq = (pd.DataFrame({'feature': feats,
                          'in_frozen_set': [frozen_counts[f] for f in feats]})
            .assign(frequency=lambda d: (d.in_frozen_set / n).round(3))
            .sort_values('in_frozen_set', ascending=False)
            .reset_index(drop=True))
    return summary, freq
