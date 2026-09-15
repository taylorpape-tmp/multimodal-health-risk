"""Multimodal fusion model + the controls that prove fusion does real work.

Late fusion: each modality gets its own base learner, their out-of-fold
predictions become the meta-features, and a meta-learner combines them into the
final prediction of the target (latent z on the virtual cohort, or real SSPG on
the linked cohort). Out-of-fold stacking (not in-fold) keeps the meta-learner
from seeing leaked base-model fits.

Truth is always external to the model:
  - virtual cohort: the pre-set hidden latent z (regression) or iris=1[z>0]
  - real 22:        measured SSPG (regression) or IRIS (classification)

Controls (compare_controls):
  - unimodal floor: best single modality alone -> fusion must beat it
  - scramble:       break the shared-z coupling -> fusion advantage must vanish
  - additive:       a standardized hand-sum of modality scores as baseline-to-beat
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


def _oof_predictions(X, y, n_splits=5, seed=0):
    #out-of-fold predictions from a RandomForest base learner on one modality
    oof = np.zeros(len(y))
    kf = KFold(n_splits=min(n_splits, len(y)), shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        est = RandomForestRegressor(n_estimators=200, max_depth=4,
                                    random_state=seed, n_jobs=1)
        est.fit(X[tr], y[tr])
        oof[te] = est.predict(X[te])
    return oof


def modality_oof(cohort, y, n_splits=5, seed=0):
    #one out-of-fold prediction column per modality -> the meta-feature matrix
    meta = {}
    for name, cols in cohort.modality_cols.items():
        Xm = cohort.X[cols].to_numpy(dtype=float)
        meta[name] = _oof_predictions(Xm, y, n_splits=n_splits, seed=seed)
    return pd.DataFrame(meta, index=cohort.X.index)


def fuse(cohort, y, n_splits=5, seed=0):
    """Late-fusion stack. Returns dict with fused out-of-fold predictions and
    each modality's standalone out-of-fold prediction (for the ablation floor).
    """
    meta = modality_oof(cohort, y, n_splits=n_splits, seed=seed)
    #meta-learner: ridge on the per-modality oof predictions, itself out-of-fold
    fused = np.zeros(len(y))
    kf = KFold(n_splits=min(n_splits, len(y)), shuffle=True, random_state=seed)
    M = meta.to_numpy(dtype=float)
    for tr, te in kf.split(M):
        ml = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(M[tr], y[tr])
        fused[te] = ml.predict(M[te])
    return {"fused": fused, "meta": meta}


def _score(y, pred):
    return {"R2": float(r2_score(y, pred)),
            "Spearman": float(spearmanr(y, pred)[0]),
            "RMSE": float(np.sqrt(np.mean((y - pred) ** 2)))}


def additive_baseline(cohort, y):
    #hand-built score: standardized sum of each modality's mean feature. this is
    #the BASELINE TO BEAT, never a training label. a learned fusion that beats it
    #shows it captured cross-modal interaction, not just addition.
    parts = []
    for cols in cohort.modality_cols.values():
        block = cohort.X[cols].to_numpy(dtype=float)
        s = block.mean(axis=1)
        parts.append((s - s.mean()) / (s.std() + 1e-9))
    score = np.sum(parts, axis=0)
    return _score(y, score)


def compare_controls(cohort, scramble_fn, y=None, n_splits=5, seed=0):
    """Run the fusion and its three controls, returning a comparison table.

    y defaults to the cohort's latent z (the virtual-cohort truth). Pass real
    SSPG to score the real linked cohort instead.
    """
    y = cohort.z if y is None else np.asarray(y, dtype=float)

    fused = fuse(cohort, y, n_splits=n_splits, seed=seed)
    fused_score = _score(y, fused["fused"])

    #unimodal floor: best single modality's standalone oof prediction
    uni = {name: _score(y, fused["meta"][name].to_numpy())["R2"]
           for name in cohort.modality_cols}
    best_uni_name = max(uni, key=uni.get)

    #scramble control: rebuild fusion on the decoupled cohort
    scrambled = scramble_fn(cohort, seed=seed)
    scr = fuse(scrambled, y, n_splits=n_splits, seed=seed)
    scr_score = _score(y, scr["fused"])

    add = additive_baseline(cohort, y)

    rows = [
        {"model": "fusion (all modalities)", **fused_score},
        {"model": f"best single ({best_uni_name})",
         "R2": uni[best_uni_name], "Spearman": np.nan, "RMSE": np.nan},
        {"model": "fusion on scrambled (control)", **scr_score},
        {"model": "additive baseline", **add},
    ]
    table = pd.DataFrame(rows)
    verdict = {
        "fusion_beats_best_single": fused_score["R2"] > uni[best_uni_name],
        "scramble_advantage_vanishes": scr_score["R2"] < fused_score["R2"] - 0.05,
        "fusion_beats_additive": fused_score["R2"] > add["R2"],
    }
    return table, verdict
