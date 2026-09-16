"""Feature engineering for the wide omics matrix (S4_HealthyIQR).

S4 is the high-dimensional block (12,380 analytes by ~89 patients), reduced to a
modeling set by a pipeline of pure steps: orient, prevalence filter, variance
filter, median impute, correlation prune, PCA embed. Pruning is unsupervised and
safe on all rows, but imputation and any supervised selection must be fit on
training folds only, so build_highdim_matrix returns just the label-free reduced
matrix and supervised selection (feature_selection.nested_consensus) happens in
the CV loop.
"""
import re

import numpy as np
import pandas as pd

_PATIENT = re.compile(r"\d{2,3}-\d{3}")
_SUMMARY = ("Unnamed: 0", "All", "Expression_Mean", "Individual_Mean",
            "Individual_SD", "Num_Outlier_byIQR", "Outlier_Subject_byIQR", "Type")


def orient(s4, analyte_col="Unnamed: 0"):
    """Transpose S4 from analytes x patients to patients x analytes.

    Returns (X, site_codes, zcodes): X is indexed by Zcode with analyte columns,
    plus the site code and Zcode parsed from each 'NN-NNN/Zcode' patient header.
    """
    analytes = s4[analyte_col].astype(str).values
    patient_cols = [c for c in s4.columns if _PATIENT.search(str(c))]
    block = s4[patient_cols].T
    block.columns = analytes
    site_codes = [str(c).split("/")[0] for c in patient_cols]
    zcodes = [str(c).split("/")[1] if "/" in str(c) else str(c) for c in patient_cols]
    block.index = zcodes
    #de-duplicate analyte columns (some names repeat) by keeping the first
    block = block.loc[:, ~block.columns.duplicated()]
    return block.apply(pd.to_numeric, errors="coerce"), site_codes, zcodes


def prevalence_filter(X, max_missing=0.5):
    #drop analytes missing in more than max_missing fraction of patients
    keep = X.isna().mean(axis=0) <= max_missing
    return X.loc[:, keep]


def variance_filter(X, min_var=1e-8):
    #drop near-constant analytes (no discriminative signal)
    v = X.var(axis=0, skipna=True)
    return X.loc[:, v > min_var]


def impute(X, medians=None):
    #median-impute; pass precomputed medians (from train fold) to avoid leakage
    if medians is None:
        medians = X.median(axis=0)
    return X.fillna(medians), medians


def correlation_prune(X, threshold=0.95):
    #collapse redundant analytes: within a highly-correlated cluster keep one
    corr = np.corrcoef(X.values, rowvar=False)
    corr = np.abs(np.nan_to_num(corr))
    cols = list(X.columns)
    drop = set()
    for i in range(len(cols)):
        if cols[i] in drop:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] not in drop and corr[i, j] > threshold:
                drop.add(cols[j])
    return X.drop(columns=list(drop)), sorted(drop)


def pca_embed(X, n_components=20, seed=0):
    #dense PCA components as compact features (standardize first)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(X.values)
    k = min(n_components, *Z.shape)
    pca = PCA(n_components=k, random_state=seed).fit(Z)
    comps = pca.transform(Z)
    df = pd.DataFrame(comps, index=X.index,
                      columns=[f"omics_pc{i}" for i in range(k)])
    return df, pca.explained_variance_ratio_


def build_highdim_matrix(s4, max_missing=0.5, corr_threshold=0.95,
                         n_pca=20, analyte_col="Unnamed: 0", seed=0):
    """Reduce S4 to a compact modeling matrix, for reporting and EDA.

    Returns a dict with the reduced analyte matrix, the PCA-embedded matrix, the
    explained-variance vector, and a step-by-step count trace. Imputation and PCA
    are fit on all rows here, which is fine for reporting but a mild leak for
    scoring, so inside CV call the step functions per fold instead (fit on train
    rows, apply to held-out rows).
    """
    X, site_codes, zcodes = orient(s4, analyte_col=analyte_col)
    trace = {"start_analytes": X.shape[1], "n_patients": X.shape[0]}

    X = prevalence_filter(X, max_missing=max_missing)
    trace["after_prevalence"] = X.shape[1]

    X = variance_filter(X)
    trace["after_variance"] = X.shape[1]

    X, _ = impute(X)
    X, dropped = correlation_prune(X, threshold=corr_threshold)
    trace["after_correlation_prune"] = X.shape[1]
    trace["correlation_dropped"] = len(dropped)

    pcs, evr = pca_embed(X, n_components=n_pca, seed=seed)
    trace["n_pca_components"] = pcs.shape[1]
    trace["pca_variance_explained"] = float(np.sum(evr))

    return {"reduced": X, "pca": pcs, "explained_variance": evr,
            "site_codes": site_codes, "zcodes": zcodes, "trace": trace}
