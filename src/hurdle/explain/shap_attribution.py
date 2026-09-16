"""Model-based feature attribution for the omics -> SSPG regression.

The consensus panel (reports/consensus_panel_sspg.csv) ranks analytes by how
often the four-family nested selector re-picks them across leave-one-out folds.
That is a *selection-stability* signal. This module adds an independent,
model-based *attribution* layer on top of it:

  fit_and_explain()  -> fits one XGBoost regressor on the full real S8 matrix and
                        computes exact TreeExplainer SHAP values. Returns a
                        per-feature mean(|SHAP|) ranking (global importance), the
                        raw SHAP value matrix (for beeswarm / bar plots), XGBoost's
                        native gain-based feature_importances_, and the fitted model.
  compare_rankings() -> joins SHAP mean|value|, native gain, and consensus
                        selection frequency for the top analytes, so agreement
                        across three independent methods can be shown directly.

Everything here operates on the real matrix produced by
hurdle.features.omics.build_feature_matrix; nothing is synthetic. SHAP's
TreeExplainer is exact for tree ensembles (no sampling), so mean(|SHAP|) is a
deterministic function of the fitted model.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor

#fixed XGBoost config for a single full-data fit. Shallow trees + modest depth
#suit the wide-small-n omics regime (59 patients x ~86 analytes) and match the
#tree_models.py grid's centre of mass; a fixed seed makes the SHAP ranking
#reproducible run to run.
_XGB_DEFAULTS = {
    "n_estimators": 300,
    "max_depth": 2,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 0,
    "objective": "reg:squarederror",
    "importance_type": "gain",
    "n_jobs": 1,
}


def _fit_xgboost(X, y, params=None):
    """Fit one XGBoost regressor on the full matrix (no CV: this is the model we
    explain, not a performance estimate)."""
    cfg = dict(_XGB_DEFAULTS)
    if params:
        cfg.update(params)
    model = XGBRegressor(**cfg)
    model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
    return model


def fit_and_explain(X, y, model="xgboost", params=None):
    """Fit a tree model on (X, y) and attribute the prediction with SHAP.

    X: DataFrame (or array) of analytes; if a DataFrame, its columns name the
       features. y: continuous SSPG target. model: only 'xgboost' is supported.

    Returns a dict:
      shap_ranking     DataFrame(analyte, mean_abs_shap, rank_shap) sorted desc
      gain_ranking     DataFrame(analyte, native_gain, rank_gain)   sorted desc
      shap_values      np.ndarray (n_samples, n_features) raw SHAP values
      feature_names    list[str] column order matching shap_values
      model            the fitted estimator
      X                the feature matrix actually fitted (as DataFrame)
    """
    if model != "xgboost":
        raise ValueError(f"unsupported model {model!r}; only 'xgboost' is implemented")

    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X_df = X.astype(float)
    else:
        X_df = pd.DataFrame(np.asarray(X, dtype=float))
        feature_names = [str(c) for c in X_df.columns]

    est = _fit_xgboost(X_df, y, params=params)

    #TreeExplainer is exact for tree ensembles: no background sampling, no
    #approximation, single-threaded and fast on this matrix size.
    explainer = shap.TreeExplainer(est)
    shap_values = explainer.shap_values(X_df)
    shap_values = np.asarray(shap_values, dtype=float)

    #global importance = mean absolute SHAP value per feature across all patients
    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_ranking = (
        pd.DataFrame({"analyte": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    shap_ranking["rank_shap"] = np.arange(1, len(shap_ranking) + 1)

    #second, independent importance: XGBoost native gain (total loss reduction
    #contributed by splits on each feature). feature_importances_ is normalised.
    gain = np.asarray(est.feature_importances_, dtype=float)
    gain_ranking = (
        pd.DataFrame({"analyte": feature_names, "native_gain": gain})
        .sort_values("native_gain", ascending=False)
        .reset_index(drop=True)
    )
    gain_ranking["rank_gain"] = np.arange(1, len(gain_ranking) + 1)

    return {
        "shap_ranking": shap_ranking,
        "gain_ranking": gain_ranking,
        "shap_values": shap_values,
        "feature_names": feature_names,
        "model": est,
        "X": X_df,
    }


def _load_consensus(consensus_csv):
    """Read the consensus panel CSV -> DataFrame(analyte, consensus_freq)."""
    path = Path(consensus_csv)
    if not path.exists():
        raise FileNotFoundError(f"consensus CSV not found: {path}")
    con = pd.read_csv(path)
    if "analyte" not in con.columns or "selection_frequency" not in con.columns:
        raise ValueError(
            f"{path} must have 'analyte' and 'selection_frequency' columns; "
            f"got {list(con.columns)}"
        )
    return con[["analyte", "selection_frequency"]].rename(
        columns={"selection_frequency": "consensus_freq"}
    )


def compare_rankings(shap_rank, gain_rank, consensus_csv, top_n=20):
    """Join SHAP, native-gain, and consensus rankings for the top analytes.

    shap_rank / gain_rank: the DataFrames returned by fit_and_explain
    (columns analyte + mean_abs_shap/rank_shap and native_gain/rank_gain).
    consensus_csv: path to reports/consensus_panel_sspg.csv.

    Returns a DataFrame of the top_n analytes *by SHAP*, each carrying its
    mean|SHAP|, native gain, consensus selection frequency, and each method's
    rank. Analytes absent from the consensus file get consensus_freq = NaN
    (never a fabricated 0). Sorted by rank_shap.
    """
    con = _load_consensus(consensus_csv)
    merged = (
        shap_rank.merge(gain_rank, on="analyte", how="left")
        .merge(con, on="analyte", how="left")
        .sort_values("rank_shap")
        .reset_index(drop=True)
    )
    cols = [
        "analyte",
        "mean_abs_shap",
        "native_gain",
        "consensus_freq",
        "rank_shap",
        "rank_gain",
    ]
    return merged[cols].head(top_n).reset_index(drop=True)


def signal_concentration(shap_rank, thresholds=(0.80, 0.90, 0.95)):
    """How concentrated is the SHAP signal: how few features carry most of it.

    Returns a dict with the total mean|SHAP| mass, a per-feature cumulative
    fraction, and, for each threshold, the number of top features whose
    cumulative mean|SHAP| first reaches that fraction of the total.
    """
    vals = shap_rank.sort_values("mean_abs_shap", ascending=False)["mean_abs_shap"].values
    total = float(vals.sum())
    if total <= 0:
        raise ValueError("total mean|SHAP| is zero; the model attributes nothing")
    cum = np.cumsum(vals) / total
    n_for = {t: int(np.searchsorted(cum, t) + 1) for t in thresholds}
    return {
        "total_mean_abs_shap": total,
        "cumulative_fraction": cum,
        "n_features_for_fraction": n_for,
        "n_features_total": int(len(vals)),
    }
