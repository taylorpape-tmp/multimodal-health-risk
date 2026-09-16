"""Model-based feature attribution for the omics -> SSPG regression.

Where the consensus panel measures selection stability, this adds a model-based
attribution layer: fit_and_explain fits one XGBoost regressor and computes exact
TreeExplainer SHAP values, returning a mean(|SHAP|) ranking, the raw SHAP matrix,
native gain importances, and the model. compare_rankings joins SHAP, native gain, and
consensus selection frequency so agreement across the three methods can be shown
directly.
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
    """Fit one XGBoost regressor on the full matrix (no CV; this is the model we
    explain, not a performance estimate)."""
    cfg = dict(_XGB_DEFAULTS)
    if params:
        cfg.update(params)
    model = XGBRegressor(**cfg)
    model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
    return model


def fit_and_explain(X, y, model="xgboost", params=None):
    """Fit a tree model on (X, y) and attribute predictions with SHAP.

    X is a DataFrame (or array) of analytes, y the continuous SSPG target; only
    'xgboost' is supported. Returns a dict with shap_ranking and gain_ranking
    DataFrames, the raw shap_values matrix, feature_names, the fitted model, and the
    feature matrix X.
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

    Takes the two ranking DataFrames from fit_and_explain and the consensus panel CSV
    path, and returns the top_n analytes by SHAP with their mean|SHAP|, native gain,
    consensus frequency, and per-method ranks (sorted by rank_shap). Analytes absent
    from the consensus file get consensus_freq = NaN, never a fabricated 0.
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
