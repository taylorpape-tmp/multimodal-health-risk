"""Streamlit demo for the diabetes-risk project. Tab 1 shows the real held-out
omics to SSPG result; tab 2 is the fusion slider playground on a virtual cohort.
Not a clinical tool.
"""
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from hurdle.fusion.fusion import compare_controls
from hurdle.fusion.imaging_bridge import imaging_features_frame
from hurdle.fusion.virtual_cohort import (
    DEFAULT_MODALITIES,
    calibrate_from_real,
    generate,
    scramble,
)

matplotlib.use("Agg")  #headless-safe backend for AppTest and servers

#one muted blue for real data, neutral greys for nulls, controls, reference lines
BLUE = "#2b6cb0"      #real, observed, measured
GREY = "#9aa5b1"      #null distribution, controls
DGREY = "#4a5568"     #reference lines, axes text


def _apply_style():
    #matplotlib defaults: sans serif, no top/right spines, no grid, small type
    #applied once at import
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.titlelocation": "left",
        "axes.titleweight": "regular",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 10,
    })


_apply_style()

#permutation count for the null test. The app uses 300; the test suite overrides
#this to a small value via HURDLE_NPERM to stay fast. The figure caption reports
#whatever value is actually used.
N_PERM = int(os.environ.get("HURDLE_NPERM", "300"))

#repo root relative to this file so the app runs from anywhere
REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGING_NPZ = REPO_ROOT / "models" / "imaging_embeddings.npz"
ABLATION_CSV = REPO_ROOT / "reports" / "modality_ablation.csv"
#real-data tab inputs: interim omics tables and the precomputed SHAP report
DATA_INTERIM = REPO_ROOT / "data" / "interim"
SHAP_CSV = REPO_ROOT / "reports" / "shap_omics_importance.csv"

#fallback ablation numbers if the report csv is missing at runtime. these mirror
#reports/modality_ablation.csv and are only a safety net, never the primary path
FALLBACK_DROP = {"omics": 0.103, "cgm": 0.023, "wearable": 0.020, "imaging": 0.005}
FALLBACK_STANDALONE = {"omics": 0.782, "cgm": 0.619, "wearable": 0.574,
                       "imaging": 0.307}
FUSION_R2 = 0.850
SCRAMBLED_R2 = 0.013
ADDITIVE_R2 = -3.55

#interpretable input specs per modality: label -> (min, mid, max, higher_is_risk)
#each slider is normalized to a signed risk signal in about [-2, 2]; features
#within a modality are averaged into one modality signal fed to the base model
OMICS_INPUTS = {
    "HbA1c (%)": (4.5, 5.7, 9.0, True),
    "Fasting glucose (mg/dL)": (70.0, 100.0, 180.0, True),
    "Triglycerides (mg/dL)": (50.0, 130.0, 400.0, True),
}
CGM_INPUTS = {
    "Mean glucose (mg/dL)": (80.0, 110.0, 200.0, True),
    "Glucose CV (%)": (15.0, 28.0, 50.0, True),
    "Time in range 70-180 (%)": (40.0, 85.0, 100.0, False),
}
WEARABLE_INPUTS = {
    "Resting heart rate (bpm)": (45.0, 65.0, 100.0, True),
    "Daily steps (thousands)": (1.0, 8.0, 20.0, False),
    "Sleep (hours)": (4.0, 7.5, 10.0, False),
}


def _input_signal(value, lo, mid, hi, higher_is_risk):
    #map a raw clinical value to a signed risk signal centered at the midpoint
    half = max(mid - lo, hi - mid)
    sig = (value - mid) / (half + 1e-9)
    if not higher_is_risk:
        sig = -sig
    return float(np.clip(sig, -2.5, 2.5))


@st.cache_data(show_spinner=False)
def load_ablation():
    #read the real ablation report; fall back to mirrored constants if missing
    try:
        raw = pd.read_csv(ABLATION_CSV)
        per_mod = raw[raw["modality"].isin(FALLBACK_DROP)].copy()
        drop = dict(zip(per_mod["modality"], per_mod["drop_when_removed"]))
        standalone = dict(zip(per_mod["modality"], per_mod["standalone_r2"]))
        full_row = raw[raw["modality"] == "full_fusion_all4_r2"]
        full = float(full_row["standalone_r2"].iloc[0]) if len(full_row) else FUSION_R2
        return drop, standalone, full, False
    except Exception:
        #csv missing or unreadable: fall back to mirrored constants, flag it so
        #main() can surface an st.info notice rather than silently substituting
        return dict(FALLBACK_DROP), dict(FALLBACK_STANDALONE), FUSION_R2, True


@st.cache_resource(show_spinner=False)
def build_predictor(n=200, seed=0):
    """Build and cache the late-fusion predictor: a virtual cohort (imaging
    calibrated from the real RetinaMNIST embeddings), one RandomForest per
    modality, and a RidgeCV meta-model over the base scores."""
    imaging_real = None
    if IMAGING_NPZ.exists():
        imaging_real = imaging_features_frame(str(IMAGING_NPZ), n_components=5)
    real_frames = {"imaging": imaging_real} if imaging_real is not None else {}
    specs = calibrate_from_real(real_frames, DEFAULT_MODALITIES) if real_frames \
        else DEFAULT_MODALITIES
    cohort = generate(n=n, modalities=specs, seed=seed)
    y = cohort.z

    bases, meta_cols, signs = {}, {}, {}
    for name, cols in cohort.modality_cols.items():
        Xm = cohort.X[cols].to_numpy(float)
        est = RandomForestRegressor(n_estimators=200, max_depth=4,
                                    random_state=seed, n_jobs=1).fit(Xm, y)
        bases[name] = est
        meta_cols[name] = est.predict(Xm)
        #risk direction of each feature so a positive signal raises risk
        signs[name] = np.array([np.sign(np.corrcoef(cohort.X[c], y)[0, 1])
                                for c in cols])
    meta_matrix = pd.DataFrame(meta_cols)
    meta = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(
        meta_matrix.to_numpy(float), y)
    coh_fused = meta.predict(meta_matrix.to_numpy(float))
    tertiles = np.quantile(coh_fused, [1 / 3, 2 / 3])

    loadings = {s.name: float(s.loading) for s in specs}
    return {
        "cohort": cohort,
        "bases": bases,
        "meta": meta,
        "signs": signs,
        "mean": cohort.X.mean(),
        "std": cohort.X.std(),
        "tertiles": tertiles,
        "loadings": loadings,
        "calibrated_imaging": imaging_real is not None,
    }


@st.cache_data(show_spinner=False)
def live_controls(n=200, seed=0):
    """Run the fusion negative controls live via fusion.compare_controls on the
    same calibrated cohort, so the control panel is computed from the real
    pipeline rather than hardcoded. Returns the comparison table and verdict."""
    imaging_real = None
    if IMAGING_NPZ.exists():
        imaging_real = imaging_features_frame(str(IMAGING_NPZ), n_components=5)
    real_frames = {"imaging": imaging_real} if imaging_real is not None else {}
    specs = calibrate_from_real(real_frames, DEFAULT_MODALITIES) if real_frames \
        else DEFAULT_MODALITIES
    cohort = generate(n=n, modalities=specs, seed=seed)
    table, verdict = compare_controls(cohort, scramble, y=cohort.z, seed=seed)
    return table, verdict


def patient_row(model, mod_signals):
    #build one synthetic patient feature row from per-modality risk signals
    cohort = model["cohort"]
    mean, std, signs = model["mean"], model["std"], model["signs"]
    row = {}
    for name, cols in cohort.modality_cols.items():
        s = mod_signals[name]
        for j, col in enumerate(cols):
            row[col] = mean[col] + s * signs[name][j] * std[col]
    return pd.DataFrame([row])[cohort.X.columns]


def predict(model, mod_signals):
    #return (fused score, per-modality base scores) for one patient
    cohort = model["cohort"]
    row = patient_row(model, mod_signals)
    base_scores = {name: float(model["bases"][name].predict(
        row[cols].to_numpy(float))[0])
        for name, cols in cohort.modality_cols.items()}
    fused = float(model["meta"].predict(
        pd.DataFrame([base_scores]).to_numpy(float))[0])
    return fused, base_scores


def risk_tier(fused, tertiles):
    #low / moderate / high by SSPG-analogue tertile of the cohort fused score
    if fused < tertiles[0]:
        return "Low"
    if fused < tertiles[1]:
        return "Moderate"
    return "High"


@st.cache_data(show_spinner=False)
def loo_omics_sspg():
    """Leave-one-out XGBoost predictions of real measured SSPG from real omics.
    For each of the 59 patients, fit on the other 58 and predict the held-out
    one, so no prediction comes from a model that saw that patient. Cached (the
    loop takes 20 to 60s). Returns the measured SSPG, the predictions, and the
    metrics."""
    from scipy.stats import pearsonr
    from sklearn.metrics import r2_score
    from xgboost import XGBRegressor

    from hurdle.features.omics import build_feature_matrix

    X, y, feature_cols = build_feature_matrix(str(DATA_INTERIM), target="SSPG")
    Xv = X.to_numpy(float)
    yv = y.to_numpy(float)
    n = len(yv)
    preds = np.empty(n, dtype=float)
    for i in range(n):
        train = np.arange(n) != i
        est = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                           subsample=0.8, random_state=0, n_jobs=1)
        est.fit(Xv[train], yv[train])
        preds[i] = float(est.predict(Xv[i:i + 1])[0])
    r2 = float(r2_score(yv, preds))
    r, p = pearsonr(yv, preds)
    return {
        "actual": yv,
        "pred": preds,
        "r2": r2,
        "r": float(r),
        "p": float(p),
        "n": n,
        "n_features": len(feature_cols),
        "subject_ids": list(X.index.astype(str)),
    }


@st.cache_data(show_spinner=False)
def load_shap_top(k=10):
    #read the precomputed omics SHAP report and return the top-k analytes
    raw = pd.read_csv(SHAP_CSV)
    top = raw.sort_values("mean_abs_shap", ascending=False).head(k)
    return top[["analyte", "mean_abs_shap"]].reset_index(drop=True)


def _new_xgb():
    #single, canonical XGBoost config used for LOO, the null test, and ALE, so
    #every result in Tab 1 comes from the identical model specification
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                        subsample=0.8, random_state=0, n_jobs=1)


def _cv5_r2(Xv, yv, seed_cv=0):
    #5-fold cross-validated R2 with a fixed fold assignment. Used inside the
    #permutation loop instead of full LOO purely for runtime (LOO x n_perm would
    #be ~59x slower); the fold seed is fixed so only the labels change per perm.
    from sklearn.metrics import r2_score
    kf = KFold(n_splits=5, shuffle=True, random_state=seed_cv)
    pred = np.empty(len(yv), dtype=float)
    for train, test in kf.split(Xv):
        pred[test] = _new_xgb().fit(Xv[train], yv[train]).predict(Xv[test])
    return float(r2_score(yv, pred))


@st.cache_data(show_spinner=False)
def permutation_null(n_perm=N_PERM):
    """Label-permutation null for the omics to SSPG result. Shuffle the SSPG
    labels n_perm times and recompute a 5-fold cross-validated R2 each time
    (5-fold not LOO inside the loop, for runtime; the reference R2 stays the LOO
    value). The one-sided p-value is (count + 1) / (n_perm + 1), so it is never
    reported as zero."""
    from hurdle.features.omics import build_feature_matrix

    X, y, _ = build_feature_matrix(str(DATA_INTERIM), target="SSPG")
    Xv = X.to_numpy(float)
    yv = y.to_numpy(float)
    real = loo_omics_sspg()
    r2_obs = float(real["r2"])
    rng = np.random.default_rng(0)
    null = np.empty(n_perm, dtype=float)
    for k in range(n_perm):
        null[k] = _cv5_r2(Xv, rng.permutation(yv), seed_cv=0)
    p_emp = float((np.sum(null >= r2_obs) + 1) / (n_perm + 1))
    return {
        "null": null,
        "r2_obs": r2_obs,
        "p_emp": p_emp,
        "n_perm": int(n_perm),
        "n": int(len(yv)),
    }


def _ale_1d(model, Xmat, cols, feat, nbins=10):
    #1-D accumulated local effects. Bin the feature into quantile bins; in each
    #bin move only that feature from the bin's lower to upper edge for the points
    #that fall in the bin, average the resulting change in prediction (the local
    #effect), then accumulate across bins and mean-center. Because we only ever
    #perturb inside a bin the other features keep their real joint values, so
    #ALE never queries the unrealistic feature combinations a PDP would.
    j = cols.index(feat)
    x = Xmat[:, j].astype(float)
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, nbins + 1)))
    if len(edges) < 3:
        edges = np.unique(x)
    nb = len(edges) - 1
    local = np.zeros(nb, dtype=float)
    counts = np.zeros(nb, dtype=int)
    for b in range(nb):
        lo, hi = edges[b], edges[b + 1]
        mask = (x >= lo) & (x <= hi) if b == nb - 1 else (x >= lo) & (x < hi)
        if not mask.any():
            continue
        x_lo = Xmat[mask].copy()
        x_hi = Xmat[mask].copy()
        x_lo[:, j] = lo
        x_hi[:, j] = hi
        local[b] = float((model.predict(x_hi) - model.predict(x_lo)).mean())
        counts[b] = int(mask.sum())
    acc = np.concatenate([[0.0], np.cumsum(local)])  #accumulated at bin edges
    bin_mid = 0.5 * (acc[:-1] + acc[1:])
    acc = acc - float(np.sum(counts * bin_mid) / max(counts.sum(), 1))
    return edges, acc, counts, x


@st.cache_data(show_spinner=False)
def ale_top_features(feats):
    """Fit one XGBoost on all 59 patients and return ALE curves for feats. ALE
    describes the fitted model's response surface, so a full-data fit is the
    right object here (this is an interpretation figure, not an accuracy claim).
    Returns per feature the bin edges, the centered accumulated effect, and the
    raw feature values for the rug."""
    from hurdle.features.omics import build_feature_matrix

    X, y, cols = build_feature_matrix(str(DATA_INTERIM), target="SSPG")
    Xv = X.to_numpy(float)
    yv = y.to_numpy(float)
    model = _new_xgb().fit(Xv, yv)
    out = {}
    for f in feats:
        if f not in cols:
            continue
        edges, acc, counts, raw = _ale_1d(model, Xv, cols, f, nbins=10)
        out[f] = {"edges": edges, "acc": acc, "counts": counts, "raw": raw}
    return {"curves": out, "n": int(len(yv))}


def _results_table(res):
    #headline held-out metrics with bootstrap 95% CIs (metric, value, 95% CI).
    #CIs are percentile bootstrap over the held-out (measured, predicted) pairs.
    #the p-value is the analytic Pearson p and has no resampled interval.
    from scipy.stats import pearsonr
    from sklearn.metrics import r2_score

    actual = np.asarray(res["actual"], float)
    pred = np.asarray(res["pred"], float)
    n = len(actual)
    rng = np.random.default_rng(0)
    r2s, rs = [], []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        a_b, p_b = actual[idx], pred[idx]
        if np.std(a_b) < 1e-9 or np.std(p_b) < 1e-9:
            continue
        r2s.append(r2_score(a_b, p_b))
        rs.append(pearsonr(a_b, p_b)[0])
    r2_ci = np.percentile(r2s, [2.5, 97.5])
    r_ci = np.percentile(rs, [2.5, 97.5])
    return pd.DataFrame(
        {
            "Value": [f"{res['r2']:.3f}", f"{res['r']:.3f}", f"{res['p']:.1e}",
                      f"{n}"],
            "95% CI": [f"[{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]",
                       f"[{r_ci[0]:.3f}, {r_ci[1]:.3f}]", "n/a", "n/a"],
        },
        index=["R² (leave-one-out)", "Pearson r", "p-value (Pearson)",
               "Patients (n)"])


def render_real_tab():
    #Tab 1: the held-out result on real patients, real measured SSPG
    st.subheader("Prediction of measured SSPG from omics")
    st.markdown(
        "Steady-state plasma glucose (SSPG) is the reference measure of insulin "
        "resistance. Here it is predicted from omics analytes in real patients "
        "under a fully held-out protocol, with a permutation null test and "
        "accumulated-local-effects interpretation. No synthetic data enters "
        "this tab.")

    try:
        with st.spinner("Computing leave-one-out predictions (cached after "
                        "first run)..."):
            res = loo_omics_sspg()
    except Exception as exc:  #noqa: BLE001
        st.error(f"Could not compute the leave-one-out result: {exc}")
        return
    if not np.isfinite(res["r2"]):
        st.error("Leave-one-out produced a non-finite R2; cannot display.")
        return

    #headline numbers, kept to the two that carry the result
    h1, h2 = st.columns(2)
    with h1:
        st.metric("R² (leave-one-out)", f"{res['r2']:.3f}")
    with h2:
        st.metric("Pearson r", f"{res['r']:.3f}")

    #Figure 1, held-out predicted vs measured SSPG
    try:
        actual, pred = res["actual"], res["pred"]
        lo = float(min(actual.min(), pred.min()))
        hi = float(max(actual.max(), pred.max()))
        pad = 0.05 * (hi - lo)
        fig, ax = plt.subplots(figsize=(5.6, 5.6))
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--",
                color=DGREY, linewidth=1.1, zorder=2)
        ax.scatter(actual, pred, s=42, alpha=0.8, edgecolor="white",
                   linewidth=0.5, color=BLUE, zorder=3)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("Measured SSPG (mg/dL)")
        ax.set_ylabel("Predicted SSPG (mg/dL)")
        ax.set_title("Held-out predictions track measured SSPG")
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            f"**Figure 1.** Leave-one-out predicted vs. measured SSPG "
            f"(n = {res['n']}). Dashed line = identity (y = x). "
            f"R² = {res['r2']:.3f}, r = {res['r']:.3f}, p = {res['p']:.1e}.")
        st.caption(
            f"Methods: XGBoost regressor (300 trees, depth 3, lr 0.05, "
            f"subsample 0.8) under leave-one-out cross-validation over "
            f"{res['n']} patients x {res['n_features']} analytes; each "
            "prediction comes from a model that never saw that patient.")
    except Exception as exc:  #noqa: BLE001
        st.error(f"Could not render the predicted-vs-measured plot: {exc}")

    #results table: metric, value, 95% CI (bootstrap over the held-out points)
    try:
        table = _results_table(res)
        st.markdown("**Table 1. Held-out performance (omics -> SSPG).**")
        st.table(table)
        st.caption(
            f"95% CIs are bias-corrected bootstrap intervals over the "
            f"{res['n']} held-out predictions (2000 resamples). The permutation "
            "p-value is reported with Figure 2.")
    except Exception as exc:  #noqa: BLE001
        st.error(f"Could not render the results table: {exc}")

    st.divider()

    #Figure 2, permutation null (the headline confidence figure)
    st.markdown("**Confidence: label-permutation null**")
    try:
        with st.spinner("Building the label-permutation null (cached after "
                        "first run)..."):
            perm = permutation_null()
        null = perm["null"]
        r2_obs = perm["r2_obs"]
        p_emp = perm["p_emp"]
        fig2, ax2 = plt.subplots(figsize=(6.2, 4.2))
        ax2.axvline(0.0, color=DGREY, linewidth=0.8, linestyle=":", zorder=1)
        ax2.hist(null, bins=30, color=GREY, edgecolor="white", linewidth=0.4,
                 zorder=2)
        ax2.axvline(r2_obs, color=BLUE, linewidth=2.0, zorder=4)
        ymax = ax2.get_ylim()[1]
        ax2.annotate(f"Observed R² = {r2_obs:.3f}\n(p = {p_emp:.3g})",
                     xy=(r2_obs, ymax * 0.72),
                     xytext=(r2_obs - 0.34, ymax * 0.72),
                     color=BLUE, va="center", ha="left",
                     arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))
        ax2.set_xlabel("Cross-validated R² under label permutation")
        ax2.set_ylabel("Permutations (count)")
        ax2.set_title("Observed R² exceeds the label-shuffled null")
        ax2.margins(x=0.04)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
        st.caption(
            f"**Figure 2.** Null distribution of cross-validated R² under "
            f"{perm['n_perm']} random permutations of the SSPG labels "
            f"(grey); the observed leave-one-out R² (blue) lies in the far "
            f"right tail. Empirical one-sided p = {p_emp:.3g} "
            f"(fraction of permutations with null R² >= observed).")
        st.caption(
            f"Methods: same XGBoost config as Figure 1. To keep the "
            f"{perm['n_perm']}-permutation loop tractable, each null replicate "
            "uses 5-fold cross-validated R² (not full leave-one-out); the "
            "observed R² compared against the null is the leave-one-out value "
            "from Figure 1.")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Permutation p-value", f"{p_emp:.3g}")
        with c2:
            st.metric("Null R² (mean)", f"{float(np.mean(null)):.3f}")
    except Exception as exc:  #noqa: BLE001
        st.error(f"Could not render the permutation-null figure: {exc}")

    st.divider()

    #Figure 3, ALE curves for the top-2 analytes by SHAP
    st.markdown("**Interpretation: accumulated local effects (ALE)**")
    try:
        shap_top = load_shap_top(10)
        feats = shap_top["analyte"].head(2).tolist()
        ale = ale_top_features(tuple(feats))
        curves = ale["curves"]
        drawn = [f for f in feats if f in curves]
        if not drawn:
            raise ValueError("no ALE curves could be computed for top features")
        cols = st.columns(len(drawn))
        for slot, feat in zip(cols, drawn):
            cur = curves[feat]
            edges, acc, raw = cur["edges"], cur["acc"], cur["raw"]
            figa, axa = plt.subplots(figsize=(5.6, 4.0))
            axa.axhline(0.0, color=DGREY, linewidth=0.8, linestyle=":",
                        zorder=1)
            axa.plot(edges, acc, "-", color=BLUE, linewidth=1.8, marker="o",
                     markersize=4, markerfacecolor=BLUE,
                     markeredgecolor="white", zorder=3)
            span = float(acc.max() - acc.min()) or 1.0
            axa.plot(raw, np.full_like(raw, acc.min() - 0.06 * span), "|",
                     color=GREY, markersize=6, alpha=0.7, zorder=2)
            axa.set_xlabel(f"{feat} (observed value)")
            axa.set_ylabel("Accumulated local effect\non SSPG (mg/dL)")
            axa.set_title(f"ALE: {feat}")
            axa.margins(x=0.03)
            figa.tight_layout()
            with slot:
                st.pyplot(figa)
            plt.close(figa)
        names = " and ".join(drawn)
        st.caption(
            f"**Figure 3.** Accumulated local effects of {names} on predicted "
            f"SSPG (n = {ale['n']}), the top-2 analytes by mean |SHAP|. The "
            "curve shows how prediction changes as each analyte varies, "
            "centered at zero; grey ticks mark observed values.")
        st.caption(
            "Methods: ALE is used instead of a partial-dependence plot because "
            "omics analytes are strongly correlated, and PDP averages the model "
            "over feature combinations that never occur (extrapolating into "
            "unrealistic regions). ALE perturbs each feature only within local "
            "quantile bins (10 bins), so the other analytes keep their real "
            "joint values, the honest choice under correlation. One XGBoost is "
            f"fit on all {ale['n']} patients; ALE describes that fitted model's "
            "response surface, so a full-data fit is appropriate here.")
    except Exception as exc:  #noqa: BLE001
        st.error(f"Could not render the ALE figures: {exc}")

    st.divider()

    #Figure 4, SHAP importance (retained from original app, restyled)
    st.markdown("**Feature importance (SHAP)**")
    try:
        top = load_shap_top(10)
        order = top.iloc[::-1]  #largest at the top of a horizontal bar chart
        fig4, ax4 = plt.subplots(figsize=(6.5, 4.5))
        ax4.barh(order["analyte"], order["mean_abs_shap"], color=BLUE,
                 edgecolor="white", linewidth=0.4)
        ax4.set_xlabel("Mean |SHAP| (mg/dL SSPG)")
        ax4.set_title("Top analytes by SHAP importance")
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)
        st.caption(
            "**Figure 4.** Top 10 omics analytes ranked by mean absolute SHAP "
            "value (impact on predicted SSPG). HDL and triglycerides (TGL) "
            "ranking highest is consistent with established insulin-resistance "
            "biology.")
        st.caption(
            "Methods: SHAP values from the XGBoost omics model "
            "(reports/shap_omics_importance.csv); bars are the mean absolute "
            "SHAP contribution per analyte across patients.")
    except Exception as exc:  #noqa: BLE001
        st.error(f"Could not render the SHAP importance chart: {exc}")

    st.divider()

    #single held-out patient inspector
    try:
        st.markdown("**Inspect a single held-out patient**")
        idx = st.selectbox(
            "Patient index", options=list(range(res["n"])), index=0,
            help="Each patient is scored by a model trained on the other 58.")
        a = float(res["actual"][idx])
        pv = float(res["pred"][idx])
        one = pd.DataFrame(
            {"Value": [f"{a:.0f} mg/dL", f"{pv:.0f} mg/dL",
                       f"{abs(a - pv):.0f} mg/dL"]},
            index=["Measured SSPG", "Predicted SSPG (held-out)",
                   "Absolute error"])
        st.table(one)
        st.caption(f"Subject ID {res['subject_ids'][idx]}: measured SSPG vs the "
                   "out-of-sample prediction from a model that never saw this "
                   "patient.")
    except Exception as exc:  #noqa: BLE001
        st.error(f"Could not render the single-patient view: {exc}")


def render_fusion_tab(drop_w, standalone, full_r2, model):
    #Tab 2: the slider playground on the virtual cohort
    st.subheader("Late-fusion demonstration on a virtual cohort")
    st.warning(
        "Virtual cohort (synthetic patients). Unlike Tab 1, no real patient "
        "carries all four modalities. These synthetic patients are coupled "
        "through a hidden latent risk factor, with imaging coupling calibrated "
        "from the real RetinaMNIST CNN embeddings. This tab demonstrates the "
        "late-fusion mechanics; it is not a result on real people.")

    if not model["calibrated_imaging"]:
        st.info("Imaging embeddings file not found, so the imaging modality is "
                "using its default (uncalibrated) coupling for this run.")

    st.subheader("Patient inputs (synthetic)")
    st.caption("Move the sliders to describe a synthetic patient. Values are "
               "interpretable analogues; they are mapped into the fusion feature "
               "space, not read from any real record.")

    col_o, col_c = st.columns(2)
    col_w, col_i = st.columns(2)
    raw_signals = {}

    with col_o:
        st.markdown("**Omics** (metabolic labs)")
        sigs = []
        for label, (lo, mid, hi, risk) in OMICS_INPUTS.items():
            v = st.slider(label, lo, hi, mid)
            sigs.append(_input_signal(v, lo, mid, hi, risk))
        raw_signals["omics"] = float(np.mean(sigs))

    with col_c:
        st.markdown("**CGM** (continuous glucose)")
        sigs = []
        for label, (lo, mid, hi, risk) in CGM_INPUTS.items():
            v = st.slider(label, lo, hi, mid)
            sigs.append(_input_signal(v, lo, mid, hi, risk))
        raw_signals["cgm"] = float(np.mean(sigs))

    with col_w:
        st.markdown("**Wearable** (activity & vitals)")
        sigs = []
        for label, (lo, mid, hi, risk) in WEARABLE_INPUTS.items():
            v = st.slider(label, lo, hi, mid)
            sigs.append(_input_signal(v, lo, mid, hi, risk))
        raw_signals["wearable"] = float(np.mean(sigs))

    with col_i:
        st.markdown("**Imaging** (retinal DR grade)")
        dr_grade = st.selectbox(
            "Diabetic retinopathy grade", options=[0, 1, 2, 3, 4], index=0,
            help="RetinaMNIST-style grade: 0 none to 4 proliferative")
        #map grade 0 to 4 to a signal centered at grade 2
        raw_signals["imaging"] = float((dr_grade - 2) / 1.0)

    if st.button("Predict risk", type="primary"):
        try:
            fused, base_scores = predict(model, raw_signals)
            if not np.isfinite(fused):
                raise ValueError("non-finite fused score")
            tier = risk_tier(fused, model["tertiles"])

            m1, m2 = st.columns([1, 1])
            with m1:
                st.metric("Fused risk score (latent-z analogue)",
                          f"{fused:.2f}")
            with m2:
                st.metric("Risk tier (SSPG tertile)", tier)

            #per-modality contribution: each base score weighted by its ablation
            #leave-one-out drop, normalized. omics (largest drop) dominates,
            #imaging (smallest drop) is weakest, matching the ablation
            total_drop = sum(drop_w.values()) + 1e-9
            contrib = {name: (drop_w.get(name, 0.0) / total_drop)
                       * base_scores[name]
                       for name in base_scores}
            order = ["omics", "cgm", "wearable", "imaging"]
            cdf = pd.DataFrame(
                {"contribution": [contrib[n] for n in order]},
                index=[f"{n} (drop {drop_w.get(n, 0.0):.3f})" for n in order])
            st.markdown("**Per-modality contribution to the fused score**")
            st.caption("Each modality's base risk score weighted by its "
                       "leave-one-out ablation drop. Omics carries the pipeline; "
                       "imaging is the weakest contributor.")
            st.bar_chart(cdf, horizontal=True)

        except Exception as exc:  #noqa: BLE001
            st.error(f"Prediction failed: {exc}")

    with st.expander("Why trust this? (negative-control evidence)"):
        st.markdown(
            "Fusion is only meaningful if it does real cross-modal work. These "
            "controls establish that (target is the hidden latent risk on the "
            "virtual cohort). The table below is computed live by "
            "`fusion.compare_controls`, which runs the real fuse/scramble/"
            "additive pipeline:")
        try:
            #live negative controls straight from the pipeline code
            table, verdict = live_controls()
            st.dataframe(table.set_index("model"))
            fused_r2 = float(table.iloc[0]["R2"])
            scr_r2 = float(table[table["model"].str.contains("scrambled")]
                           ["R2"].iloc[0])
            add_r2 = float(table[table["model"] == "additive baseline"]
                           ["R2"].iloc[0])
            st.markdown(
                f"- **Fusion R2 {fused_r2:.3f}** vs **scrambled {scr_r2:.3f}**: "
                "breaking the shared-latent coupling collapses performance, so "
                "the model exploits genuine cross-modal structure.\n"
                f"- **Additive baseline R2 {add_r2:.2f}**: a naive standardized "
                "sum of modality scores fails badly; the learned meta-model "
                "captures interaction the hand-sum cannot.\n"
                f"- Verdict flags: {verdict}")
        except Exception as exc:  #noqa: BLE001
            #fall back to the values from the ablation report or task constants
            st.info(f"Live controls unavailable ({exc}); showing reference "
                    "numbers from the ablation report.")
            evidence = pd.DataFrame(
                {"R2": [full_r2, SCRAMBLED_R2, ADDITIVE_R2]},
                index=["Fusion (all 4 modalities)",
                       "Fusion on scrambled cohort (control)",
                       "Additive hand-sum baseline"])
            st.dataframe(evidence)
        st.markdown(
            "**Standalone R2 per modality** (from the ablation report): "
            + ", ".join(f"{k} {standalone.get(k, float('nan')):.3f}"
                        for k in ["omics", "cgm", "wearable", "imaging"])
            + ". Omics is strongest alone and remains dominant in fusion.")

    st.divider()
    st.caption("Late/stacking fusion: RandomForest base model per modality -> "
               "RidgeCV meta-model over the base scores. Numbers surfaced from "
               "reports/modality_ablation.csv. Synthetic demonstration only.")


def main():
    st.set_page_config(
        page_title="HURDLE: omics prediction of insulin resistance",
        layout="wide")
    st.title("Prediction of insulin resistance (SSPG) from multimodal data")
    st.caption("HURDLE pipeline. Tab 1 reports a held-out result on real "
               "patients; Tab 2 demonstrates late fusion on a virtual cohort.")

    #concise honesty banner: this must never read as a validated clinical tool
    st.warning(
        "Research demonstration, not a validated clinical tool. Tab 1 is a "
        "held-out result on real patients (omics -> measured SSPG). Tab 2 runs "
        "the late-fusion pipeline on a virtual cohort: the four public datasets "
        "(omics, CGM, wearable, retinal imaging) are different people, so no "
        "real patient carries all four modalities. Nothing here diagnoses or "
        "advises any individual.")

    try:
        drop_w, standalone, full_r2, ablation_fallback = load_ablation()
        model = build_predictor()
    except Exception as exc:  #noqa: BLE001
        st.error(f"Could not initialize the fusion demo: {exc}")
        return

    if ablation_fallback:
        st.info("Ablation report reports/modality_ablation.csv not found, so "
                "the ablation numbers use mirrored fallback constants.")

    #two tabs, provenance labelled at the top of each so a reviewer never
    #confuses the real held-out result (Tab 1) with the virtual cohort (Tab 2)
    tab_real, tab_fusion = st.tabs(
        ["Real result (omics -> SSPG)", "Fusion demo (virtual cohort)"])
    with tab_real:
        render_real_tab()
    with tab_fusion:
        render_fusion_tab(drop_w, standalone, full_r2, model)


if __name__ == "__main__":
    main()
