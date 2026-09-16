"""HURDLE multimodal diabetes-risk pipeline: portfolio demonstration UI.

Two tabs:
  Tab 1 "Real result (omics -> SSPG)"  -- the honest proof on REAL patients:
      leave-one-out XGBoost predictions of real, lab-measured SSPG from real
      omics analytes. Nothing synthetic in this tab.
  Tab 2 "Fusion demo (virtual cohort)" -- the late-fusion slider playground on
      a VIRTUAL cohort (four disjoint public datasets coupled through a hidden
      latent risk factor). No real patient carries all four modalities.

This app is not a validated clinical tool. See the honesty banner rendered at
the top of the page and docs/DEMO.md for the full disclosure.
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV

from hurdle.fusion.fusion import compare_controls
from hurdle.fusion.imaging_bridge import imaging_features_frame
from hurdle.fusion.virtual_cohort import (
    DEFAULT_MODALITIES,
    calibrate_from_real,
    generate,
    scramble,
)

matplotlib.use("Agg")  #headless-safe backend for AppTest and servers

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
    """Build the deployable late-fusion predictor once and cache it.

    Generates a virtual cohort whose imaging loading is calibrated from the real
    RetinaMNIST CNN embeddings, fits one RandomForest base model per modality
    (matching fusion.py's estimator config) and a RidgeCV meta-model over the
    base scores. Returns everything the single-patient predictor needs.
    """
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
    """Run the real fusion controls via fusion.compare_controls.

    Rebuilds the same calibrated virtual cohort and calls compare_controls,
    which internally runs fuse()/modality_oof() and the scramble + additive
    baselines. Returns the comparison table and verdict computed from the
    actual pipeline code, so the negative-control panel is live, not hardcoded.
    """
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
    """Leave-one-out XGBoost predictions of REAL measured SSPG from REAL omics.

    Loads the 59-patient x 86-analyte S8 table via the feature builder, then for
    each patient fits an XGBRegressor on the other 58 and predicts the held-out
    one — so every prediction comes from a model that never saw that patient.
    Cached so the ~20-60s loop runs once per session. Returns a dict with the
    measured SSPG vector, the LOO predictions, and the honest metrics.
    """
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


def render_real_tab():
    #Tab 1: the honest proof on REAL patients / REAL measured SSPG
    st.subheader("Real result — omics -> measured SSPG")
    st.success(
        "**REAL DATA.** 59 real patients, 86 real omics analytes, and their "
        "real lab-measured SSPG (steady-state plasma glucose, 40-276 mg/dL, the "
        "gold-standard insulin-resistance measure). Every point below is a "
        "leave-one-out prediction: each patient is scored by an XGBoost model "
        "that never saw them. This is the honest, held-out result — no virtual "
        "cohort, no synthetic coupling.")

    try:
        with st.spinner("Computing leave-one-out predictions on 59 real "
                        "patients (cached after first run)..."):
            res = loo_omics_sspg()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not compute the real-data LOO result: {exc}")
        return
    if not np.isfinite(res["r2"]):
        st.error("Real-data LOO produced a non-finite R2; cannot display.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("R2 (leave-one-out)", f"{res['r2']:.3f}")
    with c2:
        st.metric("Pearson r", f"{res['r']:.3f}")
    with c3:
        st.metric("p-value", f"{res['p']:.2e}")
    st.caption(f"{res['n']} real patients x {res['n_features']} real analytes. "
               "Each prediction is fully held out (leave-one-out).")

    #THE MONEY PLOT: predicted vs actual measured SSPG, 59 held-out points
    try:
        actual, pred = res["actual"], res["pred"]
        lo = float(min(actual.min(), pred.min()))
        hi = float(max(actual.max(), pred.max()))
        pad = 0.05 * (hi - lo)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(actual, pred, s=45, alpha=0.75, edgecolor="black",
                   linewidth=0.5, color="#2b6cb0", zorder=3)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--",
                color="#718096", linewidth=1.3, label="y = x (perfect)",
                zorder=2)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("Measured SSPG (mg/dL)")
        ax.set_ylabel("LOO-predicted SSPG (mg/dL)")
        ax.set_title(f"Held-out prediction of real SSPG\n"
                     f"R2 = {res['r2']:.3f}   r = {res['r']:.3f}   "
                     f"p = {res['p']:.1e}")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper left", frameon=False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Points near the dashed y=x line are patients the model got "
                   "right. Higher SSPG = more insulin-resistant.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not render the predicted-vs-actual plot: {exc}")

    #pick a real patient: one concrete held-out case
    try:
        st.markdown("**Inspect one real patient (held-out)**")
        idx = st.selectbox(
            "Patient index (0-58)", options=list(range(res["n"])), index=0,
            help="Each choice is a real patient scored by a model that never "
                 "saw them.")
        a = float(res["actual"][idx])
        pv = float(res["pred"][idx])
        err = abs(a - pv)
        p1, p2, p3 = st.columns(3)
        with p1:
            st.metric("Actual SSPG (measured)", f"{a:.0f} mg/dL")
        with p2:
            st.metric("Predicted SSPG (held-out)", f"{pv:.0f} mg/dL")
        with p3:
            st.metric("Absolute error", f"{err:.0f} mg/dL")
        st.caption(f"Subject ID: {res['subject_ids'][idx]}. This is one real "
                   "person's real SSPG vs the model's out-of-sample guess.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not render the single-patient view: {exc}")

    #SHAP top analytes: which omics features drive the real prediction
    try:
        st.markdown("**Top analytes driving the omics model (SHAP)**")
        top = load_shap_top(10)
        order = top.iloc[::-1]  #largest at the top of a horizontal bar chart
        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        ax2.barh(order["analyte"], order["mean_abs_shap"], color="#38a169",
                 edgecolor="black", linewidth=0.4)
        ax2.set_xlabel("Mean |SHAP| (impact on predicted SSPG)")
        ax2.set_title("Top 10 omics analytes by SHAP importance")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
        st.caption("HDL and triglycerides (TGL) topping the list is exactly "
                   "what insulin-resistance biology predicts — the model is "
                   "leaning on the right analytes, not spurious noise.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not render the SHAP importance chart: {exc}")


def render_fusion_tab(drop_w, standalone, full_r2, model):
    #Tab 2: the existing slider playground on the VIRTUAL cohort, unchanged
    st.subheader("Fusion demo — virtual cohort")
    st.info(
        "**VIRTUAL COHORT (synthetic patients).** Unlike Tab 1, no real patient "
        "carries all four modalities. These synthetic patients are coupled "
        "through a hidden latent risk factor (imaging coupling calibrated from "
        "the real RetinaMNIST CNN embeddings). This tab demonstrates the "
        "late-fusion mechanics; it is not a result on real people.")

    if not model["calibrated_imaging"]:
        st.info("Imaging embeddings file not found — imaging modality is using "
                "its default (uncalibrated) coupling for this run.")

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
            help="RetinaMNIST-style grade: 0 none ... 4 proliferative")
        #map grade 0..4 to a signal centered at grade 2
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

            #per-modality contribution: weight each base score by its ablation
            #leave-one-out drop, normalized. omics (largest drop) dominates;
            #imaging (smallest drop) is visibly weakest, matching the ablation
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

        except Exception as exc:  # noqa: BLE001
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
        except Exception as exc:  # noqa: BLE001
            #fall back to the values from the ablation report / task constants
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
            + " — omics is strongest alone and remains dominant in fusion.")

    st.divider()
    st.caption("Late/stacking fusion: RandomForest base model per modality -> "
               "RidgeCV meta-model over the base scores. Numbers surfaced from "
               "reports/modality_ablation.csv. Synthetic demonstration only.")


def main():
    st.set_page_config(page_title="HURDLE multimodal diabetes-risk demo",
                       layout="wide")
    st.title("HURDLE — multimodal diabetes-risk fusion (demo)")

    #prominent honesty banner: this must never read as a validated clinical tool
    st.warning(
        "**Portfolio demonstration — NOT a validated clinical tool.** "
        "This app runs the HURDLE late-fusion pipeline on a **virtual cohort**: "
        "synthetic patients carrying all four modalities at once. The four real "
        "public datasets (omics, CGM, wearable, retinal imaging) are **different "
        "people** — no real patient has all four. The synthetic patients are "
        "coupled through a hidden latent risk factor, with the imaging modality's "
        "coupling calibrated from the real RetinaMNIST CNN embeddings. Nothing "
        "here diagnoses, screens, or advises any individual. Do not use it for "
        "medical decisions.")

    try:
        drop_w, standalone, full_r2, ablation_fallback = load_ablation()
        model = build_predictor()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not initialize the fusion demo: {exc}")
        return

    if ablation_fallback:
        st.info("Ablation report reports/modality_ablation.csv not found — "
                "surfacing mirrored fallback constants for the ablation numbers.")

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
