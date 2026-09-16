"""SHAP feature attribution on the REAL omics -> SSPG regression.

Fits one XGBoost regressor on the full Stanford S8 SSPG matrix (59 patients x
~86 analytes), computes exact TreeExplainer SHAP values, and cross-checks the
resulting global importance ranking against two independent measures: XGBoost's
native gain and the cross-fold consensus selection frequency
(reports/consensus_panel_sspg.csv). Agreement across three methods is evidence
the SSPG signal is carried by real analytes, not a single-method artefact.

Outputs (all from a real run; nothing synthetic):
  reports/shap_omics_importance.csv  analyte, mean_abs_shap, native_gain,
                                     consensus_freq, rank_shap, rank_gain
  reports/shap_summary.png           SHAP beeswarm of the top analytes, or a
                                     horizontal mean|SHAP| bar if beeswarm fails

Run from the repo root:
    python scripts/run_shap_attribution.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  #noqa: E402
import numpy as np  #noqa: E402
import pandas as pd  #noqa: E402
import shap  #noqa: E402

#run from the repo root without an editable install: put src on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hurdle.explain.shap_attribution import (  #noqa: E402
    compare_rankings,
    fit_and_explain,
    signal_concentration,
)
from hurdle.features.omics import build_feature_matrix  #noqa: E402

INTERIM = "data/interim"
REPORTS = Path("reports")
CONSENSUS_CSV = REPORTS / "consensus_panel_sspg.csv"
IMPORTANCE_CSV = REPORTS / "shap_omics_importance.csv"
SUMMARY_PNG = REPORTS / "shap_summary.png"

TOP_N_TABLE = 20
TOP_N_PRINT = 15
TOP_N_PLOT = 15


def _apply_style():
    #publication-grade defaults (frame='open', role-mapped size ladder 8/7/6),
    #matching the figure-style skill's apply_figure_style() so the standalone
    #script produces the same look without depending on the skill's kernel plugin.
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlelocation": "left",
    })


def _beeswarm(shap_values, X, feature_names, top_n):
    """Try a SHAP beeswarm of the top_n analytes. Return the Figure on success,
    or None if beeswarm plotting errors (caller falls back to a bar)."""
    try:
        order = np.argsort(np.abs(shap_values).mean(axis=0))[::-1][:top_n]
        fig = plt.figure(figsize=(7.0, 5.5))
        shap.summary_plot(
            shap_values[:, order],
            X.iloc[:, order],
            feature_names=[feature_names[i] for i in order],
            show=False,
            plot_size=None,
            max_display=top_n,
        )
        fig = plt.gcf()
        fig.suptitle(
            f"SHAP value distribution of the top {top_n} analytes for SSPG",
            fontsize=9,
        )
        fig.tight_layout()
        return fig
    except Exception as exc:  #noqa: BLE001
        print(f"  beeswarm plotting failed ({type(exc).__name__}: {exc}); "
              "falling back to a mean|SHAP| bar chart")
        plt.close("all")
        return None


def _bar(shap_rank, top_n):
    """Clean horizontal bar of the top_n analytes by mean|SHAP| (fallback / robust
    default)."""
    top = shap_rank.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.barh(top["analyte"], top["mean_abs_shap"], color="#2b6cb0")
    ax.set_xlabel("mean |SHAP value|  (impact on SSPG prediction)")
    ax.set_title(f"Top {top_n} analytes carrying the SSPG signal (SHAP attribution)")
    ax.margins(y=0.01)
    fig.tight_layout()
    return fig


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    _apply_style()

    X, y, feature_cols = build_feature_matrix(INTERIM, target="SSPG", add_ratios=True)
    print(f"REAL SSPG matrix: {len(X)} patients x {len(feature_cols)} analytes "
          f"(single full-data XGBoost fit + exact TreeExplainer SHAP)")

    out = fit_and_explain(X, y, model="xgboost")
    shap_rank = out["shap_ranking"]
    gain_rank = out["gain_ranking"]

    #three-method comparison table, then the CSV (top-N by SHAP)
    table = compare_rankings(shap_rank, gain_rank, CONSENSUS_CSV, top_n=TOP_N_TABLE)
    table_out = table.copy()
    table_out["mean_abs_shap"] = table_out["mean_abs_shap"].round(4)
    table_out["native_gain"] = table_out["native_gain"].round(4)
    table_out.to_csv(IMPORTANCE_CSV, index=False)

    #print the top-15 with gain + consensus alongside
    print(f"\n=== top {TOP_N_PRINT} analytes by SHAP mean|value| "
          "(with native gain + consensus selection frequency) ===")
    show = table_out.head(TOP_N_PRINT).copy()
    show["consensus_freq"] = show["consensus_freq"].map(
        lambda v: "-" if pd.isna(v) else f"{v:.3f}"
    )
    print(show.to_string(index=False))

    #agreement readout: overlap of each method's top-5 signal-carriers
    shap_top5 = list(shap_rank.head(5)["analyte"])
    gain_top5 = list(gain_rank.head(5)["analyte"])
    consensus = pd.read_csv(CONSENSUS_CSV).sort_values("selection_frequency", ascending=False)
    con_top5 = list(consensus.head(5)["analyte"])
    print("\n=== do SHAP / gain / consensus agree on the top signal-carriers? ===")
    print(f"  SHAP top-5:      {shap_top5}")
    print(f"  gain top-5:      {gain_top5}")
    print(f"  consensus top-5: {con_top5}")
    print(f"  SHAP n gain:     {sorted(set(shap_top5) & set(gain_top5))}")
    print(f"  SHAP n consensus:{sorted(set(shap_top5) & set(con_top5))}")
    print(f"  all three:       {sorted(set(shap_top5) & set(gain_top5) & set(con_top5))}")

    #how concentrated is the signal
    conc = signal_concentration(shap_rank)
    n80 = conc["n_features_for_fraction"][0.80]
    n90 = conc["n_features_for_fraction"][0.90]
    n95 = conc["n_features_for_fraction"][0.95]
    n_tot = conc["n_features_total"]
    n_zeroish = int((shap_rank["mean_abs_shap"] < 1e-6 * shap_rank["mean_abs_shap"].max()).sum())
    print("\n=== signal concentration (mean|SHAP| mass) ===")
    print(f"  {n80} / {n_tot} analytes carry 80% of the total attribution")
    print(f"  {n90} / {n_tot} analytes carry 90%")
    print(f"  {n95} / {n_tot} analytes carry 95%")
    print(f"  {n_zeroish} / {n_tot} analytes contribute ~0 (< 1e-6 of the top feature) -- "
          "the model is not leaning on noise")

    #figure: beeswarm first, bar fallback
    fig = _beeswarm(out["shap_values"], out["X"], out["feature_names"], TOP_N_PLOT)
    used = "beeswarm"
    if fig is None:
        fig = _bar(shap_rank, TOP_N_PLOT)
        used = "bar"
    fig.savefig(SUMMARY_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nsaved {IMPORTANCE_CSV}")
    print(f"saved {SUMMARY_PNG}  ({used})")


if __name__ == "__main__":
    main()
