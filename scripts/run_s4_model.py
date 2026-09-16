"""Model the wide S4 omics block (12k analytes) against real SSPG labels, HONESTLY.

Attaches the S8 SSPG target to the S4 patients (join S4 Zcode -> S8 SubjectID),
then runs the leak-safe cross-validated reduction+model (omics_highdim_cv) with
Ridge and XGBoost. For each it also runs the LEAKY all-rows reduction so the
optimism gap is reported honestly side by side.

Real data only: if an input file is missing, the run is SKIPPED and noted; no
numbers are fabricated. Run from the repo root:
    python scripts/run_s4_model.py

Outputs:
    reports/s4_highdim_results.csv   one row per (model, mode) with real metrics
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from hurdle.features import omics_highdim as hd  # noqa: E402
from hurdle.features.omics_highdim_cv import optimism_gap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
S4_XLSX = ROOT / "data/raw/clinical/time_molecule_comparison.xlsx"
CROSSWALK_CSV = ROOT / "data/interim/cgm_omics_shared_patients.csv"
SSPG_CSV = ROOT / "data/interim/omics_S8_sspg_clean.csv"
OUT_CSV = ROOT / "reports/s4_highdim_results.csv"

N_PCA = 20
MODELS = ("ridge", "xgboost")


def load_s4_with_sspg():
    """Return (X, y, n_labeled, n_crosswalk) — oriented S4 rows with an SSPG label.

    Join path (as named in the S4 task): the crosswalk
    (cgm_omics_shared_patients.csv) is the site-code<->Zcode spine that ties S4
    patients to a target. The crosswalk carries an SSPG column for 19 of the S4
    patients; the S8 panel (omics_S8_sspg_clean.csv) is the authoritative SSPG
    source keyed on the same Zcode (SubjectID). We take SSPG from the crosswalk
    where present and fill the remaining S4 patients from S8 (the two agree
    exactly on the 19 they share), then keep every S4 patient that ends up with
    a non-null SSPG. This honors the crosswalk as the join spine while not
    discarding the S8-only labels the crosswalk simply never listed.

    n_crosswalk is the crosswalk-only coverage (SSPG present in the crosswalk);
    n_labeled is the crosswalk-spine + S8-fill coverage actually modeled.
    """
    s4 = pd.read_excel(S4_XLSX, sheet_name="S4_HealthyIQR")
    X, _sites, _z = hd.orient(s4)                          #patients x analytes, indexed by Zcode

    xw = pd.read_csv(CROSSWALK_CSV).dropna(subset=["SSPG"])
    xw_sspg = xw.set_index(xw["Zcode"].astype(str))["SSPG"].astype(float)

    s8 = pd.read_csv(SSPG_CSV).dropna(subset=["SSPG"])
    s8_sspg = s8.set_index(s8["SubjectID"].astype(str))["SSPG"].astype(float)

    #crosswalk SSPG first, S8 fills the rest (combine_first keeps crosswalk on overlap)
    label = xw_sspg.combine_first(s8_sspg)

    n_crosswalk = len([zc for zc in X.index if zc in xw_sspg.index])
    keep = [zc for zc in X.index if zc in label.index]
    X = X.loc[keep]
    y = label.loc[keep].astype(float).values
    return X, y, len(keep), n_crosswalk


def main():
    for path in (S4_XLSX, CROSSWALK_CSV, SSPG_CSV):
        if not path.exists():
            print(f"SKIP: missing {path.relative_to(ROOT)}")
            return

    X, y, n_labeled, n_crosswalk = load_s4_with_sspg()
    print(f"S4 patients with an SSPG label: {n_labeled} "
          f"(crosswalk-only coverage: {n_crosswalk}; remainder filled from S8)")
    print(f"S4 analytes (pre-reduction): {X.shape[1]}")
    print(f"SSPG target: mean={y.mean():.1f}  range={y.min():.0f}-{y.max():.0f}\n")

    rows = []
    for model in MODELS:
        gap = optimism_gap(X, y, n_pca=N_PCA, model=model)
        honest, leaky = gap["honest"], gap["leaky"]
        pcs = honest["fold_trace"][0]["n_pca_components"]
        print(f"=== {model.upper()}  (leave-one-out, {pcs} PCs/fold) ===")
        print(f"  HONEST (per-fold reduction):  R2={honest['R2']:+.4f}  "
              f"Pearson={honest['Pearson']:+.4f}  RMSE={honest['RMSE']:.2f}")
        print(f"  LEAKY  (all-rows reduction):  R2={leaky['R2']:+.4f}  "
              f"Pearson={leaky['Pearson']:+.4f}  RMSE={leaky['RMSE']:.2f}")
        print(f"  optimism gap:  dR2={gap['gap_R2']:+.4f}  "
              f"dPearson={gap['gap_Pearson']:+.4f}\n")
        for mode, res in (("honest", honest), ("leaky", leaky)):
            rows.append({
                "model": model, "mode": mode, "n_patients": res["n"],
                "n_analytes": X.shape[1], "n_pca": res["n_pca"],
                "R2": res["R2"], "Pearson": res["Pearson"],
                "Spearman": res["Spearman"], "MAE": res["MAE"], "RMSE": res["RMSE"],
            })
        rows.append({
            "model": model, "mode": "optimism_gap", "n_patients": honest["n"],
            "n_analytes": X.shape[1], "n_pca": N_PCA,
            "R2": gap["gap_R2"], "Pearson": gap["gap_Pearson"],
            "Spearman": float("nan"), "MAE": float("nan"), "RMSE": float("nan"),
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV.relative_to(ROOT)}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
