"""real statistics + evaluation report for the omics models.

runs the leave-one-out cross-validation via the real MLModel, then attaches
uncertainty (95% bootstrap CIs) and significance (label-permutation p-values
that rerun the full CV on permuted labels, respecting the CV structure).

outputs:
  reports/stats_report.csv          one row per model/target/metric
  reports/calibration_omics_iris.png reliability diagram for the IRIS classifier

real data only: if an interim csv is missing, that section is SKIPPED and noted.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hurdle.features.omics import build_feature_matrix
from hurdle.ml.linear_models import RidgeModel
from hurdle.ml.tree_models import XGBoostModel
from hurdle.stats import (
    brier_score,
    evaluate_model,
    expected_calibration_error,
    reliability_diagram,
)

try:
    from figure_style import apply_figure_style
except Exception:
    def apply_figure_style(*a, **k):
        return None

INTERIM = Path("data/interim")
REPORTS = Path("reports")
N_BOOT = 2000
N_PERM = 200


class LightXGBoostModel(XGBoostModel):
    #fixed shallow config, NO inner GridSearchCV tuning (param_grid=None -> one
    #fit per train fold). the production XGBoostModel tunes an 18-combo grid with
    #5-fold inner CV, which is >10 min per LOO pass and so infeasible for the
    #N_PERM full-LOO permutation reruns. shallow trees + a small ensemble is
    #appropriate regularization for the wide small-n omics matrix (n=59, ~85
    #features); still fully leak-safe (fit only on each train fold). the report
    #labels this row 'XGBoost (fixed n=50,d=2)' and tuned=False so the artifact
    #self-discloses it is NOT the GridSearchCV-tuned production model.
    name = "XGBoost (fixed n=50,d=2)"
    param_grid = None

    def _estimator(self):
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=0, objective="reg:squarederror",
                            n_estimators=50, max_depth=2, learning_rate=0.1,
                            tree_method="hist")


def _fmt(d):
    return f"{d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]  p={d['p']:.3g}"


def run_regression(rows):
    f = INTERIM / "omics_S8_sspg_clean.csv"
    if not f.exists():
        print(f"SKIP regression: {f} missing")
        return None
    X, y, cols = build_feature_matrix(INTERIM, target="SSPG")
    print(f"\nS8 SSPG regression: n={len(y)}, {len(cols)} features")
    #Ridge runs the production contract (RidgeCV self-tunes alpha per fold).
    #XGBoost uses a fixed shallow config (LightXGBoostModel) instead of the
    #GridSearchCV-tuned production XGBoostModel: the tuned grid is >10 min per LOO
    #pass, so N_PERM full-LOO permutation reruns of it are infeasible. the tuned
    #column records this so the CSV self-discloses which rows are tuned.
    specs = [("XGBoost (fixed n=50,d=2)", LightXGBoostModel, None, False),
             ("Ridge", RidgeModel, "default", True)]
    for label, cls, grid, tuned in specs:
        res = evaluate_model(cls, X, y, task="regression", param_grid=grid,
                             n_boot=N_BOOT, n_perm=N_PERM, seed=0)
        print(f"  {label} (tuned={tuned}):")
        for m in ("R2", "RMSE", "Pearson"):
            print(f"    {m:8s} {_fmt(res[m])}")
            d = res[m]
            rows.append(dict(target="SSPG", task="regression", model=label,
                             tuned=tuned, metric=m, point=d["point"],
                             ci_lo=d["lo"], ci_hi=d["hi"], p_value=d["p"],
                             n=res["n"]))
    return True


def run_classification(rows):
    f = INTERIM / "omics_S9_isir_clean.csv"
    if not f.exists():
        print(f"SKIP classification: {f} missing")
        return None
    X, y, cols = build_feature_matrix(INTERIM, target="IRIS")
    print(f"\nS9 IRIS classification: n={len(y)}, {len(cols)} features")
    res = evaluate_model(RidgeModel, X, y, task="classification",
                         n_boot=N_BOOT, n_perm=N_PERM, seed=0)
    print("  Ridge (classification):")
    for m in ("AUROC", "F1"):
        print(f"    {m:6s} {_fmt(res[m])}")
        d = res[m]
        rows.append(dict(target="IRIS", task="classification", model="Ridge",
                         tuned=True, metric=m, point=d["point"], ci_lo=d["lo"],
                         ci_hi=d["hi"], p_value=d["p"], n=res["n"]))

    #calibration on the LOO out-of-fold scores; clip to [0,1] for prob semantics
    scores = np.clip(res["preds"], 0.0, 1.0)
    yv = np.asarray(y)
    brier = brier_score(yv, scores)
    ece = expected_calibration_error(yv, scores, n_bins=10)
    print(f"    Brier={brier:.4f}  ECE={ece:.4f}")
    _plot_reliability(yv, scores, brier, ece)
    return dict(brier=brier, ece=ece)


def _plot_reliability(y, scores, brier, ece):
    apply_figure_style()
    pm, obs, counts = reliability_diagram(y, scores, n_bins=10)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=1, label="perfect calibration")
    ax.plot(pm, obs, "o-", color="#1f77b4", label="IRIS classifier (LOO)")
    for x, yv, c in zip(pm, obs, counts):
        ax.annotate(str(c), (x, yv), textcoords="offset points", xytext=(4, 4),
                    fontsize=6, color="0.4")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency of IRIS+")
    ax.set_title(f"Reliability: omics IRIS\nBrier={brier:.3f}  ECE={ece:.3f}")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    fig.tight_layout()
    out = REPORTS / "calibration_omics_iris.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"    wrote {out}")


def main():
    REPORTS.mkdir(exist_ok=True)
    rows = []
    run_regression(rows)
    cal = run_classification(rows)
    if rows:
        df = pd.DataFrame(rows)
        out = REPORTS / "stats_report.csv"
        df.to_csv(out, index=False)
        print(f"\nwrote {out}  ({len(df)} rows)")
        print(df.to_string(index=False))
    else:
        print("\nno rows produced; all sections skipped")
    if cal:
        print(f"\nomics IRIS calibration: Brier={cal['brier']:.4f}  ECE={cal['ece']:.4f}")


if __name__ == "__main__":
    main()
