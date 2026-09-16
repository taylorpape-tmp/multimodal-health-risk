"""Nested-CV rigor checks and trivial baselines for the omics -> SSPG model.

Shows the reported XGBoost SSPG score isn't inflated by hyperparameter-selection leakage
(non-nested leaky vs nested honest pooled-LOO R2, plus the optimism gap; the nested number
is the reportable one) and that the model beats naive predictors (training-mean,
shuffled-feature, and random-noise baselines on the same LOO split). Real omics data only:
if the interim csv is missing the run is skipped. Writes reports/rigor_checks.csv and
reports/rigor_nested_vs_baselines.png. GridSearchCV runs n_jobs=1 with a small inner grid
so nested LOO finishes quickly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBRegressor

from hurdle.features.omics import build_feature_matrix
from hurdle.stats import (
    dummy_mean_r2,
    nested_cv_r2,
    nonnested_cv_r2,
    random_features_r2,
    shuffled_features_r2,
)

try:
    from figure_style import apply_figure_style
except Exception:
    def apply_figure_style(*a, **k):
        return None

INTERIM = Path("data/interim")
REPORTS = Path("reports")

#small inner grid so nested LOO is tractable; n_jobs=1 set on GridSearchCV in rigor.py
PARAM_GRID = {
    "model__n_estimators": [100, 300],
    "model__max_depth": [2, 3],
}


def _make_estimator():
    #bare pipeline: trees are scale-invariant so no transform; grid keys use 'model__'
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("model", XGBRegressor(random_state=0, objective="reg:squarederror",
                               learning_rate=0.1, subsample=0.8,
                               colsample_bytree=0.8, tree_method="hist")),
    ])


def run(rows):
    f = INTERIM / "omics_S8_sspg_clean.csv"
    if not f.exists():
        print(f"SKIP rigor checks: {f} missing -- no real omics data, nothing run")
        return None
    X, y, cols = build_feature_matrix(INTERIM, target="SSPG")
    n = len(y)
    print(f"\nomics -> SSPG (XGBoost): n={n}, {len(cols)} features")
    print(f"inner grid: {PARAM_GRID}  (GridSearchCV n_jobs=1)")

    #1. nested vs non-nested
    print("\n[1] nested vs non-nested CV (pooled outer LOO)")
    nn_r2, _, best = nonnested_cv_r2(_make_estimator, PARAM_GRID, X, y)
    ne_r2, _ = nested_cv_r2(_make_estimator, PARAM_GRID, X, y)
    gap = nn_r2 - ne_r2
    print(f"  non-nested (leaky, tuned on ALL data) R2 = {nn_r2:+.4f}  best={best}")
    print(f"  nested     (honest, tuned per fold)    R2 = {ne_r2:+.4f}")
    print(f"  optimism gap (non-nested - nested)     = {gap:+.4f}")
    print("  -> the reportable headline number is the NESTED R2; the gap is the "
          "leakage nesting removes")
    rows.append(dict(check="non_nested_cv", model="XGBoost", metric="R2",
                     value=nn_r2, n=n))
    rows.append(dict(check="nested_cv", model="XGBoost", metric="R2",
                     value=ne_r2, n=n))
    rows.append(dict(check="optimism_gap", model="XGBoost", metric="R2_diff",
                     value=gap, n=n))

    #2. trivial baselines on the same LOO split
    print("\n[2] trivial baselines (same outer LOO split)")
    b_mean = dummy_mean_r2(X, y)
    b_shuf = shuffled_features_r2(_make_estimator, PARAM_GRID, X, y)
    b_rand = random_features_r2(_make_estimator, PARAM_GRID, X, y)
    print(f"  predict-the-training-mean R2 = {b_mean:+.4f}")
    print(f"  shuffled-features         R2 = {b_shuf:+.4f}")
    print(f"  random-noise-features     R2 = {b_rand:+.4f}")
    rows.append(dict(check="baseline_predict_mean", model="DummyRegressor",
                     metric="R2", value=b_mean, n=n))
    rows.append(dict(check="baseline_shuffled_features", model="XGBoost",
                     metric="R2", value=b_shuf, n=n))
    rows.append(dict(check="baseline_random_features", model="XGBoost",
                     metric="R2", value=b_rand, n=n))

    beats = ne_r2 > max(b_mean, b_shuf, b_rand)
    print(f"\n  real (nested) model beats ALL three baselines: {beats}")

    _plot(nn_r2, ne_r2, b_mean, b_shuf, b_rand)
    return dict(nested=ne_r2, non_nested=nn_r2, gap=gap, mean=b_mean,
                shuffled=b_shuf, random=b_rand, beats=beats, n=n)


def _plot(nn, ne, b_mean, b_shuf, b_rand):
    apply_figure_style()
    labels = ["non-nested\n(leaky)", "nested\n(honest)", "predict\nmean",
              "shuffled\nfeatures", "random\nfeatures"]
    vals = [nn, ne, b_mean, b_shuf, b_rand]
    #focal (nested, honest) saturated; leaky sibling and the three baselines muted
    colors = ["#9ecae1", "#08519c", "0.6", "0.6", "0.6"]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.bar(labels, vals, color=colors, edgecolor="0.2", linewidth=0.6)
    ax.axhline(0.0, color="0.2", lw=0.8)
    ax.set_ylabel("pooled leave-one-out $R^2$")
    ax.set_title("omics -> SSPG: nested-CV is not inflated and beats trivial baselines")
    for x, v in enumerate(vals):
        off = 0.012 if v >= 0 else -0.03
        ax.annotate(f"{v:+.3f}", (x, v + off), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=7)
    ax.margins(y=0.15)
    fig.tight_layout()
    out = REPORTS / "rigor_nested_vs_baselines.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    REPORTS.mkdir(exist_ok=True)
    rows = []
    run(rows)
    if rows:
        df = pd.DataFrame(rows)
        out = REPORTS / "rigor_checks.csv"
        df.to_csv(out, index=False)
        print(f"\nwrote {out}  ({len(df)} rows)")
        print(df.to_string(index=False))
    else:
        print("\nno rows produced; rigor checks skipped")


if __name__ == "__main__":
    main()
