"""Run the four-family nested consensus selector on the REAL omics matrices.

This produces an actual consensus-selected analyte panel from the Stanford
S8 SSPG regression table (not synthetic data). For every leave-one-out outer
fold the four selector families (filter/wrapper/embedded/explain) re-vote from
scratch, a feature is frozen when >= vote_threshold families back it, and we
report how often each analyte entered the frozen set across all folds.

S9 IRIS is a binary classification target. run_nested_consensus is a
regression-only method (Pearson p-values, OLS leave-one-out R^2, LASSO/
ElasticNet regression, standardised-OLS-coef explain, R^2/MAE/RMSE final
scoring); it has no classification path, so IRIS is skipped honestly rather
than forced through a regressor.

SHAP is not installed and the task forbids adding heavy dependencies, so the
explainability readout for the final panel is the module's own built-in
standardised-OLS-coefficient explainer (the same mechanism as explain_vote),
fit once on the full real data over the consensus panel.

Run from the repo root:
    python scripts/run_consensus_panel.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#run from the repo root without an editable install: put src on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hurdle.feature_selection.nested_consensus import (  # noqa: E402
    ConsensusConfig,
    run_nested_consensus,
)
from hurdle.features.omics import as_model_frame, build_feature_matrix  # noqa: E402

INTERIM = "data/interim"
REPORTS = Path("reports")
PANEL_CSV = REPORTS / "consensus_panel_sspg.csv"


def standardized_ols_explainer(frame, panel, target_col="target"):
    """Built-in explainer: standardised OLS coefficients for the final panel.

    This is the module's explain-family mechanism (StandardScaler + OLS via the
    pseudo-inverse) applied once to the full real data on the consensus panel,
    used in place of SHAP. Returns a frame of (analyte, std_ols_coef, abs_coef)
    sorted by magnitude, so a larger |coef| means a stronger standardised linear
    association with SSPG.
    """
    X = StandardScaler().fit_transform(frame[panel].values.astype(float))
    y = frame[target_col].values.astype(float)
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return (pd.DataFrame({"analyte": panel, "std_ols_coef": beta})
            .assign(abs_coef=lambda d: d["std_ols_coef"].abs().round(4),
                    std_ols_coef=lambda d: d["std_ols_coef"].round(4))
            .sort_values("abs_coef", ascending=False)
            .reset_index(drop=True))


def run_sspg_panel():
    """Build the real SSPG matrix, run nested consensus, save + print the panel."""
    X, y, feature_cols = build_feature_matrix(INTERIM, target="SSPG", add_ratios=True)
    frame = as_model_frame(X, y, target_col="target")
    cfg = ConsensusConfig()
    print(f"REAL SSPG matrix: {len(X)} patients x {len(feature_cols)} analytes "
          f"(leave-one-out, vote_threshold={cfg.vote_threshold})")

    summary, freq = run_nested_consensus(frame, feature_cols, "target", cfg)

    #panel CSV: analyte + how many outer folds it entered the frozen set + frequency
    panel_table = (freq.rename(columns={"feature": "analyte",
                                        "in_frozen_set": "vote_count",
                                        "frequency": "selection_frequency"})
                   [["analyte", "vote_count", "selection_frequency"]])
    REPORTS.mkdir(parents=True, exist_ok=True)
    panel_table.to_csv(PANEL_CSV, index=False)

    #the consensus panel = analytes frozen in EVERY outer fold (frequency == 1.0),
    #i.e. those that survived the vote_threshold with no dependence on any held-out
    #patient. Report this stable panel, then the full ranked selection frequencies.
    n = len(y)
    stable = panel_table[panel_table["vote_count"] == n]["analyte"].tolist()

    print("\n=== external nested-LOO performance (final models) ===")
    print(summary.to_string(index=False))

    print(f"\n=== consensus panel: analytes frozen in all {n} folds "
          f"(selection_frequency == 1.0) ===")
    if stable:
        for a in stable:
            print(f"  {a}")
    else:
        print("  (none reached every fold; see top selection frequencies below)")

    print("\n=== top analytes by selection frequency across folds ===")
    print(panel_table.head(15).to_string(index=False))

    #built-in standardised-OLS-coef explainer on the final panel (SHAP not used)
    panel = stable or panel_table.head(cfg.explain_topk)["analyte"].tolist()
    expl = standardized_ols_explainer(frame, panel, "target")
    print("\n=== standardised-OLS-coef explainer for the consensus panel "
          "(SHAP skipped: not installed) ===")
    print(expl.to_string(index=False))

    return panel_table, stable


def main():
    run_sspg_panel()
    #S9 IRIS is classification; run_nested_consensus is regression-only, so it is
    #intentionally not run here (skipped honestly, not forced through a regressor).
    print("\nS9 IRIS classification SKIPPED: run_nested_consensus is regression-only "
          "(no classification vote/score path); forcing it would fabricate a result.")


if __name__ == "__main__":
    main()
