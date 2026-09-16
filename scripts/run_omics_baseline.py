"""Smoke integration for the omics modality: build the SSPG matrix, run XGBoost and Ridge
through the MLModel wrappers under leave-one-out CV, print metrics.

Deliberately quick: no hyperparameter grids (param_grid=None), so each model is a single LOO
pass over ~59 patients.

    python scripts/run_omics_baseline.py
"""
import sys
from pathlib import Path

#run from the repo root without an editable install: put src on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hurdle.features.omics import as_model_frame, build_feature_matrix  #noqa: E402
from hurdle.ml.linear_models import RidgeModel  #noqa: E402
from hurdle.ml.tree_models import XGBoostModel  #noqa: E402

INTERIM = "data/interim"


def main():
    #SSPG is the continuous regression target the tree/linear wrappers expect
    X, y, feature_cols = build_feature_matrix(INTERIM, target="SSPG", add_ratios=True)
    frame = as_model_frame(X, y, target_col="target")
    print(f"omics SSPG matrix: {len(X)} patients x {len(feature_cols)} features")

    #param_grid=None -> plain LOO fits, no nested tuning, so this stays a smoke test
    for cls in (XGBoostModel, RidgeModel):
        model = cls(frame, feature_cols=feature_cols, target_col="target",
                    param_grid=None)
        model.run(verbose=True)


if __name__ == "__main__":
    main()
