"""Omics feature builder for the two cleaned interim tables.

Both interim files are already patient-rows (transposed), with SubjectID as the
first column and the label in its own column:
  omics_S9_isir_clean.csv  -> 152 analytes + IRIS   (binary classification)
  omics_S8_sspg_clean.csv  -> 85 analytes  + SSPG   (continuous regression)

Because each row is a distinct patient (no repeated SubjectIDs), no cross-file
join is needed and CV is leave-one-out (no subject groups). The S9 file carries
one patient with a missing IRIS label; build_feature_matrix drops that patient
for the classification target rather than inventing a label.

Outputs are shaped for the MLModel base and run_nested_consensus:
  build_feature_matrix() -> (X_df, y, feature_cols)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ID_COL = "SubjectID"
IRIS_LABEL = "IRIS"
SSPG_LABEL = "SSPG"

#target -> (filename, label column, task)
_TARGETS = {
    "IRIS": ("omics_S9_isir_clean.csv", IRIS_LABEL, "classification"),
    "SSPG": ("omics_S8_sspg_clean.csv", SSPG_LABEL, "regression"),
}


def _read(interim_dir, filename):
    path = Path(interim_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"expected interim file not found: {path}")
    df = pd.read_csv(path)
    if ID_COL not in df.columns:
        raise ValueError(f"{filename} missing {ID_COL} column")
    return df.set_index(ID_COL)


def load_omics(interim_dir):
    """Load both cleaned frames with SubjectID as the index and label preserved.

    Returns (s9, s8): s9 is the IRIS classification frame, s8 the SSPG
    regression frame. Feature columns + label are kept as-is; no rows dropped.
    """
    s9 = _read(interim_dir, _TARGETS["IRIS"][0])
    s8 = _read(interim_dir, _TARGETS["SSPG"][0])
    return s9, s8


def _add_ratio_features(X):
    #clinically-motivated ratios, added only when both inputs are present and the
    #denominator is nonzero. TG/HDL is an established insulin-resistance proxy.
    if "TGL" in X.columns and "HDL" in X.columns and (X["HDL"] != 0).all():
        X["TG_HDL_ratio"] = X["TGL"] / X["HDL"]
    return X


def build_feature_matrix(interim_dir, target="SSPG", add_ratios=True):
    """Return (X_df, y, feature_cols) ready for MLModel / run_nested_consensus.

    target: 'IRIS' (binary classification, S9) or 'SSPG' (regression, S8).
    add_ratios: if True, derive TG_HDL_ratio when TGL and HDL both exist.

    Rows with a missing label are dropped (never imputed). X and y share the
    SubjectID index; feature_cols is exactly list(X_df.columns).
    """
    if target not in _TARGETS:
        raise ValueError(f"unknown target {target!r}; choose from {list(_TARGETS)}")
    filename, label, task = _TARGETS[target]
    df = _read(interim_dir, filename)

    #drop patients without a label rather than fabricate one
    df = df[df[label].notna()]

    y = df[label].copy()
    if task == "classification":
        y = y.astype(int)
    else:
        y = y.astype(float)

    X = df.drop(columns=[label]).astype(float)
    if add_ratios:
        X = _add_ratio_features(X)

    feature_cols = list(X.columns)
    return X, y, feature_cols


def as_model_frame(X, y, target_col="target"):
    """Fold X and y back into one frame with a named target column, matching the
    (frame, feature_cols, target_col) contract of MLModel and run_nested_consensus.
    """
    frame = X.copy()
    frame[target_col] = np.asarray(y)
    return frame
