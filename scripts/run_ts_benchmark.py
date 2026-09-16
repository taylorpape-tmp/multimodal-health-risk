"""Honest benchmark: hand-crafted time-series features vs foundation-model
embeddings vs both, under the same leave-one-out Ridge harness.

Two modalities, two DIFFERENT kinds of target, labelled honestly:

  CGM  (real clinical target).  57 Hall-2018 CGM subjects; 19 of them link to
       the omics cohort by site code and carry a measured SSPG (steady-state
       plasma glucose, the gold-standard insulin-resistance readout). We
       regress SSPG on: (a) the hand-crafted CGM variability features
       (MAGE/CONGA/MODD/...), (b) the MOMENT embedding of the raw glucose
       trace, (c) both concatenated. This is a genuine diabetes-risk signal.

  WEARABLE (representation-quality proxy -- NOT a diabetes result).  The 43
       Basis-watch subjects (SubjectN) are DISJOINT from the omics Zcodes, so
       no metabolic label exists for them. Inventing one would be dishonest.
       Instead we ask a self-supervised question: does the foundation-model
       embedding of the raw HR series CAPTURE known circadian structure? We
       hold out the hand-crafted cosinor HR amplitude as the target and see
       how well each feature set predicts it under LOO. High skill means the
       embedding encodes the circadian signal the hand-crafted feature
       measures -- a representation-quality check, explicitly not a risk score.

Both sections use RidgeModel under leave-one-out CV (param_grid=None, so
RidgeCV self-tunes alpha inside each fold). Metrics: R2, MAE, RMSE, Pearson,
Spearman. Outputs reports/ts_benchmark.csv and reports/ts_benchmark.png.

Run from the repo root (after building the embedding parquets):
    python scripts/run_ts_benchmark.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

#run from the repo root without an editable install: put src on the path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hurdle.features.cgm import build_cgm_matrix  #noqa: E402
from hurdle.features.wearable import build_wearable_matrix  #noqa: E402
from hurdle.ml.linear_models import RidgeModel  #noqa: E402

INTERIM = ROOT / "data" / "interim"
CGM_RAW = ROOT / "data" / "raw" / "cgm" / "hall2018_cgm_S1_Data"
WEAR_DIR = INTERIM / "wearable"
WEAR_EMB = INTERIM / "ts_embeddings_wearable.parquet"
CGM_EMB = INTERIM / "ts_embeddings_cgm.parquet"
SHARED = INTERIM / "cgm_omics_shared_patients.csv"
REPORTS = ROOT / "reports"
OUT_CSV = REPORTS / "ts_benchmark.csv"
OUT_PNG = REPORTS / "ts_benchmark.png"

#the hand-crafted wearable feature held out as the representation-probe target;
#cosinor HR amplitude is a clean circadian quantity the raw HR series encodes
WEAR_PROXY_TARGET = "hr_amplitude"


def _loo_ridge(feature_frame, target, feature_cols):
    #run one leave-one-out RidgeCV pass and return the metric dict. the frame
    #carries the features plus a 'target' column; RidgeModel does LOO by default.
    frame = feature_frame[feature_cols].copy()
    frame["target"] = np.asarray(target, dtype=float)
    model = RidgeModel(frame, feature_cols=feature_cols, target_col="target",
                       param_grid=None)
    res = model.run(verbose=False)
    return {k: res[k] for k in ("R2", "MAE", "RMSE", "Pearson", "Spearman")}


def _bench_three(hand_df, emb_df, target, tag, target_name, n):
    #run the three feature sets (hand-crafted / embeddings / both) on a shared,
    #aligned index and return a list of result rows for the csv
    hand_cols = list(hand_df.columns)
    emb_cols = list(emb_df.columns)
    both = pd.concat([hand_df, emb_df], axis=1)
    rows = []
    for method, frame, cols in (
        ("hand_crafted", hand_df, hand_cols),
        ("embeddings", emb_df, emb_cols),
        ("both", both, hand_cols + emb_cols),
    ):
        m = _loo_ridge(frame, target, cols)
        rows.append({
            "modality": tag, "method": method, "n": n,
            "n_features": len(cols), "target": target_name, **m,
        })
    return rows


def cgm_benchmark():
    #CGM: real SSPG clinical target on the 19 CGM subjects that link to omics
    hand = build_cgm_matrix(CGM_RAW)                       #57 x 14
    emb = pd.read_parquet(CGM_EMB)                         #57 x 512
    shared = pd.read_csv(SHARED)
    #CGM subjectId '1636-69-001' -> omics site_code '69-001'
    site = {sid: sid.replace("1636-", "") for sid in emb.index}
    sspg = shared.set_index("site_code")["SSPG"]
    target_map = {sid: sspg.get(sc) for sid, sc in site.items()}
    keep = [sid for sid, v in target_map.items()
            if v is not None and np.isfinite(v)]
    keep = sorted(keep)
    y = np.array([float(target_map[s]) for s in keep])
    hand_k = hand.loc[keep]
    emb_k = emb.loc[keep]
    #drop any hand-crafted column that is non-finite for a kept subject so the
    #linear harness sees a clean matrix (CONGA/MODD can be NaN on short traces)
    good = hand_k.columns[np.isfinite(hand_k.values).all(axis=0)]
    hand_k = hand_k[good]
    print(f"[CGM] real target = SSPG (steady-state plasma glucose); "
          f"n={len(keep)} subjects with a measured label")
    print(f"[CGM] hand-crafted features: {len(good)}  embedding dims: "
          f"{emb_k.shape[1]}")
    return _bench_three(hand_k, emb_k, y, "cgm",
                        "SSPG (insulin resistance, mg/dL)", len(keep))


def wearable_benchmark():
    #WEARABLE: representation-quality proxy -- predict the held-out cosinor HR
    #amplitude from the embedding. NOT a diabetes-risk result.
    hand = build_wearable_matrix(WEAR_DIR)                 #43 x 32
    emb = pd.read_parquet(WEAR_EMB)                        #43 x 512
    keep = sorted(set(hand.index) & set(emb.index))
    y = hand.loc[keep, WEAR_PROXY_TARGET].values.astype(float)
    #the target must NOT be among the hand-crafted predictors, or it trivially
    #predicts itself; drop it (and any degenerate all-constant column)
    hand_pred = hand.loc[keep].drop(columns=[WEAR_PROXY_TARGET])
    nunique = hand_pred.nunique()
    hand_pred = hand_pred[nunique[nunique > 1].index]
    emb_k = emb.loc[keep]
    print(f"[WEAR] representation-quality proxy target = held-out cosinor "
          f"{WEAR_PROXY_TARGET}; n={len(keep)} subjects")
    print("[WEAR] NOTE: this is a representation-quality check, NOT a "
          "diabetes-risk result (Basis subjects have no metabolic label)")
    print(f"[WEAR] hand-crafted predictors: {hand_pred.shape[1]}  "
          f"embedding dims: {emb_k.shape[1]}")
    return _bench_three(hand_pred, emb_k, y, "wearable",
                        f"held-out cosinor {WEAR_PROXY_TARGET} "
                        "(representation probe)", len(keep))


def _plot(df):
    #grouped bars: R2 by method within each modality, two panels sharing the
    #method colour thread. representation-probe panel is explicitly titled.
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    #publication-grade defaults inline (the figure-style skill's kernel helper
    #is not importable from a standalone script): a 3-size role ladder, clean
    #spines, and a semantic-zero baseline drawn below
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 120,
    })
    methods = ["hand_crafted", "embeddings", "both"]
    labels = {"hand_crafted": "hand-crafted", "embeddings": "FM embedding",
              "both": "both"}
    #one bound colour per method, threaded across both panels
    palette = {"hand_crafted": "#4C72B0", "embeddings": "#DD8452",
               "both": "#55A868"}
    panels = [
        ("cgm", "CGM \u2192 SSPG  (real insulin-resistance target)"),
        ("wearable", "Wearable HR \u2192 held-out cosinor amplitude\n"
                     "(representation-quality probe, not a risk result)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for ax, (tag, title) in zip(axes, panels):
        sub = df[df["modality"] == tag].set_index("method")
        vals = [sub.loc[m, "R2"] for m in methods]
        n = int(sub["n"].iloc[0])
        x = np.arange(len(methods))
        colors = [palette[m] for m in methods]
        ax.bar(x, vals, color=colors, width=0.62, edgecolor="black",
               linewidth=0.6)
        ax.axhline(0.0, color="0.4", linewidth=0.8)
        for xi, v in zip(x, vals):
            off = 0.012 if v >= 0 else -0.012
            va = "bottom" if v >= 0 else "top"
            ax.text(xi, v + off, f"{v:+.2f}", ha="center", va=va, fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[m] for m in methods])
        ax.set_title(f"{title}\n(n={n}, LOO Ridge)", fontsize=8)
        ax.set_ylabel("R\u00b2 (leave-one-out)")
        ax.margins(y=0.15)
    fig.suptitle("Time-series foundation-model embeddings vs hand-crafted "
                 "features", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    #geometric verify: no text box escapes the figure
    r = fig.canvas.get_renderer()
    for t in fig.findobj(mpl.text.Text):
        if t.get_text().strip() and t.get_visible():
            bb = t.get_window_extent(r)
            assert bb.x1 <= fig.bbox.x1 + 1 and bb.x0 >= -1, \
                f"text {t.get_text()!r} escapes figure"
    fig.savefig(OUT_PNG, dpi=200)
    return fig


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = cgm_benchmark() + wearable_benchmark()
    df = pd.DataFrame(rows)
    metric_cols = ["R2", "MAE", "RMSE", "Pearson", "Spearman"]
    out = df[["modality", "method", "target", "n", "n_features", *metric_cols]]
    out.to_csv(OUT_CSV, index=False)

    #print the real numbers
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    print("\n=== ts_benchmark results (real runs) ===")
    for tag in ("cgm", "wearable"):
        block = out[out["modality"] == tag]
        print(f"\n{tag.upper()}  target: {block['target'].iloc[0]}")
        print(block[["method", "n", "n_features", *metric_cols]]
              .to_string(index=False))

    _plot(df)
    print(f"\nwrote {OUT_CSV}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
