"""publication-quality showcase figures for the hurdle multimodal portfolio.

runs end to end from the repo root against the real data under data/. every
annotated numeric result in the analytic figures (wearable cosinor, CGM TIR,
S4 funnel counts and PCA scree, SSPG CV metrics, retina class balance) is
computed here from the loaded data, never a hardcoded stand-in. the two
schematic figures (data_landscape, pipeline_architecture) instead carry
descriptive cohort-size ranges and infrastructure labels as static captions.
writes one png per figure to reports/figures/.

    python scripts/make_showcase_figures.py
"""
import collections
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hurdle.features.cgm import extract_cgm_features  #noqa: E402
from hurdle.features.omics_highdim import build_highdim_matrix  #noqa: E402
from hurdle.features.wearable import _cosinor  #noqa: E402

FIGDIR = ROOT / "reports" / "figures"
CACHE = FIGDIR / ".cache"

#okabe-ito colorblind-safe palette, bound to modalities and reused everywhere
OKABE = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermillion": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "grey": "#999999", "black": "#222222",
}
MODALITY = {
    "omics": OKABE["blue"], "cgm": OKABE["orange"],
    "wearable": OKABE["green"], "imaging": OKABE["purple"],
}


def apply_figure_style(*, frame="open", sizes=(9, 8, 7), grid=False):
    #publication-grade mechanics: role-mapped font ladder, outward ticks,
    #frameless legends, 300-dpi save (inlined from the figure-style skill)
    base, secondary, tick = sizes
    boxed = frame == "boxed"
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": base,
        "axes.labelsize": base, "axes.titlesize": base,
        "legend.fontsize": secondary, "xtick.labelsize": tick, "ytick.labelsize": tick,
        "axes.linewidth": 0.6, "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid), "legend.frameon": False,
        "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.titleweight": "normal", "axes.titlelocation": "left",
        "lines.linewidth": 1.2, "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def set_frame(ax, style="open"):
    #set spine visibility on an existing axes (inlined from figure-style)
    show = {"open": (False, False, True, True),
            "boxed": (True, True, True, True),
            "none": (False, False, False, False)}[style]
    for side, vis in zip(("top", "right", "bottom", "left"), show):
        ax.spines[side].set_visible(vis)
        if vis:
            ax.spines[side].set_linewidth(0.6)
    ax.tick_params(direction="out", length=0 if style == "none" else 3, width=0.6)


def _apply_style():
    apply_figure_style(frame="open", sizes=(9, 8, 7))


def _bbox_check(fig):
    #geometric qa: any two visible text boxes overlapping is a layout bug.
    #draw first so text window-extents are finalized, and skip legend-internal
    #labels (stacked entries share a column and are not a collision)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    legs = {t for lg in fig.legends for t in lg.get_texts()}
    for ax in fig.axes:
        lg = ax.get_legend()
        if lg is not None:
            legs.update(lg.get_texts())
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible() and t not in legs]
    bad = [(a.get_text(), b.get_text())
           for i, (a, ba) in enumerate(texts)
           for b, bb in texts[i + 1:] if ba.overlaps(bb)]
    return bad


def _load_cgm():
    df = pd.read_csv(ROOT / "data/raw/cgm/hall2018_cgm_S1_Data", sep="\t", dtype=str)
    df["gv"] = pd.to_numeric(df["GlucoseValue"], errors="coerce")
    df["t"] = pd.to_datetime(df["DisplayTime"], errors="coerce")
    return df


def fig_data_landscape():
    #four modality blocks sized by real n, with the 22-patient omics-cgm overlap
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    blocks = [
        ("Multi-omics", "59-89 patients\n85-12,024 analytes", 6, 55, 40, 30, MODALITY["omics"]),
        ("CGM", "57 subjects\n~105k glucose readings", 54, 55, 40, 30, MODALITY["cgm"]),
        ("Wearable", "43 subjects\n1-min HR / accel / temp", 6, 12, 40, 30, MODALITY["wearable"]),
        ("Retinal imaging", "1,600 fundus images\n5 DR grades", 54, 12, 40, 30, MODALITY["imaging"]),
    ]
    for title, sub, x, y, w, h, col in blocks:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2.5",
                             linewidth=1.6, edgecolor=col, facecolor=col, alpha=0.16)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h - 7, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color=col)
        ax.text(x + w / 2, y + h / 2 - 4, sub, ha="center", va="center",
                fontsize=8.5, color=OKABE["black"])

    #overlap callout: omics and cgm share 22 patients (19 with SSPG)
    ax.annotate("22 patients shared\n(omics \u2229 CGM; 19 with SSPG)",
                xy=(50, 70), xytext=(50, 95), ha="center", va="center",
                fontsize=8.5, color=OKABE["black"],
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=OKABE["grey"], lw=1.0),
                arrowprops=dict(arrowstyle="-", color=OKABE["grey"], lw=1.2))
    ax.plot([46, 54], [70, 70], color=OKABE["grey"], lw=1.2, zorder=1)

    ax.text(50, 47, "wearable and imaging cohorts share no patients with any modality",
            ha="center", va="center", fontsize=7.5, style="italic", color=OKABE["grey"])
    ax.set_title("Four modalities, largely disjoint cohorts", loc="center",
                 fontsize=11, pad=8)
    fig.tight_layout()
    return fig


def fig_wearable_circadian():
    #real basis subject with strong, well-covered circadian hr rhythm
    subject = "Basis_007"
    d = pd.read_parquet(ROOT / f"data/interim/wearable/{subject}.parquet")
    hr = d["hr"].astype(float)
    hod = d.index.hour + d.index.minute / 60.0
    mesor, amp, acro = _cosinor(hod.values, hr.values)

    prof = hr.groupby(d.index.hour)
    hours = np.arange(24)
    means = prof.mean().reindex(hours).values
    sems = (prof.std() / np.sqrt(prof.count())).reindex(hours).values

    tt = np.linspace(0, 24, 400)
    w = 2.0 * math.pi / 24.0
    fit = mesor + amp * np.cos(w * tt - w * acro)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.errorbar(hours, means, yerr=sems, fmt="o", ms=4.5, color=MODALITY["wearable"],
                ecolor=OKABE["grey"], elinewidth=0.8, capsize=2, label="hourly mean HR (\u00b1SEM)")
    ax.plot(tt, fit, color=OKABE["vermillion"], lw=2.2, label="fitted cosinor")
    ax.axhline(mesor, color=OKABE["grey"], lw=1.0, ls="--")
    ax.annotate(f"MESOR = {mesor:.1f} bpm", xy=(0.3, mesor), xytext=(0.3, mesor + 1.2),
                fontsize=7.5, color=OKABE["grey"], va="bottom")
    ax.annotate("", xy=(acro, mesor + amp), xytext=(acro, mesor),
                arrowprops=dict(arrowstyle="<->", color=OKABE["vermillion"], lw=1.2))
    ax.text(acro + 0.4, mesor + amp / 2, f"amplitude\n{amp:.1f} bpm",
            fontsize=7.5, color=OKABE["vermillion"], va="center")
    ax.axvline(acro, color=OKABE["vermillion"], lw=1.0, ls=":")
    ax.text(acro, ax.get_ylim()[0], f" acrophase {acro:.1f} h", rotation=90,
            fontsize=7.5, color=OKABE["vermillion"], va="bottom", ha="right")

    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("hour of day (UTC)")
    ax.set_ylabel("heart rate (bpm)")
    ax.set_title(f"Circadian heart-rate rhythm, subject {subject}", loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=7.5)
    set_frame(ax, "open")
    ax.margins(0.04)
    fig.tight_layout()
    return fig, dict(subject=subject, mesor=round(mesor, 2), amplitude=round(amp, 2),
                     acrophase_h=round(acro, 2))


def fig_cgm_trace(cgm):
    #one real subject, densest ~60h contiguous window, tir/tar/tbr bands
    subject = "2133-039"
    sub = cgm[cgm.subjectId == subject].dropna(subset=["gv", "t"]).sort_values("t")
    feat = extract_cgm_features(sub)
    t = sub.t.values
    best = None
    for i in range(len(sub)):
        j = np.searchsorted(t, t[i] + np.timedelta64(60, "h"))
        if best is None or (j - i) > best[0]:
            best = (j - i, i, j)
    _, i, j = best
    w = sub.iloc[i:j]
    hrs = (w.t - w.t.iloc[0]).dt.total_seconds() / 3600.0

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.axhspan(0, 70, color=OKABE["vermillion"], alpha=0.12)
    ax.axhspan(70, 180, color=MODALITY["cgm"], alpha=0.10)
    ax.axhspan(180, 400, color=OKABE["purple"], alpha=0.12)
    ax.plot(hrs, w.gv.values, color=OKABE["black"], lw=1.1)
    ax.axhline(70, color=OKABE["vermillion"], lw=1.0, ls="--")
    ax.axhline(180, color=OKABE["purple"], lw=1.0, ls="--")

    xmax = float(hrs.max())
    ax.text(xmax, 35, "TBR <70", ha="right", va="center", fontsize=7.5, color=OKABE["vermillion"])
    ax.text(xmax, 125, "TIR 70-180", ha="right", va="center", fontsize=7.5, color=OKABE["orange"])
    ax.text(xmax, 200, "TAR >180", ha="right", va="center", fontsize=7.5, color=OKABE["purple"])

    ax.set_ylim(40, max(210, float(w.gv.max()) + 10))
    ax.set_xlim(0, xmax)
    ax.set_xlabel("time within window (h)")
    ax.set_ylabel("glucose (mg/dL)")
    ax.set_title(f"CGM trace, subject {subject}", loc="left")
    ax.text(0.02, 0.96, f"full-record TIR = {feat['tir'] * 100:.1f}%  (n={len(sub)} readings)",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=OKABE["grey"], lw=0.9))
    set_frame(ax, "open")
    fig.tight_layout()
    return fig, dict(subject=subject, tir_pct=round(feat["tir"] * 100, 1),
                     tar_pct=round(feat["tar"] * 100, 1), tbr_pct=round(feat["tbr"] * 100, 1),
                     mean_glucose=round(feat["mean_glucose"], 1), n_readings=int(len(sub)))


def _s4_reduction():
    #cache the slow excel read + reduction so reruns are fast; still runs cold
    CACHE.mkdir(parents=True, exist_ok=True)
    trace_f = CACHE / "s4_trace.npz"
    if trace_f.exists():
        z = np.load(trace_f, allow_pickle=True)
        return z["trace"].item(), z["evr"]
    s4p = CACHE / "s4.parquet"
    if s4p.exists():
        s4 = pd.read_parquet(s4p)
    else:
        s4 = pd.read_excel(ROOT / "data/raw/clinical/time_molecule_comparison.xlsx",
                           sheet_name="S4_HealthyIQR")
        s4.to_parquet(s4p)
    res = build_highdim_matrix(s4)
    trace, evr = res["trace"], res["explained_variance"]
    np.savez(trace_f, trace=np.array(trace, dtype=object), evr=evr)
    return trace, evr


def fig_s4_funnel():
    trace, evr = _s4_reduction()
    steps = [
        ("start", trace["start_analytes"]),
        ("prevalence", trace["after_prevalence"]),
        ("variance", trace["after_variance"]),
        ("corr-prune", trace["after_correlation_prune"]),
        ("PCA", trace["n_pca_components"]),
    ]
    labels = [s[0] for s in steps]
    counts = [s[1] for s in steps]

    fig, (axf, axs) = plt.subplots(1, 2, figsize=(9.6, 4.4),
                                   gridspec_kw=dict(width_ratios=[1.0, 1.1], wspace=0.32))

    ypos = np.arange(len(steps))[::-1]
    axf.barh(ypos[:-1], counts[:-1], color=MODALITY["omics"], alpha=0.85, height=0.62)
    axf.barh(ypos[-1], max(counts[-1], 120), color=OKABE["vermillion"], alpha=0.9, height=0.62)
    for yp, lab, c in zip(ypos, labels, counts):
        axf.text(200, yp, f"{lab}: {c:,}", va="center", ha="left", fontsize=8,
                 color=OKABE["black"])
    axf.set_yticks([])
    axf.set_xlabel("analytes retained")
    axf.set_title("S4 reduction funnel", loc="left")
    set_frame(axf, "open")
    axf.margins(y=0.06)

    k = np.arange(1, len(evr) + 1)
    cum = np.cumsum(evr)
    axs.bar(k, evr * 100, color=MODALITY["omics"], alpha=0.8, label="per-PC")
    axs2 = axs.twinx()
    axs2.plot(k, cum * 100, color=OKABE["vermillion"], marker="o", ms=3.5, lw=1.8,
              label="cumulative")
    axs2.axhline(cum[-1] * 100, color=OKABE["grey"], ls="--", lw=1.0)
    axs2.text(len(evr), 30, f"{cum[-1] * 100:.1f}%\nin {len(evr)} PCs",
              ha="right", va="center", fontsize=7.5, color=OKABE["vermillion"])
    axs.set_xlabel("principal component")
    axs.set_ylabel("variance explained (%)")
    axs2.set_ylabel("cumulative (%)", color=OKABE["vermillion"])
    axs2.tick_params(axis="y", colors=OKABE["vermillion"])
    axs2.set_ylim(0, 100)
    axs.set_title("PCA scree", loc="left")
    set_frame(axs, "open")
    axs2.spines["top"].set_visible(False)

    fig.tight_layout()
    return fig, dict(trace={k2: int(v) for k2, v in trace.items()
                            if isinstance(v, (int, np.integer))},
                     cumulative_variance_pct=round(float(cum[-1] * 100), 1))


def fig_sspg_scatter():
    #leave-one-out xgboost on the real s8 matrix; predicted vs measured sspg
    o = pd.read_csv(ROOT / "data/interim/omics_S8_sspg_clean.csv")
    feat_cols = [c for c in o.columns if c not in ("SubjectID", "SSPG")]
    x = o[feat_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(o["SSPG"], errors="coerce")
    mask = y.notna()
    x, y = x[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
    x = x.fillna(x.median())

    preds = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(x):
        m = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=4)
        m.fit(x.iloc[tr], y.iloc[tr])
        preds[te] = m.predict(x.iloc[te])
    r2 = r2_score(y, preds)
    r, p = pearsonr(y, preds)

    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    lo = min(y.min(), preds.min()) - 10
    hi = max(y.max(), preds.max()) + 10
    ax.plot([lo, hi], [lo, hi], color=OKABE["grey"], ls="--", lw=1.2, label="y = x")
    ax.scatter(y, preds, s=34, color=MODALITY["omics"], alpha=0.8, edgecolor="white", lw=0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("measured SSPG (mg/dL)")
    ax.set_ylabel("predicted SSPG (mg/dL)")
    ax.set_title("Omics baseline predicts insulin resistance", loc="left")
    ax.text(0.04, 0.96, f"$R^2$ = {r2:.2f}\nPearson r = {r:.2f}\nn = {len(y)}",
            transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=OKABE["grey"], lw=0.9))
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    set_frame(ax, "open")
    fig.tight_layout()
    return fig, dict(r2=round(float(r2), 3), pearson=round(float(r), 3),
                     p_value=float(p), n=int(len(y)), n_analytes=len(feat_cols),
                     model="XGBoost (LOO-CV)")


def fig_retina_grid():
    npz = np.load(ROOT / "data/raw/imaging/retinamnist.npz")
    imgs = np.concatenate([npz["train_images"], npz["val_images"], npz["test_images"]])
    labs = np.concatenate([npz["train_labels"], npz["val_labels"], npz["test_labels"]]).ravel()
    counts = collections.Counter(labs.tolist())
    total = len(labs)
    grade_names = ["no DR", "mild", "moderate", "severe", "proliferative"]

    ncol = 6
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(5, ncol, figsize=(8.4, 7.4),
                             gridspec_kw=dict(wspace=0.06, hspace=0.06))
    for g in range(5):
        idx = np.where(labs == g)[0]
        pick = rng.choice(idx, size=ncol, replace=False)
        for c in range(ncol):
            ax = axes[g, c]
            ax.imshow(imgs[pick[c]])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(MODALITY["imaging"])
                sp.set_linewidth(1.2)
        pct = counts[g] / total * 100
        axes[g, 0].set_ylabel(f"grade {g}\n{grade_names[g]}\nn={counts[g]} ({pct:.1f}%)",
                              rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle(f"RetinaMNIST fundus samples by DR grade (N={total:,}, imbalanced)",
                 fontsize=11, x=0.55, y=0.985)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig, {f"grade_{g}_pct": round(counts[g] / total * 100, 1) for g in range(5)}


def fig_pipeline():
    #left-to-right schematic: raw -> cleaning -> feature builders -> sql ->
    #fusion/virtual-cohort -> risk, over a cloud band
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.set_xlim(0, 118)
    ax.set_ylim(0, 60)
    ax.axis("off")

    #cloud infrastructure band underneath
    band = FancyBboxPatch((2, 2), 114, 9, boxstyle="round,pad=0.4,rounding_size=2",
                          facecolor=OKABE["sky"], alpha=0.16, edgecolor=OKABE["sky"], lw=1.2)
    ax.add_patch(band)
    ax.text(59, 6.5, "cloud infrastructure:  AWS S3 storage + GPU training  \u00b7  GCP orchestration",
            ha="center", va="center", fontsize=8.5, color=OKABE["blue"], fontstyle="italic")

    def box(x, y, w, h, title, sub, col):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6",
                           facecolor=col, alpha=0.16, edgecolor=col, lw=1.5)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h - 4.2, title, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=col)
        if sub:
            ax.text(x + w / 2, y + (h - 8) / 2 + 2, sub, ha="center", va="center",
                    fontsize=6.8, color=OKABE["black"])
        return (x + w, y + h / 2), (x, y + h / 2)

    #column 1: raw data (four stacked modality tiles)
    mods = [("omics", "omics"), ("CGM", "cgm"), ("wearable", "wearable"), ("imaging", "imaging")]
    raw_right = []
    for i, (name, key) in enumerate(mods):
        yy = 44 - i * 8.5
        b = FancyBboxPatch((3, yy), 15, 7, boxstyle="round,pad=0.3,rounding_size=1.2",
                           facecolor=MODALITY[key], alpha=0.18, edgecolor=MODALITY[key], lw=1.3)
        ax.add_patch(b)
        ax.text(10.5, yy + 3.5, name, ha="center", va="center", fontsize=8,
                fontweight="bold", color=MODALITY[key])
        raw_right.append((18, yy + 3.5))
    ax.text(10.5, 53.5, "raw data", ha="center", fontsize=9, fontweight="bold",
            color=OKABE["black"])

    r_clean = box(26, 20, 15, 22, "cleaning", "coerce \u00b7 QC\nresample \u00b7 dedup", OKABE["grey"])
    r_feat = box(49, 20, 17, 22, "feature builders", "cosinor \u00b7 CGM TIR\nomics reduction\nimage encoder", OKABE["green"])
    r_sql = box(74, 20, 14, 22, "SQL layer", "per-modality\nfeature tables", OKABE["blue"])
    r_fus = box(96, 24, 18, 16, "fusion +\nvirtual cohort", "align shared\npatients", OKABE["orange"])

    #arrows: raw tiles -> cleaning
    for (rx, ry) in raw_right:
        ax.add_patch(FancyArrowPatch((rx, ry), (26, 31), arrowstyle="-|>",
                     mutation_scale=11, color=OKABE["grey"], lw=1.1, alpha=0.8))
    for a, b in [(r_clean[0], r_feat[1]), (r_feat[0], r_sql[1]), (r_sql[0], r_fus[1])]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13,
                     color=OKABE["black"], lw=1.4))

    #risk prediction terminal box
    risk = FancyBboxPatch((99, 44), 15, 11, boxstyle="round,pad=0.4,rounding_size=1.6",
                          facecolor=OKABE["vermillion"], alpha=0.18,
                          edgecolor=OKABE["vermillion"], lw=1.6)
    ax.add_patch(risk)
    ax.text(106.5, 49.5, "risk\nprediction", ha="center", va="center", fontsize=8.5,
            fontweight="bold", color=OKABE["vermillion"])
    ax.add_patch(FancyArrowPatch((105, 40), (106.5, 44), arrowstyle="-|>",
                 mutation_scale=13, color=OKABE["black"], lw=1.4))

    ax.set_title("HURDLE multimodal pipeline", loc="left", fontsize=11, x=0.02, y=0.98)
    fig.tight_layout()
    return fig


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    _apply_style()
    real = {}

    fig = fig_data_landscape()
    assert not _bbox_check(fig), "data_landscape text overlap"
    fig.savefig(FIGDIR / "data_landscape.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, real["wearable_circadian"] = fig_wearable_circadian()
    assert not _bbox_check(fig), "wearable_circadian text overlap"
    fig.savefig(FIGDIR / "wearable_circadian.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    cgm = _load_cgm()
    fig, real["cgm_trace"] = fig_cgm_trace(cgm)
    fig.savefig(FIGDIR / "cgm_trace.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, real["s4_reduction_funnel"] = fig_s4_funnel()
    fig.savefig(FIGDIR / "s4_reduction_funnel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, real["omics_sspg_scatter"] = fig_sspg_scatter()
    fig.savefig(FIGDIR / "omics_sspg_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, real["retina_grade_grid"] = fig_retina_grid()
    fig.savefig(FIGDIR / "retina_grade_grid.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig = fig_pipeline()
    fig.savefig(FIGDIR / "pipeline_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    for name, vals in real.items():
        print(name, vals)


if __name__ == "__main__":
    main()
