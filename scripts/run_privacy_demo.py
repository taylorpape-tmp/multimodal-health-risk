"""Privacy-preserving ML demo on the real omics data: DP epsilon-sweep + FedAvg.

Runs from the repo root:  python scripts/run_privacy_demo.py

Three real-data measurements, no fabricated numbers:
  1. DP downstream utility  -- IRIS (IS/IR) held-out accuracy of an
     output-perturbed logistic regression as epsilon sweeps 0.1..inf.
  2. DP per-feature release -- mean absolute error of a DP summary table over the
     SSPG omics matrix, showing the cost of splitting the budget across features
     by basic composition.
  3. Federated vs central   -- FedAvg over the real IRIS patients partitioned
     across n_clients simulated sites, compared to a central model on the same
     held-out split. Federated shares only model params, never raw rows.

Outputs (created under reports/):
  privacy_utility_tradeoff.png   metric-vs-epsilon (DP) and federated-vs-central
  privacy_report.csv             every epsilon/metric row + the fed/central rows

If the interim omics csvs are absent the run aborts with a clear message rather
than inventing data.
"""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hurdle.features.omics import build_feature_matrix  #noqa: E402
from hurdle.privacy.dp import dp_output_perturbed_logistic, dp_summary_table  #noqa: E402
from hurdle.privacy.federated import simulate_federated_training  #noqa: E402

INTERIM = "data/interim"
REPORTS = Path("reports")
DELTA = 1e-5
EPS = [0.1, 0.5, 1.0, 2.0, 5.0, np.inf]
LOGIT_REPS = 50
SUMMARY_REPS = 25
LAM = 0.5
TEST_FRAC = 0.3
SEED = 0


def _apply_style():
    #self-contained publication-grade rcParams (spines, role-mapped size ladder)
    #so the script runs standalone after `pip install -e .` with no skill module
    mpl.rcParams.update({
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.titlelocation": "left",
        "font.family": "sans-serif",
    })


def _eps_label(e):
    return "inf" if e == np.inf else f"{e:g}"


def dp_logistic_sweep(Xi, yi):
    #output-perturbed logistic on IRIS: held-out accuracy vs epsilon, averaged over
    #LOGIT_REPS noise draws. standardize on TRAIN only (leak-safe), then apply the
    #same L2 row-normalization the DP fit uses so test rows match the ||x||<=1 domain.
    A = Xi.values
    y = yi.values.astype(int)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y))
    nte = int(round(TEST_FRAC * len(y)))
    te, tr = perm[:nte], perm[nte:]
    sc = StandardScaler().fit(A[tr])
    Xtr, Xte = sc.transform(A[tr]), sc.transform(A[te])
    ytr, yte = y[tr], y[te]
    nn = np.linalg.norm(Xte, axis=1, keepdims=True)
    nn[nn == 0] = 1.0
    Xte_n = Xte / nn

    rows = []
    for eps in EPS:
        reps = LOGIT_REPS if eps != np.inf else 1
        accs = []
        for r in range(reps):
            w, sigma = dp_output_perturbed_logistic(Xtr, ytr, eps, DELTA, lam=LAM, seed=500 + r)
            pred = (Xte_n @ w >= 0).astype(int)
            accs.append(accuracy_score(yte, pred))
        rows.append({"epsilon": eps, "metric": "iris_dp_logistic_accuracy",
                     "value": float(np.mean(accs)), "n_test": int(nte), "n_features": ""})
    return rows


def dp_summary_sweep(Xs):
    #per-feature DP mean release over the SSPG omics matrix: mean |dp-true| across
    #features, averaged over SUMMARY_REPS draws, plus a scale-free relative error.
    true_scale = float(np.abs(Xs.mean(axis=0)).mean())
    rows = []
    for eps in EPS:
        reps = SUMMARY_REPS if eps != np.inf else 1
        maes = [dp_summary_table(Xs, epsilon=eps, delta=DELTA, seed=2000 + r)["abs_error"].mean()
                for r in range(reps)]
        mae = float(np.mean(maes))
        rows.append({"epsilon": eps, "metric": "sspg_dp_mean_abs_error",
                     "value": mae, "n_test": "", "n_features": Xs.shape[1]})
        rows.append({"epsilon": eps, "metric": "sspg_dp_mean_rel_error",
                     "value": mae / true_scale, "n_test": "", "n_features": Xs.shape[1]})
    return rows


def federated_sweep(Xi, yi, clients=(2, 4, 8), rounds=10):
    #FedAvg vs central on the real IRIS patients across increasing site counts
    rows = []
    central = None
    for k in clients:
        res = simulate_federated_training(Xi.values, yi.values, n_clients=k,
                                          rounds=rounds, task="classification", seed=SEED)
        central = res["central_metric"]
        rows.append({"n_clients": k, "central_accuracy": res["central_metric"],
                     "federated_accuracy": res["federated_metric"], "gap": res["gap"],
                     "client_sizes": "|".join(str(s) for s in res["client_sizes"]),
                     "n_train": res["n_train"], "n_test": res["n_test"]})
    return rows, central


def make_figure(dp_acc, dp_err, fed_rows, central_acc, path):
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    ax0, ax1, ax2 = axes

    #panel A: DP downstream accuracy vs epsilon (finite eps on log-x; inf as reference)
    fin = [(r["epsilon"], r["value"]) for r in dp_acc if r["epsilon"] != np.inf]
    inf_acc = next(r["value"] for r in dp_acc if r["epsilon"] == np.inf)
    xs, ys_ = zip(*fin)
    ax0.plot(xs, ys_, "o-", color="C0")
    ax0.axhline(inf_acc, ls="--", color="0.35")
    ax0.text(xs[0], inf_acc + 0.008, f"non-private = {inf_acc:.2f}", fontsize=7, color="0.35")
    ax0.axhline(0.5, ls=":", color="0.6")
    ax0.text(xs[0], 0.5 + 0.008, "chance", fontsize=7, color="0.6")
    ax0.set_xscale("log")
    ax0.set_xlabel("privacy budget epsilon (log)")
    ax0.set_ylabel("IRIS held-out accuracy")
    ax0.set_title("DP logistic: utility falls as privacy tightens")

    #panel B: DP per-feature mean-release relative error vs epsilon (log-log)
    rel = [(r["epsilon"], r["value"]) for r in dp_err
           if r["metric"] == "sspg_dp_mean_rel_error" and r["epsilon"] != np.inf]
    xs2, ys2 = zip(*rel)
    ax1.plot(xs2, ys2, "s-", color="C1")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("privacy budget epsilon (log)")
    ax1.set_ylabel("mean rel. error of DP feature means")
    ax1.set_title("DP summary table: 85-feature\nbudget split inflates noise")

    #panel C: federated vs central accuracy across site counts
    ks = [r["n_clients"] for r in fed_rows]
    feds = [r["federated_accuracy"] for r in fed_rows]
    xpos = np.arange(len(ks))
    ax2.bar(xpos, feds, color="C2", width=0.6, label="federated (FedAvg)")
    ax2.axhline(central_acc, ls="--", color="0.2")
    ax2.text(xpos[0] - 0.3, central_acc + 0.008, f"central = {central_acc:.2f}",
             fontsize=7, color="0.2")
    ax2.set_xticks(xpos)
    ax2.set_xticklabels([f"{k} sites" for k in ks])
    ax2.set_ylabel("IRIS held-out accuracy")
    ax2.set_ylim(0, 1)
    ax2.set_title("Federated nears central\nwithout sharing rows")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    interim = Path(INTERIM)
    needed = ["omics_S8_sspg_clean.csv", "omics_S9_isir_clean.csv"]
    missing = [f for f in needed if not (interim / f).exists()]
    if missing:
        print(f"SKIP: interim omics files missing: {missing}. Nothing fabricated; aborting.")
        return

    Xs, _, _ = build_feature_matrix(INTERIM, target="SSPG", add_ratios=False)
    Xi, yi, _ = build_feature_matrix(INTERIM, target="IRIS", add_ratios=False)
    print(f"omics loaded: SSPG {Xs.shape}, IRIS {Xi.shape}")

    dp_acc = dp_logistic_sweep(Xi, yi)
    dp_err = dp_summary_sweep(Xs)
    fed_rows, central_acc = federated_sweep(Xi, yi)

    REPORTS.mkdir(exist_ok=True)
    #long-format csv: DP sweep rows + federated rows share a file via a 'section' tag
    dp_df = pd.DataFrame(dp_acc + dp_err)
    dp_df.insert(0, "section", "dp_sweep")
    dp_df["epsilon"] = dp_df["epsilon"].map(_eps_label)
    fed_df = pd.DataFrame(fed_rows)
    fed_df.insert(0, "section", "federated_vs_central")
    csv_path = REPORTS / "privacy_report.csv"
    pd.concat([dp_df, fed_df], ignore_index=True).to_csv(csv_path, index=False)

    png_path = REPORTS / "privacy_utility_tradeoff.png"
    make_figure(dp_acc, dp_err, fed_rows, central_acc, png_path)

    print("\n=== DP epsilon-sweep: IRIS output-perturbed logistic accuracy ===")
    for r in dp_acc:
        print(f"  epsilon={_eps_label(r['epsilon']):>4}  accuracy = {r['value']:.3f}"
              f"  (n_test={r['n_test']})")
    print("\n=== DP epsilon-sweep: SSPG per-feature DP mean release error ===")
    for r in dp_err:
        if r["metric"] == "sspg_dp_mean_abs_error":
            rel = next(x["value"] for x in dp_err
                       if x["metric"] == "sspg_dp_mean_rel_error" and x["epsilon"] == r["epsilon"])
            print(f"  epsilon={_eps_label(r['epsilon']):>4}  mean|dp-true| = {r['value']:9.3f}"
                  f"  rel = {rel:.3f}")
    print("\n=== Federated (FedAvg) vs central: IRIS accuracy ===")
    for r in fed_rows:
        print(f"  {r['n_clients']} sites: central = {r['central_accuracy']:.3f}"
              f"  federated = {r['federated_accuracy']:.3f}  gap = {r['gap']:+.3f}"
              f"  sizes=[{r['client_sizes']}]")
    print(f"\nwrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
