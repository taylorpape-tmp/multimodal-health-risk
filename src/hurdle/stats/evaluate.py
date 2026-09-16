"""Run LOO cross-validation via MLModel with bootstrap CIs and label-permutation
p-values.

The permutation reruns the full leave-one-out on the permuted labels, so it sits
outside the CV and cannot leak. All metrics share one permutation loop (one CV
rerun per permutation), which is cheaper and gives a single coherent null.
"""
import numpy as np
import pandas as pd

from . import metrics as M
from .metrics import GREATER_IS_BETTER
from .resampling import bootstrap_ci

#task -> [(metric name, metric_fn)] reported with CI + permutation p
_REG_METRICS = [("R2", M.r2), ("RMSE", M.rmse), ("Pearson", M.pearson)]
_CLF_METRICS = [("AUROC", M.auroc), ("F1", M.f1)]


def _build_model(model_cls, X, y, target_col, groups, param_grid):
    frame = X.copy()
    frame[target_col] = np.asarray(y)
    feature_cols = [c for c in frame.columns if c != target_col]
    return model_cls(frame, feature_cols=feature_cols, target_col=target_col,
                     groups=groups, param_grid=param_grid)


def _shared_permutation_pvalues(y, predict_fn, metric_list, observed, n_perm, seed):
    #one CV rerun per permutation; score every metric on the same permuted preds
    rng = np.random.default_rng(seed)
    counts = {name: 0 for name, _ in metric_list}
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        preds = predict_fn(y_perm)
        for name, fn in metric_list:
            v = float(fn(y_perm, preds))
            if not np.isfinite(v):
                continue
            better = (v >= observed[name]) if GREATER_IS_BETTER.get(name, True) \
                else (v <= observed[name])
            if better:
                counts[name] += 1
    return {name: (1 + counts[name]) / (1 + n_perm) for name, _ in metric_list}


def evaluate_model(model_cls, X, y, task, groups=None, target_col="target",
                   param_grid="default", n_boot=2000, n_perm=200, seed=0):
    """Run LOO for model_cls on (X, y) and return per-metric 95% bootstrap CIs and
    label-permutation p-values.

    X is a DataFrame, y a 1d array/Series, task 'regression' or 'classification', and
    n_perm reruns the full CV each time so keep it modest. Returns
    {'name', 'task', 'n', 'preds', <metric>: {'point','lo','hi','p'}}.
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    y = np.asarray(y)
    metric_list = _REG_METRICS if task == "regression" else _CLF_METRICS

    #predict_fn(labels) reruns the whole LOO on the given labels
    def predict_fn(labels):
        model = _build_model(model_cls, X, labels, target_col, groups, param_grid)
        return model.run(verbose=False)["preds"]

    preds = predict_fn(y)
    observed = {name: float(fn(y, preds)) for name, fn in metric_list}
    pvals = _shared_permutation_pvalues(y, predict_fn, metric_list, observed,
                                        n_perm, seed)

    out = {"name": model_cls.name, "task": task, "n": int(len(y)), "preds": preds}
    for name, fn in metric_list:
        point, lo, hi = bootstrap_ci(y, preds, fn, n_boot=n_boot, seed=seed)
        out[name] = {"point": point, "lo": lo, "hi": hi, "p": pvals[name]}
    return out
