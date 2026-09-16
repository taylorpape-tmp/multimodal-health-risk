"""Metric callables with a uniform metric_fn(y_true, y_pred) -> float signature.

Thin wrappers over sklearn/scipy metrics so bootstrap_ci and permutation_test can
plug in any of them. Classification metrics accept probability scores and threshold
at 0.5 for hard-label metrics like f1.
"""
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, mean_squared_error, r2_score, roc_auc_score


def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def pearson(y_true, y_pred):
    return float(pearsonr(y_true, y_pred)[0])


def auroc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def _harden(y_pred):
    #accept probabilities or hard labels; threshold probs at 0.5
    y_pred = np.asarray(y_pred)
    uniq = set(np.unique(y_pred).tolist())
    return y_pred.astype(int) if uniq <= {0, 1} else (y_pred >= 0.5).astype(int)


def f1(y_true, y_pred):
    return f1_score(y_true, _harden(y_pred), average="macro")


#metric -> whether a larger value means a better model (drives permutation tail)
GREATER_IS_BETTER = {"R2": True, "RMSE": False, "Pearson": True, "AUROC": True, "F1": True}
