"""Federated learning as a compact simulation (numpy/sklearn only).

FedAvg (McMahan et al. 2017): each site trains a local logistic or ridge model on
its own patients and shares only parameters, which the server averages weighted by
sample size, so raw rows never leave a site; here the real omics patients are
partitioned across n_clients sites. One StandardScaler fit on the pooled train split
is shared with every client, so the comparison isolates the effect of federating the
fit rather than differing scalers.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler


def federated_average(client_params, client_sizes):
    """Sample-size weighted average of per-client parameter vectors (FedAvg step).

    Takes a list of equal-length flattened parameter arrays and a list of local
    sample counts, and returns sum_k (n_k / sum n_k) * params_k, so sites with more
    patients pull the global model harder. Only parameters are combined, no raw data.
    """
    if len(client_params) == 0:
        raise ValueError("need at least one client")
    if len(client_params) != len(client_sizes):
        raise ValueError("client_params and client_sizes length mismatch")
    P = np.asarray([np.asarray(p, dtype=float).ravel() for p in client_params])
    if len({p.shape for p in P}) != 1:
        raise ValueError("all client params must share one shape")
    sizes = np.asarray(client_sizes, dtype=float)
    if sizes.sum() <= 0 or (sizes < 0).any():
        raise ValueError("client_sizes must be non-negative with positive sum")
    w = sizes / sizes.sum()
    return (w[:, None] * P).sum(axis=0)


def _partition(n, n_clients, rng):
    #shuffle indices then split into n_clients near-equal shards (one site's
    #patients per shard); returns a list of index arrays, none empty
    if n_clients < 1:
        raise ValueError("n_clients must be >= 1")
    if n_clients > n:
        raise ValueError(f"n_clients={n_clients} exceeds n={n}")
    idx = rng.permutation(n)
    return [s for s in np.array_split(idx, n_clients)]


def _extract(est, task):
    #flatten a fitted linear estimator to a parameter vector [coef..., intercept]
    return np.concatenate([est.coef_.ravel(), np.atleast_1d(est.intercept_).ravel()])


def _make_local(task):
    if task == "classification":
        return LogisticRegression(solver="lbfgs", max_iter=2000)
    return Ridge(alpha=1.0)


def _predict_with(task, params, Xs):
    #apply flattened params [coef..., intercept] to scaled features Xs
    d = Xs.shape[1]
    coef, intercept = params[:d], params[d]
    z = Xs @ coef + intercept
    if task == "classification":
        return (z >= 0).astype(int)  #logit>=0 <=> prob>=0.5
    return z


def central_logistic(X, y, task="classification", test_frac=0.3, seed=0):
    """Train one central model on the pooled train split, score on the held-out test.

    The baseline for the federated comparison (all rows pooled at one site). Returns
    (metric, params, scaler, split), where metric is test accuracy (classification)
    or R^2 (regression).
    """
    return _fit_central(X, y, task, test_frac, seed)


def _fit_central(X, y, task, test_frac, seed):
    A = np.asarray(X, dtype=float)
    yv = np.asarray(y).ravel()
    n = len(yv)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(test_frac * n)))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    scaler = StandardScaler().fit(A[train_idx])
    Xtr, Xte = scaler.transform(A[train_idx]), scaler.transform(A[test_idx])
    ytr, yte = yv[train_idx], yv[test_idx]
    est = _make_local(task)
    est.fit(Xtr, ytr if task == "regression" else ytr.astype(int))
    params = _extract(est, task)
    metric = _score(task, yte, _predict_with(task, params, Xte))
    return metric, params, scaler, (train_idx, test_idx)


def _score(task, y_true, y_pred):
    if task == "classification":
        return float(accuracy_score(y_true.astype(int), y_pred))
    return float(r2_score(y_true, y_pred))


def simulate_federated_training(X, y, n_clients=4, rounds=5, task="classification",
                                test_frac=0.3, seed=0):
    """FedAvg over the real patients vs a central model on the same train/test split.

    Each round broadcasts the global params, has every site fit locally on its own
    disjoint train patients, and FedAvg-aggregates weighted by local n; after `rounds`
    the aggregated model is scored on the held-out test and compared to
    central_logistic on the identical split. Returns a dict with central_metric,
    federated_metric, gap, per_round_metric, n_clients, rounds, task, and client_sizes.
    """
    A = np.asarray(X, dtype=float)
    yv = np.asarray(y).ravel()
    if task == "classification":
        yv = yv.astype(int)

    central_metric, _, _, split = _fit_central(X, y, task, test_frac, seed)
    train_idx, test_idx = split

    #reuse the exact central split + scaler so the only difference is federation
    scaler = StandardScaler().fit(A[train_idx])
    Xtr_all, Xte = scaler.transform(A[train_idx]), scaler.transform(A[test_idx])
    ytr_all, yte = yv[train_idx], yv[test_idx]

    rng = np.random.default_rng(seed + 1)
    shards = _partition(len(train_idx), n_clients, rng)
    client_sizes = [int(s.size) for s in shards]

    d = A.shape[1]
    global_params = np.zeros(d + 1)  #[coef..., intercept]
    per_round = []
    for _ in range(rounds):
        local_params = []
        for s in shards:
            Xs, ys = Xtr_all[s], ytr_all[s]
            est = _make_local(task)
            #a site with a single class cannot fit logistic; fall back to the
            #broadcast params so it still contributes its (uninformative) weight
            if task == "classification" and len(np.unique(ys)) < 2:
                local_params.append(global_params.copy())
                continue
            est = _warm_fit(est, Xs, ys, task, global_params)
            local_params.append(_extract(est, task))
        global_params = federated_average(local_params, client_sizes)
        per_round.append(_score(task, yte, _predict_with(task, global_params, Xte)))

    federated_metric = per_round[-1]
    return {
        "central_metric": central_metric,
        "federated_metric": federated_metric,
        "gap": central_metric - federated_metric,
        "per_round_metric": per_round,
        "n_clients": n_clients,
        "rounds": rounds,
        "task": task,
        "client_sizes": client_sizes,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }


def _warm_fit(est, Xs, ys, task, init_params):
    #warm-start from the broadcast global params when the estimator supports it,
    #so successive FedAvg rounds refine rather than restart (McMahan et al. FedAvg
    #with E local epochs); ridge has no warm start so it simply refits locally
    if task == "classification":
        d = Xs.shape[1]
        est.set_params(warm_start=True)
        est.coef_ = init_params[:d].reshape(1, -1).copy()
        est.intercept_ = np.atleast_1d(init_params[d]).copy()
        est.classes_ = np.array([0, 1])
        est.fit(Xs, ys.astype(int))
    else:
        est.fit(Xs, ys)
    return est
