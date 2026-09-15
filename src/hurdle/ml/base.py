"""Base model contract, ported from the dom_study MLModel architecture.

Adaptations for HURDLE:
  - target column is configurable (SSPG regression, IRIS classification), not
    hardcoded to dom_z_mean
  - the cross-validation splitter is pluggable: LeaveOneOut for tiny-n omics,
    or subject-wise LeaveOneGroupOut when repeated observations share a subject
  - a task flag ('regression' | 'classification') selects the metric set

A subclass overrides: name, task, param_grid (None = no tuning), _estimator(),
and optionally _fold_meta() to record per-fold chosen hyperparameters.
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut
from sklearn.pipeline import Pipeline

from .transforms import DEFAULT_FOR_FAMILY, build_transform


class MLModel:
    #subclass overrides: name, task, family, param_grid (None = no tuning),
    #                    _estimator() [returns a BARE estimator], _fold_meta()
    name = 'model'
    task = 'regression'                 #'regression' | 'classification'
    family = 'linear'                   #'tree' | 'linear' | 'kernel' -> default transform
    param_grid = None                   #keys use the 'model__' pipeline-step prefix
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=0)
    scoring = None                      #None -> chosen from task in _fit_fold
    verbose_folds = False

    def __init__(self, frame, feature_cols=None, target_col='target',
                 groups=None, transform='auto', param_grid='default'):
        if feature_cols is None:
            feature_cols = [c for c in frame.columns if c != target_col]
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.X = frame[self.feature_cols].values
        self.y = frame[target_col].values
        #groups: optional per-row subject id for subject-wise CV (None -> LOO)
        self.groups = None if groups is None else np.asarray(groups)
        #transform: 'auto' picks the data-dependent default for this model family
        self.transform = DEFAULT_FOR_FAMILY[self.family] if transform == 'auto' else transform
        #param_grid: 'default' uses the class grid; pass a dict to override (tests
        #pass a tiny grid so tuned models fit in seconds without touching the
        #production grid on the class)
        if param_grid != 'default':
            self.param_grid = param_grid

    def _estimator(self):
        #subclass returns a BARE estimator; the base wraps it with the transform
        raise NotImplementedError

    def _pipeline(self):
        #pipeline = transform step -> model; param_grid keys use 'model__'
        return Pipeline([
            ('transform', build_transform(self.transform, n_samples=len(self.y))),
            ('model',     self._estimator()),
        ])

    def _fold_meta(self, est):
        #override to record per-fold chosen hyperparams as {name: value}
        return {}

    def _default_scoring(self):
        return 'neg_mean_squared_error' if self.task == 'regression' else 'accuracy'

    def _splitter(self):
        #subject-wise LeaveOneGroupOut when groups given; else leave-one-out
        if self.groups is not None:
            from sklearn.model_selection import LeaveOneGroupOut
            return LeaveOneGroupOut().split(self.X, self.y, self.groups)
        return LeaveOneOut().split(self.X)

    def _fit_fold(self, X_train, y_train):
        pipe = self._pipeline()
        if self.param_grid is None:
            pipe.fit(X_train, y_train)
            return pipe
        scoring = self.scoring or self._default_scoring()
        #n_jobs=1: nested CV on tiny-n doesn't benefit from process parallelism,
        #and process-based joblib backends aren't available in all sandboxes
        gs = GridSearchCV(pipe, self.param_grid, cv=self.inner_cv,
                          scoring=scoring, n_jobs=1)
        gs.fit(X_train, y_train)
        return gs.best_estimator_

    def run(self, verbose=True):
        n = len(self.y)
        preds = np.zeros(n, dtype=float)
        meta = defaultdict(list)
        for i, (train_idx, test_idx) in enumerate(self._splitter(), 1):
            est = self._fit_fold(self.X[train_idx], self.y[train_idx])
            preds[test_idx] = est.predict(self.X[test_idx])
            for k, v in self._fold_meta(est).items():
                meta[k].append(v)
            if verbose and self.verbose_folds:
                print(f"  fold {i} done", flush=True)

        results = {
            'name':      self.name,
            'task':      self.task,
            'n':         n,
            'features':  self.feature_cols,
            'preds':     preds,
            'fold_meta': dict(meta),
        }
        results.update(self._metrics(self.y, preds))
        if verbose:
            self._print(results)
        return results

    def _metrics(self, y, preds):
        if self.task == 'regression':
            return {
                'R2':       r2_score(y, preds),
                'MAE':      mean_absolute_error(y, preds),
                'RMSE':     np.sqrt(mean_squared_error(y, preds)),
                'Pearson':  pearsonr(y, preds)[0],
                'Spearman': spearmanr(y, preds)[0],
            }
        #classification: preds may be continuous scores or hard labels
        hard = (preds >= 0.5).astype(int) if set(np.unique(y)) <= {0, 1} else np.rint(preds)
        out = {'Accuracy': accuracy_score(y, hard), 'F1': f1_score(y, hard, average='macro')}
        try:
            out['AUROC'] = roc_auc_score(y, preds)
        except ValueError:
            out['AUROC'] = float('nan')
        return out

    def _print(self, r):
        print(f"\n{r['name']}  (n={r['n']}, {len(r['features'])} features, {r['task']})")
        for k, v in r.items():
            if k in ('name', 'task', 'n', 'features', 'preds', 'fold_meta'):
                continue
            print(f"{k}: {v:+.4f}")
        if r['fold_meta']:
            print("hyperparameters chosen across folds:")
            for k, vals in r['fold_meta'].items():
                self._print_param(k, vals)

    def _print_param(self, name, vals):
        s = pd.Series(vals)
        if s.dtype.kind in 'fc' and s.nunique() > 10:
            print(f"  {name}: median={np.median(s):.3g}  range={s.min():.3g} to {s.max():.3g}")
        else:
            counts = s.value_counts().sort_index()
            print(f"  {name}: " + ", ".join(f"{v}:{c}" for v, c in counts.items()))
