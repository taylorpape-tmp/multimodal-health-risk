"""Pluggable feature transforms, a 'transform ladder' alongside the model ladder.

The right transform depends on both data and model: trees are scale-invariant
('none'), linear/kernel models are scale-sensitive, and skewed features (clinical
labs, metabolite abundances) want robust/log/power rather than plain z-score.
Each factory returns a fresh sklearn transformer (or 'passthrough') to drop into
a Pipeline step named 'transform'.
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

#the named ladder; sweep over these the way you sweep over models
TRANSFORMS = ('none', 'standard', 'robust', 'log', 'power', 'quantile')

#data-dependent default per model family
DEFAULT_FOR_FAMILY = {'tree': 'none', 'linear': 'standard', 'kernel': 'standard'}


def build_transform(name, n_samples=None):
    """Return a fresh transformer for `name`, or 'passthrough' for none.

    n_samples lets the quantile transformer cap its quantile count so it stays
    valid at small n.
    """
    if name in (None, 'none'):
        return 'passthrough'
    if name == 'standard':
        return StandardScaler()
    if name == 'robust':
        return RobustScaler()
    if name == 'log':
        #log1p then z-score; log1p needs x > -1, so this is for non-negative data
        return Pipeline([
            ('log',   FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
            ('scale', StandardScaler()),
        ])
    if name == 'power':
        #Yeo-Johnson makes data more Gaussian and handles negatives
        return PowerTransformer(method='yeo-johnson')
    if name == 'quantile':
        nq = 1000 if n_samples is None else max(2, min(1000, n_samples))
        return QuantileTransformer(output_distribution='normal',
                                   n_quantiles=nq, random_state=0)
    raise ValueError(f"unknown transform {name!r}; choose from {TRANSFORMS}")
