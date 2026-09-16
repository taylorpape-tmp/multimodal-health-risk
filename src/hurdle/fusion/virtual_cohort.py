"""Virtual-cohort generator for the multimodal fusion demo.

The four public datasets are different people (omics and CGM share 22 patients
via the crosswalk; wearable and imaging share none), so no real
all-four-modalities-per-patient matrix exists. This builds a synthetic cohort
whose patients carry all four modalities at once, purely to demonstrate the
pipeline end to end, and it is never presented as real individuals.

It avoids the circular trap of summing the modalities into a target the model
then re-learns. Instead a hidden latent risk z ~ N(0,1) is drawn first, every
modality is generated from z through noisy nonlinear maps, and the model sees
only those features (never z). Recovering z therefore requires genuinely fusing
the modalities. Coupling strengths are calibrated to the real 22, and the
controls that check fusion does real work live in fusion.py.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ModalitySpec:
    #one modality's synthetic feature block: how many features, how strongly the
    #block loads on the shared latent z, and how much independent noise it carries
    name: str
    n_features: int
    loading: float          #strength of coupling to z (0 = independent of z)
    noise: float = 1.0      #independent gaussian noise sd on top of the z signal
    nonlinear: bool = True  #pass the z signal through a nonlinearity per feature


DEFAULT_MODALITIES = (
    #loadings default to moderate; calibrate_from_real() can overwrite them
    ModalitySpec("omics",    n_features=20, loading=0.8, noise=1.0),
    ModalitySpec("cgm",      n_features=8,  loading=0.7, noise=1.0),
    ModalitySpec("wearable", n_features=12, loading=0.5, noise=1.0),
    ModalitySpec("imaging",  n_features=6,  loading=0.6, noise=1.0),
)


@dataclass
class VirtualCohort:
    X: pd.DataFrame                 #all modality features concatenated
    z: np.ndarray                   #hidden latent truth (the regression target)
    iris: np.ndarray                #binary label derived by thresholding z (>0)
    modality_cols: dict = field(default_factory=dict)   #name -> [feature cols]


def generate(n=300, modalities=DEFAULT_MODALITIES, seed=0, coupling=1.0):
    """Generate a virtual cohort of n synthetic patients.

    coupling scales every modality's loading at once; coupling=0 makes the
    features independent of z, matching the scramble control.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)                      #hidden truth, drawn FIRST
    blocks, cols = [], {}
    for m in modalities:
        eff_loading = m.loading * coupling
        #per-feature random sign/scale so features are not carbon copies of z
        w = rng.normal(size=m.n_features) * eff_loading
        signal = np.outer(z, w)                 #n x n_features, each col loads on z
        if m.nonlinear:
            #a smooth nonlinearity so z is not linearly trivial to read off
            signal = np.tanh(signal) + 0.3 * signal
        noise = rng.normal(size=(n, m.n_features)) * m.noise
        block = signal + noise
        names = [f"{m.name}_f{i}" for i in range(m.n_features)]
        cols[m.name] = names
        blocks.append(pd.DataFrame(block, columns=names))
    X = pd.concat(blocks, axis=1)
    iris = (z > 0).astype(int)
    return VirtualCohort(X=X, z=z, iris=iris, modality_cols=cols)


def scramble(cohort, seed=0):
    """Negative control: independently permute each modality block's rows.

    Within-modality column structure is kept, but the modalities no longer share
    a patient's z, so fusion should gain nothing over a single modality.
    """
    rng = np.random.default_rng(seed)
    X = cohort.X.copy()
    for name, feature_cols in cohort.modality_cols.items():
        perm = rng.permutation(len(X))
        X[feature_cols] = X[feature_cols].values[perm]
    return VirtualCohort(X=X.reset_index(drop=True), z=cohort.z,
                         iris=cohort.iris, modality_cols=cohort.modality_cols)


def calibrate_from_real(real_frames, base_modalities=DEFAULT_MODALITIES):
    """Set each modality's loading from the real cross-modal signal.

    real_frames maps modality name to a (subjects x features) DataFrame for the
    real linked patients (e.g. the 22 omics+CGM). Each modality's loading is set
    proportional to how strongly its features correlate with a shared component
    (the first singular vector of the concatenated standardized blocks), keeping
    the synthetic coupling in the observed range. Modalities with no real data
    keep their default loading. Returns a new tuple of ModalitySpec.
    """
    #build a shared latent proxy from the concatenated standardized real blocks
    aligned = [f for f in real_frames.values() if f is not None and len(f)]
    if not aligned:
        return base_modalities
    n = min(len(f) for f in aligned)
    mats = []
    for f in aligned:
        m = f.iloc[:n].to_numpy(dtype=float)
        m = (m - np.nanmean(m, axis=0)) / (np.nanstd(m, axis=0) + 1e-9)
        m = np.nan_to_num(m)
        mats.append(m)
    concat = np.hstack(mats)
    #first singular vector as the shared component; project each modality onto it
    u = np.linalg.svd(concat, full_matrices=False)[0][:, 0]
    out = []
    for spec in base_modalities:
        real = real_frames.get(spec.name)
        if real is None or not len(real):
            out.append(spec)
            continue
        m = real.iloc[:n].to_numpy(dtype=float)
        m = np.nan_to_num((m - np.nanmean(m, axis=0)) / (np.nanstd(m, axis=0) + 1e-9))
        #mean absolute correlation of this modality's features with the shared u
        corrs = [abs(np.corrcoef(m[:, j], u)[0, 1]) for j in range(m.shape[1])
                 if np.std(m[:, j]) > 0]
        loading = float(np.clip(np.nanmean(corrs) if corrs else spec.loading, 0.1, 1.5))
        out.append(ModalitySpec(spec.name, spec.n_features, loading,
                                spec.noise, spec.nonlinear))
    return tuple(out)
