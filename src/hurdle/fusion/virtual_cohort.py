"""Virtual-cohort generator for the multimodal fusion demonstration.

THE HONEST FRAMING
------------------
The four public datasets are different people. Omics and CGM share 22 real
patients (via the crosswalk); wearable and retinal imaging share no patients
with anyone. So a real all-four-modalities-per-patient matrix does not exist in
public data. This module constructs a VIRTUAL cohort — synthetic patients that
carry all four modalities at once — so the fusion pipeline can be demonstrated
end to end. It is explicitly synthetic and never presented as real individuals.

WHY IT IS NOT CIRCULAR
----------------------
The naive approach (invent a risk score by summing the modalities, then train a
model to predict it) is circular: the model just re-learns the sum. Instead:

  1. draw a hidden latent risk z ~ N(0,1) for each synthetic patient FIRST
  2. generate every modality's features FROM z through noisy, calibrated maps
     (high z -> more insulin-resistant omics, more glucose variability, worse
     retinal grade) — noisy and nonlinear so no single modality reveals z
  3. the model sees ONLY the generated features, never z
  4. truth = z, which was set before any feature existed and is hidden

The model must FUSE the noisy modalities to recover z — a genuine task. The
coupling strengths are calibrated to the real 22 where cross-modal correlation
can actually be measured, so the synthetic patients are not arbitrary.

Controls that prove fusion does real work live in fusion.py:
  - ablation: fusion must beat the best single modality
  - scramble: break the shared-z coupling -> the advantage must vanish
  - additive baseline: a hand-summed score is the baseline to beat, never a label
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

    coupling scales every modality's loading at once; coupling=0 reproduces the
    scramble control (features independent of z). Returns a VirtualCohort.
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
    """Break the shared-z coupling: independently permute the ROWS of each
    modality block. Column structure within a modality is preserved, but the
    modalities no longer share a patient's z -> fusion should gain nothing.
    Truth z stays aligned to nothing in particular, so we also return shuffled z
    for the block that keeps its own rows. Used as the negative control.
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

    real_frames maps modality name -> a (subjects x features) DataFrame for the
    REAL linked patients (e.g. the 22 omics+CGM). We measure how strongly each
    modality's leading principal direction correlates with a shared component
    (here approximated by the first singular vector of the concatenated,
    standardized real blocks) and set the loading proportional to that. This
    keeps the synthetic coupling in the range actually observed rather than
    invented. Modalities with no real linked data keep their default loading.

    Returns a new tuple of ModalitySpec with calibrated loadings.
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
