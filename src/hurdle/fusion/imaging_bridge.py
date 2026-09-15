"""Bridge the real retinal CNN into the fusion layer.

train_imaging.py exports per-image embeddings (models/imaging_embeddings.npz).
This module turns those raw embeddings (e.g. 2048-dim for ResNet-50) into a
compact imaging feature block and uses their statistics to anchor the virtual
cohort's imaging modality — so the imaging block is driven by the REAL CNN, not
a purely synthetic draw. The per-patient join remains virtual (RetinaMNIST is
anonymized and shares no patients with omics/CGM), and that is stated plainly.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_embeddings(npz_path, split="train"):
    #load one split's embedding matrix (n_images x embed_dim) from the export
    data = np.load(npz_path)
    if split not in data:
        raise KeyError(f"split {split!r} not in {list(data.keys())}")
    return data[split]


def reduce_embeddings(emb, n_components=6, seed=0):
    """Standardize then PCA-reduce raw CNN embeddings to n_components imaging
    features. Returns (reduced matrix, fitted scaler, fitted pca) so the same
    transform can be reapplied to other splits.
    """
    scaler = StandardScaler().fit(emb)
    z = scaler.transform(emb)
    k = min(n_components, z.shape[1], z.shape[0])
    pca = PCA(n_components=k, random_state=seed).fit(z)
    return pca.transform(z), scaler, pca


def imaging_features_frame(npz_path, n_components=6, split="train", seed=0):
    #a real imaging feature block (images x components) from the CNN embeddings
    emb = load_embeddings(npz_path, split=split)
    reduced, _, _ = reduce_embeddings(emb, n_components=n_components, seed=seed)
    cols = [f"imaging_f{i}" for i in range(reduced.shape[1])]
    return pd.DataFrame(reduced, columns=cols)


def real_frames_with_imaging(omics_df=None, cgm_df=None, imaging_npz=None,
                             n_components=6, split="train", seed=0):
    """Assemble the real_frames dict for virtual_cohort.calibrate_from_real,
    including the real CNN imaging block when an embeddings file is given.
    Modalities that are None are simply omitted (their loading stays default).
    """
    frames = {}
    if omics_df is not None:
        frames["omics"] = omics_df
    if cgm_df is not None:
        frames["cgm"] = cgm_df
    if imaging_npz is not None:
        frames["imaging"] = imaging_features_frame(
            imaging_npz, n_components=n_components, split=split, seed=seed)
    return frames
