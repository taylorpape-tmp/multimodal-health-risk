"""Time-series foundation-model embeddings for wearable and CGM traces.

The hand-crafted feature paths (wearable.py, cgm.py) reduce each subject's raw
sequence to a handful of scalar summaries (cosinor, IS/IV/RA, MAGE/CONGA/MODD).
That summarisation may discard sequential structure. This module offers the
complementary path: a pretrained time-series foundation model embeds the RAW
univariate sequence natively into a fixed-length vector, which downstream
tabular models can consume alongside (or instead of) the hand-crafted features.

Model: AutonLab/MOMENT-1-small (a ~40M-parameter T5-style encoder pretrained
for time-series representation). It takes a univariate series of a FIXED length
(512) and returns a 512-dim embedding. It applies reversible instance
normalisation (RevIN) internally, so raw un-normalised values may be passed in.

Preprocessing (documented, because the choice matters):
  embed_series resamples ANY 1-D numeric series to the model's 512-sample input.
  Downsampling uses contiguous bin means (anti-aliased); upsampling uses linear
  interpolation. build_ts_embeddings does the modality-specific pre-resampling
  BEFORE that step:
    wearable - the worn HR series is resampled to HOURLY means first, so the
               512-sample window preserves multi-day circadian structure rather
               than aliasing minute-level noise.
    cgm      - the cleaned, time-ordered glucose trace (5-min Dexcom) is passed
               as-is and binned to 512, keeping excursion-scale detail.

Public API:
  load_ts_model(name)                 -> load+cache the foundation model once
  embed_series(model, series_1d, ...) -> fixed-length embedding for one series
  build_ts_embeddings(wear_dir, cgm)  -> per-subject embedding frames + parquets
"""
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MODEL = "AutonLab/MOMENT-1-small"
FALLBACK_MODEL = "amazon/chronos-bolt-small"
SEQ_LEN = 512                       #MOMENT-1 fixed univariate input length
EMBED_DIM = 512                     #MOMENT-1-small embedding width
EMBED_PREFIX = "ts"                 #embedding columns are ts000..ts511

#module-level cache so repeated calls reuse one loaded model (weights load once)
_MODEL_CACHE = {}


def load_ts_model(name=DEFAULT_MODEL):
    #load the foundation model once and cache it by name. MOMENT is loaded in
    #'embedding' task mode and set to eval so the forward pass is deterministic.
    #torch/momentfm are imported lazily so importing this module stays cheap and
    #does not require the heavy deps until an embedding is actually requested.
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]
    if name != DEFAULT_MODEL:
        raise ValueError(
            f"only {DEFAULT_MODEL} is wired up here; got {name!r}. "
            f"the documented fallback is {FALLBACK_MODEL} (different API)."
        )
    import torch
    from momentfm import MOMENTPipeline

    model = MOMENTPipeline.from_pretrained(
        name, model_kwargs={"task_name": "embedding"}
    )
    model.init()
    model.eval()
    #pin threads off the grad path; eval+no_grad already gives determinism
    torch.set_grad_enabled(False)
    _MODEL_CACHE[name] = model
    return model


def _resample_to_length(values, target_len=SEQ_LEN):
    #resample a 1-D numeric series to exactly target_len samples. non-finite
    #values are dropped first. downsampling averages contiguous bins (anti-alias);
    #upsampling interpolates linearly; an empty series returns zeros; a single
    #finite value returns that constant. index-space resampling (not time-space):
    #the caller is responsible for any time-regularisation beforehand.
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.zeros(target_len, dtype=float)
    if n == 1:
        return np.full(target_len, x[0], dtype=float)
    if n == target_len:
        return x
    if n > target_len:
        #contiguous near-equal bins; mean within each bin
        edges = np.linspace(0, n, target_len + 1).astype(int)
        out = np.empty(target_len, dtype=float)
        for i in range(target_len):
            lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
            out[i] = x[lo:hi].mean()
        return out
    #upsample: linear interpolation onto a denser regular grid
    old = np.linspace(0.0, 1.0, n)
    new = np.linspace(0.0, 1.0, target_len)
    return np.interp(new, old, x)


def embed_series(model, series_1d, seq_len=SEQ_LEN, name=DEFAULT_MODEL):
    """Embed one univariate series into a fixed-length vector.

    The series is resampled to `seq_len` samples (see _resample_to_length),
    shaped to MOMENT's [batch=1, channel=1, seq_len] input, and passed through
    the model in eval/no-grad mode. Returns a 1-D float64 numpy array of length
    EMBED_DIM. Deterministic for the same input; always finite (a constant or
    empty series still yields a finite, fixed-length vector).
    """
    import torch

    resampled = _resample_to_length(series_1d, seq_len)
    x = torch.tensor(resampled, dtype=torch.float32).reshape(1, 1, seq_len)
    mask = torch.ones(1, seq_len, dtype=torch.float32)
    with torch.no_grad():
        out = model(x_enc=x, input_mask=mask)
    emb = out.embeddings.reshape(-1).to(torch.float64).cpu().numpy()
    #guard: replace any non-finite entry (e.g. from a degenerate input) with 0
    return np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)


def _embed_columns(dim=EMBED_DIM):
    #stable embedding column names ts000..ts(dim-1)
    return [f"{EMBED_PREFIX}{i:03d}" for i in range(dim)]


def _wearable_hr_hourly(parquet_path):
    #one wearable subject -> the hourly-mean HR series (a regular time grid that
    #preserves circadian shape) as a 1-D array in chronological order
    df = pd.read_parquet(parquet_path)
    hr = df["hr"].astype(float)
    hourly = hr.resample("1h").mean()
    return hourly.values


def _cgm_glucose_trace(sub_df, glucose_col="GlucoseValue", time_col="DisplayTime"):
    #one CGM subject -> the cleaned, time-ordered glucose trace as a 1-D array.
    #reuses the same cleaning contract as cgm.extract_cgm_features.
    from .cgm import _clean_series

    g, _t = _clean_series(sub_df, glucose_col, time_col)
    return g.values


def build_ts_embeddings(interim_wear_dir, cgm_path,
                        model=None, name=DEFAULT_MODEL,
                        wear_out=None, cgm_out=None,
                        cgm_subject_col="subjectId",
                        cgm_glucose_col="GlucoseValue",
                        cgm_time_col="DisplayTime"):
    """Embed every wearable and CGM subject; return (wear_df, cgm_df).

    One row per subject, EMBED_DIM embedding columns (ts000..). The wearable
    frame is indexed by file stem (Basis_NNN); the CGM frame by subjectId. When
    wear_out/cgm_out are given the frames are written there as parquet.
    """
    if model is None:
        model = load_ts_model(name)

    interim_wear_dir = Path(interim_wear_dir)
    cols = _embed_columns()

    #--- wearable: hourly-mean HR series per subject -------------------------
    wear_files = sorted(interim_wear_dir.glob("*.parquet"))
    if not wear_files:
        raise FileNotFoundError(f"no parquet files in {interim_wear_dir}")
    wear_rows, wear_idx = [], []
    for p in wear_files:
        series = _wearable_hr_hourly(p)
        wear_rows.append(embed_series(model, series, name=name))
        wear_idx.append(p.stem)
    wear_df = pd.DataFrame(wear_rows, index=pd.Index(wear_idx, name="subject"),
                           columns=cols)

    #--- cgm: cleaned glucose trace per subject -----------------------------
    from .cgm import _read_cgm

    cgm_raw = _read_cgm(cgm_path)
    if cgm_subject_col not in cgm_raw.columns:
        raise ValueError(f"CGM file {cgm_path} missing {cgm_subject_col!r}")
    cgm_rows, cgm_idx = [], []
    for sid, sub in cgm_raw.groupby(cgm_subject_col, sort=True):
        series = _cgm_glucose_trace(sub, cgm_glucose_col, cgm_time_col)
        cgm_rows.append(embed_series(model, series, name=name))
        cgm_idx.append(sid)
    cgm_df = pd.DataFrame(cgm_rows, index=pd.Index(cgm_idx, name=cgm_subject_col),
                          columns=cols)

    if wear_out is not None:
        Path(wear_out).parent.mkdir(parents=True, exist_ok=True)
        wear_df.to_parquet(wear_out)
    if cgm_out is not None:
        Path(cgm_out).parent.mkdir(parents=True, exist_ok=True)
        cgm_df.to_parquet(cgm_out)

    return wear_df, cgm_df
