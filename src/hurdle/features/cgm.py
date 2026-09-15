"""Continuous-glucose-monitoring variability features (Hall 2018 CGM).

Per-subject glucose-variability metrics computed from a long-format CGM file
(5-minute glucose in mg/dL). The standard metric families are:

  dispersion   - mean, SD, CV (=SD/mean*100), IQR
  excursion    - MAGE (mean amplitude of glycemic excursions > 1 SD)
  continuity   - CONGA (SD of glucose differences n hours apart), MODD (mean
                 absolute difference between same-clock-time points 24 h apart)
  composite    - J-index, GMI (glucose management indicator)
  time-in-range- TIR (70-180), TAR (>180), TBR (<70), as fractions summing to 1
  risk         - LBGI / HBGI (Kovatchev low/high blood-glucose index)

Public API:
  extract_cgm_features(df_one_subject) -> dict   one subject's metric row
  build_cgm_matrix(path)               -> DataFrame  subjects x features
"""
import numpy as np
import pandas as pd

#defaults for the Hall 2018 dexcom feed: 5-minute sampling, mg/dL
DEFAULT_SAMPLING_MIN = 5.0
TIR_LOW = 70.0
TIR_HIGH = 180.0
CONGA_HOURS = 1.0

FEATURE_ORDER = [
    "mean_glucose", "sd_glucose", "cv_glucose", "iqr_glucose",
    "mage", "conga1", "modd", "j_index",
    "tir", "tar", "tbr",
    "lbgi", "hbgi", "gmi",
]


def _clean_series(df, glucose_col, time_col):
    #coerce glucose to numeric (Hall data carries 'Low' sentinels), drop non-finite,
    #and order by time so the difference-based metrics see chronological samples
    g = pd.to_numeric(df[glucose_col], errors="coerce")
    if time_col in df.columns:
        t = pd.to_datetime(df[time_col], errors="coerce")
        order = t.argsort(kind="stable")
        g = g.iloc[order].reset_index(drop=True)
        t = t.iloc[order].reset_index(drop=True)
    else:
        t = pd.Series([pd.NaT] * len(g))
    keep = g.notna().values
    return g[keep].reset_index(drop=True).astype(float), t[keep].reset_index(drop=True)


def _sampling_min(t):
    #median inter-sample gap in minutes; fall back to the dexcom default if the
    #timestamps are missing or unparseable
    if t.isna().all():
        return DEFAULT_SAMPLING_MIN
    dt = t.diff().dt.total_seconds().dropna()
    dt = dt[dt > 0]
    if len(dt) == 0:
        return DEFAULT_SAMPLING_MIN
    return float(np.median(dt)) / 60.0


def _mage(g, sd):
    #mean amplitude of glycemic excursions: amplitudes between consecutive
    #turning points that exceed one standard deviation of the trace
    g = np.asarray(g, dtype=float)
    if sd == 0 or len(g) < 3:
        return 0.0
    tp = [0]
    for i in range(1, len(g) - 1):
        if (g[i] - g[i - 1]) * (g[i + 1] - g[i]) < 0:
            tp.append(i)
    tp.append(len(g) - 1)
    amps = np.abs(np.diff(g[tp]))
    valid = amps[amps > sd]
    return float(valid.mean()) if len(valid) else 0.0


def _conga(g, sampling_min, hours):
    #continuous overlapping net glycemic action: SD of glucose differences
    #taken `hours` apart
    k = int(round(hours * 60.0 / sampling_min))
    if k < 1 or len(g) <= k:
        return float("nan")
    diffs = g[k:].values - g[:-k].values
    return float(np.std(diffs, ddof=1)) if len(diffs) > 1 else float("nan")


def _modd(g, sampling_min):
    #mean of daily differences: mean absolute difference between points 24 h apart
    k = int(round(24.0 * 60.0 / sampling_min))
    if k < 1 or len(g) <= k:
        return float("nan")
    diffs = np.abs(g[k:].values - g[:-k].values)
    return float(np.mean(diffs)) if len(diffs) else float("nan")


def _risk_indices(g):
    #Kovatchev symmetrised blood-glucose risk function (mg/dL form)
    g = np.asarray(g, dtype=float)
    g = np.clip(g, 1.0, None)
    f = 1.509 * (np.power(np.log(g), 1.084) - 5.381)
    rl = np.where(f < 0, 10.0 * f ** 2, 0.0)
    rh = np.where(f > 0, 10.0 * f ** 2, 0.0)
    return float(np.mean(rl)), float(np.mean(rh))


def extract_cgm_features(df_one_subject, glucose_col="GlucoseValue",
                         time_col="DisplayTime", conga_hours=CONGA_HOURS,
                         tir_low=TIR_LOW, tir_high=TIR_HIGH):
    """Compute the CGM variability metric row for one subject's glucose trace.

    Returns a dict keyed by FEATURE_ORDER. TIR/TAR/TBR are fractions of valid
    readings and sum to 1. Difference-based metrics (CONGA/MODD) return NaN only
    when the trace is shorter than the required lag.
    """
    g, t = _clean_series(df_one_subject, glucose_col, time_col)
    n = len(g)
    if n == 0:
        return {k: float("nan") for k in FEATURE_ORDER}

    mean = float(g.mean())
    sd = float(g.std(ddof=1)) if n > 1 else 0.0
    cv = sd / mean * 100.0 if mean != 0 else float("nan")
    q75, q25 = np.percentile(g, [75, 25])
    iqr = float(q75 - q25)

    sampling_min = _sampling_min(t)
    mage = _mage(g, sd)
    conga1 = _conga(g, sampling_min, conga_hours)
    modd = _modd(g, sampling_min)
    j_index = 0.001 * (mean + sd) ** 2

    tbr = float((g < tir_low).mean())
    tar = float((g > tir_high).mean())
    tir = float(((g >= tir_low) & (g <= tir_high)).mean())

    lbgi, hbgi = _risk_indices(g)
    gmi = 3.31 + 0.02392 * mean

    return {
        "mean_glucose": mean, "sd_glucose": sd, "cv_glucose": cv, "iqr_glucose": iqr,
        "mage": mage, "conga1": conga1, "modd": modd, "j_index": j_index,
        "tir": tir, "tar": tar, "tbr": tbr,
        "lbgi": lbgi, "hbgi": hbgi, "gmi": gmi,
    }


def _read_cgm(path):
    #tab-separated in the Hall feed; sniff comma as a fallback
    df = pd.read_csv(path, sep="\t", dtype=str)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",", dtype=str)
    return df


def build_cgm_matrix(path, subject_col="subjectId", glucose_col="GlucoseValue",
                     time_col="DisplayTime", **kwargs):
    """Read a long-format CGM file and return a subjects x features DataFrame.

    Groups on `subject_col` (e.g. '1636-69-001') and applies
    extract_cgm_features per subject. The index is the subject id; columns are
    FEATURE_ORDER.
    """
    df = _read_cgm(path)
    missing = {subject_col, glucose_col} - set(df.columns)
    if missing:
        raise ValueError(f"CGM file {path} missing columns {sorted(missing)}; "
                         f"found {list(df.columns)}")
    rows, index = [], []
    for sid, sub in df.groupby(subject_col, sort=True):
        rows.append(extract_cgm_features(sub, glucose_col=glucose_col,
                                         time_col=time_col, **kwargs))
        index.append(sid)
    out = pd.DataFrame(rows, index=pd.Index(index, name=subject_col))
    return out[FEATURE_ORDER]
