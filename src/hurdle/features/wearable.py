"""Per-subject wearable feature engineering for the Basis-watch cohort.

Input is one parquet per subject under data/interim/wearable/: a 1-minute
resampled frame with a tz-aware UTC DatetimeIndex and columns hr (bpm),
accel_magnitude (device counts), skin_temp (degrees Fahrenheit). Only worn
minutes are present as rows; gaps in the index are non-wear.

The feature set goes well beyond means/stds, capturing temporal structure:
  circadian   - cosinor MESOR/amplitude/acrophase for hr and skin_temp,
                interdaily stability (IS), intradaily variability (IV),
                relative amplitude (RA)
  sleep/rest  - nightly minimum-HR window, resting HR, HR dip to nightly low
  activity    - MVPA-proxy minutes, sedentary minutes, sedentary-bout length,
                activity fragmentation
  hr dynamics - SDNN-like and RMSSD-like variability of the 1-min HR series,
                day-to-day regularity of the daily HR profile
  wear        - worn minutes, worn days, recording span, coverage fraction

extract_features(parquet_path) -> dict is one subject (one row);
build_wearable_matrix(interim_wear_dir) -> DataFrame stacks subjects x features.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

#device-specific accel_magnitude cutpoints (Basis watch counts). subject
#distributions differ materially (observed medians ~145-240, .90 ~1200-1450), but
#that spread is genuine between-subject activity signal we want to keep: the
#cohort shares one device and units, so a fixed absolute cutpoint measures the
#same physical movement for everyone, whereas within-subject percentiles would
#normalise away the very differences the features exist to capture. the near-floor
#(resting/still ~20-25 counts) is stable across subjects; cutpoints sit well above
#it (sedentary) and in the sustained-movement range (mvpa).
SEDENTARY_ACCEL = 100.0             #below this a minute counts as sedentary
MVPA_ACCEL = 800.0                  #above this a minute counts as MVPA-proxy
NIGHT_WINDOW_MIN = 30               #rolling window for the nightly HR low
RESTING_HR_PCT = 5                  #percentile of hr taken as resting HR
MIN_PER_DAY = 1440

#the columns extract_features returns, in stable order; build_wearable_matrix
#and the tests rely on this contract not drifting silently.
FEATURE_COLUMNS = (
    'worn_minutes', 'worn_days', 'span_days', 'coverage_fraction',
    'hr_mean', 'hr_std', 'skin_temp_mean', 'accel_mean',
    'hr_mesor', 'hr_amplitude', 'hr_acrophase_h',
    'temp_mesor', 'temp_amplitude', 'temp_acrophase_h',
    'hr_is', 'hr_iv', 'accel_is', 'accel_iv', 'accel_ra',
    'resting_hr', 'nightly_min_hr', 'hr_dip', 'hr_dip_frac',
    'mvpa_min_per_day', 'sedentary_min_per_day', 'sedentary_bout_mean_min',
    'activity_fragmentation',
    'hr_sdnn', 'hr_rmssd', 'hr_within_day_std', 'hr_day_to_day_cv',
    'hr_profile_regularity',
)


def _safe(value, fallback=0.0):
    #coerce a scalar to a finite float so the output row never carries NaN/inf
    try:
        v = float(value)
    except (TypeError, ValueError):
        return fallback
    return v if math.isfinite(v) else fallback


def _cosinor(hour_of_day, values, period=24.0):
    #single-component cosinor by OLS on [1, cos(wt), sin(wt)]; returns
    #(mesor, amplitude, acrophase_hours). acrophase is the peak time in hours.
    t = np.asarray(hour_of_day, dtype=float)
    y = np.asarray(values, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(y) < 3:
        return _safe(y.mean() if len(y) else 0.0), 0.0, 0.0
    w = 2.0 * math.pi / period
    X = np.column_stack([np.ones_like(t), np.cos(w * t), np.sin(w * t)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    mesor, b_cos, b_sin = coef
    amplitude = math.hypot(b_cos, b_sin)
    #model = M + A*cos(w t - psi) with psi = atan2(b_sin, b_cos); peak at t=psi/w
    acrophase = (math.atan2(b_sin, b_cos) % (2.0 * math.pi)) / w
    return _safe(mesor), _safe(amplitude), _safe(acrophase)


def _is_iv(hourly):
    #interdaily stability and intradaily variability on an hourly-resampled
    #series (regular grid, NaN for missing hours). IS in ~[0,1] (higher = more
    #day-to-day reproducible); IV in ~[0,2] (higher = more fragmented/noisy).
    x = np.asarray(hourly, dtype=float)
    finite = x[np.isfinite(x)]
    n = len(finite)
    if n < 24:
        return 0.0, 0.0
    xbar = finite.mean()
    tss = np.sum((finite - xbar) ** 2)
    if tss <= 0:
        return 0.0, 0.0
    hod = (np.arange(len(x)) % 24)
    prof = np.array([np.nanmean(x[hod == h]) if np.isfinite(x[hod == h]).any()
                     else np.nan for h in range(24)])
    pv = prof[np.isfinite(prof)]
    #Van Someren IS = (N * sum_h (mean_h - xbar)^2) / (p * sum_i (x_i - xbar)^2),
    #N = total data points, p = bins/day (24). the numerator scales by N, not by
    #the bin count, or IS collapses toward 0.
    is_num = n * np.sum((pv - xbar) ** 2)
    is_val = is_num / (24.0 * tss)
    d = np.diff(x)
    d = d[np.isfinite(d)]
    iv_val = (n * np.sum(d ** 2)) / ((len(d)) * tss) if len(d) else 0.0
    return _safe(is_val), _safe(iv_val)


def _relative_amplitude(hourly, most=10, least=5):
    #RA = (M10 - L5)/(M10 + L5) from the 24h profile of hourly means; the
    #contrast between the most-active M-hours and least-active L-hours.
    x = np.asarray(hourly, dtype=float)
    hod = (np.arange(len(x)) % 24)
    prof = np.array([np.nanmean(x[hod == h]) if np.isfinite(x[hod == h]).any()
                     else np.nan for h in range(24)])
    prof = prof[np.isfinite(prof)]
    if len(prof) < most + least:
        return 0.0
    s = np.sort(prof)
    l5 = s[:least].mean()
    m10 = s[-most:].mean()
    denom = m10 + l5
    return _safe((m10 - l5) / denom) if denom > 0 else 0.0


def _bout_stats(flags):
    #run-length stats over a boolean sequence in row order; returns
    #(mean True-run length, number of True runs). a time gap between rows still
    #ends a run only if the flag flips, which is the conservative choice.
    f = np.asarray(flags, dtype=bool)
    if f.size == 0:
        return 0.0, 0
    edges = np.diff(f.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if f[0]:
        starts = np.r_[0, starts]
    if f[-1]:
        ends = np.r_[ends, f.size]
    lengths = ends - starts
    return (float(lengths.mean()) if len(lengths) else 0.0), int(len(lengths))


def extract_features(parquet_path):
    #one subject -> one feature row (dict). grounds every value in the parquet;
    #never invents data. subject id is the file stem (e.g. 'Basis_001').
    parquet_path = Path(parquet_path)
    df = pd.read_parquet(parquet_path)
    subject = parquet_path.stem

    hr = df['hr'].astype(float)
    temp = df['skin_temp'].astype(float)
    accel = df['accel_magnitude'].astype(float)
    idx = df.index

    #wear/coverage: rows are worn minutes; span is first-to-last timestamp
    worn_minutes = int(hr.notna().sum())
    span_days = _safe((idx.max() - idx.min()).total_seconds() / 86400.0) if len(idx) > 1 else 0.0
    worn_days = worn_minutes / MIN_PER_DAY
    #+1 makes the span endpoint-inclusive (n points span n-1 intervals), so
    #fully-worn wear lands at exactly 1.0 rather than a hair above it
    denom_min = span_days * MIN_PER_DAY + 1
    coverage_fraction = _safe(worn_minutes / denom_min) if denom_min > 0 else 0.0

    #decimal hour-of-day for cosinor
    hod = idx.hour + idx.minute / 60.0 + idx.second / 3600.0
    hr_mesor, hr_amp, hr_acro = _cosinor(hod, hr.values)
    temp_mesor, temp_amp, temp_acro = _cosinor(hod, temp.values)

    #hourly-resampled regular grids for IS/IV/RA
    hr_hourly = hr.resample('1h').mean()
    accel_hourly = accel.resample('1h').mean()
    hr_is, hr_iv = _is_iv(hr_hourly.values)
    accel_is, accel_iv = _is_iv(accel_hourly.values)
    accel_ra = _relative_amplitude(accel_hourly.values)

    #sleep/rest proxy: resting HR from a low percentile; nightly low from the
    #per-day minimum of a rolling window; dip = distance from mean to that low
    resting_hr = _safe(np.nanpercentile(hr.values, RESTING_HR_PCT))
    roll = hr.rolling(NIGHT_WINDOW_MIN, min_periods=NIGHT_WINDOW_MIN // 2).mean()
    daily_min = roll.groupby(roll.index.date).min()
    nightly_min_hr = _safe(daily_min.mean())
    hr_mean = _safe(hr.mean())
    hr_dip = _safe(hr_mean - nightly_min_hr)
    hr_dip_frac = _safe(hr_dip / hr_mean) if hr_mean > 0 else 0.0

    #activity: absolute-cutpoint minutes normalised per worn day, plus bout
    #structure and fragmentation (active bouts per active minute)
    a = accel.dropna()
    sed_flags = (a < SEDENTARY_ACCEL).values
    mvpa_min = int((a > MVPA_ACCEL).sum())
    sed_min = int(sed_flags.sum())
    day_norm = worn_days if worn_days > 0 else 1.0
    mvpa_min_per_day = _safe(mvpa_min / day_norm)
    sedentary_min_per_day = _safe(sed_min / day_norm)
    sed_bout_mean, _ = _bout_stats(sed_flags)
    act_bout_mean, n_act_bouts = _bout_stats(~sed_flags)
    active_min = int((~sed_flags).sum())
    activity_fragmentation = _safe(n_act_bouts / active_min) if active_min > 0 else 0.0

    #hr dynamics: overall SDNN-like spread, RMSSD-like successive-minute
    #variability (only across true 1-min steps), and within-/between-day spread
    hr_std = _safe(hr.std())
    step = idx.to_series().diff().dt.total_seconds().values
    dhr = hr.diff().values
    consec = (step == 60.0) & np.isfinite(dhr)
    hr_rmssd = _safe(math.sqrt(np.mean(dhr[consec] ** 2))) if consec.any() else 0.0
    by_day = hr.groupby(hr.index.date)
    hr_within_day_std = _safe(by_day.std().mean())
    daily_mean = by_day.mean()
    hr_day_to_day_cv = (_safe(daily_mean.std() / daily_mean.mean())
                        if len(daily_mean) > 1 and daily_mean.mean() > 0 else 0.0)

    #day-to-day regularity: mean correlation of each day's 24-bin HR profile with
    #the cohort-subject's overall mean profile (1.0 = perfectly repeating day)
    hr_profile_regularity = _profile_regularity(hr)

    row = {
        'subject': subject,
        'worn_minutes': worn_minutes,
        'worn_days': _safe(worn_days),
        'span_days': span_days,
        'coverage_fraction': coverage_fraction,
        'hr_mean': hr_mean,
        'hr_std': hr_std,
        'skin_temp_mean': _safe(temp.mean()),
        'accel_mean': _safe(accel.mean()),
        'hr_mesor': hr_mesor,
        'hr_amplitude': hr_amp,
        'hr_acrophase_h': hr_acro,
        'temp_mesor': temp_mesor,
        'temp_amplitude': temp_amp,
        'temp_acrophase_h': temp_acro,
        'hr_is': hr_is,
        'hr_iv': hr_iv,
        'accel_is': accel_is,
        'accel_iv': accel_iv,
        'accel_ra': accel_ra,
        'resting_hr': resting_hr,
        'nightly_min_hr': nightly_min_hr,
        'hr_dip': hr_dip,
        'hr_dip_frac': hr_dip_frac,
        'mvpa_min_per_day': mvpa_min_per_day,
        'sedentary_min_per_day': sedentary_min_per_day,
        'sedentary_bout_mean_min': _safe(sed_bout_mean),
        'activity_fragmentation': activity_fragmentation,
        'hr_sdnn': hr_std,
        'hr_rmssd': hr_rmssd,
        'hr_within_day_std': hr_within_day_std,
        'hr_day_to_day_cv': hr_day_to_day_cv,
        'hr_profile_regularity': hr_profile_regularity,
    }
    return row


def _profile_regularity(series):
    #mean Pearson correlation of each day's hourly profile against the overall
    #mean hourly profile; a scale-free regularity score in ~[-1, 1]
    s = series.dropna()
    if s.empty:
        return 0.0
    hourly = s.resample('1h').mean()
    frame = pd.DataFrame({'v': hourly})
    frame['date'] = frame.index.date
    frame['hod'] = frame.index.hour
    prof = frame.pivot_table(index='date', columns='hod', values='v')
    if prof.shape[0] < 2:
        return 0.0
    overall = prof.mean(axis=0)
    corrs = []
    for _, day in prof.iterrows():
        ok = day.notna() & overall.notna()
        if ok.sum() >= 3 and day[ok].std() > 0 and overall[ok].std() > 0:
            corrs.append(np.corrcoef(day[ok].values, overall[ok].values)[0, 1])
    return _safe(np.mean(corrs)) if corrs else 0.0


def build_wearable_matrix(interim_wear_dir):
    #stack all per-subject parquet rows into a subjects x features DataFrame,
    #indexed by subject. skips the _wear_summary.csv sidecar.
    interim_wear_dir = Path(interim_wear_dir)
    files = sorted(p for p in interim_wear_dir.glob('*.parquet'))
    if not files:
        raise FileNotFoundError(f"no parquet files in {interim_wear_dir}")
    rows = [extract_features(p) for p in files]
    mat = pd.DataFrame(rows).set_index('subject')
    return mat[list(FEATURE_COLUMNS)]
