"""Tests for wearable per-subject feature extraction.

All tests run on a tiny synthetic 1-minute series built in-process with a known
sinusoidal HR, so the suite is fast and never touches the 43 real parquets.
"""
import numpy as np
import pandas as pd
import pytest

from hurdle.features.wearable import (
    FEATURE_COLUMNS,
    _cosinor,
    build_wearable_matrix,
    extract_features,
)

INJECT_MESOR = 65.0
INJECT_AMP = 12.0
INJECT_ACRO_H = 16.0                #hr peaks at 16:00 each day
N_DAYS = 4


def _synth_frame(n_days=N_DAYS, mesor=INJECT_MESOR, amp=INJECT_AMP,
                 acro_h=INJECT_ACRO_H, seed=0):
    #a regular 1-min UTC series: hr is a 24h cosine peaking at acro_h; accel is
    #low at night and high midday so activity/RA have real structure; skin_temp
    #is a gentle antiphase cosine. small noise keeps stds nonzero.
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2015-01-01 00:00:00', periods=n_days * 1440,
                        freq='1min', tz='UTC')
    h = idx.hour + idx.minute / 60.0
    w = 2 * np.pi / 24.0
    hr = mesor + amp * np.cos(w * (h - acro_h)) + 0.5 * rng.normal(size=len(idx))
    #accel: high around midday (10-18h), low at night -> clear day/night contrast
    accel = 30 + 900 * np.clip(np.cos(w * (h - 14.0)), 0, None) \
        + 20 * rng.random(len(idx))
    skin_temp = 92.0 - 2.0 * np.cos(w * (h - 4.0)) + 0.2 * rng.normal(size=len(idx))
    return pd.DataFrame({'hr': hr, 'accel_magnitude': accel, 'skin_temp': skin_temp},
                        index=idx)


@pytest.fixture
def synth_parquet(tmp_path):
    #write the synthetic frame as Basis_999.parquet so extract_features reads it
    #the same way it reads real subjects (subject id = file stem)
    p = tmp_path / 'Basis_999.parquet'
    _synth_frame().to_parquet(p)
    return p


def test_cosinor_recovers_injected_signal():
    #direct cosinor on the clean sinusoid recovers amplitude and peak time
    f = _synth_frame()
    hod = f.index.hour + f.index.minute / 60.0
    mesor, amp, acro = _cosinor(hod, f['hr'].values)
    assert mesor == pytest.approx(INJECT_MESOR, abs=1.0)
    assert amp == pytest.approx(INJECT_AMP, rel=0.15)
    assert acro == pytest.approx(INJECT_ACRO_H, abs=1.0)


def test_extract_features_row_has_no_nan(synth_parquet):
    row = extract_features(synth_parquet)
    for k in FEATURE_COLUMNS:
        assert k in row, f"missing feature {k}"
        assert np.isfinite(row[k]), f"non-finite feature {k}={row[k]}"


def test_extract_features_column_contract(synth_parquet):
    #the row carries exactly the contract columns plus the subject id, nothing else
    row = extract_features(synth_parquet)
    assert set(row) == set(FEATURE_COLUMNS) | {'subject'}
    assert row['subject'] == 'Basis_999'


def test_cosinor_amplitude_in_feature_row(synth_parquet):
    #the pipeline-level cosinor amplitude matches the injected amplitude too
    row = extract_features(synth_parquet)
    assert row['hr_amplitude'] == pytest.approx(INJECT_AMP, rel=0.2)
    assert row['hr_mesor'] == pytest.approx(INJECT_MESOR, abs=1.5)
    assert 0.0 <= row['hr_acrophase_h'] < 24.0


def test_coverage_math_full_wear(synth_parquet):
    #fully-worn synthetic series: worn minutes = rows, coverage ~= 1, worn_days
    #= minutes/1440. span is (n_days*1440 - 1) minutes so coverage is just under 1.
    row = extract_features(synth_parquet)
    assert row['worn_minutes'] == N_DAYS * 1440
    assert row['worn_days'] == pytest.approx(N_DAYS, abs=1e-6)
    assert row['coverage_fraction'] == pytest.approx(1.0, abs=0.01)
    assert row['coverage_fraction'] <= 1.0 + 1e-9


def test_coverage_fraction_drops_with_gaps(tmp_path):
    #drop the second half of each day: worn minutes halve. coverage is worn/span
    #and the span still ends at the last kept minute (day-4 11:59), so it lands
    #near 0.57, not 0.5 -- and must be strictly below the full-wear coverage.
    f = _synth_frame()
    keep = f.index.hour < 12
    f = f[keep]
    p = tmp_path / 'Basis_998.parquet'
    f.to_parquet(p)
    row = extract_features(p)
    assert row['worn_minutes'] == int(keep.sum())
    assert 0.5 <= row['coverage_fraction'] < 0.65


def test_relative_amplitude_positive_with_day_night_contrast(synth_parquet):
    #accel is high midday, low at night -> RA is clearly positive and bounded
    row = extract_features(synth_parquet)
    assert 0.0 < row['accel_ra'] <= 1.0


def test_interdaily_stability_high_for_repeating_days(synth_parquet):
    #every synthetic day shares the same 24h HR shape, so IS is near its ceiling
    #of 1 and IV (fragmentation) stays low
    row = extract_features(synth_parquet)
    assert row['hr_is'] > 0.7
    assert 0.0 <= row['hr_is'] <= 1.0 + 1e-6
    assert row['hr_iv'] >= 0.0


def test_hr_dip_positive(synth_parquet):
    #nightly HR low sits below the daily mean, so the dip is positive
    row = extract_features(synth_parquet)
    assert row['hr_dip'] > 0
    assert row['resting_hr'] < row['hr_mean']


def test_build_matrix_shape_and_columns(tmp_path):
    #two synthetic subjects -> 2 rows, exact feature columns, subject index
    for name, seed in (('Basis_101', 1), ('Basis_102', 2)):
        _synth_frame(seed=seed).to_parquet(tmp_path / f'{name}.parquet')
    #a stray non-parquet sidecar must be ignored
    (tmp_path / '_wear_summary.csv').write_text('subject,worn_minutes\n')
    mat = build_wearable_matrix(tmp_path)
    assert mat.shape == (2, len(FEATURE_COLUMNS))
    assert list(mat.columns) == list(FEATURE_COLUMNS)
    assert list(mat.index) == ['Basis_101', 'Basis_102']
    assert np.isfinite(mat.values).all()


def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_wearable_matrix(tmp_path / 'does_not_exist')
