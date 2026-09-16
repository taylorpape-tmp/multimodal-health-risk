"""Smoke tests for the HURDLE Streamlit demo UI.

These use streamlit.testing.v1.AppTest to run the app headlessly. The app is now
two tabs:
  Tab 1 "Real result (omics -> SSPG)" -- leave-one-out XGBoost on 59 real
        patients / real measured SSPG. Must compute a finite R2 metric.
  Tab 2 "Fusion demo (virtual cohort)" -- the slider playground; Predict must
        still run and emit a finite numeric fused score.

AppTest renders the content of ALL tabs in one pass (tabs are not lazily
evaluated by the headless runner), so widgets and metrics from both tabs are
reachable on the elements lists. Kept fast by relying on the app's caches.
"""
import re
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")

#generous timeout: first run computes the 59-patient LOO loop (~20-60s) and
#builds the virtual-cohort predictor before either is cached
_TIMEOUT = 180


def _run():
    #run the app headlessly with a timeout that covers the first LOO + build
    at = AppTest.from_file(APP_PATH, default_timeout=_TIMEOUT)
    at.run()
    return at


def _numeric_metric_values(at):
    #metric values that parse as finite floats (strip units like " mg/dL")
    out = []
    for m in at.metric:
        v = m.value
        if not isinstance(v, str):
            continue
        match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", v)
        if match:
            try:
                out.append(float(match.group()))
            except ValueError:
                pass
    return out


def test_app_loads_without_exception():
    #the app renders and surfaces the honesty banner, no uncaught exception
    at = _run()
    assert not at.exception
    banners = " ".join(w.value for w in at.warning)
    assert "NOT a validated clinical tool" in banners


def test_two_tabs_present_with_provenance_labels():
    #the restructured app exposes exactly the two named tabs
    at = _run()
    assert not at.exception
    tab_labels = [t.label for t in at.tabs]
    assert "Real result (omics -> SSPG)" in tab_labels
    assert "Fusion demo (virtual cohort)" in tab_labels


def test_tab1_real_metrics_compute_finite_r2():
    #Tab 1 must compute the real leave-one-out result: a finite R2 metric
    at = _run()
    assert not at.exception

    #the real-data provenance is labelled explicitly (green success banner)
    success_text = " ".join(s.value for s in at.success)
    assert "REAL DATA" in success_text

    #an "R2 (leave-one-out)" metric is rendered with a finite numeric value
    r2_metrics = [m for m in at.metric if "R2" in str(m.label)]
    assert r2_metrics, "no R2 metric rendered in the real-data tab"
    r2_val = float(r2_metrics[0].value)
    assert np.isfinite(r2_val), f"R2 metric not finite: {r2_metrics[0].value}"
    #real omics -> SSPG lands ~0.46-0.50; assert it's a sane, positive R2
    assert 0.2 < r2_val < 0.9, f"real LOO R2 out of expected range: {r2_val}"


def test_predict_button_produces_finite_risk():
    #clicking Predict (Tab 2) runs the fusion and emits a finite fused score
    at = _run()
    assert not at.exception
    assert len(at.button) >= 1
    at.button[0].click().run()
    assert not at.exception

    metric_values = _numeric_metric_values(at)
    assert metric_values, "no numeric metrics rendered after predict"
    assert any(np.isfinite(v) for v in metric_values), \
        f"no finite numeric risk in {metric_values}"


def test_predict_tier_is_valid_label():
    #the risk tier metric is one of the three SSPG-tertile labels
    at = _run()
    at.button[0].click().run()
    assert not at.exception
    tiers = [m.value for m in at.metric]
    assert any(v in {"Low", "Moderate", "High"} for v in tiers)
