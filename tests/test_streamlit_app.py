"""Smoke tests for the HURDLE Streamlit demo UI.

These use streamlit.testing.v1.AppTest to run the app headlessly: the app must
load without raising, the Predict button must run, and a finite numeric risk
score must be produced. Kept fast by relying on the app's cached predictor.
"""
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def _run():
    #run the app headlessly with a generous timeout for the first model build
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    return at


def test_app_loads_without_exception():
    #the app renders and surfaces the honesty banner, no uncaught exception
    at = _run()
    assert not at.exception
    banners = " ".join(w.value for w in at.warning)
    assert "NOT a validated clinical tool" in banners


def test_predict_button_produces_finite_risk():
    #clicking Predict runs the fusion and emits a finite numeric fused score
    at = _run()
    assert not at.exception
    assert len(at.button) >= 1
    at.button[0].click().run()
    assert not at.exception

    #the fused risk metric is rendered as "<number>" (2 decimals)
    metric_values = [m.value for m in at.metric]
    assert metric_values, "no metrics rendered after predict"
    finite_numeric = []
    for v in metric_values:
        try:
            finite_numeric.append(np.isfinite(float(v)))
        except (TypeError, ValueError):
            finite_numeric.append(False)
    assert any(finite_numeric), f"no finite numeric risk in {metric_values}"


def test_predict_tier_is_valid_label():
    #the risk tier metric is one of the three SSPG-tertile labels
    at = _run()
    at.button[0].click().run()
    assert not at.exception
    tiers = [m.value for m in at.metric]
    assert any(v in {"Low", "Moderate", "High"} for v in tiers)
