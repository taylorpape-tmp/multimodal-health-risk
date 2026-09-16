"""Tests for the FastAPI serving app (health + real SSPG inference).

Uses starlette's TestClient (bundled with fastapi). Two flavours of /predict:
  - a fast path that injects a tiny fitted RidgeCV via the get_model dependency
    override, so no disk/startup training is needed;
  - a real-data path, skipped unless the cleaned S8 matrix is on disk, that
    trains the actual served pipeline and asserts a plausible SSPG score.

SSPG (steady-state plasma glucose) in the real S8 cohort spans ~40-276 mg/dL;
predictions are asserted finite and within a generous physiological band, never
against a hardcoded value.
"""
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hurdle.serving.app import ServedModel, app, get_model

REAL_INTERIM = Path("data/interim")
#generous physiological band for SSPG (mg/dL): real cohort is ~40-276
SSPG_LO, SSPG_HI = 0.0, 500.0


@pytest.fixture
def tiny_model():
    #a real fitted pipeline (not a stub): tiny synthetic omics-like matrix so
    #the injected model actually runs sklearn .predict, just fast and offline.
    rng = np.random.default_rng(0)
    n, p = 30, 6
    X = rng.normal(size=(n, p))
    #SSPG-like target centered in-range so predictions land in a plausible band
    y = 150.0 + 25.0 * X[:, 0] - 15.0 * X[:, 1] + rng.normal(size=n)
    pipe = Pipeline([("transform", StandardScaler()),
                     ("model", RidgeCV(alphas=np.logspace(-3, 3, 10)))])
    pipe.fit(X, y)
    cols = [f"f{i}" for i in range(p)]
    return ServedModel(estimator=pipe, feature_cols=cols, name="hurdle-sspg-test")


@pytest.fixture
def client_with_model(tiny_model):
    #inject the tiny model through the dependency, bypassing startup/disk
    app.dependency_overrides[get_model] = lambda: tiny_model
    with TestClient(app) as c:
        yield c, tiny_model
    app.dependency_overrides.clear()


def test_health_ok():
    #health must answer 200 regardless of model state (no override needed)
    with TestClient(app) as c:
        r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_root_metadata():
    with TestClient(app) as c:
        r = c.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "hurdle-serving"
    assert "/predict" in body["endpoints"]


def test_predict_returns_finite_plausible_sspg(client_with_model):
    #a real omics feature vector -> a finite numeric SSPG in a plausible range
    client, model = client_with_model
    features = [0.5, -0.2, 0.1, 0.0, 0.3, -0.1]
    r = client.post("/predict", json={"features": features})
    assert r.status_code == 200
    body = r.json()
    pred = body["prediction"]
    assert isinstance(pred, float)
    assert np.isfinite(pred)
    assert SSPG_LO <= pred <= SSPG_HI
    assert body["model"] == "hurdle-sspg-test"
    assert body["n_features"] == len(features)
    assert body["model_loaded"] is True
    #prediction must equal the underlying pipeline output (real inference, not faked)
    expected = float(model.estimator.predict(np.asarray(features).reshape(1, -1))[0])
    assert pred == pytest.approx(expected)


def test_predict_wrong_length_is_400(client_with_model):
    #the injected model knows its 6-feature contract -> mismatched width is a 400
    client, _ = client_with_model
    r = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    assert r.status_code == 400
    assert "Expected 6 features" in r.json()["detail"]


def test_predict_503_without_model():
    #simulate the degraded no-model boot: get_model must raise 503 so the
    #service stays up (health green) but declines to score. No context manager
    #here so the startup handler does not re-train _model back to non-None.
    from hurdle.serving import app as app_mod
    app.dependency_overrides.clear()
    saved = app_mod._model
    app_mod._model = None
    try:
        c = TestClient(app)
        r = c.post("/predict", json={"features": [0.0, 0.0]})
    finally:
        app_mod._model = saved
    assert r.status_code == 503


@pytest.mark.skipif(
    not (REAL_INTERIM / "omics_S8_sspg_clean.csv").exists(),
    reason="real S8 interim matrix not present",
)
def test_predict_on_real_s8_vector():
    #train the ACTUAL served pipeline on the real S8 matrix and score a real
    #patient row through the API; assert a finite, plausible SSPG prediction.
    from hurdle.features.omics import build_feature_matrix
    from hurdle.serving.app import train_sspg_model

    model = train_sspg_model(str(REAL_INTERIM))
    X, y, cols = build_feature_matrix(REAL_INTERIM, target="SSPG", add_ratios=True)
    #the served contract must match the real feature builder exactly
    assert model.feature_cols == cols
    row = X.iloc[0].tolist()

    app.dependency_overrides[get_model] = lambda: model
    try:
        with TestClient(app) as c:
            r = c.post("/predict", json={"features": row})
    finally:
        app.dependency_overrides.clear()

    assert r.status_code == 200
    pred = r.json()["prediction"]
    assert np.isfinite(pred)
    #generous band around the observed SSPG range (40-276 mg/dL)
    assert 0.0 <= pred <= 400.0
    assert r.json()["n_features"] == len(cols)
