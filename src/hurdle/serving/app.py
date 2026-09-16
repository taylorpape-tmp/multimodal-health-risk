"""FastAPI serving app for HURDLE diabetes-risk models.

This is the entrypoint baked into the serving Docker image and pushed to ECR by
the deploy workflow. It exposes:

  GET  /health   -> liveness/readiness probe (used by load balancers / ECS)
  GET  /         -> service metadata
  POST /predict  -> score a single omics feature vector (predicted SSPG)

The prediction path serves a real fitted scikit-learn pipeline (a standardized
RidgeCV over the S8 omics analytes -- the CPU-tier HURDLE regressor for the
SSPG continuous insulin-resistance target). Model resolution at startup, in
order:

  1. If a checkpoint exists at $HURDLE_MODEL_PATH, load it (joblib).
  2. Else, if the cleaned S8 interim matrix is on disk under
     $HURDLE_INTERIM_DIR, train the RidgeCV pipeline in-process (fast: n=59,
     86 analyte features) and serve its real predictions.
  3. Else, stay in a degraded "no-model" mode that still answers /health so
     the container and CI smoke test pass without any artifact present.

Predictions are always real model output -- never a hardcoded score.

Run locally:
    uvicorn hurdle.serving.app:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("HURDLE_MODEL_PATH", "/models/model.joblib")
MODEL_NAME = os.environ.get("HURDLE_MODEL_NAME", "hurdle-sspg-ridge")
#where the cleaned S8/S9 interim matrices live for startup training (repo-local
#by default; absent inside the serving image, which triggers no-model mode)
INTERIM_DIR = os.environ.get("HURDLE_INTERIM_DIR", "data/interim")


@dataclass
class ServedModel:
    """A fitted regressor plus the feature contract it was trained against.

    Wrapping a bare sklearn pipeline this way lets /predict validate the
    incoming vector length and report the model name/feature count uniformly,
    whether the estimator came from a checkpoint or from startup training.
    """

    estimator: Any                                  #fitted object exposing .predict
    feature_cols: list[str] = field(default_factory=list)
    name: str = MODEL_NAME

    def predict(self, x: np.ndarray) -> float:
        return float(np.asarray(self.estimator.predict(x)).ravel()[0])


def _build_ridge_pipeline():
    #standardized RidgeCV: matches the linear-family default transform
    #('standard') and RidgeModel's self-tuned alpha grid, so the served model
    #is the same estimator the offline CV harness evaluates.
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from hurdle.ml.linear_models import RidgeModel

    return Pipeline([
        ("transform", StandardScaler()),
        ("model", RidgeCV(alphas=RidgeModel.ALPHAS)),
    ])


def train_sspg_model(interim_dir: str = INTERIM_DIR) -> ServedModel:
    """Fit the SSPG RidgeCV pipeline on the real S8 omics matrix.

    Uses the same feature builder the offline pipeline uses, so the served
    feature contract (85 analytes + TG_HDL_ratio) matches training exactly.
    """
    from hurdle.features.omics import build_feature_matrix

    X, y, feature_cols = build_feature_matrix(interim_dir, target="SSPG", add_ratios=True)
    pipe = _build_ridge_pipeline()
    pipe.fit(X.values, y.values)
    return ServedModel(estimator=pipe, feature_cols=list(feature_cols), name=MODEL_NAME)


#module-level model handle, populated at startup (checkpoint or fresh training)
_model: ServedModel | None = None


def _load_model() -> None:
    """Resolve a real model at boot; never raise (missing data -> no-model mode).

    A missing checkpoint AND missing training data must not stop the container
    from booting and answering /health -- that keeps CI smoke tests and rolling
    deploys green before any artifact/data is mounted.
    """
    global _model
    if os.path.exists(MODEL_PATH):
        import joblib  # local import: only needed when a checkpoint exists

        loaded = joblib.load(MODEL_PATH)
        _model = loaded if isinstance(loaded, ServedModel) else ServedModel(loaded)
    elif os.path.isdir(INTERIM_DIR):
        try:
            _model = train_sspg_model(INTERIM_DIR)
        except Exception:
            #data dir exists but S8 matrix unreadable -> degrade, keep /health up
            _model = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    #resolve the model once at startup; nothing to tear down on shutdown
    _load_model()
    yield


app = FastAPI(
    title="HURDLE diabetes-risk API",
    description="Multimodal diabetes-risk scoring (omics, wearable, CGM, retinal).",
    version="0.1.0",
    lifespan=lifespan,
)


def get_model() -> ServedModel:
    """FastAPI dependency yielding the active model.

    Isolating model access behind a dependency lets tests inject a tiny fitted
    model via app.dependency_overrides[get_model] without touching disk or the
    startup path. Raises 503 when no model is loaded.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=f"No model loaded. Expected checkpoint at {MODEL_PATH} or S8 matrix in {INTERIM_DIR}.",
        )
    return _model


class PredictRequest(BaseModel):
    """A single subject's omics feature vector.

    `features` order must match the columns the model was trained on
    (see the model's `feature_cols`). Kept as a bare list for a minimal API.
    """

    features: list[float] = Field(..., description="Feature vector in training column order.")


class PredictResponse(BaseModel):
    model: str
    prediction: float
    n_features: int
    model_loaded: bool


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness/readiness probe. 200 as soon as the process is up."""
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "hurdle-serving",
        "model_name": MODEL_NAME,
        "model_loaded": _model is not None,
        "endpoints": ["/health", "/predict"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, model: ServedModel = Depends(get_model)) -> PredictResponse:
    #validate the vector width against the training contract when known
    if model.feature_cols and len(req.features) != len(model.feature_cols):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(model.feature_cols)} features, got {len(req.features)}.",
        )
    x = np.asarray(req.features, dtype=float).reshape(1, -1)
    try:
        pred = model.predict(x)
    except Exception as exc:  # surface shape/dtype mismatches as a clean 400
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
    return PredictResponse(
        model=model.name,
        prediction=pred,
        n_features=len(req.features),
        model_loaded=True,
    )
