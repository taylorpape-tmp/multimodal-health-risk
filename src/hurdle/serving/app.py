"""FastAPI serving app for HURDLE diabetes-risk models.

This is the entrypoint baked into the serving Docker image and pushed to ECR by
the deploy workflow. It exposes:

  GET  /health   -> liveness/readiness probe (used by load balancers / ECS)
  GET  /         -> service metadata
  POST /predict  -> score a single feature vector

The prediction path loads a trained scikit-learn pipeline (any of the CPU-tier
HURDLE models from hurdle.ml, or a distilled head over RETFound embeddings)
from a checkpoint. In production the checkpoint is pulled from S3
(s3://$HURDLE_DATA_BUCKET/models/...) at startup; for local/dev the app runs
in a degraded "no-model" mode that still answers /health so the container and
CI smoke test pass without a trained artifact present.

Run locally:
    uvicorn hurdle.serving.app:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("HURDLE_MODEL_PATH", "/models/model.joblib")
MODEL_NAME = os.environ.get("HURDLE_MODEL_NAME", "hurdle-consensus")

app = FastAPI(
    title="HURDLE diabetes-risk API",
    description="Multimodal diabetes-risk scoring (omics, wearable, CGM, retinal).",
    version="0.1.0",
)

# Module-level model handle, populated at startup if a checkpoint exists.
_model: Any = None


@app.on_event("startup")
def _load_model() -> None:
    """Load the trained pipeline if present; otherwise stay in no-model mode.

    Never raise here: a missing checkpoint must not stop the container from
    booting and answering /health (that keeps CI smoke tests and rolling
    deploys green before the first model is trained).
    """
    global _model
    if os.path.exists(MODEL_PATH):
        import joblib  # local import: only needed when a checkpoint exists

        _model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    """A single subject's feature vector.

    `features` order must match the columns the model was trained on
    (see the model's `feature_cols`). Kept as a bare list for a minimal API.
    """

    features: list[float] = Field(..., description="Feature vector in training column order.")


class PredictResponse(BaseModel):
    model: str
    prediction: float
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
def predict(req: PredictRequest) -> PredictResponse:
    if _model is None:
        # 503: service is up but not ready to score (no checkpoint loaded yet).
        raise HTTPException(
            status_code=503,
            detail=f"No model loaded. Expected checkpoint at {MODEL_PATH}.",
        )
    x = np.asarray(req.features, dtype=float).reshape(1, -1)
    try:
        pred = float(np.asarray(_model.predict(x)).ravel()[0])
    except Exception as exc:  # surface shape/dtype mismatches as a clean 400
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
    return PredictResponse(model=MODEL_NAME, prediction=pred, model_loaded=True)
