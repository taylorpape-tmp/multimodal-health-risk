# HURDLE serving: Docker -> ECR -> deploy

The serving layer is a FastAPI app (`hurdle.serving.app:app`) that scores a
single omics feature vector and returns a predicted **SSPG** (steady-state
plasma glucose, mg/dL) — the continuous insulin-resistance target from the S8
matrix. This document covers building the image, pushing it to the real ECR
repository, and deploying it. **It documents commands only — nothing here
builds or pushes an image or makes an AWS call.**

## Endpoints

| Method | Path       | Purpose                                                        |
|--------|------------|----------------------------------------------------------------|
| GET    | `/health`  | Liveness/readiness probe. `200` as soon as the process is up.  |
| GET    | `/`        | Service metadata (name, model name, endpoint list).            |
| POST   | `/predict` | Score one omics vector -> `{model, prediction, n_features, model_loaded}`. |

`POST /predict` body:

```json
{ "features": [ /* 86 floats in training-column order */ ] }
```

The feature contract is the S8 SSPG matrix: **85 analytes + `TG_HDL_ratio`
= 86 features**, in the exact column order returned by
`hurdle.features.omics.build_feature_matrix(interim_dir, target="SSPG",
add_ratios=True)`. A vector of the wrong width is rejected with `400`.

## Model resolution at startup

The app resolves a **real** fitted estimator at boot (never a hardcoded score),
in this order:

1. **Checkpoint** — if `$HURDLE_MODEL_PATH` (default `/models/model.joblib`)
   exists, load it with joblib. This is the production path: the checkpoint is
   pulled from S3 and mounted at `/models/model.joblib` in the container.
2. **Startup training** — else, if `$HURDLE_INTERIM_DIR` (default
   `data/interim`) holds the cleaned S8 matrix, fit the standardized RidgeCV
   pipeline in-process (n=59, 86 features; sub-second). This is the local/dev
   path.
3. **No-model mode** — else stay up and answer `/health`, but return `503` from
   `/predict`. This keeps CI smoke tests and rolling deploys green before any
   artifact/data is present.

> **Note:** `.dockerignore` excludes `data/`, so the S8 matrix is **not** baked
> into the image. Inside the container, path (2) is therefore unavailable —
> production serving depends on the S3 checkpoint from path (1). Build the
> checkpoint once (see below) and mount/download it to `/models/model.joblib`.

## 0. Build the model checkpoint (one-time, before first deploy)

The RidgeCV pipeline is cheap to fit, so produce the checkpoint from the S8
matrix and stage it in S3. `joblib.dump` a `ServedModel` (it carries the
feature contract), e.g.:

```python
import joblib
from hurdle.serving.app import train_sspg_model
joblib.dump(train_sspg_model("data/interim"), "model.joblib")
```

```bash
# stage the checkpoint where the task pulls it at deploy time
aws s3 cp model.joblib "s3://$HURDLE_DATA_BUCKET/models/model.joblib"
```

## 1. Build the image

The multi-stage `Dockerfile` at repo root installs `requirements-serving.txt`
(pinned CPU-tier deps; no torch/CUDA) and the `hurdle` package, then runs as a
non-root user with a container HEALTHCHECK against `/health`.

```bash
# from repo root
docker build -t hurdle-serving:latest .
```

## 2. Authenticate to ECR and push

Real ECR repository:
`791973676965.dkr.ecr.us-east-1.amazonaws.com/hurdle-serving`

```bash
AWS_REGION=us-east-1
ACCOUNT_ID=791973676965
ECR_REPO=791973676965.dkr.ecr.us-east-1.amazonaws.com/hurdle-serving

# log the local docker daemon into ECR
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin \
      "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# tag + push (use an immutable tag, e.g. the git SHA, plus :latest)
GIT_SHA=$(git rev-parse --short HEAD)
docker tag hurdle-serving:latest "${ECR_REPO}:${GIT_SHA}"
docker tag hurdle-serving:latest "${ECR_REPO}:latest"
docker push "${ECR_REPO}:${GIT_SHA}"
docker push "${ECR_REPO}:latest"
```

## 3. Deploy

The task/service runs the pushed image with the model checkpoint made available
at `/models/model.joblib` and the S3 bucket set via env:

```bash
# environment the container expects at runtime
#   HURDLE_MODEL_PATH   -> /models/model.joblib   (default)
#   HURDLE_MODEL_NAME   -> hurdle-sspg-ridge       (default)
#   HURDLE_DATA_BUCKET  -> <your S3 bucket>        (source of the checkpoint)

# roll the service to the new image (ECS example)
aws ecs update-service \
  --cluster hurdle \
  --service hurdle-serving \
  --force-new-deployment \
  --region us-east-1
```

Deployment health is gated on `GET /health` (load balancer / ECS health check)
and the Dockerfile `HEALTHCHECK`. Once the checkpoint is present, `/health`
reports `"model_loaded": true` and `/predict` returns real SSPG scores.

## 4. Smoke-test a running container

```bash
# local run (dev): mounts the repo so startup training (path 2) can fire
docker run --rm -p 8080:8080 hurdle-serving:latest &

curl -s localhost:8080/health
# -> {"status":"ok","model_loaded":true|false}

curl -s -X POST localhost:8080/predict \
  -H 'content-type: application/json' \
  -d '{"features": [/* 86 floats */]}'
# -> {"model":"hurdle-sspg-ridge","prediction":<float mg/dL>,"n_features":86,"model_loaded":true}
```

## Tests

`tests/test_serving.py` covers the layer with starlette's `TestClient`:
`/health`, `/` metadata, a `/predict` happy path (a tiny fitted RidgeCV injected
through the `get_model` FastAPI dependency, so it is fast and offline), a
wrong-length `400`, a no-model `503`, and a real-data path that trains the
actual served pipeline on the on-disk S8 matrix and asserts a finite SSPG in a
plausible band. Run:

```bash
pytest tests/test_serving.py -q
```
