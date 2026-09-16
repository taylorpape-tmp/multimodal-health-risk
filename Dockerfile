#Multi-stage build for the HURDLE FastAPI serving image.
#
#Stage 1 (builder): install dependencies + the hurdle package into a venv.
#Stage 2 (runtime): copy only the venv + source, run as a non-root user.
#Result: a slim image with no build toolchain and no pip cache.

#Stage 1: builder
FROM python:3.11-slim AS builder

#Build deps for any wheels that need compiling (kept out of the final image).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

#Isolated venv we can copy wholesale into the runtime stage.
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /build

#Install runtime deps first (layer caches unless requirements change).
COPY requirements-serving.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-serving.txt

#Install the hurdle package itself. pyproject + src/ layout => `pip install .`
#picks up src/hurdle. --no-deps: deps are already pinned above.
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps .

#Stage 2: runtime
FROM python:3.11-slim AS runtime

#Non-root user for the running container.
RUN useradd --create-home --uid 10001 appuser

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
#Fail fast, unbuffered logs, no .pyc writes.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HURDLE_MODEL_PATH=/models/model.joblib

#Copy the built venv (contains the hurdle package + all runtime deps).
COPY --from=builder /opt/venv /opt/venv

#Directory where the S3 checkpoint is mounted/downloaded at deploy time.
RUN mkdir -p /models && chown appuser:appuser /models

USER appuser
WORKDIR /home/appuser

EXPOSE 8080

#Container-level healthcheck hits the app's readiness probe.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health').status==200 else 1)"

#Serve the FastAPI app. hurdle.serving.app:app is the entrypoint.
CMD ["uvicorn", "hurdle.serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
