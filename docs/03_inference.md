# Inference/API Guide

This document summarizes the inference-side implementation, including local Flask serving and Dockerization.

## What Was Implemented

### Flask API (`api/app.py`)

The inference API is implemented with Flask and exposes two endpoints:

- `GET /health`
- `POST /predict`

The app loads model artifacts at startup (see `MODEL_DIR` / `MODEL_ARTIFACT_URI` in `api/model_loader.py`); by default that is under `runs/artifacts/latest/`:

- `sample_model.joblib`
- `model_version.txt` (unless `MODEL_VERSION` is set when using URI-based bundles)

The model version is returned in both health and prediction responses.

### `GET /health`

Returns service and model status:

```json
{
  "status": "ok",
  "model_version": "v001"
}
```

### `POST /predict`

Expected JSON payload fields:

- `age`
- `income_k`
- `tenure_years`

Sample payload:

```json
{
  "age": 42,
  "income_k": 88.0,
  "tenure_years": 6
}
```

Validation behavior:

- rejects non-JSON or non-object payloads
- rejects missing required fields
- rejects non-numeric values for required fields

Successful response:

```json
{
  "prediction": 64091.97,
  "model_version": "v001"
}
```

## Logging Behavior

Logs are **structured JSON** (one object per line on **stdout**) with fields such as:

- **`http_request`** — every response: `request_id`, `method`, `path`, `status_code`, **`latency_ms`** (end-to-end for that request), `model_version`
- **`predict_success`** — successful inference: `request_id`, `predict_ms` (model forward pass only), `model_version`
- **`predict_validation_error`** — bad JSON, missing fields, or non-numeric features: `request_id`, `reason`, `model_version`
- **`predict_inference_error`** — model/runtime failure: `request_id`, `traceback`, `model_version`

In production on AWS, an **agent or log driver** (for example ECS **`awslogs`**, or **Fluent Bit** on EKS) collects these stdout lines and forwards them to **CloudWatch Logs**; the app does not call the CloudWatch API directly. See `docs/04_monitoring.md` for the short design note.

## Artifact Contract Used by API

The API loads from **`MODEL_DIR`** (default `runs/artifacts/latest`) or resolves **`MODEL_ARTIFACT_URI`** at startup (see `api/model_loader.py`). Files expected under the resolved directory:

- `sample_model.joblib`
- `model_version.txt` (unless `MODEL_VERSION` is supplied for URI-only bundles)

Training writes the same layout under `runs/artifacts/runs/vNNN/` with a `latest/` alias.

## Dockerization

### Dockerfile.local

Container behavior (local verification path):

- base image: `python:3.11-slim`
- installs dependencies from `requirements.txt`
- copies `api/` and `runs/artifacts/` into the image (for local/smoke runs that bake artifacts)
- exposes port `8080`
- starts Flask app via:
  - `python -m api.app`

Environment variable used by the app in this local image:

- `MODEL_DIR=/app/runs/artifacts/latest`

### Dockerfile.inference

Deployment-oriented, model-agnostic image behavior:

- copies only `api/` (no baked model artifacts)
- keeps model reference external at runtime via:
  - `MODEL_ARTIFACT_URI` (for example S3 URI or mounted path)
  - `MODEL_VERSION` (optional override when no version file is present)
- used by `.github/workflows/inference.yml` for CI/release image build.

### `.dockerignore`

A `.dockerignore` file was added to keep build context smaller and cleaner (excluding local virtualenv, git metadata, logs, notebooks, training artifacts not needed for serving, etc.).

## Run Locally

```bash
python -m api.app
```

Health check:

```bash
curl http://127.0.0.1:8080/health
```

Prediction:

```bash
curl -X POST http://127.0.0.1:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"age":42,"income_k":88.0,"tenure_years":6}'
```

## Run with Docker

```bash
docker build -f Dockerfile.local -t mlpipeline-api:local .
docker run --rm -p 8080:8080 mlpipeline-api:local
```

Then call the same `/health` and `/predict` endpoints on `http://127.0.0.1:8080`.
