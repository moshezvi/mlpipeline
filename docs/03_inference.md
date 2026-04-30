# Inference/API Guide

This document summarizes the inference-side implementation, including local Flask serving and Dockerization.

## What Was Implemented

### Flask API (`api/app.py`)

The inference API is implemented with Flask and exposes two endpoints:

- `GET /health`
- `POST /predict`

The app loads model artifacts at startup from:

- `artifacts/latest/regression_model.joblib`
- `artifacts/latest/model_version.txt`

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

Basic request/error logging is included:

- request method/path logging for each request
- health check success logging
- prediction validation errors
- prediction inference exceptions (with stack trace)
- prediction success logging

## Artifact Contract Used by API

The API uses the active model alias under:

- `artifacts/latest/regression_model.joblib`
- `artifacts/latest/model_version.txt`
- `artifacts/latest/metrics.json` (available for inspection, not required for prediction)

This aligns with training output layout where each run is immutable under `artifacts/runs/vNNN/` and the currently served model is mirrored to `artifacts/latest/`.

## Dockerization

### Dockerfile

Container behavior:

- base image: `python:3.11-slim`
- installs dependencies from `requirements.txt`
- copies `api/` and `artifacts/` into image
- exposes port `8080`
- starts Flask app via:
  - `python api/app.py`

Environment variable used by the app in Docker:

- `MODEL_DIR=/app/artifacts/latest`

### `.dockerignore`

A `.dockerignore` file was added to keep build context smaller and cleaner (excluding local virtualenv, git metadata, logs, notebooks, training artifacts not needed for serving, etc.).

## Run Locally

```bash
python api/app.py
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
docker build -t mlpipeline-api .
docker run --rm -p 8080:8080 mlpipeline-api
```

Then call the same `/health` and `/predict` endpoints on `http://127.0.0.1:8080`.
