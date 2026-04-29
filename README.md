# mlpipeline

Minimal end-to-end ML pipeline take-home project covering:

- model training and versioned artifacts
- Flask inference API
- Docker packaging
- CI, monitoring, and trade-off documentation

## Quick links

- Quickstart only: `QUICKSTART.md`
- Training details: `02_training.md`
- Inference/API and Docker details: `03_inference.md`
- Monitoring design: `plans/monitoring.md`
- Trade-offs: `plans/tradeoffs.md`
- Validation workflow: `.github/workflows/validation.yml`
- Train/release workflow (skeleton): `.github/workflows/train-release.yml`

## Directory structure

```text
mlpipeline/
├── api/
│   ├── app.py
│   └── model_loader.py
├── training/
│   ├── __init__.py
│   ├── train.py
│   ├── data.py
│   ├── modeling.py
│   ├── layout.py
│   ├── tracking.py
│   └── logging_utils.py
├── tests/
│   ├── conftest.py
│   ├── test_training.py
│   └── test_api.py
├── runs/
│   ├── artifacts/
│   │   ├── latest/
│   │   └── runs/
│   ├── logs/
│   └── mlruns/
├── .github/workflows/validation.yml
├── .github/workflows/train-release.yml
├── Dockerfile
├── requirements.txt
├── README.md
├── QUICKSTART.md
├── 02_training.md
└── 03_inference.md
```

## Current project workflow

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Development

- `Linting`
  - `ruff check .`
- `Unit tests`
  - `pytest -q`

Linting runs automatically in CI (`.github/workflows/validation.yml`).
Unit tests run automatically in CI.

### 2) Train model

```bash
python training/train.py --output-dir runs/artifacts
```

Training writes immutable run outputs and updates a latest alias:

- `runs/artifacts/runs/vNNN/`
  - `regression_model.joblib`
  - `metrics.json`
  - `model_version.txt`
- `runs/artifacts/latest/`
  - `regression_model.joblib`
  - `metrics.json`
  - `model_version.txt`
  - `manifest.json`

`metrics.json` includes model and quality metadata such as:

- `model_version`
- `rmse`
- `baseline_rmse`
- `passed_guardrail`
- `training_time_seconds`
- `git_commit` (when available)

Training params (`samples`, `random_seed`) are logged in MLflow run parameters.

The training step emits `runs/artifacts/latest/manifest.json` (and a run-scoped copy) used as the handoff contract for release orchestration.
The train/release workflow validates manifest keys and blocks release when `passed_guardrail` is false.

Training code is split into small modules under `training/`:

- `training/train.py` - CLI and orchestration entrypoint
- `training/data.py` - synthetic dataset generation
- `training/modeling.py` - validation, training, and metric computation
- `training/layout.py` - versioning and artifact directory/file layout
- `training/tracking.py` - MLflow + git commit tracking
- `training/logging_utils.py` - file logger setup

### 3) Run inference API locally

```bash
python api/app.py
```

Endpoints:

- `GET /health`
- `POST /predict`

Example:

```bash
curl -X POST http://127.0.0.1:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 42, "income_k": 88.0, "tenure_years": 6}'
```

### 4) Run with Docker

```bash
docker build -t mlpipeline-takehome .
docker run --rm -p 8080:8080 mlpipeline-takehome
```

## Deliverables status (take-home)

- Reproducible training entrypoint: `training/train.py`
- Model artifact + params + metrics logging: MLflow (local file backend)
- Prediction API: `api/app.py` with validation and model version response
- Dockerized API: `Dockerfile`
- Validation checks + Docker build smoke: `.github/workflows/validation.yml`
- Monitoring design: `plans/monitoring.md`
- Trade-offs and limitations: `plans/tradeoffs.md`

## Productionization considerations

This project is intentionally local-first for take-home scope. In production, core concerns are:

- training compute should run remotely on managed CPU/GPU infrastructure, not developer laptops
- experiment tracking should use a managed backend (MLflow server or SageMaker Experiments)
- model artifacts should live in durable object storage and be promoted through environments
- deployments should bind API image version and model version for traceability
- API should include auth, rate limiting, and richer structured telemetry

## Considerations

- GitHub Actions is currently used as the primary orchestrator for simplicity and transparency in this take-home implementation.
- In a future production setup, GitHub Actions may be limited to validation/release triggers while training orchestration is handled by a dedicated workflow orchestrator.
- The exact orchestrator choice can evolve based on scaling, scheduling, observability, and environment-management requirements.

## Plan for productization

1. Move training execution to managed jobs (e.g., SageMaker Training Jobs) with configurable instance types.
2. Attach managed experiment tracking/lineage (SageMaker Experiments or managed MLflow).
3. Publish model artifacts to a registry with stage transitions (`staging` -> `production`).
4. Gate promotions with automated checks (quality thresholds + integration smoke tests).
5. Deploy inference service with environment-specific configuration and secrets management.
6. Add production observability: latency/error SLOs, data drift checks, and post-deploy quality monitoring.
