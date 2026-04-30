# Take-Home Scoped Plan (3-5 hours)

## Objective
Deliver a minimal, production-oriented workflow for the assignment:
1. model training + versioning,
2. Flask inference API in Docker,
3. CI/CD thinking (light implementation),
4. monitoring/observability design with trade-offs.

## What to build (and what not to)
### In scope
- One reproducible training script from the provided notebook.
- One model tracking interface (MLflow or SageMaker Experiments interface only).
- One Flask service with `POST /predict` returning prediction + model version.
- One Docker image runnable locally.
- One lightweight GitHub Actions workflow (or clear YAML snippets).
- One short monitoring design section.

### Out of scope
- Full cloud infra provisioning.
- Advanced model optimization.
- Full production monitoring stack implementation.
- Multi-environment deployment automation.

## Deliverables checklist
- `training/train.py` (or equivalent) with deterministic training entrypoint.
- Logged artifact, parameters, and evaluation metric.
- `api/app.py` with `POST /predict`.
- Response includes `{ "prediction": ..., "model_version": ... }`.
- `Dockerfile.local` and local run instructions in `README.md`.
- `.github/workflows/ci.yml` (or `plans/cicd-design.md` with YAML snippets).
- `docs/plans/monitoring.md` describing latency/error/model/drift monitoring.
- `docs/plans/tradeoffs.md` with known limitations.

## Recommended implementation order
### Step 1 - Training + versioning (60-90 min)
- Extract notebook logic into a script with fixed random seed.
- Train simple regression model and evaluate (e.g., RMSE).
- Log:
  - model artifact,
  - params,
  - metric(s),
  using MLflow (easiest local path) or SageMaker interface stubs.
- Persist a `model_version` value (timestamp or run ID).

### Step 2 - Flask API + model version response (45-75 min)
- Build `POST /predict` that accepts JSON features.
- Load latest model artifact on startup.
- Return prediction and `model_version` in response.
- Add basic input validation and clear error JSON.

### Step 3 - Docker local run (30-45 min)
- Create Docker image for API service.
- Validate local run:
  - `docker build ...`
  - `docker run ...`
  - curl sample request to `/predict`.

### Step 4 - CI/CD light implementation (30-45 min)
- Add GitHub Actions workflow with:
  - dependency install,
  - lint or tests,
  - Docker build.
- Add conceptual "promote staging -> production model" section in workflow comments or separate doc.

### Step 5 - Monitoring design + trade-offs (30-45 min)
- Document:
  - API latency + error rate metrics,
  - post-deploy model quality checks,
  - basic drift checks (feature distribution/prediction distribution).
- Include limitations and next improvements.

## Suggested file layout
- `training/train.py`
- `training/requirements.txt` (or root `requirements.txt`)
- `api/app.py`
- `api/model_loader.py`
- `Dockerfile.local`
- `.github/workflows/ci.yml`
- `docs/plans/monitoring.md`
- `docs/plans/tradeoffs.md`

## Acceptance criteria (aligned to rubric)
- Reproducible training run produces model artifact.
- Parameters + metric + artifact are logged in chosen tracking interface.
- `POST /predict` works locally and returns model version.
- Dockerized service runs locally.
- CI/CD workflow or design clearly demonstrates lint/test/build/deploy/promotion thinking.
- Monitoring section explicitly covers latency/errors, model performance, and drift.

## Timebox guardrails
- If blocked on tooling, prefer clear design notes over deeper implementation.
- Keep each section "good enough" and rubric-complete.
- Prioritize end-to-end completeness over polish.
