# mlpipeline
Pipeline for training and deploying ML models

## Take-home quickstart

### 1) Install deps
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Development

- `Linting`
  - `ruff check .`

Linting Auto-runs on commit via `.git/hooks/pre-commit` and in CI.

- `Unit tests`
  - `pytest -q`

### 2) Train and log model
```bash
python training/train.py --output-dir runs/artifacts
```

This writes:
- `runs/artifacts/sample_model.joblib`
- `runs/artifacts/metrics.json`
- `runs/artifacts/model_version.txt`

### 3) Run Flask API locally
```bash
python -m api.app
```

Test:
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 42, "income_k": 88.0, "tenure_years": 6}'
```
Sample Output:
```bash
{"model_version":"v011","prediction":64091.97642093163}
```

Health check:
```bash
curl http://localhost:8080/health
```
Sample Output:
```bash
{"model_version":"v011","status":"ok"}
```

### 4) Run with Docker
```bash
docker build -f Dockerfile.local -t mlpipeline-api:local .
docker run --rm -p 8080:8080 mlpipeline-api:local
```


