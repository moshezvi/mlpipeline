# mlpipeline
Pipeline for training and deploying ML models

## Take-home quickstart

### 1) Install deps
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Train and log model
```bash
python training/train.py --output-dir artifacts
```

This writes:
- `artifacts/regression_model.joblib`
- `artifacts/metrics.json`
- `artifacts/model_version.txt`

### 3) Run Flask API locally
```bash
python api/app.py
```

Test:
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 42, "income_k": 88.0, "tenure_years": 6}'
```

### 4) Run with Docker
```bash
docker build -t mlpipeline-takehome .
docker run --rm -p 8080:8080 mlpipeline-takehome
```
