import importlib
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression


def _prepare_model_artifacts(base_dir: Path):
    model_dir = base_dir / "runs" / "artifacts" / "latest"
    model_dir.mkdir(parents=True, exist_ok=True)

    x = np.array([[20, 50.0, 1], [40, 80.0, 5], [60, 100.0, 8]])
    y = np.array([30000.0, 60000.0, 90000.0])
    model = LinearRegression().fit(x, y)

    joblib.dump(model, model_dir / "regression_model.joblib")
    (model_dir / "model_version.txt").write_text("vtest", encoding="utf-8")
    return model_dir


def test_health_endpoint_returns_status_and_model_version(tmp_path, monkeypatch):
    model_dir = _prepare_model_artifacts(tmp_path)
    monkeypatch.setenv("MODEL_DIR", str(model_dir))

    sys.modules.pop("api.app", None)
    app_module = importlib.import_module("api.app")
    app = app_module.app
    client = app.test_client()

    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["model_version"] == "vtest"


def test_predict_valid_and_invalid_payloads(tmp_path, monkeypatch):
    model_dir = _prepare_model_artifacts(tmp_path)
    monkeypatch.setenv("MODEL_DIR", str(model_dir))

    sys.modules.pop("api.app", None)
    app_module = importlib.import_module("api.app")
    app = app_module.app
    client = app.test_client()

    ok = client.post(
        "/predict",
        json={"age": 42, "income_k": 88.0, "tenure_years": 6},
    )
    assert ok.status_code == 200
    ok_payload = ok.get_json()
    assert isinstance(ok_payload["prediction"], float)
    assert ok_payload["model_version"] == "vtest"

    missing = client.post("/predict", json={"age": 42, "income_k": 88.0})
    assert missing.status_code == 400

    non_numeric = client.post(
        "/predict",
        json={"age": "x", "income_k": 88.0, "tenure_years": 6},
    )
    assert non_numeric.status_code == 400
