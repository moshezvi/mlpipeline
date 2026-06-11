import importlib
import sys
import tarfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from api import model_loader


def _prepare_model_artifacts(base_dir: Path):
    model_dir = base_dir / "runs" / "artifacts" / "latest"
    model_dir.mkdir(parents=True, exist_ok=True)

    x = pd.DataFrame(
        [[20, 50.0, 1], [40, 80.0, 5], [60, 100.0, 8]],
        columns=["age", "income_k", "tenure_years"],
    )
    y = np.array([30000.0, 60000.0, 90000.0])
    model = LinearRegression().fit(x, y)

    joblib.dump(model, model_dir / "sample_model.joblib")
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


def test_load_via_model_artifact_uri_local_dir_and_version_override(tmp_path, monkeypatch):
    """MODEL_ARTIFACT_URI + MODEL_VERSION when model_version.txt is absent."""
    model_dir = tmp_path / "from_uri"
    model_dir.mkdir()
    x = pd.DataFrame(
        [[20, 50.0, 1], [40, 80.0, 5]],
        columns=["age", "income_k", "tenure_years"],
    )
    y = np.array([30000.0, 60000.0])
    model = LinearRegression().fit(x, y)
    joblib.dump(model, model_dir / "sample_model.joblib")
    # No model_version.txt — version comes from env
    monkeypatch.setenv("MODEL_ARTIFACT_URI", str(model_dir))
    monkeypatch.setenv("MODEL_VERSION", "v-from-env")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "unused_default"))

    sys.modules.pop("api.app", None)
    app_module = importlib.import_module("api.app")
    client = app_module.app.test_client()

    health = client.get("/health")
    assert health.get_json()["model_version"] == "v-from-env"


def test_load_via_model_artifact_uri_tarball(tmp_path, monkeypatch):
    """Local tarball path containing sample_model.joblib + model_version.txt."""
    inner = tmp_path / "bundle"
    inner.mkdir()
    x = pd.DataFrame(
        [[20, 50.0, 1], [40, 80.0, 5]],
        columns=["age", "income_k", "tenure_years"],
    )
    y = np.array([30000.0, 60000.0])
    model = LinearRegression().fit(x, y)
    joblib.dump(model, inner / "sample_model.joblib")
    (inner / "model_version.txt").write_text("v-tar", encoding="utf-8")

    tar_path = tmp_path / "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for p in inner.iterdir():
            tf.add(p, arcname=p.name)

    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setenv("MODEL_ARTIFACT_URI", str(tar_path))
    monkeypatch.setenv("MODEL_DIR", str(staging))
    monkeypatch.delenv("MODEL_VERSION", raising=False)

    sys.modules.pop("api.app", None)
    app_module = importlib.import_module("api.app")
    client = app_module.app.test_client()

    assert client.get("/health").get_json()["model_version"] == "v-tar"


def test_load_model_accepts_legacy_regression_model_filename(tmp_path):
    model_dir = tmp_path / "legacy"
    model_dir.mkdir()
    x = pd.DataFrame(
        [[20, 50.0, 1], [40, 80.0, 5]],
        columns=["age", "income_k", "tenure_years"],
    )
    y = np.array([30000.0, 60000.0])
    model = LinearRegression().fit(x, y)
    joblib.dump(model, model_dir / "regression_model.joblib")
    (model_dir / "model_version.txt").write_text("v-legacy", encoding="utf-8")

    _model, model_version = model_loader.load_model(str(model_dir))

    assert model_version == "v-legacy"


def test_model_artifact_tarball_rejects_path_traversal(tmp_path):
    payload = tmp_path / "payload.txt"
    payload.write_text("escaped", encoding="utf-8")
    archive = tmp_path / "malicious.tar"
    outside_target = tmp_path / "outside.txt"

    with tarfile.open(archive, "w") as tf:
        tf.add(payload, arcname="../outside.txt")

    with pytest.raises(ValueError, match="Unsafe tar member"):
        model_loader._extract_tarball(archive, tmp_path / "extract")

    assert not outside_target.exists()


def test_model_artifact_tarball_rejects_links(tmp_path):
    archive = tmp_path / "malicious-link.tar"
    link_info = tarfile.TarInfo("sample_model.joblib")
    link_info.type = tarfile.SYMTYPE
    link_info.linkname = "/etc/passwd"

    with tarfile.open(archive, "w") as tf:
        tf.addfile(link_info)

    with pytest.raises(ValueError, match="links are not allowed"):
        model_loader._extract_tarball(archive, tmp_path / "extract")


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
