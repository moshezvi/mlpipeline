import importlib
import io
import sys
import tarfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression


def _prepare_model_artifacts(base_dir: Path, model_filename: str = "sample_model.joblib"):
    model_dir = base_dir / "runs" / "artifacts" / "latest"
    model_dir.mkdir(parents=True, exist_ok=True)

    x = pd.DataFrame(
        [[20, 50.0, 1], [40, 80.0, 5], [60, 100.0, 8]],
        columns=["age", "income_k", "tenure_years"],
    )
    y = np.array([30000.0, 60000.0, 90000.0])
    model = LinearRegression().fit(x, y)

    joblib.dump(model, model_dir / model_filename)
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


def test_loads_legacy_regression_model_artifacts(tmp_path, monkeypatch):
    model_dir = _prepare_model_artifacts(tmp_path, model_filename="regression_model.joblib")
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.delenv("MODEL_ARTIFACT_URI", raising=False)
    monkeypatch.delenv("MODEL_VERSION", raising=False)

    sys.modules.pop("api.app", None)
    app_module = importlib.import_module("api.app")
    client = app_module.app.test_client()

    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["model_version"] == "vtest"


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


def test_rejects_tarball_path_traversal(tmp_path):
    from api.model_loader import _extract_tarball

    tar_path = tmp_path / "malicious.tar"
    with tarfile.open(tar_path, "w") as tf:
        payload = b"overwrite"
        member = tarfile.TarInfo("../escape.txt")
        member.size = len(payload)
        tf.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError, match="Unsafe tar member path"):
        _extract_tarball(tar_path, tmp_path / "dest")

    assert not (tmp_path / "escape.txt").exists()


def test_rejects_tarball_links(tmp_path):
    from api.model_loader import _extract_tarball

    tar_path = tmp_path / "link.tar"
    with tarfile.open(tar_path, "w") as tf:
        member = tarfile.TarInfo("linked-model")
        member.type = tarfile.SYMTYPE
        member.linkname = "/etc/passwd"
        tf.addfile(member)

    with pytest.raises(ValueError, match="Unsupported tar member type"):
        _extract_tarball(tar_path, tmp_path / "dest")


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
