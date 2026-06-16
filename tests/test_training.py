import argparse
import json
from pathlib import Path

from sklearn.linear_model import LinearRegression

from training.data import FEATURES, TARGET
import training.train as train_module
import training.tracking as tracking_module


class _DummyRun:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _stub_mlflow(monkeypatch):
    monkeypatch.setattr(
        tracking_module.mlflow,
        "set_tracking_uri",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(tracking_module.mlflow, "set_experiment", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        tracking_module.mlflow,
        "start_run",
        lambda *_args, **_kwargs: _DummyRun(),
    )
    monkeypatch.setattr(tracking_module.mlflow, "log_params", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tracking_module.mlflow, "log_metric", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tracking_module.mlflow, "log_param", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tracking_module.mlflow, "log_artifact", lambda *_args, **_kwargs: None)


def test_training_writes_metrics_contract(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _stub_mlflow(monkeypatch)
    args = argparse.Namespace(
        samples=20,
        random_seed=42,
        output_dir="runs/artifacts",
        experiment_name="test-exp",
    )
    df = train_module.load_training_data(
        data_uri=None,
        samples=args.samples,
        random_seed=args.random_seed,
    )

    train_module.train_and_log(args, df)

    metrics_path = Path("runs/artifacts/latest/metrics.json")
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    required_keys = {
        "model_version",
        "rmse",
        "baseline_rmse",
        "passed_quality_evaluation",
        "training_time_seconds",
    }
    assert required_keys.issubset(metrics.keys())


def test_training_manifest_uses_immutable_run_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _stub_mlflow(monkeypatch)
    args = argparse.Namespace(
        samples=20,
        random_seed=42,
        output_dir="runs/artifacts",
        experiment_name="test-exp",
    )
    df = train_module.load_training_data(
        data_uri=None,
        samples=args.samples,
        random_seed=args.random_seed,
    )

    train_module.train_and_log(args, df)

    manifest_path = Path("runs/artifacts/latest/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_version"] == "v001"
    assert manifest["model_path"] == "runs/artifacts/runs/v001/sample_model.joblib"
    assert manifest["metrics_path"] == "runs/artifacts/runs/v001/metrics.json"
    assert Path(manifest["model_path"]).exists()
    assert Path(manifest["metrics_path"]).exists()


def test_training_increments_model_version(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _stub_mlflow(monkeypatch)
    args = argparse.Namespace(
        samples=20,
        random_seed=42,
        output_dir="runs/artifacts",
        experiment_name="test-exp",
    )
    df = train_module.load_training_data(
        data_uri=None,
        samples=args.samples,
        random_seed=args.random_seed,
    )

    train_module.train_and_log(args, df)
    first = (
        Path("runs/artifacts/latest/model_version.txt")
        .read_text(encoding="utf-8")
        .strip()
    )

    train_module.train_and_log(args, df)
    second = (
        Path("runs/artifacts/latest/model_version.txt")
        .read_text(encoding="utf-8")
        .strip()
    )

    assert first == "v001"
    assert second == "v002"


def test_failed_quality_run_does_not_update_latest_alias(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _stub_mlflow(monkeypatch)
    args = argparse.Namespace(
        samples=20,
        random_seed=42,
        output_dir="runs/artifacts",
        experiment_name="test-exp",
    )
    df = train_module.load_training_data(
        data_uri=None,
        samples=args.samples,
        random_seed=args.random_seed,
    )

    train_module.train_and_log(args, df)
    latest_version_path = Path("runs/artifacts/latest/model_version.txt")
    assert latest_version_path.read_text(encoding="utf-8").strip() == "v001"

    def train_failed_quality_model(failed_df):
        model = LinearRegression().fit(failed_df[FEATURES], failed_df[TARGET])
        return model, {
            "rmse": 10.0,
            "baseline_rmse": 10.0,
            "passed_quality_evaluation": False,
            "training_time_seconds": 0.0,
        }

    monkeypatch.setattr(
        train_module,
        "train_model_and_metrics",
        train_failed_quality_model,
    )
    train_module.train_and_log(args, df)

    failed_manifest = json.loads(
        Path("runs/artifacts/runs/v002/manifest.json").read_text(encoding="utf-8")
    )
    latest_manifest = json.loads(
        Path("runs/artifacts/latest/manifest.json").read_text(encoding="utf-8")
    )
    assert failed_manifest["passed_quality_evaluation"] is False
    assert (
        failed_manifest["model_path"] == "runs/artifacts/runs/v002/sample_model.joblib"
    )
    assert latest_version_path.read_text(encoding="utf-8").strip() == "v001"
    assert latest_manifest["model_version"] == "v001"
