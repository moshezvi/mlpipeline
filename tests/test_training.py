import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

import training.train as train_module
import training.tracking as tracking_module


class _DummyRun:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _stub_mlflow(monkeypatch):
    monkeypatch.setattr(tracking_module.mlflow, "set_tracking_uri", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tracking_module.mlflow, "set_experiment", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tracking_module.mlflow, "start_run", lambda *_args, **_kwargs: _DummyRun())
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
    first = Path("runs/artifacts/latest/model_version.txt").read_text(encoding="utf-8").strip()

    train_module.train_and_log(args, df)
    second = Path("runs/artifacts/latest/model_version.txt").read_text(encoding="utf-8").strip()

    assert first == "v001"
    assert second == "v002"


def test_failed_quality_run_does_not_replace_latest_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _stub_mlflow(monkeypatch)
    args = argparse.Namespace(
        samples=20,
        random_seed=42,
        output_dir="runs/artifacts",
        experiment_name="test-exp",
    )
    passing_df = train_module.load_training_data(
        data_uri=None,
        samples=args.samples,
        random_seed=args.random_seed,
    )
    train_module.train_and_log(args, passing_df)
    latest_version_path = Path("runs/artifacts/latest/model_version.txt")
    assert latest_version_path.read_text(encoding="utf-8").strip() == "v001"

    failing_df = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "income_k": [50.0, 60.0, 70.0, 80.0],
            "tenure_years": [1, 2, 3, 4],
            "target": [100.0, 100.0, 100.0, 100.0],
        }
    )

    with pytest.raises(RuntimeError, match="failed quality evaluation"):
        train_module.train_and_log(args, failing_df)

    assert latest_version_path.read_text(encoding="utf-8").strip() == "v001"
    latest_manifest = json.loads(
        Path("runs/artifacts/latest/manifest.json").read_text(encoding="utf-8")
    )
    failed_manifest = json.loads(
        Path("runs/artifacts/runs/v002/manifest.json").read_text(encoding="utf-8")
    )
    assert latest_manifest["model_version"] == "v001"
    assert failed_manifest["model_version"] == "v002"
    assert failed_manifest["passed_quality_evaluation"] is False
