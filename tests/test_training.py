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


def test_failed_quality_run_does_not_update_latest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _stub_mlflow(monkeypatch)
    args = argparse.Namespace(
        samples=50,
        random_seed=42,
        output_dir="runs/artifacts",
        experiment_name="test-exp",
    )
    good_df = train_module.load_training_data(
        data_uri=None,
        samples=args.samples,
        random_seed=args.random_seed,
    )
    train_module.train_and_log(args, good_df)
    latest_version_path = Path("runs/artifacts/latest/model_version.txt")
    assert latest_version_path.read_text(encoding="utf-8").strip() == "v001"

    bad_df = pd.DataFrame(
        {
            "age": [0] * 80,
            "income_k": [0.0] * 80,
            "tenure_years": [0] * 80,
            "target": [1.0 if i % 2 else -1.0 for i in range(80)],
        }
    )

    with pytest.raises(RuntimeError, match="failed quality evaluation"):
        train_module.train_and_log(args, bad_df)

    assert latest_version_path.read_text(encoding="utf-8").strip() == "v001"
    latest_manifest = json.loads(
        Path("runs/artifacts/latest/manifest.json").read_text(encoding="utf-8")
    )
    assert latest_manifest["model_version"] == "v001"

    failed_metrics = json.loads(
        Path("runs/artifacts/runs/v002/metrics.json").read_text(encoding="utf-8")
    )
    failed_manifest = json.loads(
        Path("runs/artifacts/runs/v002/manifest.json").read_text(encoding="utf-8")
    )
    assert failed_metrics["passed_quality_evaluation"] is False
    assert failed_manifest["model_path"] == "runs/artifacts/runs/v002/sample_model.joblib"
