import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


FEATURES = ["age", "income_k", "tenure_years"]
TARGET = "target"


def generate_dataset(samples: int, random_seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    age = rng.integers(18, 70, size=samples)
    income_k = rng.normal(70, 15, size=samples).clip(30, 120)
    tenure_years = rng.integers(0, 10, size=samples)

    target = (
        10000
        + 120 * age
        + 500 * income_k
        + 800 * tenure_years
        + rng.normal(0, 3000, size=samples)
    )

    return pd.DataFrame(
        {
            "age": age,
            "income_k": income_k,
            "tenure_years": tenure_years,
            "target": target,
        }
    )


def setup_file_logger() -> tuple[logging.Logger, Path]:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.FileHandler(log_path, encoding="utf-8"))
    return logger, log_path


def get_git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() or None
        )
    except Exception:
        return None


def resolve_next_model_version(runs_dir: Path) -> str:
    existing_versions = []
    for run_dir in runs_dir.glob("v*"):
        if run_dir.is_dir():
            version_part = run_dir.name.removeprefix("v")
            if version_part.isdigit():
                existing_versions.append(int(version_part))
    next_version_num = (max(existing_versions) + 1) if existing_versions else 1
    return f"v{next_version_num:03d}"


def train_and_log(args: argparse.Namespace, df: pd.DataFrame) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger, log_path = setup_file_logger()
    logger.info(
        "Training started: samples=%s, random_seed=%s, output_dir=%s, experiment_name=%s",
        args.samples,
        args.random_seed,
        output_dir,
        args.experiment_name,
    )

    model_name = "regression_model"
    runs_dir = output_dir / "runs"
    latest_dir = output_dir / "latest"
    runs_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    latest_model_path = latest_dir / f"{model_name}.joblib"
    latest_metrics_path = latest_dir / "metrics.json"
    latest_version_path = latest_dir / "model_version.txt"

    model_version = resolve_next_model_version(runs_dir)
    run_dir = runs_dir / model_version
    run_dir.mkdir(parents=True, exist_ok=True)
    run_model_path = run_dir / f"{model_name}.joblib"
    run_metrics_path = run_dir / "metrics.json"
    run_version_path = run_dir / "model_version.txt"
    logger.info("Resolved model version: %s", model_version)

    required_columns = FEATURES + [TARGET]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error("Dataset missing required columns: %s", missing_columns)
        raise ValueError(f"dataset missing required columns: {missing_columns}")
    if df.empty:
        logger.error("Dataset is empty")
        raise ValueError("dataset is empty")
    non_numeric_columns = [
        col for col in required_columns if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric_columns:
        logger.error("Dataset columns must be numeric: %s", non_numeric_columns)
        raise ValueError(f"dataset columns must be numeric: {non_numeric_columns}")

    # Prepare training data
    x_train = df[FEATURES]
    y_train = df[TARGET]
    logger.info("Dataset prepared: %s rows and %s features", len(df), len(FEATURES))

    train_start = time.perf_counter()
    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_train)
    training_time_seconds = time.perf_counter() - train_start
    rmse = float(np.sqrt(mean_squared_error(y_train, predictions)))
    logger.info("Model trained: RMSE=%.4f", rmse)
    
    # Baseline RMSE is the RMSE of the mean of the target variable - used for monitoring
    baseline_prediction = np.full_like(y_train, y_train.mean(), dtype=float)
    baseline_rmse = float(np.sqrt(mean_squared_error(y_train, baseline_prediction)))
    
    # Guardrail check: RMSE must be less than 95% of the baseline RMSE
    if rmse < 0.95 * baseline_rmse:
        passed_guardrail = True
    else:
        passed_guardrail = False
    logger.info("Computed metrics: rmse=%.4f, baseline_rmse=%.4f, passed_guardrail=%s", rmse, baseline_rmse, passed_guardrail)

    git_commit = get_git_commit()
    metrics_payload = {
        "model_version": model_version,
        "rmse": rmse,
        "baseline_rmse": baseline_rmse,
        "passed_guardrail": passed_guardrail,
        "training_time_seconds": training_time_seconds,
    }
    if git_commit:
        metrics_payload["git_commit"] = git_commit

    # Save model and metrics to run directory
    joblib.dump(model, run_model_path)
    run_metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    run_version_path.write_text(model_version, encoding="utf-8")

    joblib.dump(model, latest_model_path)
    latest_metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    latest_version_path.write_text(model_version, encoding="utf-8")

    # Log run to MLflow
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=model_version):
        mlflow.log_params({"random_seed": args.random_seed, "samples": args.samples})
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("baseline_rmse", baseline_rmse)
        mlflow.log_param("model_version", model_version)
        if git_commit:
            mlflow.log_param("git_commit", git_commit)
        mlflow.log_artifact(str(run_model_path))
        mlflow.log_artifact(str(run_metrics_path))
        mlflow.log_artifact(str(run_version_path))
        mlflow.log_artifact(str(latest_model_path))
        mlflow.log_artifact(str(latest_metrics_path))
        mlflow.log_artifact(str(latest_version_path))

    logger.info("Saved run artifacts to: %s", run_dir)
    logger.info("Updated latest artifacts in: %s", latest_dir)
    logger.info("Model version: %s", model_version)
    logger.info("Training log file: %s", log_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and version regression model.")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--experiment-name", type=str, default="mlpipeline-takehome")
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be > 0")
    return args


if __name__ == "__main__":
    try:
        parsed_args = parse_args()
        dataset = generate_dataset(samples=parsed_args.samples, random_seed=parsed_args.random_seed)
        required_columns = FEATURES + [TARGET]
        missing_columns = [col for col in required_columns if col not in dataset.columns]
        if missing_columns:
            raise ValueError(f"dataset missing required columns: {missing_columns}")
        if dataset.empty:
            raise ValueError("dataset is empty")
        non_numeric_columns = [
            col for col in required_columns if not pd.api.types.is_numeric_dtype(dataset[col])
        ]
        if non_numeric_columns:
            raise ValueError(f"dataset columns must be numeric: {non_numeric_columns}")
        train_and_log(parsed_args, dataset)
    except Exception as exc:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
        logging.exception("Training failed")
        sys.exit(1)
