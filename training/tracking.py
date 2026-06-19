import argparse
import subprocess
from pathlib import Path

import mlflow


def get_git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() or None
        )
    except Exception:
        return None


def log_to_mlflow(
    args: argparse.Namespace,
    model_version: str,
    metrics_payload: dict[str, float | bool | str],
    paths: dict[str, Path],
    git_commit: str | None,
) -> None:
    mlflow.set_tracking_uri("file:./runs/mlruns")
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=model_version):
        mlflow.log_params({"random_seed": args.random_seed, "samples": args.samples})
        mlflow.log_metric("rmse", float(metrics_payload["rmse"]))
        mlflow.log_metric("baseline_rmse", float(metrics_payload["baseline_rmse"]))
        mlflow.log_param("model_version", model_version)
        if git_commit:
            mlflow.log_param("git_commit", git_commit)
        for artifact_key in (
            "run_model_path",
            "run_metrics_path",
            "run_version_path",
            "run_manifest_path",
            "latest_model_path",
            "latest_metrics_path",
            "latest_version_path",
            "latest_manifest_path",
        ):
            artifact_path = paths[artifact_key]
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))
