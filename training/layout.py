import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.linear_model import LinearRegression


def resolve_next_model_version(runs_dir: Path) -> str:
    existing_versions = []
    for run_dir in runs_dir.glob("v*"):
        if run_dir.is_dir():
            version_part = run_dir.name.removeprefix("v")
            if version_part.isdigit():
                existing_versions.append(int(version_part))
    next_version_num = (max(existing_versions) + 1) if existing_versions else 1
    return f"v{next_version_num:03d}"


def build_dir_structure(output_dir: Path, model_version: str) -> dict[str, Path]:
    model_name = "sample_model"
    runs_dir = output_dir / "runs"
    latest_dir = output_dir / "latest"
    runs_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    run_dir = runs_dir / model_version
    run_dir.mkdir(parents=True, exist_ok=True)

    return {
        "runs_dir": runs_dir,
        "latest_dir": latest_dir,
        "run_dir": run_dir,
        "run_model_path": run_dir / f"{model_name}.joblib",
        "run_metrics_path": run_dir / "metrics.json",
        "run_version_path": run_dir / "model_version.txt",
        "run_manifest_path": run_dir / "manifest.json",
        "latest_model_path": latest_dir / f"{model_name}.joblib",
        "latest_metrics_path": latest_dir / "metrics.json",
        "latest_version_path": latest_dir / "model_version.txt",
        "latest_manifest_path": latest_dir / "manifest.json",
    }


def save_artifacts(
    model: LinearRegression,
    metrics_payload: dict[str, float | bool | str],
    model_version: str,
    paths: dict[str, Path],
) -> None:
    joblib.dump(model, paths["run_model_path"])
    paths["run_metrics_path"].write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )
    paths["run_version_path"].write_text(model_version, encoding="utf-8")

    if bool(metrics_payload["passed_quality_evaluation"]):
        joblib.dump(model, paths["latest_model_path"])
        paths["latest_metrics_path"].write_text(
            json.dumps(metrics_payload, indent=2),
            encoding="utf-8",
        )
        paths["latest_version_path"].write_text(model_version, encoding="utf-8")


def emit_manifest(
    model_version: str,
    metrics_payload: dict[str, float | bool | str],
    paths: dict[str, Path],
) -> dict[str, str | bool | float]:
    manifest = {
        "model_version": model_version,
        "model_path": str(paths["run_model_path"]),
        "metrics_path": str(paths["run_metrics_path"]),
        "passed_quality_evaluation": bool(metrics_payload["passed_quality_evaluation"]),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if "git_commit" in metrics_payload:
        manifest["git_commit"] = str(metrics_payload["git_commit"])

    paths["run_manifest_path"].write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    if manifest["passed_quality_evaluation"]:
        paths["latest_manifest_path"].write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
    return manifest
