import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.data import FEATURES, load_training_data
from training.layout import (
    build_dir_structure,
    emit_manifest,
    resolve_next_model_version,
    save_artifacts,
)
from training.logging_utils import setup_file_logger
from training.modeling import train_model_and_metrics, validate_training_dataframe
from training.tracking import get_git_commit, log_to_mlflow


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

    runs_dir = output_dir / "runs"
    model_version = resolve_next_model_version(runs_dir)
    paths = build_dir_structure(output_dir, model_version)
    logger.info("Resolved model version: %s", model_version)

    try:
        validate_training_dataframe(df)
    except ValueError as exc:
        logger.error("Dataset validation failed: %s", exc)
        raise

    # Prepare training data
    logger.info("Dataset prepared: %s rows and %s features", len(df), len(FEATURES))

    model, computed_metrics = train_model_and_metrics(df)
    logger.info("Model trained: RMSE=%.4f", computed_metrics["rmse"])
    logger.info(
        "Computed metrics: rmse=%.4f, baseline_rmse=%.4f, passed_quality_evaluation=%s",
        computed_metrics["rmse"],
        computed_metrics["baseline_rmse"],
        computed_metrics["passed_quality_evaluation"],
    )

    git_commit = get_git_commit()
    metrics_payload = {
        "model_version": model_version,
        "rmse": computed_metrics["rmse"],
        "baseline_rmse": computed_metrics["baseline_rmse"],
        "passed_quality_evaluation": computed_metrics["passed_quality_evaluation"],
        "training_time_seconds": computed_metrics["training_time_seconds"],
    }
    if git_commit:
        metrics_payload["git_commit"] = git_commit

    if not computed_metrics["passed_quality_evaluation"]:
        raise RuntimeError("Model failed quality evaluation; artifacts were not published")

    save_artifacts(model, metrics_payload, model_version, paths)
    manifest = emit_manifest(model_version, metrics_payload, paths)

    log_to_mlflow(args, model_version, metrics_payload, paths, git_commit)

    logger.info("Saved run artifacts to: %s", paths["run_dir"])
    logger.info("Updated latest artifacts in: %s", paths["latest_dir"])
    logger.info("Emitted manifest at: %s", paths["latest_manifest_path"])
    logger.info("Manifest quality evaluation: %s", manifest["passed_quality_evaluation"])
    logger.info("Model version: %s", model_version)
    logger.info("Training log file: %s", log_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and version sample model.")
    parser.add_argument("--data-uri", type=str, default=None)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="runs/artifacts")
    parser.add_argument("--experiment-name", type=str, default="mlpipeline-sample")
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be > 0")
    return args


if __name__ == "__main__":
    try:
        parsed_args = parse_args()
        dataset = load_training_data(
            data_uri=parsed_args.data_uri,
            samples=parsed_args.samples,
            random_seed=parsed_args.random_seed,
        )
        validate_training_dataframe(dataset)
        train_and_log(parsed_args, dataset)
    except Exception:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
        logging.exception("Training failed")
        sys.exit(1)
