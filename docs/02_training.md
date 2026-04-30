# Training Script Guide

This guide explains what `training/train.py` does and what it produces.

## What the script does

`training/train.py` trains a simple regression model on synthetic data and logs the run to MLflow.

High-level flow:

1. Parse CLI arguments (`samples`, `random_seed`, `output_dir`, `experiment_name`, `data_uri`).
2. Create a timestamped log file in `logs/`.
3. Resolve the next rolling version (`vNNN`) by scanning existing run directories.
4. Generate synthetic training data with features:
   - `age`
   - `income_k`
   - `tenure_years`
   and target:
   - `target`
5. Train `LinearRegression` on that data.
6. Compute:
   - `rmse`
   - `baseline_rmse` (mean predictor baseline)
7. Save run artifacts and update latest artifacts.
8. Emit a training manifest (`manifest.json`) for release handoff.
9. Log params, metrics, and artifacts to MLflow.

## Inputs

Default CLI args:

- `--samples 200`
- `--random-seed 42`
- `--output-dir runs/artifacts`
- `--experiment-name mlpipeline-sample`
- `--data-uri` optional CSV path/URI (falls back to synthetic data when omitted)

Example:

```bash
python training/train.py --samples 200 --random-seed 42 --output-dir runs/artifacts
```

## Outputs

In `runs/artifacts/`:

- `runs/vNNN/` (immutable per run), containing:
  - `sample_model.joblib`
  - `metrics.json`
  - `model_version.txt`
  - `manifest.json`
- `latest/` (active model alias), containing:
  - `sample_model.joblib`
  - `metrics.json`
  - `model_version.txt`
  - `manifest.json`

`metrics.json` includes:

- `model_version`
- `rmse`
- `baseline_rmse`
- `git_commit` (when available)

In `logs/`:

- `train-YYYYMMDD-HHMMSS.log` - timestamped training log file

In MLflow:

- experiment/run with logged params, metrics, and artifact files

## Versioning behavior

The version format is:

- `vNNN` (for example `v001`, `v002`, `v003`)

The script scans `artifacts/runs/` for existing `vNNN` directories and picks the next number.
Each new run writes immutable artifacts to `runs/artifacts/runs/vNNN/` and refreshes `runs/artifacts/latest/`.
