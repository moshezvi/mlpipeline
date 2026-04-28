from pathlib import Path

import joblib


def load_model(artifacts_dir: str):
    base = Path(artifacts_dir)
    model_path = base / "regression_model.joblib"
    version_path = base / "model_version.txt"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not version_path.exists():
        raise FileNotFoundError(f"Model version not found: {version_path}")

    model = joblib.load(model_path)
    model_version = version_path.read_text(encoding="utf-8").strip()
    return model, model_version
