import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from training.data import FEATURES, TARGET


# Compare the model's RMSE to a baseline where predictions are the mean of the target
def evaluate_quality(
    y_true: pd.Series | np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float | bool]:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(predictions, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    baseline_prediction = np.full_like(y, y.mean(), dtype=float)
    baseline_rmse = float(np.sqrt(mean_squared_error(y, baseline_prediction)))
    passed_quality_evaluation = rmse < 0.95 * baseline_rmse
    return {
        "rmse": rmse,
        "baseline_rmse": baseline_rmse,
        "passed_quality_evaluation": passed_quality_evaluation,
    }


def validate_training_dataframe(df: pd.DataFrame) -> None:
    required_columns = FEATURES + [TARGET]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"dataset missing required columns: {missing_columns}")
    if df.empty:
        raise ValueError("dataset is empty")
    non_numeric_columns = [
        col for col in required_columns if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric_columns:
        raise ValueError(f"dataset columns must be numeric: {non_numeric_columns}")


def train_model_and_metrics(df: pd.DataFrame) -> tuple[LinearRegression, dict[str, float | bool]]:
    x_train = df[FEATURES]
    y_train = df[TARGET]

    train_start = time.perf_counter()
    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_train)
    training_time_seconds = time.perf_counter() - train_start

    quality = evaluate_quality(y_train, predictions)

    return model, {
        **quality,
        "training_time_seconds": training_time_seconds,
    }
