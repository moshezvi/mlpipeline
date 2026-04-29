import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from training.data import FEATURES, TARGET


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
    rmse = float(np.sqrt(mean_squared_error(y_train, predictions)))

    baseline_prediction = np.full_like(y_train, y_train.mean(), dtype=float)
    baseline_rmse = float(np.sqrt(mean_squared_error(y_train, baseline_prediction)))
    passed_guardrail = rmse < 0.95 * baseline_rmse

    return model, {
        "rmse": rmse,
        "baseline_rmse": baseline_rmse,
        "passed_guardrail": passed_guardrail,
        "training_time_seconds": training_time_seconds,
    }
