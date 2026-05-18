# Notebook Summary: `infra-mlpipeline.ipynb`

## What Data Is Used or Generated

- The notebook generates a synthetic dataset with 200 rows.
- It creates four columns:
  - `age`
  - `income_k`
  - `tenure_years`
  - `target`
- The dataset is saved to `../Data/dummy_data.csv`.
- `target` is computed from a linear formula plus Gaussian noise:
  - `10000 + 120*age + 500*income_k + 800*tenure_years + noise`

## Input Features

- `age`
- `income_k`
- `tenure_years`

## Target Variable

- `target` (continuous numeric value).

## Model Type

- The implemented training flow uses `LinearRegression` from scikit-learn.
- The notebook text/imports mention `XGBClassifier`, but the visible training and prediction path shown is regression.

## Preprocessing Steps

- No advanced preprocessing pipeline is shown.
- Features are selected directly from CSV columns.
- In inference helper functions:
  - payload can be a Python dict or a JSON string,
  - required fields are validated,
  - values are coerced to floats,
  - a one-row pandas DataFrame is constructed for prediction.

## What `model.predict()` Expects

- A tabular input with exactly these columns:
  - `age`
  - `income_k`
  - `tenure_years`
- In the notebook, prediction is called with a one-row pandas DataFrame.
- Output is a single numeric prediction extracted from index `0`.

## Recommendations

- Tighten imports based on current visible cells:
  - **Definitely unused active import:** `from xgboost import XGBClassifier` (no visible classification training or inference call uses it).
  - **Already commented out and safe to remove entirely if not needed later:** `tarfile`, `pandas_gbq`, `google.cloud.bigquery`, `google.oauth2.service_account`.
- Keep only imports required by the visible regression path (`os`, `joblib`, `numpy`, `LinearRegression`, `json`, `pandas`) to reduce noise and avoid confusion.
- Flag and fix stale outputs before sharing:
  - Current output in the test cell includes classification fields (`probability`, `label`) even though visible `predict_regression` returns only `{"prediction": ...}`.
  - This is an inconsistency likely from stale output state or cells from an earlier classification run.
