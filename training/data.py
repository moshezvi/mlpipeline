import numpy as np
import pandas as pd


FEATURES = ["age", "income_k", "tenure_years"]
TARGET = "target"


# used for local development and testing
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


# Future enhancement: load data from a URI (e.g. GCS, S3, etc.)
def load_training_data(data_uri: str | None, samples: int, random_seed: int) -> pd.DataFrame:
    if data_uri:
        return pd.read_csv(data_uri)
    return generate_dataset(samples=samples, random_seed=random_seed)
