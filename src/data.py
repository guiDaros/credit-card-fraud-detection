"""Data loading utilities."""

from pathlib import Path

import pandas as pd

from src.config import DATA_PATH, TARGET

EXPECTED_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", TARGET]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the credit card fraud dataset and check its columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download 'creditcard.csv' from Kaggle "
            "(mlg-ulb/creditcardfraud) and place it in data/raw/."
        )

    df = pd.read_csv(path)

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    return df
