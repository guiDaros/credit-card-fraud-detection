"""Data loading utilities."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, SEED, TARGET, TEST_SIZE

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


def split_data(df: pd.DataFrame):
    """Split features and target into stratified train and test sets.

    Returns X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Stratify on y so both sets keep the same (very small) share of fraud.
    # IMPORTANT: the test set must only be used for the final evaluation.
    # Do not use it for tuning, threshold selection or any other decision.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    return X_train, X_test, y_train, y_test
