"""Data loading utilities."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, PROCESSED_PATH, SEED, TARGET, TEST_SIZE
from src.schema import validate_raw


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the credit card fraud dataset and validate it against the schema."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download 'creditcard.csv' from Kaggle "
            "(mlg-ulb/creditcardfraud) and place it in data/raw/."
        )

    df = pd.read_csv(path)

    return validate_raw(df)


def load_processed(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """Load the processed (deduplicated) dataset and validate it."""
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. "
            "Run 'python -m src.make_dataset' first."
        )

    df = pd.read_parquet(path)
    return validate_raw(df)


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
