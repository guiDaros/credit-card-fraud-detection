"""Data loading utilities."""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, PROCESSED_PATH, SEED, TARGET, TEST_SIZE
from src.schema import validate_raw

logger = logging.getLogger(__name__)

# Warn when the test set has fewer frauds than this.
MIN_TEST_FRAUDS = 50


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
    """Load the processed (deduplicated, sorted) dataset and validate it.

    Besides the pandera schema, checks the promises of make_dataset: Time is in
    non-decreasing order and there are no exact duplicate rows. Raises ValueError
    otherwise (the file is outdated or was changed by hand).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run 'python -m src.make_dataset' first."
        )

    df = validate_raw(pd.read_parquet(path))

    problems = []
    if not df["Time"].is_monotonic_increasing:  # pandas: True also allows equal values
        problems.append("Time is not in non-decreasing order")
    n_duplicates = int(df.duplicated().sum())
    if n_duplicates:
        problems.append(f"it has {n_duplicates} duplicate row(s)")
    if problems:
        raise ValueError(
            f"Processed dataset {path} is not valid: {'; '.join(problems)}. "
            "Rebuild it from data/raw with `make data`."
        )
    return df


def split_data(df: pd.DataFrame, method: str = "temporal"):
    """Split features and target into train and test sets.

    method="temporal" (default, same as the CLI): no shuffling and no stratification. The cut is the
    (1 - TEST_SIZE) quantile of Time; train is Time <= cut and test is
    Time > cut, so rows with the same Time never end up on both sides.
    method="random": stratified random split (SEED, TEST_SIZE).

    Returns X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # IMPORTANT: the test set must only be used for the final evaluation.
    # Do not use it for tuning, threshold selection or any other decision.
    if method == "random":
        # Stratify on y so both sets keep the same (very small) share of fraud.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
        )
    elif method == "temporal":
        cut = df["Time"].quantile(1 - TEST_SIZE)
        is_train = df["Time"] <= cut
        X_train, X_test = X[is_train], X[~is_train]
        y_train, y_test = y[is_train], y[~is_train]
    else:
        raise ValueError(f"Unknown split method: {method!r}. Use 'random' or 'temporal'.")

    n_test_frauds = int(y_test.sum())
    if n_test_frauds < MIN_TEST_FRAUDS:
        logger.warning(
            "Test set has only %d frauds (less than %d); metrics will be noisy.",
            n_test_frauds,
            MIN_TEST_FRAUDS,
        )

    return X_train, X_test, y_train, y_test
