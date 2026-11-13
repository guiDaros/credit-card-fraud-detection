"""Tests for the dataset cleaning step, using small synthetic data."""

import numpy as np
import pandas as pd

from src.make_dataset import clean_dataset


def make_df_with_duplicates() -> pd.DataFrame:
    """5 unique rows plus 2 exact duplicates (one of them a fraud), unsorted by Time."""
    rng = np.random.default_rng(0)
    n = 5
    data = {"Time": np.array([30.0, 10.0, 20.0, 50.0, 40.0])}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(size=n)
    data["Amount"] = rng.uniform(0, 100, size=n)
    data["Class"] = np.array([0, 1, 0, 0, 0])
    df = pd.DataFrame(data)
    # Duplicate row 1 (fraud) and row 3 (normal).
    return pd.concat([df, df.iloc[[1, 3]]], ignore_index=True)


def test_clean_dataset_removes_duplicates_and_sorts():
    df = make_df_with_duplicates()
    clean, report = clean_dataset(df)

    assert report["rows_before"] == 7
    assert report["rows_after"] == 5
    assert report["duplicates_removed"] == 2
    assert report["frauds_before"] == 2
    assert report["frauds_after"] == 1
    assert report["fraud_rate_after"] == 1 / 5

    assert not clean.duplicated().any()
    assert clean["Time"].is_monotonic_increasing
    assert list(clean["Time"]) == [10.0, 20.0, 30.0, 40.0, 50.0]
