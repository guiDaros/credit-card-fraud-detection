"""Tests for the raw dataset schema, using small synthetic data."""

import numpy as np
import pandas as pd
import pandera.errors
import pytest

from src.schema import validate_raw


def make_valid_df(n_rows: int = 5) -> pd.DataFrame:
    """Build a small synthetic DataFrame that matches the raw schema."""
    rng = np.random.default_rng(0)
    data = {"Time": np.arange(n_rows, dtype=float)}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(size=n_rows)
    data["Amount"] = rng.uniform(0, 100, size=n_rows)
    data["Class"] = np.array([0, 1, 0, 0, 1], dtype=int)[:n_rows]
    return pd.DataFrame(data)


def test_valid_df_passes():
    df = make_valid_df()
    result = validate_raw(df)
    pd.testing.assert_frame_equal(result, df)


def test_negative_amount_fails():
    df = make_valid_df()
    df.loc[0, "Amount"] = -1.0
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_raw(df)


def test_invalid_class_fails():
    df = make_valid_df()
    df.loc[0, "Class"] = 2
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_raw(df)


def test_missing_column_fails():
    df = make_valid_df().drop(columns=["V5"])
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_raw(df)
