"""Tests for feature engineering and data splitting, using small synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src.config import TARGET
from src.data import split_data
from src.features import add_features


@pytest.fixture
def synthetic_df():
    """Small fake dataset with the same columns as the real one (5% fraud)."""
    rng = np.random.default_rng(0)
    n_rows = 1000
    data = {"Time": rng.uniform(0, 172800, n_rows)}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(size=n_rows)
    data["Amount"] = rng.uniform(0, 500, n_rows)
    data[TARGET] = [1] * 50 + [0] * (n_rows - 50)
    return pd.DataFrame(data)


def test_add_features_creates_and_drops_columns(synthetic_df):
    result = add_features(synthetic_df)

    assert "Amount_log" in result.columns
    assert "Hour" in result.columns
    assert "Time" not in result.columns
    assert "Amount" not in result.columns

    assert np.allclose(result["Amount_log"], np.log1p(synthetic_df["Amount"]))
    assert result["Hour"].between(0, 23).all()

    # The original DataFrame must not be modified.
    assert "Time" in synthetic_df.columns
    assert "Amount" in synthetic_df.columns


def test_random_split_keeps_fraud_ratio(synthetic_df):
    X_train, X_test, y_train, y_test = split_data(synthetic_df, method="random")

    overall_ratio = synthetic_df[TARGET].mean()
    assert y_train.mean() == pytest.approx(overall_ratio, abs=0.005)
    assert y_test.mean() == pytest.approx(overall_ratio, abs=0.005)


@pytest.mark.parametrize("method", ["random", "temporal"])
def test_split_data_has_no_overlap(synthetic_df, method):
    X_train, X_test, y_train, y_test = split_data(synthetic_df, method=method)

    assert len(X_train) + len(X_test) == len(synthetic_df)
    assert set(X_train.index).isdisjoint(X_test.index)


def test_temporal_split_train_before_test(synthetic_df):
    X_train, X_test, _, _ = split_data(synthetic_df, method="temporal")

    assert X_train["Time"].max() < X_test["Time"].min()


def test_temporal_split_keeps_tied_times_together(synthetic_df):
    # Round Time to whole hours so many rows share the same Time value.
    df = synthetic_df.assign(Time=(synthetic_df["Time"] // 3600) * 3600)
    X_train, X_test, _, _ = split_data(df, method="temporal")

    assert X_train["Time"].max() < X_test["Time"].min()


def test_split_keeps_time_and_amount(synthetic_df):
    X_train, X_test, _, _ = split_data(synthetic_df)

    for X in (X_train, X_test):
        assert "Time" in X.columns
        assert "Amount" in X.columns


def test_add_features_after_split(synthetic_df):
    X_train, X_test, _, _ = split_data(synthetic_df, method="temporal")
    train_feats = add_features(X_train)
    test_feats = add_features(X_test)

    assert list(train_feats.columns) == list(test_feats.columns)
    assert "Time" not in train_feats.columns


def test_split_warns_when_few_test_frauds(synthetic_df, caplog):
    df = synthetic_df.copy()
    df[TARGET] = [1] * 10 + [0] * (len(df) - 10)
    with caplog.at_level("WARNING"):
        split_data(df)
    assert "frauds" in caplog.text


def test_split_data_rejects_unknown_method(synthetic_df):
    with pytest.raises(ValueError):
        split_data(synthetic_df, method="other")
