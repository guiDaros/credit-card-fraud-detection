"""Tests for scripts/compare_hour.py, using small synthetic data."""

import sys

import numpy as np
import pandas as pd

import scripts.compare_hour as compare_hour
from src.config import TARGET
from src.data import split_data

PARAMS = {"n_estimators": 5, "max_depth": 3}


def make_processed_df(n_rows: int = 400) -> pd.DataFrame:
    """Fake processed dataset (sorted by Time), ~10% fraud spread over time."""
    rng = np.random.default_rng(0)
    y = (np.arange(n_rows) % 10 == 0).astype(int)
    df = pd.DataFrame({"Time": np.sort(rng.uniform(0, 172800, n_rows))})
    for i in range(1, 29):
        df[f"V{i}"] = rng.normal(size=n_rows)
    df["V1"] += 2 * y
    df["Amount"] = rng.uniform(0, 500, n_rows)
    df[TARGET] = y
    return df


def test_compare_hour_scores_same_folds():
    df = make_processed_df()
    X, y = df.drop(columns=TARGET), df[TARGET]

    scores = compare_hour.compare_hour(X, y, PARAMS, n_splits=3)

    assert len(scores["with_hour"]) == len(scores["without_hour"]) == 3
    for values in scores.values():
        assert all(0 <= v <= 1 for v in values)


def test_main_uses_only_the_training_set(monkeypatch, capsys):
    df = make_processed_df()
    _, X_test, _, _ = split_data(df, method="temporal")
    seen_times = []
    real_oof = compare_hour.temporal_oof_proba

    def spy_oof(model_factory, X, y, n_splits):
        seen_times.append(X["Time"].to_numpy())
        return real_oof(model_factory, X, y, n_splits=n_splits)

    monkeypatch.setattr(compare_hour, "load_processed", lambda: df.copy())
    monkeypatch.setattr(compare_hour, "temporal_oof_proba", spy_oof)
    monkeypatch.setattr(compare_hour, "load_params", lambda path: (dict(PARAMS), "test"))
    monkeypatch.setattr(sys, "argv", ["compare_hour", "--n-splits", "3"])

    compare_hour.main()

    assert len(seen_times) == 2  # with and without Hour
    for times in seen_times:
        assert times.max() < X_test["Time"].min()  # no test row reaches the CV
        assert np.all(np.diff(times) >= 0)  # sorted by Time
    assert "Test set not used" in capsys.readouterr().out
