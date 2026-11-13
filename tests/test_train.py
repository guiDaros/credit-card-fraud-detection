"""Tests for tuning and threshold selection, using small synthetic data."""

import numpy as np
import pandas as pd
import pytest

import src.train as train
from src.cv import temporal_oof_proba

N_ROWS = 300
N_TRIALS = 2


@pytest.fixture
def train_data():
    """Small fake training set, already in Time order, with ~10% fraud spread over time."""
    rng = np.random.default_rng(0)
    y = pd.Series((np.arange(N_ROWS) % 10 == 0).astype(int))
    X = pd.DataFrame(
        {
            "V1": rng.normal(size=N_ROWS) + 2 * y,
            "V2": rng.normal(size=N_ROWS),
        }
    )
    return X, y


def expected_temporal_rows(X, y, n_splits):
    """Number of rows that temporal_oof_proba gives a prediction to."""
    _, mask = temporal_oof_proba(
        lambda: train.make_rf(n_estimators=5, max_depth=3), X, y, n_splits=n_splits
    )
    return int(mask.sum())


@pytest.mark.parametrize("split", ["random", "temporal"])
def test_tune_runs_and_scores_right_rows(train_data, split, monkeypatch):
    X, y = train_data
    scored_sizes = []
    real_ap = train.average_precision_score

    def spy_ap(y_true, proba):
        scored_sizes.append(len(y_true))
        return real_ap(y_true, proba)

    monkeypatch.setattr(train, "average_precision_score", spy_ap)
    study = train.tune(X, y, n_trials=N_TRIALS, split=split)

    assert len(study.trials) == N_TRIALS
    assert 0 <= study.best_value <= 1
    expected = N_ROWS if split == "random" else expected_temporal_rows(X, y, 3)
    assert scored_sizes == [expected] * N_TRIALS
    if split == "temporal":
        assert expected < N_ROWS  # the first block is never scored


@pytest.mark.parametrize("split", ["random", "temporal"])
def test_choose_threshold_uses_right_rows(train_data, split, monkeypatch):
    X, y = train_data
    seen = {}
    real_best = train.best_f1_threshold

    def spy_best(y_true, proba):
        seen["n"] = len(y_true)
        seen["has_nan"] = bool(np.isnan(proba).any())
        return real_best(y_true, proba)

    monkeypatch.setattr(train, "best_f1_threshold", spy_best)
    threshold = train.choose_threshold({"n_estimators": 20, "max_depth": 4}, X, y, split=split)

    assert 0 <= threshold <= 1
    expected = N_ROWS if split == "random" else expected_temporal_rows(X, y, 5)
    assert seen["n"] == expected
    assert not seen["has_nan"]


def test_temporal_rejects_unsorted_data(train_data):
    X, y = train_data
    X_shuffled = X.sample(frac=1, random_state=0)
    y_shuffled = y.loc[X_shuffled.index]

    with pytest.raises(ValueError, match="sorted by Time"):
        train.choose_threshold({"n_estimators": 5}, X_shuffled, y_shuffled, split="temporal")


def test_temporal_checks_time_column_when_present():
    X = pd.DataFrame({"Time": [3.0, 1.0, 2.0]})
    with pytest.raises(ValueError, match="sorted by Time"):
        train.check_time_order(X)
    train.check_time_order(X.sort_values("Time"))  # index unsorted, Time sorted: ok


def test_unknown_split_raises(train_data):
    X, y = train_data
    with pytest.raises(ValueError, match="Unknown split"):
        train.choose_threshold({"n_estimators": 5}, X, y, split="kfold")
