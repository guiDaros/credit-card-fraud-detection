"""Tests for temporal out-of-fold predictions, using small synthetic data."""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.cv import temporal_oof_proba

N_ROWS = 100
N_SPLITS = 4


def make_data(n_rows: int = N_ROWS):
    """Fake data sorted by Time, with frauds spread over the whole period."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "Time": np.arange(n_rows, dtype=float),
            "V1": rng.normal(size=n_rows),
        }
    )
    y = pd.Series(np.where(np.arange(n_rows) % 5 == 0, 1, 0))
    return X, y


class RecordingModel:
    """Tiny model that remembers which rows (by Time) it was trained on."""

    def __init__(self, log):
        self.log = log

    def fit(self, X, y):
        self.train_times = X["Time"].to_numpy()
        return self

    def predict_proba(self, X):
        self.log.append((self.train_times, X["Time"].to_numpy()))
        return np.full((len(X), 2), 0.5)


def test_mask_excludes_first_block():
    X, y = make_data()
    proba, mask = temporal_oof_proba(
        lambda: LogisticRegression(), X, y, n_splits=N_SPLITS
    )

    # TimeSeriesSplit test blocks have size n // (n_splits + 1); the first
    # training block is everything before the first test block.
    first_test_start = N_ROWS - N_SPLITS * (N_ROWS // (N_SPLITS + 1))
    assert not mask[:first_test_start].any()
    assert mask[first_test_start:].all()
    assert np.isnan(proba[~mask]).all()
    assert np.all((proba[mask] >= 0) & (proba[mask] <= 1))


def test_no_prediction_from_future_data():
    X, y = make_data()
    log = []
    _, mask = temporal_oof_proba(lambda: RecordingModel(log), X, y, n_splits=N_SPLITS)

    assert len(log) == N_SPLITS
    for train_times, test_times in log:
        # Every training row must come strictly before every predicted row.
        assert train_times.max() < test_times.min()

    predicted_times = np.concatenate([test for _, test in log])
    assert len(predicted_times) == mask.sum()


def test_skips_fold_without_both_classes(caplog):
    X, y = make_data()
    # Remove frauds from the second test block (rows 40-59 with n=100, 4 splits).
    y = y.copy()
    y.iloc[40:60] = 0

    with caplog.at_level(logging.WARNING):
        proba, mask = temporal_oof_proba(
            lambda: LogisticRegression(), X, y, n_splits=N_SPLITS
        )

    assert "Skipping fold 1" in caplog.text
    assert not mask[40:60].any()
    assert mask[60:].all()
