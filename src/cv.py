"""Temporal cross-validation helpers.

cross_val_predict does not work with TimeSeriesSplit, because its test folds
do not form a partition of the data (the first training block is never
tested). This module runs the folds in a manual loop instead.
"""

import logging

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


def _take(data, idx):
    """Select rows by position from a pandas object or a numpy array."""
    return data.iloc[idx] if hasattr(data, "iloc") else data[idx]


def temporal_oof_proba(model_factory, X, y, n_splits: int = 4):
    """Out-of-fold fraud probabilities with a forward-in-time split.

    For each TimeSeriesSplit fold, a fresh model from model_factory() is
    trained on the past rows and predicts the next block. Folds whose train
    or test part does not contain both classes are skipped with a warning.

    Assumes X (and y) are already sorted by Time.

    Returns (proba, mask): proba has one value per row (NaN where there is no
    prediction) and mask is True for rows that actually got a prediction.
    The first training block never gets a prediction.
    """
    y_values = np.asarray(y)
    proba = np.full(len(y_values), np.nan)
    mask = np.zeros(len(y_values), dtype=bool)

    splitter = TimeSeriesSplit(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X)):
        n_train_classes = len(np.unique(y_values[train_idx]))
        n_test_classes = len(np.unique(y_values[test_idx]))
        if n_train_classes < 2 or n_test_classes < 2:
            logger.warning(
                "Skipping fold %d: train has %d class(es), test has %d class(es).",
                fold,
                n_train_classes,
                n_test_classes,
            )
            continue

        model = model_factory()
        model.fit(_take(X, train_idx), _take(y, train_idx))
        proba[test_idx] = model.predict_proba(_take(X, test_idx))[:, 1]
        mask[test_idx] = True

    return proba, mask
