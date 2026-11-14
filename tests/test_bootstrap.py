"""Tests for the bootstrap confidence intervals in src/evaluate.py, on synthetic data."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from src.evaluate import (
    block_bootstrap_ci,
    block_bootstrap_indices,
    bootstrap_ci,
    bootstrap_samples,
    metrics_at,
)

N_BOOT = 200
THRESHOLD = 0.5


@pytest.fixture
def scored():
    """2,000 rows over 48 hours, ~5% fraud, scores that separate the classes imperfectly."""
    rng = np.random.default_rng(0)
    n_rows = 2000
    y = (rng.random(n_rows) < 0.05).astype(int)
    proba = np.clip(0.3 * y + rng.normal(0.35, 0.15, n_rows), 0, 1)
    time = np.sort(rng.uniform(0, 48 * 3600, n_rows))
    return y, proba, time


def point_metrics(y, proba):
    at = metrics_at(y, proba, THRESHOLD)
    return {
        "pr_auc": average_precision_score(y, proba),
        **{name: at[name] for name in ("precision", "recall", "f1")},
    }


def run_ci(kind, y, proba, time, seed=42):
    if kind == "row":
        return bootstrap_ci(y, proba, THRESHOLD, n_boot=N_BOOT, seed=seed)
    return block_bootstrap_ci(y, proba, THRESHOLD, time, n_boot=N_BOOT, seed=seed)


@pytest.mark.parametrize("kind", ["row", "block"])
def test_same_seed_gives_same_intervals(scored, kind):
    first = run_ci(kind, *scored)
    second = run_ci(kind, *scored)
    other_seed = run_ci(kind, *scored, seed=7)

    assert first == second
    assert first != other_seed


@pytest.mark.parametrize("kind", ["row", "block"])
def test_interval_contains_point_metric(scored, kind):
    y, proba, time = scored
    ci = run_ci(kind, y, proba, time)

    assert ci["n_boot"] == N_BOOT and ci["level"] == 0.95
    for name, point in point_metrics(y, proba).items():
        low, high = ci[name]
        assert low <= point <= high, name
        assert 0 <= low < high <= 1


def test_block_interval_reports_blocks(scored):
    _, _, time = scored
    ci = run_ci("block", *scored)
    assert ci["block_seconds"] == 3600
    assert ci["n_blocks"] == len(np.unique(time // 3600)) == 48


def test_block_resamples_keep_whole_hours(scored):
    _, _, time = scored
    hours = time // 3600
    block_sizes = {h: int((hours == h).sum()) for h in np.unique(hours)}

    for idx in block_bootstrap_indices(time, n_boot=20, seed=0):
        drawn_hours, counts = np.unique(hours[idx], return_counts=True)
        # Every drawn hour appears with all of its rows (possibly several times).
        for h, count in zip(drawn_hours, counts):
            assert count % block_sizes[h] == 0
        assert len(set(np.unique(hours[idx]))) <= len(block_sizes)


def test_resamples_without_fraud_are_skipped():
    y = np.array([1] + [0] * 49)
    proba = np.linspace(0, 1, 50)
    ci = bootstrap_ci(y, proba, THRESHOLD, n_boot=N_BOOT, seed=0)

    # P(no fraud in a resample of 50) = (49/50)^50, about 36%.
    assert 0 < ci["n_used"] < N_BOOT
    for name in ("pr_auc", "precision", "recall", "f1"):
        assert all(np.isfinite(ci[name]))


def test_paired_samples_are_aligned(scored):
    y, proba, time = scored
    idx_list = block_bootstrap_indices(time, n_boot=50, seed=0)
    a = bootstrap_samples(y, proba, THRESHOLD, idx_list)
    b = bootstrap_samples(y, 1 - proba, THRESHOLD, idx_list)
    assert len(a["pr_auc"]) == len(b["pr_auc"])


def test_samples_match_metrics_at(scored):
    y, proba, time = scored
    idx_list = block_bootstrap_indices(time, n_boot=30, seed=0)
    samples = bootstrap_samples(y, proba, THRESHOLD, idx_list)

    for i, idx in enumerate(idx_list):
        expected = metrics_at(y[idx], proba[idx], THRESHOLD)
        for name in ("precision", "recall", "f1"):
            assert samples[name][i] == pytest.approx(expected[name])
