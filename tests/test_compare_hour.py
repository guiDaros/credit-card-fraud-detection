"""Tests for scripts/compare_hour.py, using small synthetic data."""

import json
import sys

import numpy as np
import pandas as pd
import pytest

import scripts.compare_hour as compare_hour
import src.provenance as provenance
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


def test_build_report_pairs_folds():
    scores = {"with_hour": [0.8, 0.6, 0.7], "without_hour": [0.7, 0.6, 0.5]}
    y = np.array([0, 1, 0, 0, 1])

    report = compare_hour.build_report(scores, dict(PARAMS), "test", 3, y)

    assert report["difference"]["per_fold"] == pytest.approx([0.1, 0.0, 0.2])
    assert report["difference"]["mean"] == pytest.approx(0.1)
    assert report["difference"]["std"] == pytest.approx(0.1)
    assert report["with_hour"]["mean"] == pytest.approx(0.7)
    assert report["n_splits"] == 3
    assert report["n_scored_folds"] == 3
    assert report["n_train"] == 5
    assert report["n_train_frauds"] == 2
    assert report["params"] == PARAMS
    json.dumps(report)  # must be valid JSON (no numpy types)


def test_summarize_with_one_fold_has_no_std():
    assert compare_hour.summarize([0.5]) == {"mean": 0.5, "std": None}


def test_main_uses_only_the_training_set(monkeypatch, capsys, tmp_path):
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
    monkeypatch.setattr(provenance, "get_git_dirty", lambda: False)
    monkeypatch.setattr(compare_hour, "get_run_provenance", lambda path: {"git_dirty": False})
    output = tmp_path / "compare_hour.json"
    monkeypatch.setattr(sys, "argv", ["compare_hour", "--n-splits", "3", "--output", str(output)])

    compare_hour.main()

    assert len(seen_times) == 2  # with and without Hour
    for times in seen_times:
        assert times.max() < X_test["Time"].min()  # no test row reaches the CV
        assert np.all(np.diff(times) >= 0)  # sorted by Time
    assert "Test set not used" in capsys.readouterr().out

    report = json.loads(output.read_text())
    X_train, _, _, _ = split_data(df, method="temporal")
    assert report["n_train"] == len(X_train)
    assert report["n_scored_folds"] == len(report["with_hour"]["per_fold"])
    assert report["provenance"] == {"git_dirty": False}


def test_main_refuses_dirty_tree(monkeypatch, tmp_path):
    monkeypatch.setattr(provenance, "get_git_dirty", lambda: True)
    output = tmp_path / "compare_hour.json"
    monkeypatch.setattr(sys, "argv", ["compare_hour", "--output", str(output)])

    with pytest.raises(RuntimeError, match="--allow-dirty"):
        compare_hour.main()
    assert not output.exists()
