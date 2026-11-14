"""Tests for tuning and threshold selection, using small synthetic data."""

import numpy as np
import pandas as pd
import pytest

import src.train as train
from src.cv import temporal_oof_proba

N_ROWS = 300
N_TRIALS = 2


RAW_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def make_raw_data(n_rows: int = N_ROWS):
    """Fake raw data (Time, V1..V28, Amount) in Time order, ~10% fraud spread over time."""
    rng = np.random.default_rng(0)
    y = pd.Series((np.arange(n_rows) % 10 == 0).astype(int))
    data = {"Time": np.sort(rng.uniform(0, 172800, n_rows))}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(size=n_rows)
    data["V1"] = data["V1"] + 2 * y
    data["Amount"] = rng.uniform(0, 500, n_rows)
    return pd.DataFrame(data)[RAW_COLUMNS], y


@pytest.fixture
def train_data():
    return make_raw_data()


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


@pytest.mark.parametrize("use_hour", [True, False])
def test_pipeline_predicts_on_raw_dataframe(train_data, use_hour):
    X, y = train_data
    pipeline = train.make_pipeline(use_hour=use_hour, n_estimators=10, max_depth=4)
    pipeline.fit(X, y)

    X_new, _ = make_raw_data(20)
    proba = pipeline.predict_proba(X_new)[:, 1]

    assert proba.shape == (20,)
    assert np.all((proba >= 0) & (proba <= 1))
    assert list(pipeline.feature_names_in_) == RAW_COLUMNS
    classifier = pipeline.named_steps["classifier"]
    assert classifier.class_weight == "balanced_subsample"
    assert ("Hour" in classifier.feature_names_in_) == use_hour
    assert "Amount_log" in classifier.feature_names_in_


@pytest.mark.parametrize("column", ["Time", "Amount", "V14"])
def test_pipeline_fails_clearly_on_missing_column(train_data, column):
    X, y = train_data
    pipeline = train.make_pipeline(n_estimators=10, max_depth=4).fit(X, y)

    with pytest.raises(ValueError, match=column):
        pipeline.predict_proba(X.drop(columns=column))


@pytest.mark.parametrize("split", ["random", "temporal"])
def test_tune_searches_class_weight_and_uses_it(train_data, split, monkeypatch):
    X, y = train_data
    built_with = []
    real_make_pipeline = train.make_pipeline

    def spy_make_pipeline(use_hour=True, **params):
        built_with.append(params["class_weight"])
        return real_make_pipeline(use_hour=use_hour, **params)

    monkeypatch.setattr(train, "make_pipeline", spy_make_pipeline)
    study = train.tune(X, y, n_trials=4, split=split)

    tried = [trial.params["class_weight"] for trial in study.trials]
    assert all(weight in train.CLASS_WEIGHTS for weight in tried)
    assert "class_weight" in study.best_params
    # Every fold of a trial builds its pipeline with that trial's class_weight.
    n_folds = len(built_with) // len(tried)
    assert built_with == [weight for weight in tried for _ in range(n_folds)]


@pytest.mark.parametrize("split", ["random", "temporal"])
def test_tune_records_cv_rows(train_data, split):
    X, y = train_data
    study = train.tune(X, y, n_trials=1, split=split)
    cv = study.best_trial.user_attrs["cv"]

    expected_rows = N_ROWS if split == "random" else expected_temporal_rows(X, y, 3)
    assert cv["n_splits"] == train.TUNE_N_SPLITS
    assert cv["scheme"] == ("stratified_kfold" if split == "random" else "time_series_split")
    assert cv["n_rows"] == expected_rows
    assert 0 < cv["n_frauds"] <= int(y.sum())
    assert cv["fraud_rate"] == pytest.approx(cv["n_frauds"] / cv["n_rows"])


def test_baseline_pipeline_predicts_on_raw_dataframe(train_data):
    X, y = train_data
    pipeline = train.make_baseline_pipeline().fit(X, y)

    proba = pipeline.predict_proba(X)[:, 1]

    assert proba.shape == (N_ROWS,)
    assert list(pipeline.feature_names_in_) == RAW_COLUMNS
    assert pipeline.named_steps["classifier"].class_weight == "balanced"


def test_tune_is_reproducible_with_the_seed(train_data):
    X, y = train_data
    first = train.tune(X, y, n_trials=3, split="temporal")
    second = train.tune(X, y, n_trials=3, split="temporal")

    assert [t.params for t in first.trials] == [t.params for t in second.trials]
    assert first.best_value == second.best_value
