"""Tests for scoring new data, with a small model trained on synthetic data."""

import logging
import sys

import joblib
import numpy as np
import pandas as pd
import pandera.errors
import pytest

from src.predict import main, predict
from src.train import make_pipeline

INPUT_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
THRESHOLD = 0.4


def make_raw(n_rows: int, seed: int = 0, with_class: bool = True) -> pd.DataFrame:
    """Fake raw transactions (~10% fraud, V1 shifted for frauds)."""
    rng = np.random.default_rng(seed)
    y = (np.arange(n_rows) % 10 == 0).astype(int)
    df = pd.DataFrame({"Time": np.sort(rng.uniform(0, 172800, n_rows))})
    for i in range(1, 29):
        df[f"V{i}"] = rng.normal(size=n_rows)
    df["V1"] += 3 * y
    df["Amount"] = rng.uniform(0, 500, n_rows)
    if with_class:
        df["Class"] = y
    return df


@pytest.fixture
def bundle():
    """A small fitted pipeline, stored like src.train does."""
    train = make_raw(300)
    pipeline = make_pipeline(n_estimators=20, max_depth=4)
    pipeline.fit(train[INPUT_COLUMNS], train["Class"])
    return {
        "pipeline": pipeline,
        "threshold": THRESHOLD,
        "input_columns": INPUT_COLUMNS,
        "use_hour": True,
        "split": "temporal",
        "params": {"n_estimators": 20, "max_depth": 4},
    }


@pytest.fixture
def model_file(tmp_path, bundle):
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    return path


def test_predict_adds_score_and_flag_without_class(bundle):
    new = make_raw(50, seed=1, with_class=False)
    result = predict(new, bundle)

    assert list(result.columns) == list(new.columns) + ["score", "is_fraud"]
    assert result["score"].between(0, 1).all()
    assert (result["is_fraud"] == (result["score"] >= THRESHOLD)).all()
    assert "score" not in new.columns  # input not modified


def test_predict_keeps_extra_columns(bundle):
    new = make_raw(20, seed=1)  # has Class
    new["transaction_id"] = range(20)
    result = predict(new, bundle)

    assert "Class" in result.columns
    assert result["transaction_id"].tolist() == list(range(20))


def test_predict_rejects_missing_column(bundle):
    new = make_raw(20, seed=1, with_class=False).drop(columns=["V14"])
    with pytest.raises(pandera.errors.SchemaErrors, match="V14"):
        predict(new, bundle)


def test_predict_rejects_negative_amount(bundle):
    new = make_raw(20, seed=1, with_class=False)
    new.loc[0, "Amount"] = -5.0
    with pytest.raises(pandera.errors.SchemaErrors):
        predict(new, bundle)


@pytest.mark.parametrize("in_suffix, out_suffix", [(".csv", ".csv"), (".parquet", ".parquet"), (".csv", ".parquet")])
def test_main_writes_output_and_logs(tmp_path, model_file, monkeypatch, caplog, in_suffix, out_suffix):
    new = make_raw(40, seed=2, with_class=False)
    in_path = tmp_path / f"new{in_suffix}"
    out_path = tmp_path / f"scored{out_suffix}"
    if in_suffix == ".csv":
        new.to_csv(in_path, index=False)
    else:
        new.to_parquet(in_path, index=False)

    monkeypatch.setattr(
        sys, "argv",
        ["predict", "--input", str(in_path), "--model", str(model_file), "--output", str(out_path)],
    )
    with caplog.at_level(logging.INFO):
        main()

    out = pd.read_csv(out_path) if out_suffix == ".csv" else pd.read_parquet(out_path)
    assert list(out.columns) == INPUT_COLUMNS + ["score", "is_fraud"]
    assert len(out) == 40
    n_flagged = int(out["is_fraud"].sum())
    assert f"Flagged {n_flagged} of 40 rows" in caplog.text


def test_main_rejects_unsupported_output(tmp_path, model_file, monkeypatch):
    in_path = tmp_path / "new.csv"
    make_raw(10, with_class=False).to_csv(in_path, index=False)
    monkeypatch.setattr(
        sys, "argv",
        ["predict", "--input", str(in_path), "--model", str(model_file), "--output", str(tmp_path / "out.xlsx")],
    )
    with pytest.raises(ValueError, match="Unsupported file type"):
        main()
