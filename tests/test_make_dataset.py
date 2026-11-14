"""Tests for the dataset cleaning step, using small synthetic data."""

import json
import sys

import numpy as np
import pandas as pd
import pytest

import src.make_dataset as make_dataset
from src.data import load_processed
from src.make_dataset import clean_dataset
from src.provenance import file_sha256


def make_df_with_duplicates() -> pd.DataFrame:
    """5 unique rows plus 2 exact duplicates (one of them a fraud), unsorted by Time."""
    rng = np.random.default_rng(0)
    n = 5
    data = {"Time": np.array([30.0, 10.0, 20.0, 50.0, 40.0])}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(size=n)
    data["Amount"] = rng.uniform(0, 100, size=n)
    data["Class"] = np.array([0, 1, 0, 0, 0])
    df = pd.DataFrame(data)
    # Duplicate row 1 (fraud) and row 3 (normal).
    return pd.concat([df, df.iloc[[1, 3]]], ignore_index=True)


def test_clean_dataset_removes_duplicates_and_sorts():
    df = make_df_with_duplicates()
    clean, report = clean_dataset(df)

    assert report["rows_before"] == 7
    assert report["rows_after"] == 5
    assert report["duplicates_removed"] == 2
    assert report["frauds_before"] == 2
    assert report["frauds_after"] == 1
    assert report["fraud_rate_after"] == 1 / 5

    assert not clean.duplicated().any()
    assert clean["Time"].is_monotonic_increasing
    assert list(clean["Time"]) == [10.0, 20.0, 30.0, 40.0, 50.0]


def test_clean_dataset_reports_time_range():
    _, report = clean_dataset(make_df_with_duplicates())

    assert report["time_min"] == 10.0
    assert report["time_max"] == 50.0
    assert report["duration_hours"] == pytest.approx(40 / 3600)


@pytest.fixture
def raw_setup(tmp_path, monkeypatch):
    """Synthetic raw CSV and output paths in tmp_path, wired into make_dataset."""
    raw = tmp_path / "raw" / "creditcard.csv"
    raw.parent.mkdir()
    make_df_with_duplicates().to_csv(raw, index=False)
    monkeypatch.setattr(make_dataset, "DATA_PATH", raw)
    monkeypatch.setattr(make_dataset, "PROCESSED_PATH", tmp_path / "processed" / "t.parquet")
    monkeypatch.setattr(make_dataset, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(make_dataset, "DATA_REPORT_PATH", tmp_path / "reports" / "r.json")
    return tmp_path, raw


def test_main_refuses_raw_file_with_different_hash(raw_setup, monkeypatch):
    tmp_path, _ = raw_setup
    monkeypatch.setattr(sys, "argv", ["make_dataset"])  # real RAW_SHA256: will not match

    with pytest.raises(ValueError, match="--skip-hash-check"):
        make_dataset.main()
    assert not (tmp_path / "processed").exists()
    assert not (tmp_path / "reports").exists()


def test_main_skip_hash_check_processes_anyway(raw_setup, monkeypatch, capsys):
    tmp_path, raw = raw_setup
    monkeypatch.setattr(sys, "argv", ["make_dataset", "--skip-hash-check"])

    make_dataset.main()

    assert "WARNING" in capsys.readouterr().out
    report = json.loads((tmp_path / "reports" / "r.json").read_text())
    assert report["provenance"]["raw_sha256"] == file_sha256(raw)  # the file actually used
    assert report["rows_after"] == 5
    assert len(load_processed(tmp_path / "processed" / "t.parquet")) == 5


def test_main_accepts_raw_file_with_expected_hash(raw_setup, monkeypatch):
    tmp_path, raw = raw_setup
    monkeypatch.setattr(make_dataset, "RAW_SHA256", file_sha256(raw))
    monkeypatch.setattr(sys, "argv", ["make_dataset"])

    make_dataset.main()

    assert (tmp_path / "processed" / "t.parquet").exists()


def test_check_raw_hash_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        make_dataset.check_raw_hash(tmp_path / "missing.csv")


def write_processed(df: pd.DataFrame, tmp_path):
    path = tmp_path / "transactions.parquet"
    df.to_parquet(path, index=False)
    return path


def test_load_processed_accepts_clean_data(tmp_path):
    clean, _ = clean_dataset(make_df_with_duplicates())
    assert len(load_processed(write_processed(clean, tmp_path))) == 5


def test_load_processed_rejects_unsorted_time(tmp_path):
    clean, _ = clean_dataset(make_df_with_duplicates())
    unsorted = clean.iloc[[1, 0, 2, 3, 4]].reset_index(drop=True)

    with pytest.raises(ValueError, match="non-decreasing.*make data"):
        load_processed(write_processed(unsorted, tmp_path))


def test_load_processed_rejects_duplicates(tmp_path):
    clean, _ = clean_dataset(make_df_with_duplicates())
    # Duplicate the last row at the end, so Time is still in order.
    with_duplicate = pd.concat([clean, clean.iloc[[-1]]], ignore_index=True)

    with pytest.raises(ValueError, match="1 duplicate row.*make data"):
        load_processed(write_processed(with_duplicate, tmp_path))
