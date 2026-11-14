"""Tests for run naming, CLI options and the train -> evaluate scripts."""

import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

import src.config as config
import src.data as data
import src.evaluate as evaluate
import src.provenance as provenance
import src.train as train
from src.config import TARGET, add_run_args, metrics_path, model_path, pr_curve_path, run_suffix


def parse(argv):
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    return parser.parse_args(argv)


def test_run_args_defaults_to_temporal_with_hour():
    args = parse([])
    assert args.split == "temporal"
    assert args.no_hour is False


def test_run_args_parses_options():
    args = parse(["--split", "random", "--no-hour"])
    assert args.split == "random"
    assert args.no_hour is True
    with pytest.raises(SystemExit):
        parse(["--split", "kfold"])


@pytest.mark.parametrize(
    "split, use_hour, suffix",
    [
        ("temporal", True, "temporal"),
        ("random", True, "random"),
        ("temporal", False, "temporal_nohour"),
        ("random", False, "random_nohour"),
    ],
)
def test_output_names(split, use_hour, suffix):
    assert run_suffix(split, use_hour) == suffix
    assert model_path(split, use_hour).name == f"model_{suffix}.joblib"
    assert model_path(split, use_hour).parent.name == "models"
    assert metrics_path(split, use_hour).name == f"metrics_{suffix}.json"
    assert pr_curve_path(split, use_hour).name == f"pr_curve_{suffix}.png"
    assert metrics_path(split, use_hour).parent.name == "reports"


def make_processed_df(n_rows: int = 400) -> pd.DataFrame:
    """Fake processed dataset (sorted by Time, reset index), ~10% fraud spread over time."""
    rng = np.random.default_rng(0)
    y = (np.arange(n_rows) % 10 == 0).astype(int)
    df = pd.DataFrame({"Time": np.sort(rng.uniform(0, 172800, n_rows))})
    for i in range(1, 29):
        df[f"V{i}"] = rng.normal(size=n_rows)
    df["V1"] += 2 * y
    df["Amount"] = rng.uniform(0, 500, n_rows)
    df[TARGET] = y
    return df


@pytest.mark.parametrize(
    "cli, suffix", [([], "temporal"), (["--split", "random", "--no-hour"], "random_nohour")]
)
def test_train_then_evaluate_writes_named_outputs(tmp_path, monkeypatch, cli, suffix):
    df = make_processed_df()
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(train, "load_processed", lambda: df.copy())
    monkeypatch.setattr(data, "load_processed", lambda: df.copy())

    report_path = tmp_path / "data_report.json"
    report_path.write_text(json.dumps({"provenance": {"raw_sha256": "abc123"}}))
    monkeypatch.setattr(evaluate, "DATA_REPORT_PATH", report_path)
    monkeypatch.setattr(train, "DATA_REPORT_PATH", report_path)
    monkeypatch.setattr(provenance, "get_git_dirty", lambda: False)

    monkeypatch.setattr(sys, "argv", ["train", "--trials", "1", *cli])
    train.main()
    model_file = tmp_path / "models" / f"model_{suffix}.joblib"
    assert model_file.exists()

    bundle = joblib.load(model_file)
    train_prov = bundle["provenance"]
    assert train_prov["n_trials"] == 1
    assert train_prov["class_weight"] in train.CLASS_WEIGHTS
    assert train_prov["class_weight"] == bundle["params"]["class_weight"]
    classifier = bundle["pipeline"].named_steps["classifier"]
    assert classifier.class_weight == train_prov["class_weight"]
    assert 0 <= train_prov["cv_pr_auc"] <= 1
    assert train_prov["seed"] == config.SEED
    assert train_prov["split"] == ("random" if "random" in suffix else "temporal")
    assert train_prov["use_hour"] == ("nohour" not in suffix)
    assert train_prov["dataset_sha256"] == "abc123"
    assert train_prov["git_dirty"] is False
    for key in ["python", "pandas", "numpy", "scikit-learn", "optuna", "git_commit"]:
        assert key in train_prov

    monkeypatch.setattr(sys, "argv", ["evaluate", *cli])
    evaluate.main()
    assert (tmp_path / "reports" / f"pr_curve_{suffix}.png").exists()

    metrics = json.loads((tmp_path / "reports" / f"metrics_{suffix}.json").read_text())
    assert metrics["split"] == ("random" if "random" in suffix else "temporal")
    assert metrics["use_hour"] == ("nohour" not in suffix)
    assert metrics["n_train"] + metrics["n_test"] == len(df)
    assert metrics["n_train_frauds"] + metrics["n_test_frauds"] == df[TARGET].sum()
    assert metrics["provenance"]["dataset_sha256"] == "abc123"
    for key in ["python", "pandas", "numpy", "scikit-learn", "optuna", "git_commit"]:
        assert key in metrics["provenance"]
    assert metrics["provenance"]["git_dirty"] is False
    assert metrics["training_provenance"] == train_prov
    assert metrics["cv_pr_auc"] == train_prov["cv_pr_auc"]
    for key in ("bootstrap_ci", "block_bootstrap_ci"):
        for name in ("pr_auc", "precision", "recall", "f1"):
            low, high = metrics[key][name]
            assert 0 <= low <= high <= 1
    assert metrics["block_bootstrap_ci"]["block_seconds"] == 3600


@pytest.mark.parametrize("script", ["train", "evaluate"])
def test_scripts_refuse_dirty_tree(tmp_path, monkeypatch, script):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(provenance, "get_git_dirty", lambda: True)
    monkeypatch.setattr(sys, "argv", [script])

    with pytest.raises(RuntimeError, match="--allow-dirty"):
        {"train": train, "evaluate": evaluate}[script].main()
    assert not (tmp_path / "models").exists()
    assert not (tmp_path / "reports").exists()


def test_allow_dirty_trains_and_records_dirty(tmp_path, monkeypatch):
    df = make_processed_df()
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(train, "load_processed", lambda: df.copy())
    report_path = tmp_path / "data_report.json"
    report_path.write_text(json.dumps({"provenance": {"raw_sha256": "abc123"}}))
    monkeypatch.setattr(train, "DATA_REPORT_PATH", report_path)
    monkeypatch.setattr(provenance, "get_git_dirty", lambda: True)

    monkeypatch.setattr(sys, "argv", ["train", "--trials", "1", "--allow-dirty"])
    train.main()

    bundle = joblib.load(tmp_path / "models" / "model_temporal.joblib")
    assert bundle["provenance"]["git_dirty"] is True


def test_build_metrics_without_training_provenance():
    # Models saved before provenance was recorded have no "provenance" key.
    bundle = {"split": "temporal", "use_hour": True, "threshold": 0.5, "params": {}}
    y = np.array([0, 0, 1, 0, 1])
    proba = np.array([0.1, 0.2, 0.9, 0.3, 0.6])

    time = np.array([0.0, 100.0, 4000.0, 5000.0, 9000.0])

    metrics = evaluate.build_metrics(bundle, y, y, proba, time, provenance={"git_dirty": False})

    assert metrics["training_provenance"] is None
    assert metrics["cv_pr_auc"] is None


def test_evaluate_rejects_mismatched_model(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(provenance, "get_git_dirty", lambda: False)
    joblib.dump({"split": "random", "use_hour": True}, model_path("temporal", True))
    monkeypatch.setattr(sys, "argv", ["evaluate"])

    with pytest.raises(ValueError, match="split='random'"):
        evaluate.main()
