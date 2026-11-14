"""Compare the Random Forest with and without the Hour feature, on the training set only.

Run from the repository root with:
    python -m scripts.compare_hour [--n-splits 5] [--params-from models/model_temporal.joblib]
        [--output reports/compare_hour.json] [--allow-dirty]

Uses only the training part of the temporal split and forward-in-time
cross-validation (temporal_oof_proba). The test set is never loaded into the
comparison, so this can be used to decide about Hour without touching it.

Both configurations use the same folds, so the per-fold difference (with Hour minus
without Hour) is a paired comparison. Prints the mean and standard deviation of the
per-fold PR-AUC for each configuration and for the difference, and saves them
(with the parameters, n_splits and provenance) to reports/compare_hour.json.
Refuses to run if the git working tree has uncommitted changes, unless
--allow-dirty is passed.

Random Forest parameters come from --params-from (the parameters of a saved model,
which were tuned on the training set only) or, if that file does not exist, from
DEFAULT_PARAMS. Note: parameters tuned with Hour may slightly favour the with-Hour
configuration.
"""

import argparse
import json
from functools import partial
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit

from src.config import DATA_REPORT_PATH, ROOT_DIR, compare_hour_path, model_path
from src.cv import temporal_oof_proba
from src.data import load_processed, split_data
from src.provenance import ensure_clean_tree, get_run_provenance
from src.train import make_pipeline

DEFAULT_PARAMS = {"n_estimators": 200, "class_weight": "balanced_subsample"}


def per_fold_pr_auc(X, y, use_hour: bool, params: dict, n_splits: int) -> list[float]:
    """PR-AUC of each forward-in-time fold, for one configuration.

    Predictions come from temporal_oof_proba; each fold's PR-AUC is computed on the
    rows of that fold's test block. Folds that temporal_oof_proba skipped (one class
    only) are left out. X must be sorted by Time.
    """
    proba, mask = temporal_oof_proba(
        partial(make_pipeline, use_hour=use_hour, **params), X, y, n_splits=n_splits
    )
    y_values = np.asarray(y)
    scores = []
    for _, test_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
        if mask[test_idx].all():
            scores.append(float(average_precision_score(y_values[test_idx], proba[test_idx])))
    return scores


def compare_hour(X, y, params: dict, n_splits: int = 5) -> dict:
    """Per-fold PR-AUC with and without Hour, on the same folds.

    Returns {"with_hour": [...], "without_hour": [...]}.
    """
    return {
        "with_hour": per_fold_pr_auc(X, y, True, params, n_splits),
        "without_hour": per_fold_pr_auc(X, y, False, params, n_splits),
    }


def load_params(path: Path) -> tuple[dict, str]:
    """Random Forest parameters from a saved model file, or DEFAULT_PARAMS.

    The source is returned relative to the repository when possible, so the saved
    report does not contain local absolute paths.
    """
    if path.exists():
        resolved = path.resolve()
        source = resolved.relative_to(ROOT_DIR) if resolved.is_relative_to(ROOT_DIR) else path
        return dict(joblib.load(path)["params"]), str(source)
    return dict(DEFAULT_PARAMS), "DEFAULT_PARAMS"


def summarize(values) -> dict:
    """Mean and sample standard deviation (None with fewer than 2 values)."""
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()) if len(values) else None,
        "std": float(values.std(ddof=1)) if len(values) > 1 else None,
    }


def build_report(scores: dict, params: dict, params_source: str, n_splits: int, y) -> dict:
    """Report saved to compare_hour.json.

    scores is the output of compare_hour. The difference is with Hour minus without
    Hour, fold by fold (paired). n_train and n_train_frauds describe the training set
    the folds come from (the test set is not used).
    """
    with_hour = np.array(scores["with_hour"])
    without_hour = np.array(scores["without_hour"])
    difference = with_hour - without_hour
    return {
        "metric": "pr_auc",
        "split": "temporal",
        "cv_scheme": "time_series_split",
        "n_splits": n_splits,
        "n_scored_folds": int(len(with_hour)),
        "n_train": int(len(y)),
        "n_train_frauds": int(np.sum(y)),
        "params": params,
        "params_source": params_source,
        "with_hour": {"per_fold": with_hour.tolist(), **summarize(with_hour)},
        "without_hour": {"per_fold": without_hour.tolist(), **summarize(without_hour)},
        "difference": {"per_fold": difference.tolist(), **summarize(difference)},
    }


def main() -> None:
    """Load the temporal training set, run the comparison and print the summary."""
    parser = argparse.ArgumentParser(description="Compare with and without Hour (train CV only).")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of temporal folds.")
    parser.add_argument(
        "--params-from",
        type=Path,
        default=model_path("temporal", True),
        help="Saved model whose Random Forest parameters are used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=compare_hour_path(),
        help="Where to save the JSON report.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Write the report even if the git working tree has uncommitted changes.",
    )
    args = parser.parse_args()

    ensure_clean_tree(args.allow_dirty)

    # Only the training part of the temporal split is kept; the test part is discarded.
    X_train, _, y_train, _ = split_data(load_processed(), method="temporal")
    order = X_train["Time"].sort_values(kind="stable").index
    X_train = X_train.loc[order].reset_index(drop=True)
    y_train = y_train.loc[order].reset_index(drop=True)

    params, source = load_params(args.params_from)
    print(f"Training rows: {len(X_train):,} ({int(y_train.sum())} frauds). Test set not used.")
    print(f"Parameters ({source}): {params}")

    scores = compare_hour(X_train, y_train, params, n_splits=args.n_splits)
    with_hour, without_hour = np.array(scores["with_hour"]), np.array(scores["without_hour"])

    print(f"\nPR-AUC per fold ({len(with_hour)} scored folds):")
    for name, values in [("with Hour", with_hour), ("without Hour", without_hour)]:
        folds = ", ".join(f"{v:.4f}" for v in values)
        print(f"  {name:<13} mean {values.mean():.4f}  std {values.std(ddof=1):.4f}  [{folds}]")
    diff = with_hour - without_hour
    print(f"  {'difference':<13} mean {diff.mean():+.4f}  std {diff.std(ddof=1):.4f}")
    print(
        "\nIf the mean difference is small compared with its std, the folds do not show "
        "that Hour helps or hurts."
    )

    report = build_report(scores, params, source, args.n_splits, y_train)
    report["provenance"] = get_run_provenance(DATA_REPORT_PATH)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
