"""Compare the Random Forest with and without the Hour feature, on the training set only.

Run from the repository root with:
    python -m scripts.compare_hour [--n-splits 5] [--params-from models/model_temporal.joblib]

Uses only the training part of the temporal split and forward-in-time
cross-validation (temporal_oof_proba). The test set is never loaded into the
comparison, so this can be used to decide about Hour without touching it.

Both configurations use the same folds, so the per-fold difference (with Hour minus
without Hour) is a paired comparison. Prints the mean and standard deviation of the
per-fold PR-AUC for each configuration and for the difference.

Random Forest parameters come from --params-from (the parameters of a saved model,
which were tuned on the training set only) or, if that file does not exist, from
DEFAULT_PARAMS. Note: parameters tuned with Hour may slightly favour the with-Hour
configuration.
"""

import argparse
from functools import partial
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit

from src.config import model_path
from src.cv import temporal_oof_proba
from src.data import load_processed, split_data
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
    """Random Forest parameters from a saved model file, or DEFAULT_PARAMS."""
    if path.exists():
        return dict(joblib.load(path)["params"]), str(path)
    return dict(DEFAULT_PARAMS), "DEFAULT_PARAMS"


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
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
