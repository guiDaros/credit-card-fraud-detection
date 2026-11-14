"""Logistic regression baseline on the temporal split.

Run with:
    python -m src.baseline [--allow-dirty]

Fits feature step + StandardScaler + LogisticRegression (make_baseline_pipeline,
not tuned) on the temporal training set and evaluates it on the temporal test set,
the same way as the Random Forest:
- cv_pr_auc: out-of-fold PR-AUC on the same TUNE_N_SPLITS forward-in-time folds
  that Optuna uses for the Random Forest;
- threshold: best out-of-fold F1 on THRESHOLD_N_SPLITS forward-in-time folds, as
  in choose_threshold;
- test metrics and both bootstrap intervals: build_metrics, as in evaluate.

Writes reports/metrics_baseline_temporal.json (same keys as metrics_temporal.json,
plus "model"). The model itself is not saved. Refuses to run if the git working
tree has uncommitted changes, unless --allow-dirty is passed.
"""

import argparse
import json

from sklearn.metrics import average_precision_score

from src.config import DATA_REPORT_PATH, SEED, baseline_metrics_path
from src.data import load_processed, split_data
from src.evaluate import best_f1_threshold, build_metrics
from src.provenance import ensure_clean_tree, get_run_provenance
from src.train import (
    BASELINE_PARAMS,
    THRESHOLD_N_SPLITS,
    TUNE_N_SPLITS,
    cv_summary,
    make_baseline_pipeline,
    oof_proba,
)

SPLIT = "temporal"
USE_HOUR = True  # same features as the Random Forest in model_temporal


def main() -> None:
    """Fit the baseline on the temporal training set, evaluate it and save the report."""
    parser = argparse.ArgumentParser(description="Logistic regression baseline (temporal split).")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Write the report even if the git working tree has uncommitted changes.",
    )
    args = parser.parse_args()

    ensure_clean_tree(args.allow_dirty)

    X_train, X_test, y_train, y_test = split_data(load_processed(), method=SPLIT)
    # Temporal CV needs rows in Time order (same as in train.main).
    order = X_train["Time"].sort_values(kind="stable").index
    X_train = X_train.loc[order].reset_index(drop=True)
    y_train = y_train.loc[order].reset_index(drop=True)

    def factory():
        return make_baseline_pipeline(use_hour=USE_HOUR)

    # Training set only: CV PR-AUC and threshold, with the Random Forest's folds.
    y_cv, proba_cv = oof_proba(factory, X_train, y_train, split=SPLIT, n_splits=TUNE_N_SPLITS)
    cv_pr_auc = float(average_precision_score(y_cv, proba_cv))
    y_thr, proba_thr = oof_proba(
        factory, X_train, y_train, split=SPLIT, n_splits=THRESHOLD_N_SPLITS
    )
    threshold = best_f1_threshold(y_thr, proba_thr)
    print(f"CV PR-AUC: {cv_pr_auc:.4f}  threshold (out-of-fold F1): {threshold:.4f}")

    pipeline = factory().fit(X_train, y_train)

    # Final evaluation: the only place the test set is used.
    proba = pipeline.predict_proba(X_test[list(X_train.columns)])[:, 1]
    provenance = get_run_provenance(DATA_REPORT_PATH)
    # Same structure as a saved Random Forest bundle, so build_metrics can be reused.
    bundle = {
        "split": SPLIT,
        "use_hour": USE_HOUR,
        "threshold": threshold,
        "params": dict(BASELINE_PARAMS),
        "provenance": {
            "cv_pr_auc": cv_pr_auc,
            "cv": cv_summary(y_cv, SPLIT, TUNE_N_SPLITS),
            "seed": SEED,
            "split": SPLIT,
            "use_hour": USE_HOUR,
            **provenance,
        },
    }
    metrics = {
        "model": "logistic_regression",
        **build_metrics(bundle, y_train, y_test, proba, X_test["Time"], provenance),
    }

    out = baseline_metrics_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
