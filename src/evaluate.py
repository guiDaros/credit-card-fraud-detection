"""Evaluation helpers based on precision and recall (no accuracy).

Evaluate a saved model on the test set with:
    python -m src.evaluate [--split {random,temporal}] [--no-hour] [--allow-dirty]

Reads models/model_<split>[_nohour].joblib and writes
reports/metrics_<split>[_nohour].json and reports/pr_curve_<split>[_nohour].png.
Refuses to write them if the git working tree has uncommitted changes, unless
--allow-dirty is passed.
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from src.config import (
    DATA_REPORT_PATH,
    add_run_args,
    metrics_path,
    model_path,
    pr_curve_path,
)


def best_f1_threshold(y_true, proba) -> float:
    """Return the probability threshold that gives the highest F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)

    # The last precision/recall pair has no threshold, so drop it.
    precision, recall = precision[:-1], recall[:-1]

    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[np.argmax(f1)])


def metrics_at(y_true, proba, threshold: float) -> dict:
    """Compute precision, recall, F1 and confusion matrix counts at a threshold."""
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def plot_pr_curve(y_true, proba, path) -> None:
    """Save the Precision-Recall curve with the fraud-rate baseline to `path`."""
    precision, recall, _ = precision_recall_curve(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)
    baseline = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#d1495b", linewidth=2, label=f"Model (AP = {pr_auc:.3f})")
    ax.axhline(
        baseline,
        color="#555555",
        linestyle="--",
        linewidth=1,
        label=f"Baseline (fraud rate = {baseline:.4f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def build_metrics(bundle: dict, y_train, y_test, proba, provenance: dict) -> dict:
    """Build the metrics report for one run configuration.

    Includes the test metrics, the run configuration (split, use_hour, params),
    row and fraud counts for train and test, and two provenance blocks:
    - provenance: this evaluation run (dataset hash, versions, git commit, git_dirty);
    - training_provenance: copied from the model file (None for models saved before
      it was recorded). cv_pr_auc comes from it too, next to the test pr_auc.
    """
    training_provenance = bundle.get("provenance")
    cv_pr_auc = training_provenance["cv_pr_auc"] if training_provenance else None
    return {
        "split": bundle["split"],
        "use_hour": bundle["use_hour"],
        "n_train": int(len(y_train)),
        "n_train_frauds": int(np.sum(y_train)),
        "n_test": int(len(y_test)),
        "n_test_frauds": int(np.sum(y_test)),
        "fraud_rate": float(np.mean(y_test)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "cv_pr_auc": cv_pr_auc,
        "at_0.5": metrics_at(y_test, proba, 0.5),
        "at_chosen_threshold": metrics_at(y_test, proba, bundle["threshold"]),
        "params": bundle["params"],
        "provenance": provenance,
        "training_provenance": training_provenance,
    }


def main() -> None:
    """Evaluate the saved model on the test set and save the reports."""
    # Imported here to keep the helper functions above free of data dependencies.
    from src.data import load_processed, split_data
    from src.provenance import ensure_clean_tree, get_run_provenance

    parser = argparse.ArgumentParser(description="Evaluate a saved model on the test set.")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Write the reports even if the git working tree has uncommitted changes.",
    )
    add_run_args(parser)
    args = parser.parse_args()
    use_hour = not args.no_hour

    ensure_clean_tree(args.allow_dirty)

    bundle = joblib.load(model_path(args.split, use_hour))
    if bundle["split"] != args.split or bundle["use_hour"] != use_hour:
        raise ValueError(
            f"Model file says split={bundle['split']!r}, use_hour={bundle['use_hour']}, "
            f"but evaluate was called with split={args.split!r}, use_hour={use_hour}."
        )

    # Use the same split as in training. The pipeline computes the features
    # from the raw columns. Final evaluation: the only place the test set is used.
    _, X_test, y_train, y_test = split_data(load_processed(), method=args.split)
    proba = bundle["pipeline"].predict_proba(X_test[bundle["input_columns"]])[:, 1]

    metrics = build_metrics(bundle, y_train, y_test, proba, get_run_provenance(DATA_REPORT_PATH))

    out_metrics = metrics_path(args.split, use_hour)
    out_curve = pr_curve_path(args.split, use_hour)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_pr_curve(y_test, proba, out_curve)

    print(json.dumps(metrics, indent=2))
    print(f"Saved {out_metrics} and {out_curve}")


if __name__ == "__main__":
    main()
