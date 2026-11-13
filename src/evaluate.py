"""Evaluation helpers based on precision and recall (no accuracy).

Evaluate the saved model on the test set with:
    python -m src.evaluate
"""

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

from src.config import MODELS_DIR, REPORTS_DIR


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


def main() -> None:
    """Evaluate the saved model on the test set and save the reports."""
    # Imported here to keep the helper functions above free of data dependencies.
    from src.data import load_processed, split_data

    bundle = joblib.load(MODELS_DIR / "model.joblib")
    pipeline = bundle["pipeline"]
    threshold = bundle["threshold"]

    # Use the same split as in training. The pipeline computes the features
    # from the raw columns. Final evaluation: the only place the test set is used.
    _, X_test, _, y_test = split_data(load_processed(), method=bundle["split"])
    X_test = X_test[bundle["input_columns"]]

    proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "pr_auc": float(average_precision_score(y_test, proba)),
        "fraud_rate": float(np.mean(y_test)),
        "n_test": int(len(y_test)),
        "at_0.5": metrics_at(y_test, proba, 0.5),
        "at_chosen_threshold": metrics_at(y_test, proba, threshold),
        "params": bundle["params"],
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_pr_curve(y_test, proba, REPORTS_DIR / "pr_curve.png")

    print(json.dumps(metrics, indent=2))
    print(f"Saved {metrics_path} and {REPORTS_DIR / 'pr_curve.png'}")


if __name__ == "__main__":
    main()
