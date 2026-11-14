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
    SEED,
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


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
#
# Both intervals below describe one fixed model and one fixed threshold. They answer:
# "how much would the test metrics change if the test set were a different sample
# from the same period?" They capture the sampling variance of the test set only.
# They do NOT capture:
# - training variance: a different random seed, Optuna run or training sample would
#   give a different model;
# - threshold variance: the threshold was chosen once on the training data and is
#   kept fixed here, although a different training run would choose another one;
# - change over time: the test period is resampled, not a future period.
# ---------------------------------------------------------------------------

BOOTSTRAP_METRICS = ("pr_auc", "precision", "recall", "f1")
N_BOOT = 1000
CI_PERCENTILES = (2.5, 97.5)  # 95% percentile interval
BLOCK_SECONDS = 3600  # one-hour blocks for the block bootstrap


def bootstrap_indices(n_rows: int, n_boot: int = N_BOOT, seed: int = SEED) -> list:
    """Row indices of n_boot resamples of n_rows rows, drawn with replacement."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n_rows, n_rows) for _ in range(n_boot)]


def block_bootstrap_indices(
    time, n_boot: int = N_BOOT, seed: int = SEED, block_seconds: int = BLOCK_SECONDS
) -> list:
    """Row indices of n_boot block resamples, with blocks of block_seconds of Time.

    Rows are grouped by block = Time // block_seconds (e.g. hour 40 since the first
    transaction). Each resample draws as many blocks as there are, with replacement,
    and keeps all the rows of each drawn block, so its size can vary.
    """
    blocks = np.asarray(time) // block_seconds
    rows_by_block = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
    rng = np.random.default_rng(seed)
    resamples = []
    for _ in range(n_boot):
        drawn = rng.integers(0, len(rows_by_block), len(rows_by_block))
        resamples.append(np.concatenate([rows_by_block[b] for b in drawn]))
    return resamples


def bootstrap_samples(y_true, proba, threshold: float, idx_list) -> dict:
    """PR-AUC, precision, recall and F1 at `threshold` on each resample in idx_list.

    Resamples without any fraud are skipped (PR-AUC is undefined there). Returns a
    dict of numpy arrays, one value per kept resample; resamples are kept or skipped
    based on y_true alone, so two models scored on the same y_true and idx_list get
    aligned arrays (paired bootstrap).
    """
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    samples = {name: [] for name in BOOTSTRAP_METRICS}
    for idx in idx_list:
        y_b, p_b = y_true[idx], proba[idx]
        if y_b.sum() == 0:
            continue
        # Same values as metrics_at (zero_division=0), counted directly with numpy
        # because metrics_at is slow when called thousands of times.
        flagged = p_b >= threshold
        tp = int(np.sum(flagged & (y_b == 1)))
        fp = int(np.sum(flagged & (y_b == 0)))
        fn = int(np.sum(~flagged & (y_b == 1)))
        samples["pr_auc"].append(average_precision_score(y_b, p_b))
        samples["precision"].append(tp / (tp + fp) if tp + fp else 0.0)
        samples["recall"].append(tp / (tp + fn) if tp + fn else 0.0)
        samples["f1"].append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return {name: np.array(values) for name, values in samples.items()}


def _percentile_ci(samples: dict, n_boot: int, seed: int) -> dict:
    """Summary of bootstrap samples: 95% percentile interval for each metric."""
    n_used = len(samples["pr_auc"])
    if n_used == 0:
        raise ValueError("No bootstrap resample contained a fraud; cannot compute intervals.")
    result = {"n_boot": n_boot, "n_used": n_used, "seed": seed, "level": 0.95}
    for name in BOOTSTRAP_METRICS:
        low, high = np.percentile(samples[name], CI_PERCENTILES)
        result[name] = [float(low), float(high)]
    return result


def bootstrap_ci(y_true, proba, threshold: float, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """95% percentile bootstrap intervals for PR-AUC, precision, recall and F1.

    Resamples the test rows independently, with replacement (n_boot resamples, fixed
    seed), and recomputes the metrics at the fixed `threshold`. Resamples without any
    fraud are skipped; n_used says how many were kept.

    Captures: the sampling variance of the test set, treating rows as independent.
    Does NOT capture: training/seed variance, threshold variance (see the note
    above), or the correlation between transactions close in time; for the latter
    see block_bootstrap_ci, which is usually wider.

    Returns {"n_boot", "n_used", "seed", "level", "pr_auc": [low, high], ...}.
    """
    samples = bootstrap_samples(
        y_true, proba, threshold, bootstrap_indices(len(y_true), n_boot, seed)
    )
    return _percentile_ci(samples, n_boot, seed)


def block_bootstrap_ci(
    y_true,
    proba,
    threshold: float,
    time,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    block_seconds: int = BLOCK_SECONDS,
) -> dict:
    """95% block bootstrap intervals for PR-AUC, precision, recall and F1.

    Like bootstrap_ci, but resamples whole blocks of block_seconds (default one hour)
    of test transactions, using the `time` of each row (the Time column). Transactions
    in the same hour (and frauds in the same burst) are kept together, which respects
    their correlation in time.

    Captures: the sampling variance of the test set, allowing for correlation within
    each hour. With few blocks (the temporal test set spans about 8 hours) the
    interval itself is rough. Does NOT capture: training/seed variance or threshold
    variance (see the note above), nor correlation between different hours.

    Returns the same keys as bootstrap_ci plus "block_seconds" and "n_blocks".
    """
    idx_list = block_bootstrap_indices(time, n_boot, seed, block_seconds)
    result = _percentile_ci(bootstrap_samples(y_true, proba, threshold, idx_list), n_boot, seed)
    result["block_seconds"] = block_seconds
    result["n_blocks"] = int(len(np.unique(np.asarray(time) // block_seconds)))
    return result


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


def describe_test_period(time) -> dict:
    """First and last Time of the test set, in seconds and in hours.

    Time is in seconds since the first transaction of the dataset, so "hour" here
    is hours since that first transaction and "hour_of_day" is (hour % 24), the
    same value as the Hour feature.
    """
    time = np.asarray(time)
    first, last = float(time.min()), float(time.max())
    return {
        "first_time": first,
        "last_time": last,
        "first_hour": int(first // 3600),
        "last_hour": int(last // 3600),
        "first_hour_of_day": int(first // 3600 % 24),
        "last_hour_of_day": int(last // 3600 % 24),
    }


def build_metrics(bundle: dict, y_train, y_test, proba, test_time, provenance: dict) -> dict:
    """Build the metrics report for one run configuration.

    Includes the test metrics, two sets of 95% intervals at the chosen threshold
    (bootstrap_ci: rows resampled independently; block_bootstrap_ci: one-hour blocks
    resampled, using test_time, the Time of each test row), the run configuration
    (split, use_hour, params), row and fraud counts for train and test, the test
    period (describe_test_period(test_time)), and two
    provenance blocks:
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
        "test_period": describe_test_period(test_time),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "cv_pr_auc": cv_pr_auc,
        "at_0.5": metrics_at(y_test, proba, 0.5),
        "at_chosen_threshold": metrics_at(y_test, proba, bundle["threshold"]),
        "bootstrap_ci": bootstrap_ci(y_test, proba, bundle["threshold"]),
        "block_bootstrap_ci": block_bootstrap_ci(y_test, proba, bundle["threshold"], test_time),
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

    metrics = build_metrics(
        bundle, y_train, y_test, proba, X_test["Time"], get_run_provenance(DATA_REPORT_PATH)
    )

    out_metrics = metrics_path(args.split, use_hour)
    out_curve = pr_curve_path(args.split, use_hour)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_pr_curve(y_test, proba, out_curve)

    print(json.dumps(metrics, indent=2))
    print(f"Saved {out_metrics} and {out_curve}")


if __name__ == "__main__":
    main()
