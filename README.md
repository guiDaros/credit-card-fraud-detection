---
license: mit
language: en
library_name: sklearn
pipeline_tag: tabular-classification
tags:
  - tabular-classification
  - fraud-detection
  - random-forest
  - scikit-learn
  - optuna
  - imbalanced-classification
metrics:
  - average_precision
  - precision
  - recall
  - f1
---

# Credit Card Fraud Detection

Personal machine learning project focused on financial fraud detection.

## Dataset

TODO

## Approach

TODO

## Results

> **Preliminary results.** The current model was tuned with only 3 Optuna trials.
> It will be retrained and tested more thoroughly later, so these numbers may change.

Evaluated on the held-out test set (56,962 transactions, 98 frauds, ~0.17% fraud rate).

**Overall (PR-AUC)**

| Model | PR-AUC |
|---|---|
| No-skill baseline (fraud rate) | 0.002 |
| Logistic Regression (baseline) | 0.713 |
| **Random Forest** | **0.854** |

**Random Forest at different thresholds**

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| 0.5 (default) | 0.833 | 0.816 | 0.825 | 80 | 16 | 18 |
| **0.4525 (chosen on training data)** | **0.822** | **0.847** | **0.834** | **83** | **18** | **15** |

At the chosen threshold the model catches 83 of the 98 frauds, with 18 false alarms
out of about 57,000 legitimate transactions. Full metrics are in `reports/metrics.json`
and the Precision-Recall curve in `reports/pr_curve.png`.

## How to run

TODO

## Limitations

- **Limited tuning:** only 3 Optuna trials were run. Cross-validation already found a better
  candidate with more trials (PR-AUC 0.841 vs 0.826), so the model will be tuned and tested further.
- **Few frauds in the test set:** with only 98 frauds, each one changes recall by about 1 point,
  so the metrics have noticeable uncertainty.
- **Duplicate rows:** the dataset has 1,081 duplicate rows that were not removed. Copies may end
  up in both train and test, which could make the results slightly optimistic.
- **Time feature:** `Hour` is relative to the first transaction (the dataset covers only two days),
  not the real time of day.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
