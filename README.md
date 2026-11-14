---
license: mit
library_name: sklearn
tags:
  - tabular-classification
  - fraud-detection
  - random-forest
  - imbalanced-data
datasets:
  - mlg-ulb/creditcardfraud
metrics:
  - precision
  - recall
  - f1
  - average_precision
---

# Credit Card Fraud Detection

## Summary

Personal machine learning project focused on financial fraud detection. A Random Forest
classifier is trained to flag fraudulent credit card transactions in a highly imbalanced
dataset (about 0.17% fraud). Hyperparameters are tuned with Optuna to maximize PR-AUC,
and the decision threshold is chosen with cross-validation on the training data.

This is a learning project, not a production system. The reported results use a
temporal train/test split (train on the past, test on the most recent 20% of the time
range).

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
from Kaggle (`mlg-ulb/creditcardfraud`):

- 284,807 transactions made by European cardholders over two days in September 2013.
- 492 frauds (about 0.17% of all transactions).
- Features `V1`-`V28` are the result of a PCA transformation; the original features are
  not available for confidentiality reasons. `Time` (seconds since the first transaction)
  and `Amount` are not transformed. The target is `Class` (1 = fraud, 0 = legitimate).
- License: [Database Contents License (DbCL) v1.0](https://opendatacommons.org/licenses/dbcl/1-0/).

The dataset is not included in this repository. Download `creditcard.csv` from Kaggle and
place it in `data/raw/`.

**Citation:**

> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.
> *Calibrating Probability with Undersampling for Unbalanced Classification.*
> In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.

## Approach

- **Data cleaning:** exact duplicate rows are removed and the data is sorted by `Time`
  (`src/make_dataset.py`), producing `data/processed/transactions.parquet`.
- **Features:** `Amount` is replaced by `Amount_log = log1p(Amount)` and `Time` by
  `Hour = (Time // 3600) % 24` (hour relative to the first transaction). `Hour` can be
  turned off with `--no-hour`, to compare models with and without it. `V1`-`V28` are
  used as they are. Each new feature uses only its own row, so there is no leakage.
- **Pipeline:** the saved model is a scikit-learn `Pipeline` (feature step + Random
  Forest), so it takes the raw columns (`Time`, `V1`-`V28`, `Amount`) and features are
  computed inside each cross-validation fold.
- **Split** (`--split`, default `temporal`):
  - `temporal`: the last 20% of the time range is the test set; train is everything
    before it.
  - `random`: stratified 80/20 split with a fixed seed.

  The test set is used only once, for the final evaluation.
- **Class imbalance:** handled with `class_weight="balanced_subsample"` in the Random
  Forest, with no resampling. Accuracy is not used, because always predicting
  "legitimate" would already score about 99.8%.
- **Cross-validation:** with `--split temporal`, out-of-fold predictions come from a
  forward-in-time split (`TimeSeriesSplit`): each fold is trained on the past and
  predicts the next block, and the first block is never scored. With `--split random`,
  stratified K-fold is used.
- **Tuning:** Optuna (TPE sampler, fixed seed) maximizes PR-AUC (average precision),
  computed from 3-fold out-of-fold predictions on the training set. Search space:
  `n_estimators` 100-400, `max_depth` 6-24, `min_samples_leaf` 1-10, `max_features` in
  {`sqrt`, `log2`, 0.3}.
- **Threshold:** chosen to maximize F1 on 5-fold out-of-fold probabilities on the
  training set, never on the test set.
- **Final model:** trained on the full training set with the best parameters.

## Results

Configuration: temporal split, with `Hour` (`models/model_temporal.joblib`,
`reports/metrics_temporal.json`). Tuned with 20 Optuna trials.

- **Train:** 226,982 transactions, 399 frauds (fraud rate 0.0018).
- **Test:** the last 20% of the time range, 56,744 transactions, 74 frauds (fraud rate
  0.0013).

**PR-AUC (average precision)**

| Model | PR-AUC |
|---|---|
| Random baseline (≈ fraud rate) | 0.0013 |
| Random Forest, temporal CV on train (best Optuna trial) | 0.780 |
| **Random Forest, test set** | **0.817** |

The test PR-AUC is a bit higher than the CV estimate. With only 74 test frauds this gap
is within the expected noise.

**Random Forest at two thresholds (test set)**

| Threshold | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|
| 0.5 (default) | 0.929 | 0.703 | 0.800 | 52 | 4 | 22 | 56,666 |
| 0.3594 (chosen via CV) | 0.877 | 0.770 | 0.820 | 57 | 8 | 17 | 56,662 |

At the chosen threshold, the model detects 57 of the 74 frauds in the test set and
raises 8 false alarms out of 56,670 legitimate transactions.

**Model parameters:** `n_estimators=122`, `max_depth=17`, `min_samples_leaf=6`,
`max_features="log2"`, `class_weight="balanced_subsample"`.

![Precision-Recall curve](reports/pr_curve_temporal.png)

A run without `Hour` (`--no-hour`) and a run with the random split have not been
evaluated yet with the current code.

## How to run

```bash
# 1. Create a virtual environment and install the dependencies
python -m venv .venv
source .venv/bin/activate        # Windows (Git Bash): source .venv/Scripts/activate
pip install -r requirements.txt

# 2. Download creditcard.csv from Kaggle into data/raw/

# 3. Run the tests
pytest

# 4. Remove duplicates and sort by Time -> data/processed/transactions.parquet
python -m src.make_dataset

# 5. Tune, choose the threshold and train the final model -> models/model_temporal.joblib
python -m src.train --trials 20

# 6. Evaluate on the test set -> reports/metrics_temporal.json, reports/pr_curve_temporal.png
python -m src.evaluate

# 7. (Optional) Publish to the Hugging Face Hub (requires `hf auth login`)
python -m src.publish --repo your-username/credit-card-fraud-rf
```

`train`, `evaluate` and `publish` accept `--split {random,temporal}` (default `temporal`)
and `--no-hour`. Use the same options in all three. The file names follow the
configuration: `model_temporal.joblib`, `model_random.joblib`, `model_temporal_nohour.joblib`,
and the same suffixes for `metrics_*.json` and `pr_curve_*.png`. Each metrics file also
records the split, `use_hour`, train/test row and fraud counts, the dataset sha256, the
library versions and the git commit.

The notebooks in `notebooks/` contain the data exploration and the model comparison.

## How to use the model

> `model.joblib` is a pickle-based file. Loading a pickle can run arbitrary code, so
> only load files from sources you trust.

```python
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="guilhermedaros/credit-card-fraud-rf", filename="model_temporal.joblib"
)
bundle = joblib.load(path)

pipeline = bundle["pipeline"]        # fitted Pipeline: features + RandomForestClassifier
threshold = bundle["threshold"]      # decision threshold chosen via CV
columns = bundle["input_columns"]    # raw columns: Time, V1..V28, Amount

# The pipeline computes the features itself; just pass the raw columns.
df = pd.read_csv("transactions.csv")
proba = pipeline.predict_proba(df[columns])[:, 1]
is_fraud = proba >= threshold
```

The bundle also stores `use_hour`, `split` and `params`. The model needs this project's
`src` package on the Python path (the feature step calls `src.features.add_features`) and
a compatible scikit-learn version (see `requirements.txt`).

## Project structure

```
├── data/raw/            # creditcard.csv (not versioned)
├── data/processed/      # transactions.parquet (deduplicated, sorted by Time)
├── models/              # model_<split>[_nohour].joblib (not versioned)
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_modeling.ipynb
├── reports/             # data_report.json, metrics_*.json, pr_curve_*.png
├── src/
│   ├── config.py        # paths, constants, run options and output names
│   ├── schema.py        # data schema validation
│   ├── make_dataset.py  # raw -> processed (deduplication, sort by Time)
│   ├── provenance.py    # data hash, library versions, git commit
│   ├── data.py          # load_data, load_processed, split_data
│   ├── features.py      # add_features
│   ├── cv.py            # temporal out-of-fold predictions
│   ├── train.py         # baseline, pipeline, Optuna tuning, threshold, main
│   ├── evaluate.py      # metrics, PR curve, main
│   └── publish.py       # upload to the Hugging Face Hub
├── tests/
├── requirements.txt
└── LICENSE
```

## Limitations

- **Anonymized features:** `V1`-`V28` are PCA components, so the model cannot be
  interpreted in terms of real transaction attributes, and it cannot be applied to data
  that was not transformed in the same way.
- **Narrow time window:** the data covers only two days of September 2013 from European
  cardholders. Fraud patterns change over time, so performance on newer or different
  data is unknown.
- **Random split may be optimistic:** with `--split random`, train and test are mixed in
  time, while in a real setting the model predicts future transactions. The default
  `--split temporal` avoids this, but its test set covers only the last part of a
  two-day window. The 1,081 exact duplicate rows are removed before splitting.
- **Few frauds, high variance:** the test set has only 74 frauds, so each one changes
  recall by about 1.4 points. The metrics have noticeable uncertainty.
- **Limited tuning:** the results above used 20 Optuna trials.
- **No calibration and no fairness analysis:** the predicted probabilities are not
  calibrated and should not be read as true fraud probabilities. No fairness or bias
  analysis was done.
- **Not for production use:** this is a personal learning project. Do not use it to make
  real decisions about transactions or customers.

## License

The code and model in this repository are released under the MIT License. See
[LICENSE](LICENSE). The dataset is licensed separately under the DbCL v1.0.
