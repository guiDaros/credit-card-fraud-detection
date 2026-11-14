# Data

The data files are not versioned (see `.gitignore`); only this README and the
`.gitkeep` files are. This page says where the data comes from, what it contains and
what each folder promises.

## Source

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
on Kaggle (`mlg-ulb/creditcardfraud`): transactions made by European cardholders over
two days in September 2013.

Download `creditcard.csv` and place it at:

```
data/raw/creditcard.csv
```

Then build the processed dataset with `make data`.

## Columns

| Column | Type | Meaning |
|---|---|---|
| `Time` | float, >= 0 | Seconds since the first transaction in the dump (not a calendar date or time of day). |
| `V1` ... `V28` | float | Anonymized principal components (PCA) of the original features, which are not published for confidentiality reasons. |
| `Amount` | float, >= 0 | Transaction amount (not transformed). |
| `Class` | int, 0 or 1 | Target: 1 = fraud, 0 = legitimate. |

These rules are checked by the pandera schema in `src/schema.py` every time the raw or
processed data is loaded.

## Expected numbers

From `reports/data_report.json` (written by `make data`):

| | Rows | Frauds |
|---|---|---|
| Raw `creditcard.csv` | 284,807 | 492 |
| `processed/transactions.parquet` | 283,726 | 473 |

- Exact duplicate rows removed: 1,081.
- sha256 of the raw `creditcard.csv`:
  `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`

Check that you have the same file with:

```bash
sha256sum data/raw/creditcard.csv
```

## Folder contract

### `raw/`

- Holds `creditcard.csv` exactly as downloaded from Kaggle.
- Immutable: never edited by hand and never overwritten by the code. Every cleaning step
  happens in code and writes to `processed/`.
- Its sha256 is recorded in `reports/data_report.json` and copied into the provenance of
  every trained model and metrics report.

### `processed/`

- Holds `transactions.parquet`, written by `make data` (`src/make_dataset.py`).
- Validated with the same pandera schema as the raw data, without exact duplicate rows
  (first occurrence kept) and sorted by `Time` (stable sort, so ties keep their raw order).
- Regenerable: it can be deleted at any time and rebuilt from `raw/` with `make data`.

In data lake terms, `processed/` is the "trusted" (or "silver") layer: cleaned and
validated, but not yet shaped for a specific model. There is no "gold" layer (a
feature table): the features (`Amount_log` and `Hour`) are computed inside the model's
scikit-learn `Pipeline`, so the saved model takes these columns directly and training and
prediction always compute the features the same way.

## License and citation

The dataset is released under the
[Database Contents License (DbCL) v1.0](https://opendatacommons.org/licenses/dbcl/1-0/).

> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.
> *Calibrating Probability with Undersampling for Unbalanced Classification.*
> In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.
