"""Build the processed dataset: remove duplicates and sort by Time.

Run with: python -m src.make_dataset
"""

import json

import pandas as pd

from src.config import DATA_PATH, DATA_REPORT_PATH, PROCESSED_PATH, REPORTS_DIR, TARGET
from src.data import load_data
from src.provenance import get_provenance


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop duplicate rows (keep first) and sort by Time with a stable sort.

    Returns the cleaned DataFrame and a small report dict.
    """
    n_duplicates = int(df.duplicated(keep="first").sum())

    clean = df.drop_duplicates(keep="first")
    clean = clean.sort_values("Time", kind="stable").reset_index(drop=True)

    report = {
        "rows_before": int(len(df)),
        "rows_after": int(len(clean)),
        "duplicates_removed": n_duplicates,
        "frauds_before": int(df[TARGET].sum()),
        "frauds_after": int(clean[TARGET].sum()),
        "fraud_rate_before": float(df[TARGET].mean()),
        "fraud_rate_after": float(clean[TARGET].mean()),
    }
    return clean, report


def main():
    """Clean the raw CSV and save the processed parquet and the data report."""
    df = load_data()
    clean, report = clean_dataset(df)
    report["provenance"] = get_provenance(DATA_PATH)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(PROCESSED_PATH, index=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved {PROCESSED_PATH}")
    print(f"Saved {DATA_REPORT_PATH}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
