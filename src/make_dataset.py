"""Build the processed dataset: remove duplicates and sort by Time.

Run with: python -m src.make_dataset [--skip-hash-check]

Before processing, the sha256 of the raw CSV is compared with RAW_SHA256 (the file
the results were produced with); a different file stops the run unless
--skip-hash-check is passed.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import (
    DATA_PATH,
    DATA_REPORT_PATH,
    PROCESSED_PATH,
    RAW_SHA256,
    REPORTS_DIR,
    TARGET,
)
from src.data import load_data
from src.provenance import file_sha256, get_environment


def check_raw_hash(path: Path, expected: str = RAW_SHA256, skip: bool = False) -> str:
    """Return the sha256 of the raw CSV; raise ValueError if it is not `expected`.

    With skip=True a different hash only prints a warning. The returned hash is the
    real one, so the data report always records the file that was actually used.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download 'creditcard.csv' from Kaggle "
            "(mlg-ulb/creditcardfraud) and place it in data/raw/."
        )
    digest = file_sha256(path)
    if digest != expected:
        message = (
            f"The sha256 of {path} is {digest}, but the expected one is {expected} "
            "(RAW_SHA256 in src/config.py). This is not the file the results were "
            "produced with: it may be a different version of the dataset, a broken "
            "download or an edited file."
        )
        if not skip:
            raise ValueError(
                f"{message} Download it again from Kaggle, or pass --skip-hash-check "
                "to process it anyway."
            )
        print(f"WARNING: {message} Continuing because of --skip-hash-check.")
    return digest


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
        # Time is in seconds since the first transaction of the dump.
        "time_min": float(clean["Time"].min()),
        "time_max": float(clean["Time"].max()),
        "duration_hours": float((clean["Time"].max() - clean["Time"].min()) / 3600),
    }
    return clean, report


def main():
    """Check the raw CSV hash, clean it and save the processed parquet and the data report."""
    parser = argparse.ArgumentParser(description="Build the processed dataset.")
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Process the raw CSV even if its sha256 is not RAW_SHA256.",
    )
    args = parser.parse_args()

    raw_sha256 = check_raw_hash(DATA_PATH, RAW_SHA256, skip=args.skip_hash_check)
    df = load_data(DATA_PATH)
    clean, report = clean_dataset(df)
    report["provenance"] = {"raw_sha256": raw_sha256, **get_environment()}

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
