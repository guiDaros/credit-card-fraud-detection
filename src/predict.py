"""Score new transactions with a saved model.

Usage:
    python -m src.predict --input new.csv --model models/model_temporal.joblib --output scored.csv

The input (CSV or parquet) must have the raw columns Time, V1..V28 and Amount; Class
is not needed. The output has the original columns plus:
- score: the model's fraud probability
- is_fraud: True when score >= the threshold stored in the model file

Note: model files are pickle-based. Loading one can run arbitrary code, so only load
files you trust.
"""

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd

from src.schema import validate_input

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".csv", ".parquet")


def check_suffix(path: Path) -> None:
    """Raise ValueError if the file is not .csv or .parquet."""
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type {path.suffix!r} for {path}. Use .csv or .parquet.")


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or parquet file."""
    path = Path(path)
    check_suffix(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def write_table(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV or parquet, depending on the file extension."""
    path = Path(path)
    check_suffix(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def predict(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Return a copy of df with the score and is_fraud columns added.

    df is validated against the input schema first.
    """
    df = validate_input(df)
    scores = bundle["pipeline"].predict_proba(df[bundle["input_columns"]])[:, 1]

    result = df.copy()
    result["score"] = scores
    result["is_fraud"] = scores >= bundle["threshold"]
    return result


def main() -> None:
    """Read the input file, score it with the saved model and write the output file."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Score transactions with a saved model.")
    parser.add_argument("--input", required=True, type=Path, help="Input CSV or parquet file.")
    parser.add_argument("--model", required=True, type=Path, help="Saved model (.joblib).")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV or parquet file.")
    args = parser.parse_args()

    # Check the output type before doing any work.
    check_suffix(args.output)

    df = read_table(args.input)
    logger.info("Read %d rows from %s", len(df), args.input)

    bundle = joblib.load(args.model)
    result = predict(df, bundle)

    write_table(result, args.output)
    n_flagged = int(result["is_fraud"].sum())
    share = n_flagged / len(result) if len(result) else 0.0
    logger.info(
        "Flagged %d of %d rows as fraud (%.2f%%) at threshold %.4f",
        n_flagged,
        len(result),
        100 * share,
        bundle["threshold"],
    )
    logger.info("Saved %s", args.output)


if __name__ == "__main__":
    main()
