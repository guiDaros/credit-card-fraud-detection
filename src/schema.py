"""Pandera schema for the raw credit card fraud dataset."""

import pandas as pd
import pandera.pandas as pa

from src.config import TARGET

RAW_SCHEMA = pa.DataFrameSchema(
    {
        "Time": pa.Column(float, pa.Check.ge(0)),
        **{f"V{i}": pa.Column(float, nullable=False) for i in range(1, 29)},
        "Amount": pa.Column(float, pa.Check.ge(0), nullable=False),
        TARGET: pa.Column(int, pa.Check.isin([0, 1])),
    },
)

# Schema for new data to score: the model's input columns, without Class.
# Values are coerced to float, so integer-looking columns (e.g. Time = 10) are accepted.
# Extra columns (including Class, if present) are allowed and left untouched.
INPUT_SCHEMA = RAW_SCHEMA.remove_columns([TARGET]).update_columns(
    {col: {"coerce": True} for col in ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]}
)


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the raw dataset and return it.

    Uses lazy=True so all errors are reported together.
    """
    return RAW_SCHEMA.validate(df, lazy=True)


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate data to be scored (no Class column needed) and return it.

    Uses lazy=True so all errors are reported together.
    """
    return INPUT_SCHEMA.validate(df, lazy=True)
