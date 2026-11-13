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


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the raw dataset and return it.

    Uses lazy=True so all errors are reported together.
    """
    return RAW_SCHEMA.validate(df, lazy=True)
