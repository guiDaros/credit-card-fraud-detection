"""Feature engineering."""

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame with engineered features.

    Adds:
    - Amount_log: log1p(Amount)
    - Hour: (Time // 3600) % 24, the hour relative to the first transaction

    Then drops the original Time and Amount columns. The input DataFrame
    is not modified.

    Each new value is computed only from the same row, with no statistics
    from other rows (no mean, scaling or fitting), so there is no data leakage.
    It is applied after the train/test split, separately on train and test,
    because the split needs the original Time column (temporal split).
    """
    result = df.copy()
    result["Amount_log"] = np.log1p(result["Amount"])
    result["Hour"] = (result["Time"] // 3600 % 24).astype(int)
    return result.drop(columns=["Time", "Amount"])
