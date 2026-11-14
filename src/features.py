"""Feature engineering.

Two implementations of the same features:
- make_feature_step: the step used inside the saved model pipeline. It is built only
  from scikit-learn, numpy and the standard library, so a saved model can be loaded
  without this project's code.
- add_features: a plain pandas function, kept for the notebooks.

Both produce the columns in this order:
    V1, V2, ..., V28, Amount_log, Hour   (Hour only when use_hour is True)
"""

from operator import methodcaller

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

V_COLUMNS = [f"V{i}" for i in range(1, 29)]


def feature_names(use_hour: bool = True) -> list[str]:
    """Output columns of the feature step, in order."""
    return [*V_COLUMNS, "Amount_log", *(["Hour"] if use_hour else [])]


def make_feature_step(use_hour: bool = True) -> ColumnTransformer:
    """Create the (unfitted) feature step: raw columns in, feature_names(use_hour) out.

    - V1..V28 are passed through unchanged.
    - Amount_log = np.log1p(Amount).
    - Hour = (Time // 3600) % 24, only when use_hour is True. numpy ufuncs cannot
      take their second operand as a keyword, so the `//` and `%` operators are
      called with operator.methodcaller, which (unlike a lambda) can be pickled.
    - Time and Amount themselves are dropped (remainder="drop").

    Each feature has its own transformer, named after the output column, and
    verbose_feature_names_out="{transformer_name}" gives the columns those names.
    The output is a pandas DataFrame, so the classifier keeps the feature names.
    Nothing is learned from the data (no statistics across rows), so fitting it on
    the training set cannot leak information.
    """
    transformers = [(name, "passthrough", [name]) for name in V_COLUMNS]
    transformers.append(
        ("Amount_log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), ["Amount"])
    )
    if use_hour:
        hour = Pipeline(
            [
                (
                    "hours_since_start",
                    FunctionTransformer(
                        methodcaller("__floordiv__", 3600), feature_names_out="one-to-one"
                    ),
                ),
                (
                    "hour_of_day",
                    FunctionTransformer(
                        methodcaller("__mod__", 24), feature_names_out="one-to-one"
                    ),
                ),
            ]
        )
        transformers.append(("Hour", hour, ["Time"]))

    step = ColumnTransformer(
        transformers, remainder="drop", verbose_feature_names_out="{transformer_name}"
    )
    return step.set_output(transform="pandas")


def add_features(df: pd.DataFrame, use_hour: bool = True) -> pd.DataFrame:
    """Return a new DataFrame with engineered features (used by the notebooks).

    Same features as make_feature_step, which is what the saved model uses. Adds:
    - Amount_log: log1p(Amount)
    - Hour: (Time // 3600) % 24, the hour relative to the first transaction
      (only when use_hour is True)

    use_hour=False lets us compare models with and without Hour: in the
    temporal split the test set only has hours already seen in training,
    but from a different day.

    Then drops the original Time and Amount columns. The input DataFrame
    is not modified.

    Each new value is computed only from the same row, with no statistics
    from other rows (no mean, scaling or fitting), so there is no data leakage.
    It is applied after the train/test split, separately on train and test,
    because the split needs the original Time column (temporal split).
    """
    missing = [col for col in ("Time", "Amount") if col not in df.columns]
    if missing:
        raise ValueError(f"add_features: missing required column(s): {missing}")

    result = df.copy()
    result["Amount_log"] = np.log1p(result["Amount"])
    if use_hour:
        result["Hour"] = (result["Time"] // 3600 % 24).astype(int)
    return result.drop(columns=["Time", "Amount"])
