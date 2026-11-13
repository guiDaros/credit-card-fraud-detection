"""Model training functions."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import SEED


def train_baseline(X_train, y_train) -> Pipeline:
    """Train a baseline: StandardScaler + LogisticRegression with balanced class weights.

    The scaler is inside the pipeline, so it is fitted only on the training data.
    """
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def make_rf(**params) -> RandomForestClassifier:
    """Create an (unfitted) RandomForestClassifier with a fixed seed and all CPU cores.

    Any keyword argument is passed to RandomForestClassifier.
    """
    return RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
