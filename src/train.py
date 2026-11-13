"""Model training functions.

Run the full training pipeline with:
    python -m src.train --trials 20
"""

import argparse

import joblib
import optuna
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, SEED
from src.cv import temporal_oof_proba
from src.data import load_processed, split_data
from src.evaluate import best_f1_threshold
from src.features import add_features


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


SPLITS = ("random", "temporal")


def check_time_order(X) -> None:
    """Raise ValueError if the rows of X are not in Time order.

    Uses the Time column when present. After add_features, Time is dropped,
    so the index is checked instead: the processed dataset is sorted by Time
    with a reset index, so increasing index means increasing Time.
    """
    if "Time" in X.columns:
        is_sorted = X["Time"].is_monotonic_increasing
    else:
        is_sorted = X.index.is_monotonic_increasing
    if not is_sorted:
        raise ValueError("Temporal split needs X sorted by Time (oldest first).")


def oof_proba(model_factory, X, y, split: str, n_splits: int):
    """Out-of-fold probabilities for the chosen split.

    Returns (y_eval, proba): the labels and probabilities of the rows that got
    an out-of-fold prediction. With split="random" (StratifiedKFold +
    cross_val_predict) that is every row; with split="temporal"
    (temporal_oof_proba) it is only the rows in the mask.
    """
    if split == "random":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        proba = cross_val_predict(model_factory(), X, y, cv=cv, method="predict_proba")[:, 1]
        return y, proba
    if split == "temporal":
        check_time_order(X)
        proba, mask = temporal_oof_proba(model_factory, X, y, n_splits=n_splits)
        if not mask.any():
            raise ValueError("No temporal fold had both classes; cannot score.")
        return y[mask], proba[mask]
    raise ValueError(f"Unknown split: {split!r}. Use 'random' or 'temporal'.")


def tune(X, y, n_trials: int, split: str = "random") -> optuna.Study:
    """Tune a Random Forest with Optuna to maximize PR-AUC (average precision).

    Each trial is scored with 3-fold out-of-fold predictions on the data passed
    in: StratifiedKFold for split="random", or a forward-in-time split for
    split="temporal" (X must be sorted by Time; PR-AUC uses only the rows that
    got a prediction). Pass ONLY the training set here; the test set must
    never be used for tuning.
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 6, 24),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3]),
        }
        y_eval, proba = oof_proba(
            lambda: make_rf(class_weight="balanced_subsample", **params),
            X,
            y,
            split=split,
            n_splits=3,
        )
        return average_precision_score(y_eval, proba)

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study


def choose_threshold(model_params: dict, X_train, y_train, split: str = "random") -> float:
    """Choose the decision threshold that maximizes F1 on out-of-fold predictions.

    A Random Forest is built with make_rf(**model_params), and probabilities are
    generated with 5-fold out-of-fold predictions on the training set:
    StratifiedKFold for split="random", or a forward-in-time split for
    split="temporal" (X_train must be sorted by Time; only the rows that got
    a prediction are used).
    """
    # Why the threshold is chosen on the training set and not on the test set:
    # the threshold is a model decision, just like a hyperparameter. If we picked
    # it on the test set, the test metrics would be optimistic, because the test
    # set would have been used to make a choice. Out-of-fold probabilities are
    # predictions on rows the model did not see during fitting, so they behave
    # like unseen data while keeping the test set untouched until the final evaluation.
    y_eval, proba = oof_proba(
        lambda: make_rf(**model_params), X_train, y_train, split=split, n_splits=5
    )
    return best_f1_threshold(y_eval, proba)


def main() -> None:
    """Tune, choose the threshold, train the final model and save it."""
    parser = argparse.ArgumentParser(description="Train the fraud detection Random Forest.")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument(
        "--split", choices=SPLITS, default="random", help="Train/test split and CV scheme."
    )
    args = parser.parse_args()

    # Split first (on the original Time/Amount), then add features to each set.
    # The test set is not used anywhere in this script.
    X_train, X_test, y_train, y_test = split_data(load_processed(), method=args.split)
    if args.split == "temporal":
        # Temporal CV needs rows in Time order; Time is dropped by add_features,
        # so sort now and reset the index (check_time_order relies on it).
        order = X_train["Time"].sort_values(kind="stable").index
        X_train = X_train.loc[order].reset_index(drop=True)
        y_train = y_train.loc[order].reset_index(drop=True)
    X_train = add_features(X_train)

    print(f"Tuning with {args.trials} trials...")
    study = tune(X_train, y_train, n_trials=args.trials, split=args.split)
    params = {**study.best_params, "class_weight": "balanced_subsample"}
    print(f"Best CV PR-AUC: {study.best_value:.4f}")
    print(f"Best params: {params}")

    threshold = choose_threshold(params, X_train, y_train, split=args.split)
    print(f"Chosen threshold (out-of-fold F1): {threshold:.4f}")

    model = make_rf(**params)
    model.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "model.joblib"
    bundle = {
        "model": model,
        "threshold": threshold,
        "features": list(X_train.columns),
        "params": params,
    }
    joblib.dump(bundle, model_path, compress=3)

    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"Saved {model_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
