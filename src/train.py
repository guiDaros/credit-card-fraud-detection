"""Model training functions.

Run the full training pipeline with:
    python -m src.train --trials 20 [--split {random,temporal}] [--no-hour] [--allow-dirty]

Refuses to run if the git working tree has uncommitted changes, unless --allow-dirty
is passed. The model file stores a `provenance` dict (trials, CV PR-AUC, seed,
versions, git commit, git_dirty, dataset sha256).

The model is saved as models/model_<split>[_nohour].joblib.
"""

import argparse

import joblib
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DATA_REPORT_PATH, SEED, add_run_args, model_path
from src.cv import temporal_oof_proba
from src.data import load_processed, split_data
from src.evaluate import best_f1_threshold
from src.features import make_feature_step
from src.provenance import ensure_clean_tree, get_run_provenance

# class_weight options searched by tune(). Notebook 02 found no clear difference
# between them, so the choice is left to the tuning instead of being fixed.
CLASS_WEIGHTS = [None, "balanced", "balanced_subsample"]


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


def make_pipeline(use_hour: bool = True, **rf_params) -> Pipeline:
    """Create an (unfitted) pipeline: feature step + Random Forest.

    The pipeline takes the raw columns (Time, V1..V28, Amount), so whoever uses
    the model does not need to recreate the features. The feature step
    (make_feature_step) uses only scikit-learn, numpy and the standard library, so
    the saved pipeline loads without this project's `src` package. The Random Forest
    uses class_weight="balanced_subsample" by default; other keyword arguments go to
    make_rf.
    """
    rf_params = {"class_weight": "balanced_subsample", **rf_params}
    return Pipeline(
        [
            ("features", make_feature_step(use_hour)),
            ("classifier", make_rf(**rf_params)),
        ]
    )


def check_time_order(X) -> None:
    """Raise ValueError if the rows of X are not in Time order.

    Uses the Time column when present. Without it (for example, data that
    already went through add_features), the index is checked instead: the
    processed dataset is sorted by Time with a reset index, so increasing
    index means increasing Time.
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


def tune(X, y, n_trials: int, split: str = "random", use_hour: bool = True) -> optuna.Study:
    """Tune the pipeline (feature step + Random Forest) with Optuna to maximize PR-AUC.

    X has the raw columns; features are computed inside each fold.

    Each trial is scored with 3-fold out-of-fold predictions on the data passed
    in: StratifiedKFold for split="random", or a forward-in-time split for
    split="temporal" (X must be sorted by Time; PR-AUC uses only the rows that
    got a prediction). Pass ONLY the training set here; the test set must
    never be used for tuning.

    The search space includes class_weight (CLASS_WEIGHTS), so study.best_params
    always has a "class_weight" key.
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 6, 24),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3]),
            "class_weight": trial.suggest_categorical("class_weight", CLASS_WEIGHTS),
        }
        y_eval, proba = oof_proba(
            lambda: make_pipeline(use_hour=use_hour, **params),
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


def choose_threshold(
    model_params: dict, X_train, y_train, split: str = "random", use_hour: bool = True
) -> float:
    """Choose the decision threshold that maximizes F1 on out-of-fold predictions.

    A pipeline is built with make_pipeline(use_hour, **model_params) and X_train
    has the raw columns, so features are computed inside each fold. Probabilities are
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
        lambda: make_pipeline(use_hour=use_hour, **model_params),
        X_train,
        y_train,
        split=split,
        n_splits=5,
    )
    return best_f1_threshold(y_eval, proba)


def main() -> None:
    """Tune, choose the threshold, train the final pipeline and save it."""
    parser = argparse.ArgumentParser(description="Train the fraud detection Random Forest.")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Train even if the git working tree has uncommitted changes.",
    )
    add_run_args(parser)
    args = parser.parse_args()
    use_hour = not args.no_hour

    # Checked before the (long) training, not only before saving.
    ensure_clean_tree(args.allow_dirty)

    # X_train keeps the raw columns; the pipeline computes the features itself.
    # The test set is not used anywhere in this script.
    # Reads data/processed (the deduplicated dataset made by make_dataset).
    X_train, X_test, y_train, y_test = split_data(load_processed(), method=args.split)
    if args.split == "temporal":
        # Temporal CV needs rows in Time order (checked by check_time_order).
        order = X_train["Time"].sort_values(kind="stable").index
        X_train = X_train.loc[order].reset_index(drop=True)
        y_train = y_train.loc[order].reset_index(drop=True)

    print(f"Tuning with {args.trials} trials...")
    study = tune(X_train, y_train, n_trials=args.trials, split=args.split, use_hour=use_hour)
    params = study.best_params
    print(f"Best CV PR-AUC: {study.best_value:.4f}")
    print(f"Best params: {params}")

    threshold = choose_threshold(params, X_train, y_train, split=args.split, use_hour=use_hour)
    print(f"Chosen threshold (out-of-fold F1): {threshold:.4f}")

    pipeline = make_pipeline(use_hour=use_hour, **params)
    pipeline.fit(X_train, y_train)

    path = model_path(args.split, use_hour)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "threshold": threshold,
        "input_columns": list(X_train.columns),
        "use_hour": use_hour,
        "split": args.split,
        # The tuned parameters, including the chosen class_weight.
        "params": dict(params),
        "provenance": {
            "n_trials": args.trials,
            # Best out-of-fold PR-AUC found by Optuna on the training set
            # (forward-in-time folds for split="temporal").
            "cv_pr_auc": float(study.best_value),
            # Chosen by the tuning among CLASS_WEIGHTS (None means no weighting).
            "class_weight": params["class_weight"],
            "seed": SEED,
            "split": args.split,
            "use_hour": use_hour,
            # dataset_sha256, library versions, git_commit and git_dirty.
            **get_run_provenance(DATA_REPORT_PATH),
        },
    }
    joblib.dump(bundle, path, compress=3)

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Saved {path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
