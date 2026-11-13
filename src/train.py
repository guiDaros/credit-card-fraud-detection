"""Model training functions."""

import optuna
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import SEED
from src.evaluate import best_f1_threshold


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


def tune(X, y, n_trials: int) -> optuna.Study:
    """Tune a Random Forest with Optuna to maximize PR-AUC (average precision).

    Each trial is scored with stratified 3-fold cross_val_predict on the data
    passed in. Pass ONLY the training set here; the test set must never be
    used for tuning.
    """
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 6, 24),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3]),
        }
        model = make_rf(class_weight="balanced_subsample", **params)
        proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        return average_precision_score(y, proba)

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study


def choose_threshold(model_params: dict, X_train, y_train) -> float:
    """Choose the decision threshold that maximizes F1 on out-of-fold predictions.

    A Random Forest is built with make_rf(**model_params), and probabilities are
    generated with stratified 5-fold cross_val_predict on the training set.
    """
    # Why the threshold is chosen on the training set and not on the test set:
    # the threshold is a model decision, just like a hyperparameter. If we picked
    # it on the test set, the test metrics would be optimistic, because the test
    # set would have been used to make a choice. Out-of-fold probabilities are
    # predictions on rows the model did not see during fitting, so they behave
    # like unseen data while keeping the test set untouched until the final evaluation.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    model = make_rf(**model_params)
    proba = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    return best_f1_threshold(y_train, proba)
