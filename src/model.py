from __future__ import annotations

from itertools import product
from typing import Any, Iterator

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .data_processing import DEFAULT_RANDOM_SEED

MODEL_CHOICES = ("logistic_regression", "random_forest")


def build_model(
    model_name: str,
    random_seed: int = DEFAULT_RANDOM_SEED,
    **overrides: Any,
):
    if model_name == "logistic_regression":
        params = {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 2000,
            "random_state": random_seed,
            "solver": "liblinear",
        }
        params.update(overrides)
        return LogisticRegression(**params)

    if model_name == "random_forest":
        params = {
            "class_weight": "balanced_subsample",
            "max_depth": None,
            "min_samples_leaf": 1,
            "n_estimators": 300,
            "n_jobs": -1,
            "random_state": random_seed,
        }
        params.update(overrides)
        return RandomForestClassifier(**params)

    raise ValueError(f"Unsupported model: {model_name}")


def get_default_sweep_grid(model_name: str) -> dict[str, list[Any]]:
    if model_name == "logistic_regression":
        return {
            "C": [0.25, 0.5, 1.0, 2.0],
        }

    if model_name == "random_forest":
        return {
            "n_estimators": [200, 300, 500],
            "max_depth": [None, 12, 20],
            "min_samples_leaf": [1, 3],
        }

    raise ValueError(f"Unsupported model: {model_name}")


def iter_sweep_configs(model_name: str) -> Iterator[dict[str, Any]]:
    grid = get_default_sweep_grid(model_name)
    keys = list(grid.keys())
    for values in product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values))


def predict_scores(model, features):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(features)
    raise ValueError("Model must expose predict_proba or decision_function.")
