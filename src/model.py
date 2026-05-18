from __future__ import annotations

from itertools import product
from typing import Any, Iterator

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

MODEL_CHOICES = ("logistic_regression", "random_forest", "xgboost", "lightgbm")
TRAINING_RANDOM_SEED = 42

DEFAULT_LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 2000,
    "solver": "liblinear",
}

DEFAULT_RANDOM_FOREST_PARAMS = {
    "class_weight": "balanced_subsample",
    "max_depth": None,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "n_estimators": 300,
    "n_jobs": -1,
}

DEFAULT_XGBOOST_PARAMS = {
    "colsample_bytree": 0.8,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "n_estimators": 300,
    "n_jobs": -1,
    "subsample": 0.8,
}

DEFAULT_LIGHTGBM_PARAMS = {
    "class_weight": "balanced",
    "colsample_bytree": 0.8,
    "learning_rate": 0.05,
    "max_depth": -1,
    "min_child_samples": 20,
    "n_estimators": 300,
    "n_jobs": -1,
    "num_leaves": 31,
    "subsample": 0.8,
    "verbosity": -1,
}

_OPENMP_IMPORT_HINT = (
    "XGBoost and LightGBM require the OpenMP runtime. On macOS run: brew install libomp. "
    "On Debian/Ubuntu run: sudo apt install libgomp1. Or use the Docker setup in README.md."
)


def _import_xgb_classifier():
    try:
        from xgboost import XGBClassifier
    except OSError as exc:
        raise ImportError(_OPENMP_IMPORT_HINT) from exc
    return XGBClassifier


def _import_lgbm_classifier():
    try:
        from lightgbm import LGBMClassifier
    except OSError as exc:
        raise ImportError(_OPENMP_IMPORT_HINT) from exc
    return LGBMClassifier


def build_model(
    model_name: str,
    random_seed: int = TRAINING_RANDOM_SEED,
    **overrides: Any,
):
    if model_name == "logistic_regression":
        params = DEFAULT_LOGISTIC_REGRESSION_PARAMS.copy()
        params.update(overrides)
        params["random_state"] = TRAINING_RANDOM_SEED
        return LogisticRegression(**params)

    if model_name == "random_forest":
        params = DEFAULT_RANDOM_FOREST_PARAMS.copy()
        params.update(overrides)
        params["random_state"] = TRAINING_RANDOM_SEED
        return RandomForestClassifier(**params)

    if model_name == "xgboost":
        XGBClassifier = _import_xgb_classifier()
        params = DEFAULT_XGBOOST_PARAMS.copy()
        params.update(overrides)
        params["random_state"] = TRAINING_RANDOM_SEED
        return XGBClassifier(**params)

    if model_name == "lightgbm":
        LGBMClassifier = _import_lgbm_classifier()
        params = DEFAULT_LIGHTGBM_PARAMS.copy()
        params.update(overrides)
        params["random_state"] = TRAINING_RANDOM_SEED
        return LGBMClassifier(**params)

    raise ValueError(f"Unsupported model: {model_name}")


def get_default_sweep_grid(model_name: str) -> dict[str, list[Any]]:
    if model_name == "logistic_regression":
        return {
            "C": [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
        }

    if model_name == "random_forest":
        return {
            "n_estimators": [200, 300, 500],
            "max_depth": [None, 12, 20],
            "min_samples_leaf": [3],
        }

    if model_name == "xgboost":
        return {
            "n_estimators": [200, 300, 500],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
        }

    if model_name == "lightgbm":
        return {
            "n_estimators": [200, 300, 500],
            "num_leaves": [31, 63],
            "learning_rate": [0.03, 0.05, 0.1],
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
