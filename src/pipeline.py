from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .data_processing import (
    DEFAULT_RANDOM_SEED,
    PreparedDatasetMetadata,
    apply_scaler,
    ensure_directory,
    fit_scaler,
    load_metadata,
    load_prepared_frame,
    prepare_dataset,
    resample_training_frame,
    split_features_and_target,
    temporal_train_test_split,
)
from .evaluation import compute_metrics, save_evaluation_report, save_precision_recall_curve
from .model import build_model, iter_sweep_configs, predict_scores


def prepare_pipeline(
    input_csv: str | None,
    output_dir: str | Path,
    test_size: float = 0.2,
    sampling_strategy: float = 0.25,
    random_seed: int = DEFAULT_RANDOM_SEED,
    download_url: str | None = None,
) -> PreparedDatasetMetadata:
    kwargs = {
        "input_csv": input_csv,
        "output_dir": output_dir,
        "test_size": test_size,
        "sampling_strategy": sampling_strategy,
        "random_seed": random_seed,
    }
    if download_url:
        kwargs["download_url"] = download_url
    return prepare_dataset(**kwargs)


def load_training_data(prepared_dir: str | Path) -> tuple[pd.DataFrame, pd.Series, PreparedDatasetMetadata]:
    metadata = load_metadata(prepared_dir)
    train_frame = load_prepared_frame(prepared_dir, "train_resampled.csv")
    features, target = split_features_and_target(train_frame, target_column=metadata.target_column)
    return features[metadata.feature_columns], target, metadata


def load_test_data(prepared_dir: str | Path) -> tuple[pd.DataFrame, pd.Series, PreparedDatasetMetadata]:
    metadata = load_metadata(prepared_dir)
    test_frame = load_prepared_frame(prepared_dir, "test_processed.csv")
    features, target = split_features_and_target(test_frame, target_column=metadata.target_column)
    return features[metadata.feature_columns], target, metadata


def train_model_pipeline(
    prepared_dir: str | Path,
    model_name: str,
    output_dir: str | Path,
    model_params: dict[str, object] | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, str]:
    features, target, metadata = load_training_data(prepared_dir)
    requested_params = dict(model_params or {})
    model = build_model(model_name, random_seed=random_seed, **requested_params)
    model.fit(features, target)

    output_path = ensure_directory(Path(output_dir))
    model_path = output_path / f"{model_name}.joblib"
    summary_path = output_path / f"{model_name}_training_summary.json"

    summary_payload = {
        "model_name": model_name,
        "prepared_dir": str(Path(prepared_dir).resolve()),
        "random_seed": random_seed,
        "training_rows": int(len(features)),
        "feature_count": len(metadata.feature_columns),
        "requested_parameters": requested_params,
        "resolved_parameters": model.get_params(),
    }
    setattr(model, "training_metadata", summary_payload)
    joblib.dump(model, model_path)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return {
        "model_path": str(model_path),
        "summary_path": str(summary_path),
    }


def evaluate_model_pipeline(
    prepared_dir: str | Path,
    model_path: str | Path,
    report_dir: str | Path,
    threshold: float = 0.5,
) -> dict[str, str | float]:
    features, target, metadata = load_test_data(prepared_dir)
    model_file = Path(model_path)
    model = joblib.load(model_file)
    scores = predict_scores(model, features[metadata.feature_columns])
    result = compute_metrics(target, scores, threshold=threshold)

    stem = model_file.stem
    artifacts = save_precision_recall_curve(target, scores, report_dir, stem)
    report_path = save_evaluation_report(
        result,
        report_dir,
        stem,
        extra_payload={
            "model_name": stem,
            "model_path": str(model_file.resolve()),
            "prepared_dir": str(Path(prepared_dir).resolve()),
        },
    )

    artifacts["report_json"] = str(report_path)
    artifacts["average_precision"] = result.average_precision
    artifacts["f1_score"] = result.f1_score
    artifacts["precision"] = result.precision
    artifacts["recall"] = result.recall
    return artifacts


def sweep_model_pipeline(
    prepared_dir: str | Path,
    model_name: str,
    output_dir: str | Path,
    validation_size: float = 0.2,
    sampling_strategy: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    threshold: float = 0.5,
) -> dict[str, str]:
    metadata = load_metadata(prepared_dir)
    train_raw = load_prepared_frame(prepared_dir, "train_raw.csv")
    train_split, validation_split, _, _ = temporal_train_test_split(
        train_raw,
        test_size=validation_size,
    )

    feature_columns = metadata.feature_columns
    scaler = fit_scaler(train_split, feature_columns)
    train_scaled = apply_scaler(train_split, scaler, feature_columns)
    validation_scaled = apply_scaler(validation_split, scaler, feature_columns)

    effective_sampling_strategy = (
        sampling_strategy if sampling_strategy is not None else metadata.sampling_strategy
    )
    train_resampled = resample_training_frame(
        train_scaled,
        feature_columns=feature_columns,
        target_column=metadata.target_column,
        sampling_strategy=effective_sampling_strategy,
        random_seed=random_seed,
    )

    x_train, y_train = split_features_and_target(train_resampled, target_column=metadata.target_column)
    x_validation, y_validation = split_features_and_target(
        validation_scaled,
        target_column=metadata.target_column,
    )

    rows: list[dict] = []
    for config in iter_sweep_configs(model_name):
        model = build_model(model_name, random_seed=random_seed, **config)
        model.fit(x_train[feature_columns], y_train)
        scores = predict_scores(model, x_validation[feature_columns])
        metrics = compute_metrics(y_validation, scores, threshold=threshold)
        row = {
            "model_name": model_name,
            "threshold": threshold,
            "validation_size": validation_size,
            "sampling_strategy": effective_sampling_strategy,
            **config,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "average_precision": metrics.average_precision,
        }
        rows.append(row)

    results_frame = pd.DataFrame(rows).sort_values(
        by=["average_precision", "f1_score", "recall"],
        ascending=False,
    )

    output_path = ensure_directory(Path(output_dir))
    results_path = output_path / f"{model_name}_sweep_results.csv"
    summary_path = output_path / f"{model_name}_sweep_summary.json"
    results_frame.to_csv(results_path, index=False)

    best_row = results_frame.iloc[0].to_dict()
    summary_payload = {
        "model_name": model_name,
        "prepared_dir": str(Path(prepared_dir).resolve()),
        "random_seed": random_seed,
        "validation_size": validation_size,
        "sampling_strategy": effective_sampling_strategy,
        "best_configuration": best_row,
        "evaluated_configurations": int(len(results_frame)),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return {
        "results_csv": str(results_path),
        "summary_json": str(summary_path),
    }
