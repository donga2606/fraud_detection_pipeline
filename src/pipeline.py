from __future__ import annotations

import json
import shlex
import time
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
from .evaluation import (
    compute_metrics,
    save_confusion_matrix,
    save_evaluation_report,
    save_precision_recall_curve,
)
from .model import build_model, iter_sweep_configs, predict_scores
from .run_history import (
    create_run_id,
    extract_run_context_from_artifact_path,
    infer_artifact_root,
    rebuild_run_history_index,
    resolve_run_stage_dir,
    upsert_run_manifest,
    utc_now_iso,
)


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


def resolve_saved_model_path(models_dir: str | Path, model_name: str) -> Path:
    models_root = Path(models_dir)
    candidates = list(models_root.glob(f"*/{model_name}.joblib"))
    legacy_path = models_root / f"{model_name}.joblib"
    if legacy_path.exists():
        candidates.append(legacy_path)

    candidates = sorted(candidates, key=lambda candidate: candidate.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No saved model found for {model_name} under {models_root.resolve()}")
    return candidates[0]


SWEEP_TRAIN_PARAM_FLAGS = {
    "logistic_regression": {
        "C": "--lr-c",
        "max_iter": "--lr-max-iter",
        "solver": "--lr-solver",
    },
    "random_forest": {
        "n_estimators": "--rf-n-estimators",
        "max_depth": "--rf-max-depth",
        "min_samples_split": "--rf-min-samples-split",
        "min_samples_leaf": "--rf-min-samples-leaf",
    },
    "xgboost": {
        "n_estimators": "--xgb-n-estimators",
        "max_depth": "--xgb-max-depth",
        "learning_rate": "--xgb-learning-rate",
        "subsample": "--xgb-subsample",
        "colsample_bytree": "--xgb-colsample-bytree",
        "min_child_weight": "--xgb-min-child-weight",
    },
    "lightgbm": {
        "n_estimators": "--lgbm-n-estimators",
        "max_depth": "--lgbm-max-depth",
        "learning_rate": "--lgbm-learning-rate",
        "num_leaves": "--lgbm-num-leaves",
        "min_child_samples": "--lgbm-min-child-samples",
        "subsample": "--lgbm-subsample",
        "colsample_bytree": "--lgbm-colsample-bytree",
    },
}


def shell_quote(value: object) -> str:
    return shlex.quote(str(value))


def build_sweep_followup_commands(
    model_name: str,
    prepared_dir: str | Path,
    artifact_root: str | Path,
    best_configuration: dict[str, object],
    threshold: float,
) -> dict[str, str]:
    models_dir = Path(artifact_root) / "models"
    reports_dir = Path(artifact_root) / "reports"
    train_parts = [
        "python",
        "-m",
        "src.cli",
        "train",
        "--prepared-dir",
        shell_quote(prepared_dir),
        "--model",
        shell_quote(model_name),
        "--output-dir",
        shell_quote(models_dir),
    ]
    for param_name, flag in SWEEP_TRAIN_PARAM_FLAGS[model_name].items():
        value = best_configuration.get(param_name)
        if value is None:
            continue
        train_parts.extend([flag, shell_quote(value)])

    evaluate_parts = [
        "python",
        "-m",
        "src.cli",
        "evaluate",
        "--prepared-dir",
        shell_quote(prepared_dir),
        "--model",
        shell_quote(model_name),
        "--models-dir",
        shell_quote(models_dir),
        "--output-dir",
        shell_quote(reports_dir),
        "--threshold",
        shell_quote(threshold),
    ]
    return {
        "train_best_model": " ".join(train_parts),
        "evaluate_latest_model": " ".join(evaluate_parts),
    }


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
    print(
        f"Starting training for {model_name} "
        f"(rows={len(features)}, features={len(metadata.feature_columns)}).",
        flush=True,
    )
    if model_name == "random_forest" and "verbose" in model.get_params():
        model.set_params(verbose=1)
        print("Random forest progress logging is enabled.", flush=True)
    if model_name == "xgboost" and "verbose" in model.get_params():
        model.set_params(verbose=1)
        print("XGBoost progress logging is enabled.", flush=True)
    if model_name == "lightgbm" and "verbosity" in model.get_params():
        model.set_params(verbosity=1)
        print("LightGBM progress logging is enabled.", flush=True)
    fit_started_at = time.perf_counter()
    model.fit(features, target)
    fit_elapsed_seconds = time.perf_counter() - fit_started_at
    print(f"Finished fitting {model_name} in {fit_elapsed_seconds:.1f}s.", flush=True)

    run_id = create_run_id()
    created_at = utc_now_iso()
    output_path = ensure_directory(Path(output_dir))
    artifact_root = infer_artifact_root(output_path)
    run_output_path = resolve_run_stage_dir(output_path, run_id)
    model_path = run_output_path / f"{model_name}.joblib"
    summary_path = run_output_path / f"{model_name}_training_summary.json"

    summary_payload = {
        "run_id": run_id,
        "created_at": created_at,
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
    manifest_path = upsert_run_manifest(
        artifact_root,
        run_id,
        {
            "run_id": run_id,
            "created_at": created_at,
            "model_name": model_name,
            "prepared_dir": str(Path(prepared_dir).resolve()),
            "training": {
                "completed_at": created_at,
                "random_seed": random_seed,
                "training_rows": int(len(features)),
                "feature_count": len(metadata.feature_columns),
                "requested_parameters": requested_params,
                "resolved_parameters": model.get_params(),
                "model_path": str(model_path.resolve()),
                "summary_path": str(summary_path.resolve()),
            },
        },
    )

    return {
        "run_id": run_id,
        "run_dir": str(run_output_path),
        "model_path": str(model_path),
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
    }


def evaluate_model_pipeline(
    prepared_dir: str | Path,
    model_path: str | Path,
    report_dir: str | Path,
    threshold: float = 0.5,
) -> dict[str, str | float]:
    features, target, metadata = load_test_data(prepared_dir)
    model_file = Path(model_path).resolve()
    model = joblib.load(model_file)
    scores = predict_scores(model, features[metadata.feature_columns])
    result = compute_metrics(target, scores, threshold=threshold)

    report_output_root = ensure_directory(Path(report_dir))
    run_context = extract_run_context_from_artifact_path(model_file, "models")
    if run_context is not None:
        run_id = run_context["run_id"]
        artifact_root = Path(run_context["artifact_root"])
    else:
        run_id = create_run_id()
        artifact_root = infer_artifact_root(report_output_root)
    evaluated_at = utc_now_iso()
    run_report_path = resolve_run_stage_dir(report_output_root, run_id)

    stem = model_file.stem
    artifacts = save_precision_recall_curve(target, scores, run_report_path, stem)
    artifacts.update(save_confusion_matrix(target, scores, threshold, run_report_path, stem))
    report_path = save_evaluation_report(
        result,
        run_report_path,
        stem,
        extra_payload={
            "run_id": run_id,
            "evaluated_at": evaluated_at,
            "model_name": stem,
            "model_path": str(model_file.resolve()),
            "prepared_dir": str(Path(prepared_dir).resolve()),
        },
    )
    manifest_update = {
        "run_id": run_id,
        "model_name": stem,
        "prepared_dir": str(Path(prepared_dir).resolve()),
        "evaluation": {
            "completed_at": evaluated_at,
            "threshold": threshold,
            "precision": result.precision,
            "recall": result.recall,
            "f1_score": result.f1_score,
            "average_precision": result.average_precision,
            "positive_predictions": result.positive_predictions,
            "fraud_cases": result.fraud_cases,
            "true_negatives": result.true_negatives,
            "false_positives": result.false_positives,
            "false_negatives": result.false_negatives,
            "true_positives": result.true_positives,
            "model_path": str(model_file.resolve()),
            "report_json": str(report_path.resolve()),
            "curve_csv": str(Path(artifacts["curve_csv"]).resolve()),
            "curve_png": str(Path(artifacts["curve_png"]).resolve()),
            "confusion_matrix_png": str(Path(artifacts["confusion_matrix_png"]).resolve()),
        },
    }
    if run_context is None:
        manifest_update["created_at"] = evaluated_at

    manifest_path = upsert_run_manifest(
        artifact_root,
        run_id,
        manifest_update,
    )
    _, history_path = rebuild_run_history_index(artifact_root)

    artifacts["report_json"] = str(report_path)
    artifacts["run_id"] = run_id
    artifacts["manifest_path"] = str(manifest_path)
    artifacts["run_history_csv"] = str(history_path)
    artifacts["average_precision"] = result.average_precision
    artifacts["f1_score"] = result.f1_score
    artifacts["precision"] = result.precision
    artifacts["recall"] = result.recall
    artifacts["threshold"] = threshold
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

    run_id = create_run_id()
    completed_at = utc_now_iso()
    output_path = ensure_directory(Path(output_dir))
    artifact_root = infer_artifact_root(output_path)
    run_output_path = resolve_run_stage_dir(output_path, run_id)
    results_path = run_output_path / f"{model_name}_sweep_results.csv"
    summary_path = run_output_path / f"{model_name}_sweep_summary.json"
    results_frame.to_csv(results_path, index=False)

    best_row = results_frame.iloc[0].to_dict()
    recommended_commands = build_sweep_followup_commands(
        model_name=model_name,
        prepared_dir=prepared_dir,
        artifact_root=artifact_root,
        best_configuration=best_row,
        threshold=threshold,
    )
    summary_payload = {
        "run_id": run_id,
        "completed_at": completed_at,
        "model_name": model_name,
        "prepared_dir": str(Path(prepared_dir).resolve()),
        "random_seed": random_seed,
        "validation_size": validation_size,
        "sampling_strategy": effective_sampling_strategy,
        "best_configuration": best_row,
        "evaluated_configurations": int(len(results_frame)),
        "recommended_commands": recommended_commands,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    manifest_path = upsert_run_manifest(
        artifact_root,
        run_id,
        {
            "run_id": run_id,
            "created_at": completed_at,
            "model_name": model_name,
            "prepared_dir": str(Path(prepared_dir).resolve()),
            "sweep": {
                "completed_at": completed_at,
                "random_seed": random_seed,
                "validation_size": validation_size,
                "sampling_strategy": effective_sampling_strategy,
                "threshold": threshold,
                "results_csv": str(results_path.resolve()),
                "summary_json": str(summary_path.resolve()),
                "best_configuration": best_row,
                "evaluated_configurations": int(len(results_frame)),
                "recommended_commands": recommended_commands,
            },
        },
    )

    return {
        "run_id": run_id,
        "run_dir": str(run_output_path),
        "results_csv": str(results_path),
        "summary_json": str(summary_path),
        "manifest_path": str(manifest_path),
        "train_best_model_command": recommended_commands["train_best_model"],
        "evaluate_latest_model_command": recommended_commands["evaluate_latest_model"],
    }


def compare_runs_pipeline(
    artifact_root: str | Path,
    model_name: str | None = None,
    sort_by: str = "average_precision",
    top_n: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, str | int]:
    history_frame, history_path = rebuild_run_history_index(artifact_root)
    if history_frame.empty:
        return {
            "history_path": str(history_path),
            "comparison_path": "",
            "row_count": 0,
            "preview": "No evaluated runs found.",
        }

    filtered_frame = history_frame.copy()
    if model_name and model_name != "all":
        filtered_frame = filtered_frame[filtered_frame["model_name"] == model_name]

    if filtered_frame.empty:
        return {
            "history_path": str(history_path),
            "comparison_path": "",
            "row_count": 0,
            "preview": "No runs matched the requested filters.",
        }

    if sort_by not in filtered_frame.columns:
        raise ValueError(f"Unsupported sort field: {sort_by}")

    filtered_frame = filtered_frame.sort_values(
        by=[sort_by, "f1_score", "recall", "created_at"],
        ascending=[False, False, False, False],
        na_position="last",
    )
    if top_n is not None:
        filtered_frame = filtered_frame.head(top_n)

    comparison_destination = Path(output_path) if output_path else Path(artifact_root) / "comparison_report.csv"
    ensure_directory(comparison_destination.parent)
    if comparison_destination.suffix.lower() == ".json":
        comparison_destination.write_text(
            filtered_frame.to_json(orient="records", indent=2),
            encoding="utf-8",
        )
    else:
        filtered_frame.to_csv(comparison_destination, index=False)

    preview_columns = [
        column
        for column in [
            "run_id",
            "model_name",
            "average_precision",
            "f1_score",
            "precision",
            "recall",
            "threshold",
            "created_at",
        ]
        if column in filtered_frame.columns
    ]
    preview = filtered_frame[preview_columns].to_string(index=False)
    return {
        "history_path": str(history_path),
        "comparison_path": str(comparison_destination),
        "row_count": int(len(filtered_frame)),
        "preview": preview,
    }
