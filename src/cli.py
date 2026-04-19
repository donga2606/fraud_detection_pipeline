from __future__ import annotations

import argparse
from pathlib import Path

from .data_processing import DEFAULT_DATA_URL, DEFAULT_RANDOM_SEED
from .model import (
    DEFAULT_LOGISTIC_REGRESSION_PARAMS,
    DEFAULT_RANDOM_FOREST_PARAMS,
    MODEL_CHOICES,
    TRAINING_RANDOM_SEED,
)
from .pipeline import (
    evaluate_model_pipeline,
    prepare_pipeline,
    sweep_model_pipeline,
    train_model_pipeline,
)


def add_prepare_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "prepare",
        help="Download or load the dataset, apply a temporal split, scale features, and resample the training set.",
    )
    parser.add_argument("--input-csv", type=str, default=None, help="Local path to creditcard.csv.")
    parser.add_argument(
        "--download-url",
        type=str,
        default=DEFAULT_DATA_URL,
        help="Remote CSV URL used when --input-csv is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Base output directory for raw and processed data.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction reserved for the temporal test holdout.",
    )
    parser.add_argument(
        "--sampling-strategy",
        type=float,
        default=0.25,
        help="SMOTE minority/majority ratio applied on the training split.",
    )


def add_train_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "train",
        help="Train one or more fraud detection models on the prepared dataset.",
    )
    parser.add_argument(
        "--prepared-dir",
        type=str,
        default="data/processed",
        help="Directory produced by the prepare command.",
    )
    parser.add_argument(
        "--model",
        choices=[*MODEL_CHOICES, "all"],
        default="all",
        help="Which model to train.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/models",
        help="Directory where fitted model artifacts are stored.",
    )
    parser.add_argument(
        "--lr-c",
        type=float,
        default=DEFAULT_LOGISTIC_REGRESSION_PARAMS["C"],
        help="Inverse regularization strength for logistic regression.",
    )
    parser.add_argument(
        "--lr-max-iter",
        type=int,
        default=DEFAULT_LOGISTIC_REGRESSION_PARAMS["max_iter"],
        help="Maximum number of iterations for logistic regression.",
    )
    parser.add_argument(
        "--lr-solver",
        type=str,
        default=DEFAULT_LOGISTIC_REGRESSION_PARAMS["solver"],
        help="Solver used by logistic regression.",
    )
    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=DEFAULT_RANDOM_FOREST_PARAMS["n_estimators"],
        help="Number of trees in the random forest.",
    )
    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=DEFAULT_RANDOM_FOREST_PARAMS["max_depth"],
        help="Maximum tree depth for the random forest.",
    )
    parser.add_argument(
        "--rf-min-samples-split",
        type=int,
        default=DEFAULT_RANDOM_FOREST_PARAMS["min_samples_split"],
        help="Minimum samples required to split an internal node.",
    )


def add_evaluate_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained model with recall, F1, average precision, and a precision-recall curve.",
    )
    parser.add_argument(
        "--prepared-dir",
        type=str,
        default="data/processed",
        help="Directory produced by the prepare command.",
    )
    parser.add_argument(
        "--model",
        choices=[*MODEL_CHOICES, "all"],
        default="all",
        help="Which saved model to evaluate from --models-dir.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Explicit path to a saved model artifact. Overrides --model and --models-dir.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="artifacts/models",
        help="Directory that stores trained model artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/reports",
        help="Directory for evaluation reports and precision-recall curves.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold applied to fraud scores.",
    )


def add_sweep_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "sweep",
        help="Run a deterministic hyperparameter sweep using an inner temporal validation split.",
    )
    parser.add_argument(
        "--prepared-dir",
        type=str,
        default="data/processed",
        help="Directory produced by the prepare command.",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        required=True,
        help="Which model family to sweep.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/sweeps",
        help="Directory where sweep outputs are written.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Fraction of the training timeline reserved for inner validation.",
    )
    parser.add_argument(
        "--sampling-strategy",
        type=float,
        default=None,
        help="Optional SMOTE ratio override for the sweep.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold used while ranking sweep candidates.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI pipeline for the European credit card fraud detection dataset."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_prepare_command(subparsers)
    add_train_command(subparsers)
    add_evaluate_command(subparsers)
    add_sweep_command(subparsers)
    return parser


def resolve_models(model_argument: str) -> list[str]:
    if model_argument == "all":
        return list(MODEL_CHOICES)
    return [model_argument]


def get_train_model_params(args, model_name: str) -> dict[str, object]:
    if model_name == "logistic_regression":
        return {
            "C": args.lr_c,
            "max_iter": args.lr_max_iter,
            "solver": args.lr_solver,
        }

    if model_name == "random_forest":
        return {
            "n_estimators": args.rf_n_estimators,
            "max_depth": args.rf_max_depth,
            "min_samples_split": args.rf_min_samples_split,
        }

    raise ValueError(f"Unsupported model: {model_name}")


def handle_prepare(args) -> int:
    metadata = prepare_pipeline(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        test_size=args.test_size,
        sampling_strategy=args.sampling_strategy,
        random_seed=DEFAULT_RANDOM_SEED,
        download_url=args.download_url,
    )
    print(f"Prepared dataset in {Path(args.output_dir).resolve()}")
    print(
        "Temporal split:"
        f" train_rows={metadata.train_rows},"
        f" test_rows={metadata.test_rows},"
        f" split_time={metadata.split_time_value:.2f}"
    )
    print(f"Resampled training rows: {metadata.resampled_train_rows}")
    return 0


def handle_train(args) -> int:
    for model_name in resolve_models(args.model):
        result = train_model_pipeline(
            prepared_dir=args.prepared_dir,
            model_name=model_name,
            output_dir=args.output_dir,
            model_params=get_train_model_params(args, model_name),
            random_seed=TRAINING_RANDOM_SEED,
        )
        print(f"Trained {model_name}: {result['model_path']}")
    return 0


def iter_evaluation_targets(args) -> list[tuple[str, str]]:
    if args.model_path:
        model_path = Path(args.model_path)
        return [(model_path.stem, str(model_path))]

    targets = []
    for model_name in resolve_models(args.model):
        targets.append((model_name, str(Path(args.models_dir) / f"{model_name}.joblib")))
    return targets


def handle_evaluate(args) -> int:
    for model_name, model_path in iter_evaluation_targets(args):
        result = evaluate_model_pipeline(
            prepared_dir=args.prepared_dir,
            model_path=model_path,
            report_dir=args.output_dir,
            threshold=args.threshold,
        )
        print(
            f"Evaluated {model_name}: "
            f"precision={result['precision']:.4f}, "
            f"recall={result['recall']:.4f}, "
            f"f1={result['f1_score']:.4f}, "
            f"average_precision={result['average_precision']:.4f}"
        )
        print(f"Report: {result['report_json']}")
    return 0


def handle_sweep(args) -> int:
    result = sweep_model_pipeline(
        prepared_dir=args.prepared_dir,
        model_name=args.model,
        output_dir=args.output_dir,
        validation_size=args.validation_size,
        sampling_strategy=args.sampling_strategy,
        random_seed=DEFAULT_RANDOM_SEED,
        threshold=args.threshold,
    )
    print(f"Sweep results: {result['results_csv']}")
    print(f"Sweep summary: {result['summary_json']}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        return handle_prepare(args)
    if args.command == "train":
        return handle_train(args)
    if args.command == "evaluate":
        return handle_evaluate(args)
    if args.command == "sweep":
        return handle_sweep(args)

    parser.error(f"Unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
