from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

DEFAULT_DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
DEFAULT_RANDOM_SEED = 42
TARGET_COLUMN = "Class"
TIME_COLUMN = "Time"


@dataclass
class PreparedDatasetMetadata:
    dataset_path: str
    feature_columns: list[str]
    target_column: str
    time_column: str
    train_rows: int
    test_rows: int
    resampled_train_rows: int
    split_index: int
    split_time_value: float
    test_size: float
    sampling_strategy: float
    random_seed: int


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_creditcard_dataframe(dataframe: pd.DataFrame) -> None:
    required_columns = {TARGET_COLUMN, TIME_COLUMN, "Amount"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing_str}")


def load_creditcard_dataframe(csv_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)
    validate_creditcard_dataframe(dataframe)
    return dataframe.sort_values(TIME_COLUMN).reset_index(drop=True)


def download_dataset(download_url: str, destination: Path) -> Path:
    ensure_directory(destination.parent)
    urlretrieve(download_url, destination)
    return destination


def resolve_dataset_path(
    input_csv: str | None,
    raw_dir: Path,
    download_url: str = DEFAULT_DATA_URL,
) -> Path:
    if input_csv:
        csv_path = Path(input_csv).expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")
        return csv_path

    destination = raw_dir / "creditcard.csv"
    if destination.exists():
        return destination

    return download_dataset(download_url, destination)


def temporal_train_test_split(
    dataframe: pd.DataFrame,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, int, float]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    split_index = int(len(dataframe) * (1 - test_size))
    if split_index <= 0 or split_index >= len(dataframe):
        raise ValueError("Temporal split produced an empty train or test partition.")

    train_frame = dataframe.iloc[:split_index].copy()
    test_frame = dataframe.iloc[split_index:].copy()
    split_time_value = float(test_frame[TIME_COLUMN].iloc[0])
    return train_frame, test_frame, split_index, split_time_value


def split_features_and_target(
    dataframe: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    feature_frame = dataframe.drop(columns=[target_column])
    target_series = dataframe[target_column].copy()
    return feature_frame, target_series


def fit_scaler(
    train_frame: pd.DataFrame,
    feature_columns: Iterable[str],
) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_frame[list(feature_columns)])
    return scaler


def apply_scaler(
    dataframe: pd.DataFrame,
    scaler: StandardScaler,
    feature_columns: Iterable[str],
) -> pd.DataFrame:
    scaled_frame = dataframe.copy()
    scaled_frame.loc[:, list(feature_columns)] = scaler.transform(
        scaled_frame[list(feature_columns)]
    )
    return scaled_frame


def resample_training_frame(
    train_frame: pd.DataFrame,
    feature_columns: Iterable[str],
    target_column: str = TARGET_COLUMN,
    sampling_strategy: float = 0.25,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    features, target = split_features_and_target(train_frame, target_column=target_column)
    sampler = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_seed,
    )
    resampled_features, resampled_target = sampler.fit_resample(
        features[list(feature_columns)],
        target,
    )

    resampled_frame = pd.DataFrame(resampled_features, columns=list(feature_columns))
    resampled_frame[target_column] = resampled_target
    return resampled_frame


def save_dataframe(dataframe: pd.DataFrame, destination: Path) -> Path:
    ensure_directory(destination.parent)
    dataframe.to_csv(destination, index=False)
    return destination


def save_metadata(metadata: PreparedDatasetMetadata, destination: Path) -> Path:
    ensure_directory(destination.parent)
    destination.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
    return destination


def load_metadata(prepared_dir: str | Path) -> PreparedDatasetMetadata:
    metadata_path = Path(prepared_dir) / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return PreparedDatasetMetadata(**payload)


def load_prepared_frame(prepared_dir: str | Path, filename: str) -> pd.DataFrame:
    return pd.read_csv(Path(prepared_dir) / filename)


def prepare_dataset(
    input_csv: str | None,
    output_dir: str | Path,
    test_size: float = 0.2,
    sampling_strategy: float = 0.25,
    random_seed: int = DEFAULT_RANDOM_SEED,
    download_url: str = DEFAULT_DATA_URL,
) -> PreparedDatasetMetadata:
    output_path = Path(output_dir)
    raw_dir = ensure_directory(output_path / "raw")
    processed_dir = ensure_directory(output_path / "processed")

    dataset_path = resolve_dataset_path(input_csv, raw_dir, download_url=download_url)
    dataframe = load_creditcard_dataframe(dataset_path)
    feature_columns = [column for column in dataframe.columns if column != TARGET_COLUMN]

    train_raw, test_raw, split_index, split_time_value = temporal_train_test_split(
        dataframe,
        test_size=test_size,
    )
    scaler = fit_scaler(train_raw, feature_columns)
    train_processed = apply_scaler(train_raw, scaler, feature_columns)
    test_processed = apply_scaler(test_raw, scaler, feature_columns)
    train_resampled = resample_training_frame(
        train_processed,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
        sampling_strategy=sampling_strategy,
        random_seed=random_seed,
    )

    save_dataframe(train_raw, processed_dir / "train_raw.csv")
    save_dataframe(test_raw, processed_dir / "test_raw.csv")
    save_dataframe(train_processed, processed_dir / "train_processed.csv")
    save_dataframe(test_processed, processed_dir / "test_processed.csv")
    save_dataframe(train_resampled, processed_dir / "train_resampled.csv")
    joblib.dump(scaler, processed_dir / "scaler.joblib")

    metadata = PreparedDatasetMetadata(
        dataset_path=str(dataset_path),
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
        time_column=TIME_COLUMN,
        train_rows=len(train_processed),
        test_rows=len(test_processed),
        resampled_train_rows=len(train_resampled),
        split_index=split_index,
        split_time_value=split_time_value,
        test_size=test_size,
        sampling_strategy=sampling_strategy,
        random_seed=random_seed,
    )
    save_metadata(metadata, processed_dir / "metadata.json")
    return metadata
