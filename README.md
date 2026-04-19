# Credit Card Fraud Detection CLI

A modular, reproducible command-line project for fraud detection on the European cardholder dataset (`creditcard.csv`).

The implementation follows the HCMUT Internship 1 engineering constraints:

- No notebook-heavy workflow
- Modular `src/` layout with separated concerns
- Deterministic pipeline with `random_seed = 42`
- Temporal holdout split based on the `Time` column
- Evaluation focused on imbalanced classification metrics
- Hyperparameter sweep support for model comparison

## Project Structure

```text
.
|-- README.md
|-- requirements.txt
|-- src
|   |-- __init__.py
|   |-- cli.py
|   |-- data_processing.py
|   |-- evaluation.py
|   |-- model.py
|   `-- pipeline.py
```

## Pipeline Overview

1. `prepare`
   - Downloads the dataset automatically or loads a local CSV
   - Sorts rows by `Time`
   - Applies a temporal train/test split
   - Fits scaling on the training period only
   - Handles class imbalance with SMOTE on the training split
   - Saves raw and processed artifacts under `data/`

2. `train`
   - Trains one or both baseline models:
     - `logistic_regression`
     - `random_forest`
   - Accepts optional model-specific hyperparameters through CLI flags
   - Saves fitted models to `artifacts/models/`
   - Persists requested and resolved training parameters for reproducibility

3. `evaluate`
   - Scores trained models on the untouched temporal holdout
   - Reports:
     - Recall
     - Precision
     - F1-score
     - Average precision
     - Precision-recall curve (`.csv` and `.png`)

4. `sweep`
   - Runs a deterministic hyperparameter sweep
   - Uses an inner temporal validation split carved from the training timeline
   - Saves ranked sweep results for model selection

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

By default, `prepare` downloads the public TensorFlow-hosted mirror of the European credit card fraud dataset:

- Default URL: `https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`

You can also provide your own local file with `--input-csv`.

## CLI Usage

Run commands with:

```bash
python -m src.cli <command> [options]
```

### 1. Prepare the data

```bash
python -m src.cli prepare
```

Use a local dataset:

```bash
python -m src.cli prepare --input-csv /path/to/creditcard.csv
```

Tune preprocessing behavior:

```bash
python -m src.cli prepare --test-size 0.2 --sampling-strategy 0.25
```

### 2. Train models

Train both models:

```bash
python -m src.cli train --model all
```

The train command keeps the baseline defaults when no extra flags are provided, so `python -m src.cli train` reproduces the original configuration.

Train only logistic regression:

```bash
python -m src.cli train --model logistic_regression
```

Train logistic regression with custom hyperparameters:

```bash
python -m src.cli train --model logistic_regression --lr-c 0.5 --lr-max-iter 3000 --lr-solver liblinear
```

Train random forest with custom hyperparameters:

```bash
python -m src.cli train --model random_forest --rf-n-estimators 500 --rf-max-depth 12 --rf-min-samples-split 4
```

Available training flags:

- Logistic regression: `--lr-c`, `--lr-max-iter`, `--lr-solver`
- Random forest: `--rf-n-estimators`, `--rf-max-depth`, `--rf-min-samples-split`

### 3. Evaluate trained models

Evaluate all saved models:

```bash
python -m src.cli evaluate --model all
```

Evaluate a single model with a custom threshold:

```bash
python -m src.cli evaluate --model random_forest --threshold 0.35
```

Evaluate from an explicit artifact path:

```bash
python -m src.cli evaluate --model-path artifacts/models/logistic_regression.joblib
```

### 4. Run a hyperparameter sweep

Sweep logistic regression:

```bash
python -m src.cli sweep --model logistic_regression
```

Sweep random forest with a different validation fraction:

```bash
python -m src.cli sweep --model random_forest --validation-size 0.25
```

## Output Layout

After a full run, the generated directories typically look like this:

```text
.
|-- artifacts
|   |-- models
|   |   |-- logistic_regression.joblib
|   |   |-- logistic_regression_training_summary.json
|   |   |-- random_forest.joblib
|   |   `-- random_forest_training_summary.json
|   |-- reports
|   |   |-- logistic_regression_evaluation.json
|   |   |-- logistic_regression_precision_recall_curve.csv
|   |   |-- logistic_regression_precision_recall_curve.png
|   |   |-- random_forest_evaluation.json
|   |   |-- random_forest_precision_recall_curve.csv
|   |   `-- random_forest_precision_recall_curve.png
|   `-- sweeps
|       |-- logistic_regression_sweep_results.csv
|       `-- logistic_regression_sweep_summary.json
`-- data
    |-- raw
    |   `-- creditcard.csv
    `-- processed
        |-- metadata.json
        |-- scaler.joblib
        |-- test_processed.csv
        |-- test_raw.csv
        |-- train_processed.csv
        |-- train_raw.csv
        `-- train_resampled.csv
```

## Module Responsibilities

- `src/data_processing.py`: dataset I/O, validation, temporal split, scaling, imbalance handling, metadata persistence
- `src/model.py`: supported model registry and sweep search spaces
- `src/pipeline.py`: command-level orchestration for prepare, train, evaluate, and sweep
- `src/evaluation.py`: metric computation, report generation, precision-recall plotting
- `src/cli.py`: `argparse` command-line interface

## Reproducibility Notes

- All random behavior is fixed with `random_seed = 42`
- The temporal holdout prevents leakage from future transactions into training
- Scaling is fit on the training period only
- Evaluation is run on the untouched test partition
- Training summaries record both the requested CLI hyperparameters and the resolved fitted model parameters
