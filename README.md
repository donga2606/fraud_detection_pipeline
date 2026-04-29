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
|-- ui
|   `-- app.py
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
   - Creates a unique `run_id` for each training invocation
   - Saves fitted models under `artifacts/models/<run_id>/`
   - Persists requested and resolved training parameters for reproducibility

3. `evaluate`
   - Scores trained models on the untouched temporal holdout
   - Reuses the model's `run_id` when evaluating a run-managed artifact
   - Reports:
     - Recall
     - Precision
     - F1-score
     - Average precision
     - Precision-recall curve (`.csv` and `.png`)
   - Updates `artifacts/run_history.csv` for cross-run comparison

4. `sweep`
   - Runs a deterministic hyperparameter sweep
   - Uses an inner temporal validation split carved from the training timeline
   - Saves ranked sweep results under `artifacts/sweeps/<run_id>/` for later review

5. `compare`
   - Rebuilds `artifacts/run_history.csv` from saved run manifests
   - Produces a ranked CSV or JSON comparison report across evaluated runs
   - Lets you compare models and training hyperparameters without a separate UI

6. `dashboard`
   - Launches separately with Streamlit after CLI training/evaluation completes
   - Reads saved run manifests and `artifacts/run_history.csv`
   - Shows run tables, detailed artifacts, and side-by-side comparisons

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

Each `train` command now creates a unique run folder under `artifacts/models/`, so repeated training no longer overwrites older model artifacts.
The CLI also prints a training start message and elapsed fit time, and random forest training shows sklearn progress output so long runs do not appear frozen.

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
python -m src.cli evaluate --model-path artifacts/models/<run_id>/logistic_regression.joblib
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

### 5. Compare saved runs

Build a CSV comparison report from all evaluated runs:

```bash
python -m src.cli compare
```

Show only the top 5 logistic regression runs:

```bash
python -m src.cli compare --model logistic_regression --top-n 5
```

Write the comparison output as JSON:

```bash
python -m src.cli compare --output-path artifacts/comparison_report.json
```

### 6. Launch the dashboard UI

Start the Streamlit dashboard after you have trained and evaluated one or more runs:

```bash
streamlit run ui/app.py
```

Use a different artifacts directory in the sidebar if you want to inspect another experiment root.

## Output Layout

After a full run, the generated directories typically look like this:

```text
.
|-- artifacts
|   |-- comparison_report.csv
|   |-- run_history.csv
|   |-- models
|   |   `-- 20260429T065500Z-ab12cd34
|   |       |-- logistic_regression.joblib
|   |       `-- logistic_regression_training_summary.json
|   |-- reports
|   |   `-- 20260429T065500Z-ab12cd34
|   |       |-- logistic_regression_evaluation.json
|   |       |-- logistic_regression_precision_recall_curve.csv
|   |       `-- logistic_regression_precision_recall_curve.png
|   |-- runs
|   |   `-- 20260429T065500Z-ab12cd34
|   |       `-- run_manifest.json
|   `-- sweeps
|       `-- 20260429T070200Z-ef56gh78
|           |-- random_forest_sweep_results.csv
|           `-- random_forest_sweep_summary.json
|-- ui
|   `-- app.py
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
- Run manifests capture the artifact paths and evaluation metrics for each saved run
- `compare` rebuilds a comparison table from saved manifests, so historical runs remain comparable even after many experiments

## Recommended Workflow

Use the CLI for experiment execution and the dashboard for inspection:

```bash
python -m src.cli train --model logistic_regression --lr-c 0.5 --lr-max-iter 3000
python -m src.cli evaluate --model logistic_regression
streamlit run ui/app.py
```

This keeps training reproducible and scriptable while giving you a simple UI to review metrics, hyperparameters, and precision-recall artifacts afterward.
