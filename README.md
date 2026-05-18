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

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

XGBoost and LightGBM need a system OpenMP runtime (not installed by pip):

- **macOS:** `brew install libomp`
- **Debian/Ubuntu:** `sudo apt install libgomp1`

If you skip those steps, `logistic_regression` and `random_forest` still work; `xgboost` and `lightgbm` train/sweep commands will fail until OpenMP is available. For a reproducible examiner setup, use [Docker Setup](#docker-setup) below.

## Docker Setup

If you want the instructor to run the project in a consistent environment, you can use Docker instead of a local Python installation.

Build the image once from the project root:

```bash
docker build -t fraud-detection .
```

Run the CLI commands through the container while mounting `data/` and `artifacts/` so generated files stay on the host machine:

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  fraud-detection prepare
```

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  fraud-detection train --model all
```

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  fraud-detection evaluate --model all
```

Launch the Streamlit dashboard from the same image:

```bash
docker run --rm \
  -p 8501:8501 \
  -v "$(pwd)/artifacts:/app/artifacts" \
  fraud-detection dashboard
```

Then open `http://localhost:8501` in a browser.

## Dataset

By default, `prepare` downloads the public TensorFlow-hosted mirror of the European credit card fraud dataset:

- Default URL: `https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`

You can also provide your own local file with `--input-csv`.

## Workflow Summary

The project includes five CLI commands plus one dashboard entrypoint:

1. `prepare`: validate, temporally split, scale, and resample data
2. `train`: fit `logistic_regression`, `random_forest`, `xgboost`, `lightgbm`, or all
3. `evaluate`: compute metrics and generate reports/plots for a saved model
4. `sweep`: run deterministic hyperparameter search on an inner temporal split
5. `compare`: rank evaluated runs and export CSV/JSON report
6. `dashboard`: inspect saved runs in Streamlit (via `streamlit run ui/app.py` or Docker `dashboard`)

## Quick Start (Recommended)

The project exposes a single CLI entrypoint:

```bash
python -m src.cli <command> [options]
```

Use `--help` at either level to inspect available commands and flags:

```bash
python -m src.cli --help
python -m src.cli train --help
```

Run this end-to-end sequence:

```bash
python -m src.cli prepare
python -m src.cli train --model all
python -m src.cli evaluate --model all
python -m src.cli compare --output-path artifacts/comparison_report.json
streamlit run ui/app.py
```

After it finishes, you should have:

- processed data in `data/processed`
- trained model runs in `artifacts/models`
- evaluation reports, precision-recall curves, and confusion matrices in `artifacts/reports`
- a comparison report in `artifacts/comparison_report.json`

## Command Reference

### `prepare`

```bash
python -m src.cli prepare
```

What it does:

- downloads the default dataset if `--input-csv` is omitted
- performs the temporal train/test split
- applies scaling and SMOTE resampling to the training split
- writes prepared artifacts under `data/` by default

Common options:

```bash
python -m src.cli prepare --input-csv /path/to/creditcard.csv
python -m src.cli prepare --output-dir data --test-size 0.2 --sampling-strategy 0.25
```

### `train`

Train all model families with default hyperparameters:

```bash
python -m src.cli train --model all
```

Train one model with custom hyperparameters:

```bash
python -m src.cli train --model logistic_regression --lr-c 0.5 --lr-max-iter 3000 --lr-solver liblinear
python -m src.cli train --model random_forest --rf-n-estimators 500 --rf-max-depth 12 --rf-min-samples-split 4 --rf-min-samples-leaf 1
```

Available training flags:

- Logistic regression: `--lr-c`, `--lr-max-iter`, `--lr-solver`
- Random forest: `--rf-n-estimators`, `--rf-max-depth`, `--rf-min-samples-split`, `--rf-min-samples-leaf`

Each `train` command now creates a unique run folder under `artifacts/models/`, so repeated training no longer overwrites older model artifacts.
The CLI also prints a training start message and elapsed fit time, and random forest training shows sklearn progress output so long runs do not appear frozen.

### `evaluate`

Evaluate the latest saved run for all model families:

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

Useful evaluation flags:

- `--prepared-dir` to point at a different processed dataset
- `--models-dir` to choose which trained runs directory to search
- `--output-dir` to change where reports are written
- `--threshold` to change the fraud classification cutoff

### `sweep`

Sweep one model family:

```bash
python -m src.cli sweep \
  --prepared-dir data/processed \
  --model logistic_regression \
  --output-dir artifacts/sweeps \
  --validation-size 0.2 \
  --threshold 0.5
```

`sweep` saves ranked search results and a summary of the best configuration in `artifacts/sweeps`, but it does not save a trained `.joblib` model.
The sweep summary and CLI output now include copy-pasteable `train` and `evaluate` commands for the best configuration, so you do not need to convert the JSON fields by hand.

Example flow for logistic regression:

```bash
python -m src.cli sweep --model logistic_regression

# Copy the printed "Train best model" command.
# Then copy the printed "Evaluate latest trained model" command.
```

### `compare`

Print a comparison preview from all evaluated runs:

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

`compare` can also write CSV output if `--output-path` ends with `.csv`.

### `dashboard`

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
|   |       |-- logistic_regression_confusion_matrix.png
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

If you only run one command sequence, use the Quick Start block above.
For targeted experiments, a common loop is:

```bash
python -m src.cli train --model logistic_regression --lr-c 0.5 --lr-max-iter 3000
python -m src.cli evaluate --model logistic_regression
python -m src.cli compare --model logistic_regression --top-n 5
streamlit run ui/app.py
```

This keeps experiments reproducible and scriptable while making model iteration easy to inspect.