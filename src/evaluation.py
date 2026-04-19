from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


@dataclass
class EvaluationResult:
    threshold: float
    precision: float
    recall: float
    f1_score: float
    average_precision: float
    positive_predictions: int
    fraud_cases: int


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_metrics(
    y_true,
    scores,
    threshold: float = 0.5,
) -> EvaluationResult:
    predictions = (scores >= threshold).astype(int)
    return EvaluationResult(
        threshold=threshold,
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1_score=float(f1_score(y_true, predictions, zero_division=0)),
        average_precision=float(average_precision_score(y_true, scores)),
        positive_predictions=int(predictions.sum()),
        fraud_cases=int(y_true.sum()),
    )


def save_precision_recall_curve(y_true, scores, output_dir: str | Path, stem: str) -> dict[str, str]:
    import matplotlib.pyplot as plt

    output_path = ensure_directory(Path(output_dir))
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)

    curve_frame = pd.DataFrame(
        {
            "precision": precisions[:-1],
            "recall": recalls[:-1],
            "threshold": thresholds,
        }
    )
    curve_csv_path = output_path / f"{stem}_precision_recall_curve.csv"
    curve_frame.to_csv(curve_csv_path, index=False)

    figure_path = output_path / f"{stem}_precision_recall_curve.png"
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve: {stem}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()

    return {
        "curve_csv": str(curve_csv_path),
        "curve_png": str(figure_path),
    }


def save_evaluation_report(
    result: EvaluationResult,
    output_dir: str | Path,
    stem: str,
    extra_payload: dict | None = None,
) -> Path:
    output_path = ensure_directory(Path(output_dir))
    payload = asdict(result)
    if extra_payload:
        payload.update(extra_payload)

    report_path = output_path / f"{stem}_evaluation.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path
