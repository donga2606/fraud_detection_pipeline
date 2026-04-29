from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .data_processing import ensure_directory

STAGE_ROOT_NAMES = {"models", "reports", "sweeps"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def create_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def infer_artifact_root(stage_root: str | Path) -> Path:
    stage_path = Path(stage_root)
    if stage_path.name in STAGE_ROOT_NAMES:
        return stage_path.parent
    return stage_path


def resolve_run_stage_dir(stage_root: str | Path, run_id: str) -> Path:
    root_path = ensure_directory(Path(stage_root))
    return ensure_directory(root_path / run_id)


def get_run_manifest_path(artifact_root: str | Path, run_id: str) -> Path:
    return ensure_directory(Path(artifact_root) / "runs" / run_id) / "run_manifest.json"


def extract_run_context_from_artifact_path(
    artifact_path: str | Path,
    stage_root_name: str,
) -> dict[str, str] | None:
    path = Path(artifact_path).resolve()
    if path.parent.name != stage_root_name:
        if path.parent.parent.name != stage_root_name:
            return None
    if path.parent.parent.name != stage_root_name:
        return None

    run_id = path.parent.name
    artifact_root = path.parent.parent.parent
    return {
        "run_id": run_id,
        "artifact_root": str(artifact_root),
    }


def read_json_if_exists(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


def merge_manifest_payload(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_manifest_payload(merged[key], value)
            continue
        merged[key] = value
    return merged


def upsert_run_manifest(
    artifact_root: str | Path,
    run_id: str,
    updates: dict[str, Any],
) -> Path:
    manifest_path = get_run_manifest_path(artifact_root, run_id)
    current_payload = read_json_if_exists(manifest_path)
    merged_payload = merge_manifest_payload(current_payload, updates)
    manifest_path.write_text(json.dumps(merged_payload, indent=2), encoding="utf-8")
    return manifest_path


def flatten_mapping(mapping: dict[str, Any], prefix: str) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in mapping.items():
        column_name = f"{prefix}_{key}"
        if isinstance(value, (str, int, float, bool)) or value is None:
            flattened[column_name] = value
        else:
            flattened[column_name] = json.dumps(value, sort_keys=True)
    return flattened


def build_run_history_row(manifest: dict[str, Any]) -> dict[str, Any] | None:
    evaluation = manifest.get("evaluation")
    if not evaluation:
        return None

    training = manifest.get("training", {})
    created_at = training.get("completed_at") or manifest.get("created_at")
    row = {
        "run_id": manifest.get("run_id"),
        "created_at": created_at,
        "model_name": manifest.get("model_name"),
        "prepared_dir": manifest.get("prepared_dir"),
        "model_path": evaluation.get("model_path") or training.get("model_path"),
        "report_json": evaluation.get("report_json"),
        "curve_csv": evaluation.get("curve_csv"),
        "curve_png": evaluation.get("curve_png"),
        "threshold": evaluation.get("threshold"),
        "precision": evaluation.get("precision"),
        "recall": evaluation.get("recall"),
        "f1_score": evaluation.get("f1_score"),
        "average_precision": evaluation.get("average_precision"),
        "positive_predictions": evaluation.get("positive_predictions"),
        "fraud_cases": evaluation.get("fraud_cases"),
        "training_random_seed": training.get("random_seed"),
        "evaluation_completed_at": evaluation.get("completed_at"),
        "training_summary_path": training.get("summary_path"),
    }
    row.update(flatten_mapping(training.get("requested_parameters", {}), "requested"))
    row.update(flatten_mapping(training.get("resolved_parameters", {}), "resolved"))
    return row


def list_run_manifest_paths(artifact_root: str | Path) -> list[Path]:
    runs_root = Path(artifact_root) / "runs"
    if not runs_root.exists():
        return []
    return sorted(runs_root.glob("*/run_manifest.json"))


def rebuild_run_history_index(artifact_root: str | Path) -> tuple[pd.DataFrame, Path]:
    root_path = Path(artifact_root)
    rows = []
    for manifest_path in list_run_manifest_paths(root_path):
        manifest = read_json_if_exists(manifest_path)
        row = build_run_history_row(manifest)
        if row is not None:
            rows.append(row)

    history_path = root_path / "run_history.csv"
    if rows:
        history_frame = pd.DataFrame(rows).sort_values(
            by=["average_precision", "f1_score", "recall", "created_at"],
            ascending=[False, False, False, False],
        )
    else:
        history_frame = pd.DataFrame()

    ensure_directory(history_path.parent)
    history_frame.to_csv(history_path, index=False)
    return history_frame, history_path
