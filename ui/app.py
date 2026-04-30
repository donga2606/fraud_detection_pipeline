from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_history import read_json_if_exists, rebuild_run_history_index

DEFAULT_ARTIFACTS_ROOT = Path("artifacts")
METRIC_COLUMNS = ["average_precision", "f1_score", "precision", "recall", "threshold"]
SUMMARY_COLUMNS = [
    "run_id",
    "model_name",
    "average_precision",
    "f1_score",
    "precision",
    "recall",
    "threshold",
    "created_at",
]


def load_run_history(artifacts_root: Path) -> tuple[pd.DataFrame, Path]:
    history_frame, history_path = rebuild_run_history_index(artifacts_root)
    if history_frame.empty:
        return history_frame, history_path
    return history_frame.sort_values(by=["created_at"], ascending=False), history_path


def collect_parameter_columns(frame: pd.DataFrame) -> list[str]:
    prefixes = ("requested_", "resolved_")
    return sorted(column for column in frame.columns if column.startswith(prefixes))


def load_manifest(artifacts_root: Path, run_id: str) -> dict[str, Any]:
    return read_json_if_exists(artifacts_root / "runs" / run_id / "run_manifest.json")


def display_overview(history_frame: pd.DataFrame) -> None:
    st.subheader("Run Summary")
    total_runs = len(history_frame)
    model_count = int(history_frame["model_name"].nunique()) if "model_name" in history_frame else 0
    best_ap = float(history_frame["average_precision"].max()) if "average_precision" in history_frame else 0.0
    best_f1 = float(history_frame["f1_score"].max()) if "f1_score" in history_frame else 0.0

    metric_columns = st.columns(4)
    metric_columns[0].metric("Evaluated Runs", total_runs)
    metric_columns[1].metric("Model Families", model_count)
    metric_columns[2].metric("Best Average Precision", f"{best_ap:.4f}")
    metric_columns[3].metric("Best F1 Score", f"{best_f1:.4f}")


def display_runs_table(history_frame: pd.DataFrame, parameter_columns: list[str]) -> None:
    st.subheader("Saved Runs")
    visible_columns = [column for column in SUMMARY_COLUMNS if column in history_frame.columns]
    extra_columns = [column for column in parameter_columns if column in history_frame.columns]
    st.dataframe(
        history_frame[visible_columns + extra_columns],
        use_container_width=True,
        hide_index=True,
    )


def parse_curve_data(curve_csv_path: str | None) -> pd.DataFrame:
    if not curve_csv_path:
        return pd.DataFrame()
    path = Path(curve_csv_path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def display_run_details(artifacts_root: Path, history_frame: pd.DataFrame, parameter_columns: list[str]) -> None:
    st.subheader("Run Details")
    run_options = history_frame["run_id"].tolist()
    selected_run_id = st.selectbox("Select a run", run_options, index=0)
    if not selected_run_id:
        return

    selected_row = history_frame.loc[history_frame["run_id"] == selected_run_id].iloc[0]
    manifest = load_manifest(artifacts_root, selected_run_id)

    metric_data = {
        column: selected_row[column]
        for column in METRIC_COLUMNS
        if column in selected_row and pd.notna(selected_row[column])
    }
    if metric_data:
        metric_cols = st.columns(len(metric_data))
        for idx, (name, value) in enumerate(metric_data.items()):
            metric_cols[idx].metric(name.replace("_", " ").title(), f"{float(value):.4f}")

    left_column, right_column = st.columns([1, 1])
    with left_column:
        st.markdown("**Run Metadata**")
        metadata_keys = ["run_id", "model_name", "created_at", "prepared_dir", "model_path", "report_json"]
        metadata = {key: selected_row.get(key) for key in metadata_keys if key in selected_row}
        st.json(metadata)

        parameter_values = {
            column: selected_row[column]
            for column in parameter_columns
            if column in selected_row and pd.notna(selected_row[column])
        }
        if parameter_values:
            st.markdown("**Hyperparameters**")
            st.json(parameter_values)

    with right_column:
        curve_frame = parse_curve_data(selected_row.get("curve_csv"))
        if not curve_frame.empty and {"recall", "precision"}.issubset(curve_frame.columns):
            st.markdown("**Precision-Recall Curve**")
            chart_frame = curve_frame[["recall", "precision"]].rename(
                columns={"recall": "Recall", "precision": "Precision"}
            )
            st.line_chart(chart_frame.set_index("Recall"))
        curve_png = selected_row.get("curve_png")
        if curve_png and Path(curve_png).exists():
            st.image(curve_png, caption=f"Curve image for {selected_run_id}", use_container_width=True)

    with st.expander("Run Manifest"):
        st.json(manifest)


def build_comparison_table(history_frame: pd.DataFrame, selected_runs: list[str], parameter_columns: list[str]) -> pd.DataFrame:
    comparison_frame = history_frame[history_frame["run_id"].isin(selected_runs)].copy()
    visible_columns = [column for column in SUMMARY_COLUMNS if column in comparison_frame.columns]
    visible_columns.extend(column for column in parameter_columns if column in comparison_frame.columns)
    return comparison_frame[visible_columns]


def display_metric_chart(comparison_frame: pd.DataFrame) -> None:
    chart_source = comparison_frame.set_index("run_id")[
        [column for column in ["average_precision", "f1_score", "precision", "recall"] if column in comparison_frame.columns]
    ]
    if not chart_source.empty:
        st.bar_chart(chart_source)


def display_comparison_section(history_frame: pd.DataFrame, parameter_columns: list[str]) -> None:
    st.subheader("Compare Runs")
    default_selection = history_frame["run_id"].head(2).tolist()
    selected_runs = st.multiselect(
        "Select runs to compare",
        options=history_frame["run_id"].tolist(),
        default=default_selection,
    )
    if not selected_runs:
        st.info("Choose at least one run to compare.")
        return

    comparison_frame = build_comparison_table(history_frame, selected_runs, parameter_columns)
    st.dataframe(comparison_frame, use_container_width=True, hide_index=True)
    display_metric_chart(comparison_frame)


def main() -> None:
    st.set_page_config(page_title="Fraud Model Dashboard", layout="wide")
    st.title("Fraud Detection Run Dashboard")
    st.caption("Run training and evaluation from the CLI, then inspect saved runs here.")

    artifacts_root_input = st.sidebar.text_input("Artifacts root", str(DEFAULT_ARTIFACTS_ROOT))
    artifacts_root = Path(artifacts_root_input).expanduser()
    if not artifacts_root.exists():
        st.error(f"Artifacts root not found: {artifacts_root}")
        return

    history_frame, history_path = load_run_history(artifacts_root)
    st.sidebar.caption(f"Run history index: {history_path}")
    if history_frame.empty:
        st.info("No evaluated runs found yet. Run `train` and `evaluate` first, then reload this dashboard.")
        return

    model_options = ["all", *sorted(history_frame["model_name"].dropna().unique().tolist())]
    selected_model = st.sidebar.selectbox("Model filter", model_options, index=0)
    sort_by = st.sidebar.selectbox(
        "Sort by",
        [column for column in ["average_precision", "f1_score", "precision", "recall", "created_at"] if column in history_frame.columns],
        index=0,
    )

    filtered_frame = history_frame.copy()
    if selected_model != "all":
        filtered_frame = filtered_frame[filtered_frame["model_name"] == selected_model]
    filtered_frame = filtered_frame.sort_values(by=[sort_by], ascending=False, na_position="last")

    parameter_columns = collect_parameter_columns(filtered_frame)
    display_overview(filtered_frame)

    tabs = st.tabs(["Runs", "Details", "Compare"])
    with tabs[0]:
        display_runs_table(filtered_frame, parameter_columns)
    with tabs[1]:
        display_run_details(artifacts_root, filtered_frame, parameter_columns)
    with tabs[2]:
        display_comparison_section(filtered_frame, parameter_columns)

    st.sidebar.markdown("### CLI Flow")
    st.sidebar.code(
        "python -m src.cli train ...\n"
        "python -m src.cli evaluate ...\n"
        "streamlit run ui/app.py",
        language="bash",
    )


if __name__ == "__main__":
    main()
