from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

# Default column mapping for existing cleaned datasets
DEFAULT_COLUMN_MAP: Dict[str, Dict[str, str]] = {
    "livebench": {"model_col": "Model", "score_col": "Coding-Average"},
    "berkeley": {"model_col": "Model", "score_col": "Overall_Acc_Avg_2_3"},
    "aiden": {"model_col": "Model", "score_col": "Accuracy"},
    "mcpmark": {"model_col": "Model", "score_col": "Pass@1"},
}


def normalize_model_name(name: str) -> str:
    """Normalize model names to improve merge reliability."""
    cleaned = name.strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


@dataclass
class BenchmarkSpec:
    path: Path
    score_name: str
    model_column: str = "Model"
    score_column: Optional[str] = None


def load_benchmark_file(spec: BenchmarkSpec) -> pd.DataFrame:
    """Load a benchmark CSV and return standardized columns."""
    df = pd.read_csv(spec.path)
    score_col = spec.score_column or spec.score_name
    if spec.model_column not in df.columns or score_col not in df.columns:
        raise ValueError(f"Required columns missing in {spec.path}: {spec.model_column}, {score_col}")

    sliced = df[[spec.model_column, score_col]].dropna()
    sliced["model_id"] = sliced[spec.model_column].map(normalize_model_name)
    sliced = sliced.rename(columns={score_col: spec.score_name})
    return sliced[["model_id", spec.score_name]]


def merge_benchmarks(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Outer-merge benchmark frames on model_id."""
    merged: Optional[pd.DataFrame] = None
    for frame in frames:
        if "model_id" not in frame.columns:
            raise ValueError("Each frame must include a model_id column")
        merged = frame if merged is None else merged.merge(frame, on="model_id", how="outer")
    return merged if merged is not None else pd.DataFrame(columns=["model_id"])


def load_cleaned_defaults(root: Path) -> pd.DataFrame:
    """Load default cleaned benchmark CSVs from the repo."""
    frames: List[pd.DataFrame] = []
    for name, cols in DEFAULT_COLUMN_MAP.items():
        path = root / f"{name}.csv"
        if not path.exists():
            continue
        spec = BenchmarkSpec(path=path, score_name=name, model_column=cols["model_col"], score_column=cols["score_col"])
        frames.append(load_benchmark_file(spec))
    if not frames:
        raise FileNotFoundError(f"No cleaned benchmark CSVs found in {root}")
    return merge_benchmarks(frames)


def cleaned_to_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map cleaned benchmark columns into core feature names."""
    out = pd.DataFrame()
    out["model_id"] = df["model_id"]

    if "aiden" in df.columns:
        out["reasoning"] = df["aiden"]
        out["nl_instruction"] = df["aiden"]
    else:
        out["reasoning"] = pd.NA
        out["nl_instruction"] = pd.NA

    tool_cols = [c for c in ["berkeley", "mcpmark", "tool_use"] if c in df.columns]
    if tool_cols:
        out["tool_use"] = df[tool_cols].mean(axis=1)
    else:
        out["tool_use"] = pd.NA

    if "livebench" in df.columns:
        out["coding"] = df["livebench"]
    else:
        out["coding"] = pd.NA
    return out


def add_task_performance(df: pd.DataFrame, task_scores: Mapping[str, float]) -> pd.DataFrame:
    """Attach task performance by model_id."""
    task_df = pd.DataFrame([{"model_id": normalize_model_name(model), "task_performance": score} for model, score in task_scores.items()])
    merged = df.merge(task_df, on="model_id", how="left")
    return merged


def handle_missing(df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    """Handle missing values using a simple strategy."""
    if strategy == "drop":
        return df.dropna()
    if strategy == "mean":
        filled = df.copy()
        for col in filled.columns:
            if col == "model_id":
                continue
            filled[col] = filled[col].fillna(filled[col].mean())
        return filled
    raise ValueError(f"Unknown missing strategy: {strategy}")


def validate_non_empty(df: pd.DataFrame) -> None:
    """Raise if DataFrame is empty."""
    if df.empty:
        raise ValueError("Merged benchmark data is empty.")


def prepare_feature_frame(
    benchmarks: pd.DataFrame,
    required_columns: Iterable[str],
    missing_strategy: str = "drop",
) -> pd.DataFrame:
    """Ensure required columns exist and handle missing values."""
    benchmarks = benchmarks.copy()
    for col in required_columns:
        if col in benchmarks.columns:
            continue
        for suffix in ("_x", "_y"):
            candidate = f"{col}{suffix}"
            if candidate in benchmarks.columns:
                benchmarks[col] = benchmarks[candidate]
                break
    missing = [col for col in required_columns if col not in benchmarks.columns]
    if missing:
        raise ValueError(f"Missing required benchmark columns: {missing}")
    processed = handle_missing(benchmarks, strategy=missing_strategy)
    validate_non_empty(processed)
    return processed
