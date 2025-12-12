from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

from .data_loader import normalize_model_name

DEFAULT_TASK_MAP: Dict[str, Sequence[str]] = {
    "reasoning": ["gsm8k"],
    "nl_instruction": ["ifeval"],
    "tool_use": ["toolbench_function_calling"],
    "coding": ["humaneval", "mbpp"],
}


class BenchmarkClient(Protocol):
    """Protocol for benchmark execution."""

    def evaluate(self, model: str, tasks: Sequence[str]) -> Mapping[str, float]: ...


class SafetyClient(Protocol):
    """Protocol for safety benchmark execution."""

    def evaluate_safety(self, model: str, suite: Sequence[str]) -> Mapping[str, float]: ...


@dataclass
class OpenRouterConfig:
    lm_eval_executable: str = "lm_eval"
    api_key_env: str = "OPENROUTER_API_KEY"
    extra_args: Sequence[str] = field(default_factory=tuple)


class OpenRouterLMHarnessClient:
    """Thin wrapper to call LM-Eval-Harness with the OpenRouter backend."""

    def __init__(self, config: Optional[OpenRouterConfig] = None):
        self.config = config or OpenRouterConfig()

    def evaluate(self, model: str, tasks: Sequence[str]) -> Mapping[str, float]:
        env = os.environ.copy()
        if self.config.api_key_env not in env:
            raise EnvironmentError(f"{self.config.api_key_env} must be set to call OpenRouter.")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            output_path = output_dir / "results.json"
            cmd = [
                self.config.lm_eval_executable,
                "--model",
                "openrouter",
                "--model_args",
                f"model={model}",
                "--tasks",
                ",".join(tasks),
                "--output_path",
                str(output_dir),
            ]
            if self.config.extra_args:
                cmd.extend(self.config.extra_args)

            # Execute LM-Eval; callers may mock in tests to avoid the external dependency.
            subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)

            if not output_path.exists():
                raise FileNotFoundError(f"LM-Eval output not found at {output_path}")
            return parse_lm_eval_results(output_path)


def parse_lm_eval_results(path: Path) -> Mapping[str, float]:
    """Parse LM-Eval-Harness results.json into a task->score mapping."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", {})
    scores: Dict[str, float] = {}
    for task, payload in results.items():
        # Prefer exact-match accuracy; fall back to the first numeric metric present.
        for key in ("acc,exact", "acc", "f1", "score"):
            if key in payload:
                scores[task] = float(payload[key])
                break
    return scores


def aggregate_features(scores: Mapping[str, float], task_map: Mapping[str, Sequence[str]]) -> Dict[str, Optional[float]]:
    """Aggregate raw task scores into feature-level averages."""
    feature_scores: Dict[str, Optional[float]] = {}
    for feature, tasks in task_map.items():
        values = [scores[t] for t in tasks if t in scores]
        feature_scores[feature] = float(np.mean(values)) if values else None
    return feature_scores


def standardize_scores(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    """Z-score normalize feature columns."""
    standardized = df.copy()
    for col in feature_cols:
        mean = standardized[col].mean()
        std = standardized[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            standardized[col] = 0.0
        else:
            standardized[col] = (standardized[col] - mean) / std
    return standardized


def run_benchmarks(
    models: Sequence[str],
    client: BenchmarkClient,
    task_map: Mapping[str, Sequence[str]] = DEFAULT_TASK_MAP,
    safety_client: Optional[SafetyClient] = None,
    safety_suite: Optional[Sequence[str]] = None,
    standardize: bool = True,
) -> pd.DataFrame:
    """Run benchmarks via the provided client and return aggregated feature scores."""
    rows: list[dict[str, object]] = []
    all_tasks = {task for tasks in task_map.values() for task in tasks}
    for model in models:
        raw_scores = client.evaluate(model, tasks=tuple(all_tasks))
        base_features = aggregate_features(raw_scores, task_map)
        features: dict[str, object] = {k: v for k, v in base_features.items()}
        features["model_id"] = normalize_model_name(model)

        if safety_client and safety_suite:
            safety_scores = safety_client.evaluate_safety(model, safety_suite)
            for key, value in safety_scores.items():
                features[f"safety_{key}"] = value

        rows.append(features)

    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c != "model_id"]
    if standardize and feature_cols:
        df = standardize_scores(df, feature_cols=feature_cols)
    return df


def cache_results(df: pd.DataFrame, path: Path) -> None:
    """Persist benchmark results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_cached_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cached benchmark results not found at {path}")
    return pd.read_csv(path)
