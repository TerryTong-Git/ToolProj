from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import List, Sequence

import pandas as pd

from . import benchmarker, data_loader, mixed_model, visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mixed-model analysis on LLM benchmark performance.")
    parser.add_argument("--models", type=str, default="", help="Comma-separated list of model ids to benchmark via OpenRouter.")
    parser.add_argument("--generate-benchmarks", action="store_true", help="Generate benchmarks via OpenRouter/LM-Eval.")
    parser.add_argument("--cached-benchmarks", type=Path, help="Path to cached benchmark CSV (if not generating).")
    parser.add_argument("--output-dir", type=Path, default=Path("results/mixed_model"), help="Directory for outputs.")
    parser.add_argument("--missing-strategy", choices=["drop", "mean"], default="drop")
    parser.add_argument("--run-safety", action="store_true", help="Also run safety benchmarks (requires safety client).")
    parser.add_argument("--safety-suite", type=str, default="", help="Comma-separated safety tasks when --run-safety is set.")
    parser.add_argument("--task-performance-file", type=Path, help="CSV with columns: model_id, task_performance.")
    parser.add_argument(
        "--cleaned-data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "exps_correlation" / "cleaned_data",
        help="Root folder containing cleaned benchmark CSVs.",
    )
    parser.add_argument("--feature-pairs", type=str, default="", help="Optional comma list of feature pairs featureA:featureB")
    parser.add_argument("--simulate", action="store_true", help="Skip external calls; generate synthetic benchmark data.")
    return parser.parse_args()


def _parse_models(raw: str) -> List[str]:
    return [m.strip() for m in raw.split(",") if m.strip()]


def _synthetic_benchmark_df(models: Sequence[str]) -> pd.DataFrame:
    rows = []
    for idx, model in enumerate(models):
        rows.append(
            {
                "model_id": data_loader.normalize_model_name(model),
                "reasoning": 0.1 * (idx + 1),
                "nl_instruction": 0.2 * (idx + 1),
                "tool_use": 0.15 * (idx + 1),
                "coding": 0.12 * (idx + 1),
            }
        )
    return pd.DataFrame(rows)


def _load_task_performance(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "model_id" not in df.columns or "task_performance" not in df.columns:
        raise ValueError("task_performance_file must include columns: model_id, task_performance")
    df["model_id"] = df["model_id"].map(data_loader.normalize_model_name)
    return df[["model_id", "task_performance"]]


def _prepare_feature_pairs(features: Sequence[str], override: str) -> List[tuple[str, str]]:
    if override:
        pairs = []
        for item in override.split(","):
            if ":" in item:
                a, b = item.split(":")
                pairs.append((a.strip(), b.strip()))
        return pairs
    return list(combinations(features, 2))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    models = _parse_models(args.models)
    benchmark_df: pd.DataFrame

    if args.generate_benchmarks and models:
        if args.simulate:
            benchmark_df = _synthetic_benchmark_df(models)
        else:
            client = benchmarker.OpenRouterLMHarnessClient()
            safety_client = None  # Placeholder; real implementation would wrap UK AI Safety Evals.
            safety_suite = tuple(args.safety_suite.split(",")) if args.run_safety and args.safety_suite else None
            benchmark_df = benchmarker.run_benchmarks(
                models=models,
                client=client,
                task_map=benchmarker.DEFAULT_TASK_MAP,
                safety_client=safety_client,
                safety_suite=safety_suite,
                standardize=True,
            )
        benchmarker.cache_results(benchmark_df, args.output_dir / "benchmark_scores.csv")
    elif args.cached_benchmarks:
        benchmark_df = benchmarker.load_cached_results(args.cached_benchmarks)
    else:
        # Fall back to cleaned defaults
        benchmark_df = data_loader.load_cleaned_defaults(args.cleaned_data_root)

    cleaned_defaults = data_loader.load_cleaned_defaults(args.cleaned_data_root)
    feature_names = ["reasoning", "nl_instruction", "tool_use", "coding"]

    feature_frames = [data_loader.cleaned_to_feature_frame(cleaned_defaults)]

    if not set(feature_names).issubset(benchmark_df.columns):
        # Attempt to coerce to feature frame if in cleaned schema
        maybe_features = data_loader.cleaned_to_feature_frame(benchmark_df) if "model_id" in benchmark_df.columns else benchmark_df
        feature_frames.append(maybe_features)
    else:
        feature_frames.append(benchmark_df[["model_id", *feature_names]])

    merged = data_loader.merge_benchmarks(feature_frames)

    if args.task_performance_file:
        task_df = _load_task_performance(args.task_performance_file)
        merged = merged.merge(task_df, on="model_id", how="left")

    required_cols = [*feature_names, "task_performance"]
    processed = data_loader.prepare_feature_frame(merged, required_columns=required_cols, missing_strategy=args.missing_strategy)

    artifacts = mixed_model.fit_mixed_model(
        processed,
        outcome="task_performance",
        features=["reasoning", "nl_instruction", "tool_use", "coding"],
        group_col="model_id",
    )

    diagnostics = mixed_model.model_diagnostics(artifacts)
    (args.output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    feature_pairs = _prepare_feature_pairs(["reasoning", "nl_instruction", "tool_use", "coding"], args.feature_pairs)
    for feature_x, feature_y in feature_pairs:
        grid = mixed_model.prediction_grid(artifacts, feature_x=feature_x, feature_y=feature_y)
        html_path = args.output_dir / f"surface_{feature_x}_{feature_y}.html"
        png_path = args.output_dir / f"surface_{feature_x}_{feature_y}.png"
        visualization.surface_plot(
            grid=grid,
            feature_x=feature_x,
            feature_y=feature_y,
            output_html=html_path,
            output_png=png_path,
            title=f"{feature_x} vs {feature_y}",
        )


if __name__ == "__main__":
    main()
