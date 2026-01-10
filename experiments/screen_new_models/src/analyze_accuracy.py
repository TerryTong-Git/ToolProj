#!/usr/bin/env python3
"""
Analyze Code vs NL accuracy from screening results.

This script:
1. Parses all res.jsonl files from results directories
2. Computes per-model accuracy: sim_correct (simulation) vs nl_correct rate
3. Creates summary table showing Sim vs NL accuracy per model
4. Runs McNemar's test for paired statistical comparison
5. Generates visualization (bar chart comparing sim vs nl accuracy)
6. Reports code execution errors

NOTE: We use sim_correct (simulation accuracy) instead of code_correct (execution)
because the executor has known issues with EOFError/timeout that affect code_correct.
The simulation arm asks the model to trace through code without actual execution.

Usage:
    uv run python experiments/screen_new_models/src/analyze_accuracy.py \
        --results-dir src/exps_performance/results_screening/results \
        --output-dir experiments/screen_new_models/results/analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


@dataclass
class AccuracyResult:
    """Result of accuracy analysis for a single model."""

    model: str
    n_samples: int
    nl_accuracy: float
    code_accuracy: float  # code_correct (execution)
    sim_accuracy: float   # sim_correct (simulation)
    sim_minus_nl: float   # Primary comparison: sim - nl
    code_minus_nl: float  # Secondary: code - nl (may have execution issues)
    code_exec_success_rate: float  # Rate of successful code execution
    mcnemar_statistic: float | None = None
    mcnemar_pvalue: float | None = None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze Code vs NL accuracy")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing model result directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/screen_new_models/results/analysis"),
        help="Directory for output files",
    )
    parser.add_argument(
        "--exclude-kinds",
        nargs="+",
        default=["gsm8k"],
        help="Problem kinds to exclude from analysis",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum samples required per model",
    )
    parser.add_argument(
        "--compare",
        choices=["sim", "code"],
        default="sim",
        help="Compare NL with: 'sim' (simulation, recommended) or 'code' (execution)",
    )
    return parser.parse_args()


def load_all_results(results_dir: Path, exclude_kinds: list[str] | None = None) -> pd.DataFrame:
    """
    Load all res.jsonl files from results directory.

    Args:
        results_dir: Path to directory containing model result directories
        exclude_kinds: Problem kinds to exclude (default: gsm8k)

    Returns:
        DataFrame with all results combined
    """
    if exclude_kinds is None:
        exclude_kinds = ["gsm8k"]

    all_results = []
    jsonl_files = list(results_dir.rglob("res.jsonl"))

    if not jsonl_files:
        print(f"WARNING: No res.jsonl files found in {results_dir}")
        return pd.DataFrame()

    for jsonl_path in jsonl_files:
        with open(jsonl_path) as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    if result.get("kind") not in exclude_kinds:
                        all_results.append(result)
                except json.JSONDecodeError:
                    continue

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    print(f"Loaded {len(df)} results from {len(jsonl_files)} files")
    return df


def compute_model_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute accuracy metrics grouped by model.

    Returns DataFrame with columns:
    - model: model name
    - n_samples: number of samples
    - nl_accuracy: mean nl_correct
    - code_accuracy: mean code_correct (execution)
    - sim_accuracy: mean sim_correct (simulation)
    - sim_minus_nl: difference (sim - nl) - PRIMARY metric
    - code_minus_nl: difference (code - nl)
    - code_exec_success_rate: rate of successful code execution (no parse errors)
    """
    if df.empty:
        return pd.DataFrame()

    # Ensure boolean columns
    for col in ["nl_correct", "code_correct", "sim_correct", "code_parse_err"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # Group by model
    agg_dict = {
        "nl_correct": [("n_samples", "count"), ("nl_accuracy", "mean")],
        "code_correct": [("code_accuracy", "mean")],
        "sim_correct": [("sim_accuracy", "mean")],
    }

    # Add code execution success rate if parse_err column exists
    if "code_parse_err" in df.columns:
        agg_dict["code_parse_err"] = [("code_parse_err_rate", "mean")]

    summary = df.groupby("model").agg(
        n_samples=("nl_correct", "count"),
        nl_accuracy=("nl_correct", "mean"),
        code_accuracy=("code_correct", "mean"),
        sim_accuracy=("sim_correct", "mean"),
    ).reset_index()

    # Compute code execution success rate (inverse of parse error rate)
    if "code_parse_err" in df.columns:
        parse_err_rate = df.groupby("model")["code_parse_err"].mean()
        summary["code_exec_success_rate"] = summary["model"].map(
            lambda m: 1.0 - parse_err_rate.get(m, 0.0)
        )
    else:
        summary["code_exec_success_rate"] = 1.0

    summary["sim_minus_nl"] = summary["sim_accuracy"] - summary["nl_accuracy"]
    summary["code_minus_nl"] = summary["code_accuracy"] - summary["nl_accuracy"]

    # Sort by sim accuracy descending (primary metric)
    summary = summary.sort_values("sim_accuracy", ascending=False)

    return summary


def run_mcnemar_test(
    df: pd.DataFrame, model: str, compare_col: str = "sim_correct"
) -> tuple[float | None, float | None]:
    """
    Run McNemar's test for a single model.

    Tests whether there's a significant difference between nl_correct and compare_col.

    Args:
        df: DataFrame with results
        model: Model name to filter
        compare_col: Column to compare with nl_correct ('sim_correct' or 'code_correct')

    Returns:
        (statistic, pvalue) or (None, None) if test cannot be run
    """
    model_df = df[df["model"] == model].copy()

    if len(model_df) < 10:
        return None, None

    # Build contingency table
    # [[both_correct, nl_only], [compare_only, both_incorrect]]
    both_correct = ((model_df["nl_correct"]) & (model_df[compare_col])).sum()
    nl_only = ((model_df["nl_correct"]) & (~model_df[compare_col])).sum()
    compare_only = ((~model_df["nl_correct"]) & (model_df[compare_col])).sum()
    both_incorrect = ((~model_df["nl_correct"]) & (~model_df[compare_col])).sum()

    contingency = [[both_correct, nl_only], [compare_only, both_incorrect]]

    # Need discordant pairs for McNemar's test
    if nl_only + compare_only < 1:
        return None, None

    try:
        result = mcnemar(contingency, exact=True)
        return result.statistic, result.pvalue
    except Exception:
        return None, None


def analyze_all_models(
    df: pd.DataFrame, min_samples: int = 10, compare: str = "sim"
) -> list[AccuracyResult]:
    """
    Compute full analysis for all models.

    Args:
        df: DataFrame with results
        min_samples: Minimum samples required per model
        compare: What to compare with NL - 'sim' or 'code'

    Returns list of AccuracyResult objects.
    """
    if df.empty:
        return []

    summary = compute_model_accuracy(df)
    results = []

    compare_col = "sim_correct" if compare == "sim" else "code_correct"

    for _, row in summary.iterrows():
        model = row["model"]

        # Skip models with too few samples
        if row["n_samples"] < min_samples:
            continue

        # Run McNemar's test (comparing NL with sim or code)
        stat, pvalue = run_mcnemar_test(df, model, compare_col)

        results.append(
            AccuracyResult(
                model=model,
                n_samples=int(row["n_samples"]),
                nl_accuracy=row["nl_accuracy"],
                code_accuracy=row["code_accuracy"],
                sim_accuracy=row["sim_accuracy"],
                sim_minus_nl=row["sim_minus_nl"],
                code_minus_nl=row["code_minus_nl"],
                code_exec_success_rate=row.get("code_exec_success_rate", 1.0),
                mcnemar_statistic=stat,
                mcnemar_pvalue=pvalue,
            )
        )

    return results


def format_model_name(model: str) -> str:
    """Format model name for display (remove provider prefix)."""
    if "/" in model:
        return model.split("/")[-1]
    return model


def create_summary_table(results: list[AccuracyResult], compare: str = "sim") -> pd.DataFrame:
    """Create formatted summary table from results."""
    rows = []
    for r in results:
        pval_str = f"{r.mcnemar_pvalue:.4f}" if r.mcnemar_pvalue is not None else "N/A"
        sig = ""
        if r.mcnemar_pvalue is not None:
            if r.mcnemar_pvalue < 0.001:
                sig = "***"
            elif r.mcnemar_pvalue < 0.01:
                sig = "**"
            elif r.mcnemar_pvalue < 0.05:
                sig = "*"

        # Primary comparison is Sim vs NL
        diff = r.sim_minus_nl if compare == "sim" else r.code_minus_nl
        diff_label = "Sim - NL" if compare == "sim" else "Code - NL"

        rows.append(
            {
                "Model": format_model_name(r.model),
                "N": r.n_samples,
                "NL Acc": f"{r.nl_accuracy:.1%}",
                "Sim Acc": f"{r.sim_accuracy:.1%}",
                "Code Acc": f"{r.code_accuracy:.1%}",
                "Exec OK": f"{r.code_exec_success_rate:.0%}",
                diff_label: f"{diff:+.1%}",
                "p-value": pval_str,
                "Sig": sig,
            }
        )

    return pd.DataFrame(rows)


def plot_accuracy_comparison(
    results: list[AccuracyResult], output_path: Path, compare: str = "sim"
) -> None:
    """Create bar chart comparing sim/code vs NL accuracy per model."""
    if not results:
        print("No results to plot")
        return

    # Prepare data - compare NL vs Sim (primary) or Code (secondary)
    models = [format_model_name(r.model) for r in results]
    nl_acc = [r.nl_accuracy for r in results]
    compare_acc = [r.sim_accuracy if compare == "sim" else r.code_accuracy for r in results]
    compare_label = "Sim Accuracy" if compare == "sim" else "Code Accuracy"

    # Sort by compare accuracy
    sorted_indices = sorted(range(len(compare_acc)), key=lambda i: compare_acc[i], reverse=True)
    models = [models[i] for i in sorted_indices]
    nl_acc = [nl_acc[i] for i in sorted_indices]
    compare_acc = [compare_acc[i] for i in sorted_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(models))
    width = 0.35

    bars1 = ax.bar([i - width / 2 for i in x], nl_acc, width, label="NL Accuracy", color="#3498db")
    bars2 = ax.bar([i + width / 2 for i in x], compare_acc, width, label=compare_label, color="#2ecc71")

    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    title = "Simulation vs NL Accuracy" if compare == "sim" else "Code Execution vs NL Accuracy"
    ax.set_title(f"{title} by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.0%}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.0%}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_code_vs_nl_scatter(
    results: list[AccuracyResult], output_path: Path, compare: str = "sim"
) -> None:
    """Create scatter plot of sim/code vs NL accuracy."""
    if not results:
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    nl_acc = [r.nl_accuracy for r in results]
    compare_acc = [r.sim_accuracy if compare == "sim" else r.code_accuracy for r in results]
    models = [format_model_name(r.model) for r in results]
    compare_label = "Sim" if compare == "sim" else "Code"

    ax.scatter(nl_acc, compare_acc, s=100, alpha=0.7)

    # Add diagonal line (y = x)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y = x")

    # Label points
    for i, model in enumerate(models):
        ax.annotate(
            model,
            (nl_acc[i], compare_acc[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("NL Accuracy")
    ax.set_ylabel(f"{compare_label} Accuracy")
    ax.set_title(f"{compare_label} vs NL Accuracy (points above line = {compare_label.lower()} better)")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved scatter plot to {output_path}")


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print(f"\nLoading results from: {args.results_dir}")
    df = load_all_results(args.results_dir, args.exclude_kinds)

    if df.empty:
        print("ERROR: No results found!")
        return 1

    # Analyze
    print(f"\nAnalyzing {len(df)} results...")
    print(f"Comparison mode: {args.compare} vs NL")
    results = analyze_all_models(df, args.min_samples, args.compare)

    if not results:
        print("ERROR: No models with sufficient samples!")
        return 1

    # Create summary table
    summary_df = create_summary_table(results, args.compare)
    print("\n" + "=" * 100)
    print("ACCURACY SUMMARY")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print("=" * 100)

    # Save summary to CSV
    csv_path = args.output_dir / "accuracy_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved summary to {csv_path}")

    # Compute overall statistics based on compare mode
    if args.compare == "sim":
        better_count = sum(1 for r in results if r.sim_minus_nl > 0)
        worse_count = sum(1 for r in results if r.sim_minus_nl < 0)
        equal_count = sum(1 for r in results if r.sim_minus_nl == 0)
        avg_diff = sum(r.sim_minus_nl for r in results) / len(results)
        label = "Sim"
        diff_attr = "sim_minus_nl"
    else:
        better_count = sum(1 for r in results if r.code_minus_nl > 0)
        worse_count = sum(1 for r in results if r.code_minus_nl < 0)
        equal_count = sum(1 for r in results if r.code_minus_nl == 0)
        avg_diff = sum(r.code_minus_nl for r in results) / len(results)
        label = "Code"
        diff_attr = "code_minus_nl"

    print("\n" + "-" * 50)
    print(f"OVERALL {label.upper()} vs NL TREND")
    print("-" * 50)
    print(f"Models with {label} > NL: {better_count}/{len(results)}")
    print(f"Models with NL > {label}: {worse_count}/{len(results)}")
    print(f"Models with {label} = NL: {equal_count}/{len(results)}")

    print(f"\nAverage ({label} - NL): {avg_diff:+.2%}")

    # Statistical significance
    sig_better = sum(
        1 for r in results
        if r.mcnemar_pvalue is not None
        and r.mcnemar_pvalue < 0.05
        and getattr(r, diff_attr) > 0
    )
    print(f"\nModels with significant {label} > NL (p < 0.05): {sig_better}/{len(results)}")

    # Code execution error summary
    print("\n" + "-" * 50)
    print("CODE EXECUTION SUCCESS RATE")
    print("-" * 50)
    for r in sorted(results, key=lambda x: x.code_exec_success_rate, reverse=True):
        print(f"  {format_model_name(r.model):30s}: {r.code_exec_success_rate:.0%}")

    avg_exec_rate = sum(r.code_exec_success_rate for r in results) / len(results)
    print(f"\nAverage code execution success: {avg_exec_rate:.0%}")

    # Generate plots
    plot_accuracy_comparison(
        results, args.output_dir / "accuracy_comparison.png", args.compare
    )
    plot_code_vs_nl_scatter(
        results, args.output_dir / f"{args.compare}_vs_nl_scatter.png", args.compare
    )

    # Save detailed results as JSON
    json_path = args.output_dir / "analysis_results.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "compare_mode": args.compare,
                "results": [
                    {
                        "model": r.model,
                        "n_samples": r.n_samples,
                        "nl_accuracy": r.nl_accuracy,
                        "code_accuracy": r.code_accuracy,
                        "sim_accuracy": r.sim_accuracy,
                        "sim_minus_nl": r.sim_minus_nl,
                        "code_minus_nl": r.code_minus_nl,
                        "code_exec_success_rate": r.code_exec_success_rate,
                        "mcnemar_pvalue": r.mcnemar_pvalue,
                    }
                    for r in results
                ],
                "summary": {
                    "n_models": len(results),
                    f"{label.lower()}_better_count": better_count,
                    "nl_better_count": worse_count,
                    f"avg_{label.lower()}_minus_nl": avg_diff,
                    f"sig_{label.lower()}_better": sig_better,
                    "avg_code_exec_success_rate": avg_exec_rate,
                },
            },
            f,
            indent=2,
        )
    print(f"\nSaved detailed results to {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
