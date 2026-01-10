"""
Analysis script for Code Execution vs NL noise robustness experiments.

Generates:
1. Accuracy vs noise levels (aggregated across problem types)
2. Accuracy vs noise levels by noise type
3. Statistical tests for equivalence/non-inferiority

Usage:
    uv run python src/exps_performance/analysis_noise_code_vs_nl.py [results_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from scipy import stats

# Publication-quality settings
rcParams["figure.dpi"] = 300
rcParams["savefig.dpi"] = 300
rcParams["font.family"] = "sans-serif"
rcParams["axes.labelsize"] = 14
rcParams["axes.titlesize"] = 16
rcParams["legend.fontsize"] = 12
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12


def load_noise_results(results_dir: Path) -> pd.DataFrame:
    """Load all noise experiment results from JSON files."""
    files = list(results_dir.glob("*.json"))
    rows = []

    for fp in files:
        try:
            payload = json.loads(fp.read_text())
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {fp}")
            continue

        if not payload:
            continue

        # Last element is summary metadata
        *records, summary = payload
        if records:
            df = pd.DataFrame.from_records(records)
            df["model"] = summary.get("model", "unknown")
            df["seed"] = summary.get("seed", 0)
            rows.append(df)

    if not rows:
        return pd.DataFrame()

    data = pd.concat(rows, ignore_index=True)
    data["sigma"] = data["sigma"].astype(float)
    data["accuracy"] = data["accuracy"].astype(float)
    return data


def filter_arms(data: pd.DataFrame, arms: List[str] = ["nl", "code"]) -> pd.DataFrame:
    """Filter to only Code Execution (code) and NL (nl) arms."""
    return data[data["arm"].isin(arms)].copy()


def aggregate_by_problem_category(data: pd.DataFrame) -> pd.DataFrame:
    """Add problem category column for grouped analysis."""
    category_map = {
        "add": "Arithmetic",
        "sub": "Arithmetic",
        "mul": "Arithmetic",
        "lcs": "DP",
        "knap": "DP",
        "rod": "DP",
        "ilp_assign": "ILP",
        "ilp_prod": "ILP",
        "ilp_partition": "ILP",
        "tsp": "NP-Hard",
        "gcp": "NP-Hard",
        "spp": "NP-Hard",
        "bsp": "NP-Hard",
        "edp": "NP-Hard",
        "msp": "NP-Hard",
        "ksp": "NP-Hard",
        "tspd": "NP-Hard",
        "gcpd": "NP-Hard",
        "clrs30": "CLRS",
    }
    data = data.copy()
    data["category"] = data["kind"].map(category_map).fillna("Other")
    return data


def plot_accuracy_vs_noise_aggregated(data: pd.DataFrame, output_path: Path) -> None:
    """
    Generate aggregated accuracy vs noise level plot.
    Code Execution vs NL on same plot, averaged across all problem types and models.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Aggregate: mean accuracy by arm, sigma (across all kinds, models, seeds)
    agg = data.groupby(["arm", "sigma"], as_index=False)["accuracy"].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["arm", "sigma", "mean", "std", "count"]
    agg["se"] = agg["std"] / np.sqrt(agg["count"])

    colors = {"nl": "#1f77b4", "code": "#ff7f0e"}
    labels = {"nl": "NL (Arm 1)", "code": "Code Exec (Arm 3)"}
    markers = {"nl": "o", "code": "s"}

    for arm in ["nl", "code"]:
        arm_data = agg[agg["arm"] == arm].sort_values("sigma")
        ax.errorbar(
            arm_data["sigma"],
            arm_data["mean"],
            yerr=1.96 * arm_data["se"],  # 95% CI
            label=labels[arm],
            color=colors[arm],
            marker=markers[arm],
            markersize=10,
            linewidth=2,
            capsize=5,
        )

    ax.set_xlabel("Noise Level (σ)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.legend(loc="lower left")
    ax.set_title("Code Execution vs NL: Accuracy under Noise")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_by_noise_type(data: pd.DataFrame, output_path: Path) -> None:
    """
    Generate faceted plot: Accuracy vs noise level, one subplot per noise type.
    Code vs NL on same subplot.
    """
    noise_types = sorted(data["noise_type"].unique())
    n_types = len(noise_types)

    n_cols = 3
    n_rows = (n_types + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), sharey=True)
    axes = axes.flatten() if n_types > 1 else [axes]

    colors = {"nl": "#1f77b4", "code": "#ff7f0e"}
    labels = {"nl": "NL", "code": "Code Exec"}

    for idx, noise_type in enumerate(noise_types):
        ax = axes[idx]
        subset = data[data["noise_type"] == noise_type]

        agg = subset.groupby(["arm", "sigma"], as_index=False)["accuracy"].agg(["mean", "std", "count"]).reset_index()
        agg.columns = ["arm", "sigma", "mean", "std", "count"]
        agg["se"] = agg["std"] / np.sqrt(agg["count"])

        for arm in ["nl", "code"]:
            arm_data = agg[agg["arm"] == arm].sort_values("sigma")
            if arm_data.empty:
                continue
            ax.errorbar(
                arm_data["sigma"],
                arm_data["mean"],
                yerr=1.96 * arm_data["se"],
                label=labels[arm],
                color=colors[arm],
                marker="o" if arm == "nl" else "s",
                markersize=8,
                linewidth=2,
                capsize=4,
            )

        ax.set_title(noise_type.capitalize())
        ax.set_xlabel("σ")
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.set_ylabel("Accuracy")
            ax.legend(loc="lower left")

    # Hide unused subplots
    for idx in range(n_types, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Accuracy under Different Noise Types: Code Exec vs NL", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_vs_digits(data: pd.DataFrame, output_path: Path) -> None:
    """
    Generate Plot 2: Accuracy vs Digits (problem hardness) under noise.
    Aggregates over all noise types and sigma > 0.
    """
    # Filter to sigma > 0 to show effect under noise
    noisy_data = data[data["sigma"] > 0].copy()
    if noisy_data.empty:
        print("No noisy data (sigma > 0) available for digits plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # Aggregate: mean accuracy by arm, digit (across all noise types, sigmas, kinds, models, seeds)
    agg = noisy_data.groupby(["arm", "digit"], as_index=False)["accuracy"].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["arm", "digit", "mean", "std", "count"]
    agg["se"] = agg["std"] / np.sqrt(agg["count"])

    colors = {"nl": "#1f77b4", "code": "#ff7f0e"}
    labels = {"nl": "NL (Arm 1)", "code": "Code Exec (Arm 3)"}
    markers = {"nl": "o", "code": "s"}

    for arm in ["nl", "code"]:
        arm_data = agg[agg["arm"] == arm].sort_values("digit")
        if arm_data.empty:
            continue
        ax.errorbar(
            arm_data["digit"],
            arm_data["mean"],
            yerr=1.96 * arm_data["se"],  # 95% CI
            label=labels[arm],
            color=colors[arm],
            marker=markers[arm],
            markersize=10,
            linewidth=2,
            capsize=5,
        )

    ax.set_xlabel("Digits (Problem Hardness)")
    ax.set_ylabel("Accuracy (under noise)")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(sorted(noisy_data["digit"].unique()))
    ax.legend(loc="lower left")
    ax.set_title("Code vs NL: Accuracy by Problem Difficulty\n(Aggregated over σ > 0)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_degradation_slope_comparison(data: pd.DataFrame, output_path: Path) -> None:
    """
    Plot degradation curves and compute slope comparison for H1.
    Shows how accuracy degrades as noise increases for code vs NL.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Aggregate by arm and sigma
    agg = data.groupby(["arm", "sigma"], as_index=False)["accuracy"].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["arm", "sigma", "mean", "std", "count"]
    agg["se"] = agg["std"] / np.sqrt(agg["count"])

    colors = {"nl": "#1f77b4", "code": "#ff7f0e"}
    labels = {"nl": "NL", "code": "Code Exec"}

    # Left plot: Degradation curves
    ax = axes[0]
    slopes = {}
    for arm in ["nl", "code"]:
        arm_data = agg[agg["arm"] == arm].sort_values("sigma")
        if arm_data.empty:
            continue
        ax.errorbar(
            arm_data["sigma"],
            arm_data["mean"],
            yerr=1.96 * arm_data["se"],
            label=labels[arm],
            color=colors[arm],
            marker="o",
            markersize=8,
            linewidth=2,
            capsize=4,
        )
        # Compute linear regression slope
        if len(arm_data) >= 2:
            slope, intercept = np.polyfit(arm_data["sigma"], arm_data["mean"], 1)
            slopes[arm] = slope
            # Plot regression line
            x_line = np.linspace(arm_data["sigma"].min(), arm_data["sigma"].max(), 50)
            ax.plot(x_line, intercept + slope * x_line, "--", color=colors[arm], alpha=0.5)

    ax.set_xlabel("Noise Level (σ)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    ax.set_title("Degradation Curves")
    ax.grid(True, alpha=0.3)

    # Right plot: Slope comparison bar chart
    ax = axes[1]
    if slopes:
        arms = list(slopes.keys())
        slope_vals = [slopes[arm] for arm in arms]
        colors_list = [colors[arm] for arm in arms]
        bars = ax.bar(arms, slope_vals, color=colors_list, edgecolor="black")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("Degradation Slope (Δacc/Δσ)")
        ax.set_title("Slope Comparison\n(less negative = more robust)")

        # Add value labels
        for bar, val in zip(bars, slope_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.02, f"{val:.3f}", ha="center", va="top", fontsize=12, fontweight="bold")

        # Add slope ratio annotation
        if "nl" in slopes and "code" in slopes and slopes["nl"] != 0:
            ratio = slopes["code"] / slopes["nl"]
            ax.text(0.5, 0.95, f"Slope ratio (code/nl): {ratio:.2f}", transform=ax.transAxes, ha="center", fontsize=12, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_by_category(data: pd.DataFrame, output_path: Path) -> None:
    """
    Generate faceted plot: Accuracy vs noise level by problem category.
    """
    categories = sorted(data["category"].unique())
    n_cats = len(categories)

    n_cols = 3
    n_rows = (n_cats + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), sharey=True)
    axes = axes.flatten() if n_cats > 1 else [axes]

    colors = {"nl": "#1f77b4", "code": "#ff7f0e"}
    labels = {"nl": "NL", "code": "Code Exec"}

    for idx, category in enumerate(categories):
        ax = axes[idx]
        subset = data[data["category"] == category]

        agg = subset.groupby(["arm", "sigma"], as_index=False)["accuracy"].agg(["mean", "std", "count"]).reset_index()
        agg.columns = ["arm", "sigma", "mean", "std", "count"]
        agg["se"] = agg["std"] / np.sqrt(agg["count"])

        for arm in ["nl", "code"]:
            arm_data = agg[agg["arm"] == arm].sort_values("sigma")
            if arm_data.empty:
                continue
            ax.errorbar(
                arm_data["sigma"],
                arm_data["mean"],
                yerr=1.96 * arm_data["se"],
                label=labels[arm],
                color=colors[arm],
                marker="o" if arm == "nl" else "s",
                markersize=8,
                linewidth=2,
                capsize=4,
            )

        ax.set_title(category)
        ax.set_xlabel("σ")
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.set_ylabel("Accuracy")
            ax.legend(loc="lower left")

    for idx in range(n_cats, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Accuracy by Problem Category: Code Exec vs NL", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def run_statistical_tests(data: pd.DataFrame, delta: float = 0.05) -> pd.DataFrame:
    """
    Run statistical tests to determine if Code is "similar to or as good as" NL.

    Tests performed:
    1. Paired t-test: H0: mean(Code) = mean(NL)
    2. Wilcoxon signed-rank test: Non-parametric alternative
    3. Non-inferiority test: H0: mean(Code) - mean(NL) < -delta

    Args:
        data: DataFrame with arm, noise_type, sigma, accuracy columns
        delta: Non-inferiority margin (default 0.05 = 5%)

    Returns:
        DataFrame with test results per (noise_type, sigma) combination.
    """
    results = []

    for noise_type in data["noise_type"].unique():
        for sigma in sorted(data["sigma"].unique()):
            subset = data[(data["noise_type"] == noise_type) & (data["sigma"] == sigma)]

            # Pivot to get paired observations (by kind, model, seed)
            pivot = subset.pivot_table(index=["kind", "model", "seed"], columns="arm", values="accuracy").dropna()

            if len(pivot) < 3 or "nl" not in pivot.columns or "code" not in pivot.columns:
                continue

            nl_acc = pivot["nl"].values
            code_acc = pivot["code"].values
            diff = code_acc - nl_acc

            # Paired t-test
            t_stat, t_pval = stats.ttest_rel(code_acc, nl_acc)

            # Wilcoxon signed-rank test
            try:
                w_stat, w_pval = stats.wilcoxon(code_acc, nl_acc)
            except ValueError:
                w_stat, w_pval = np.nan, np.nan

            # Non-inferiority test (one-sided)
            # H0: Code - NL < -delta  vs  H1: Code - NL >= -delta
            # Equivalent to testing if (diff + delta) > 0
            ni_t_stat, ni_pval_two = stats.ttest_1samp(diff + delta, 0)
            ni_pval = ni_pval_two / 2 if ni_t_stat > 0 else 1 - ni_pval_two / 2

            # Effect size: Cohen's d
            pooled_std = np.sqrt((np.var(nl_acc, ddof=1) + np.var(code_acc, ddof=1)) / 2)
            cohens_d = np.mean(diff) / pooled_std if pooled_std > 0 else 0

            results.append(
                {
                    "noise_type": noise_type,
                    "sigma": sigma,
                    "n_pairs": len(pivot),
                    "mean_nl": np.mean(nl_acc),
                    "mean_code": np.mean(code_acc),
                    "mean_diff": np.mean(diff),
                    "std_diff": np.std(diff, ddof=1),
                    "cohens_d": cohens_d,
                    "t_stat": t_stat,
                    "t_pval": t_pval,
                    "wilcoxon_stat": w_stat,
                    "wilcoxon_pval": w_pval,
                    "noninferiority_pval": ni_pval,
                    "code_noninferior": ni_pval < 0.05,
                }
            )

    return pd.DataFrame(results)


def plot_statistical_summary(stats_df: pd.DataFrame, output_path: Path) -> None:
    """Create a heatmap visualization of statistical test results."""
    if stats_df.empty:
        print("No statistical results to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Heatmap 1: Mean difference (Code - NL)
    pivot_diff = stats_df.pivot(index="noise_type", columns="sigma", values="mean_diff")
    sns.heatmap(
        pivot_diff,
        ax=axes[0],
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".3f",
        cbar_kws={"label": "Mean Diff (Code - NL)"},
    )
    axes[0].set_title("Accuracy Difference: Code - NL")
    axes[0].set_xlabel("Noise Level (σ)")
    axes[0].set_ylabel("Noise Type")

    # Heatmap 2: Non-inferiority p-values
    pivot_pval = stats_df.pivot(index="noise_type", columns="sigma", values="noninferiority_pval")
    sns.heatmap(
        pivot_pval,
        ax=axes[1],
        cmap="RdYlGn_r",  # Red = high p-value (bad), Green = low (good)
        vmin=0,
        vmax=0.1,
        annot=True,
        fmt=".3f",
        cbar_kws={"label": "p-value"},
    )
    axes[1].set_title("Non-inferiority Test (δ=0.05)")
    axes[1].set_xlabel("Noise Level (σ)")
    axes[1].set_ylabel("Noise Type")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_effect_sizes(stats_df: pd.DataFrame, output_path: Path) -> None:
    """Create a heatmap of Cohen's d effect sizes."""
    if stats_df.empty:
        print("No statistical results to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    pivot_d = stats_df.pivot(index="noise_type", columns="sigma", values="cohens_d")
    sns.heatmap(
        pivot_d,
        ax=ax,
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Cohen's d"},
        vmin=-1,
        vmax=1,
    )
    ax.set_title("Effect Size (Cohen's d): Code - NL\n(+ve = Code better)")
    ax.set_xlabel("Noise Level (σ)")
    ax.set_ylabel("Noise Type")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def print_summary(stats_df: pd.DataFrame) -> None:
    """Print interpretation of results."""
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if stats_df.empty:
        print("No results to interpret")
        return

    noninferior_count = stats_df["code_noninferior"].sum()
    total_tests = len(stats_df)
    print(f"Non-inferiority established in {noninferior_count}/{total_tests} conditions")
    print("(Code is within 5% of NL accuracy at α=0.05)")

    # Bonferroni correction
    bonferroni_alpha = 0.05 / total_tests
    bonferroni_pass = (stats_df["noninferiority_pval"] < bonferroni_alpha).sum()
    print(f"\nWith Bonferroni correction (α={bonferroni_alpha:.4f}):")
    print(f"  Non-inferiority established in {bonferroni_pass}/{total_tests} conditions")

    # Cases where Code outperforms NL
    better = stats_df[stats_df["mean_diff"] > 0]
    print(f"\nCode outperforms NL in {len(better)}/{total_tests} conditions:")
    if not better.empty:
        print(better[["noise_type", "sigma", "mean_diff", "cohens_d"]].to_string(index=False))

    # Cases where Code underperforms significantly
    worse = stats_df[(stats_df["mean_diff"] < -0.05) & (stats_df["t_pval"] < 0.05)]
    if not worse.empty:
        print("\nCode significantly worse than NL (diff < -5%, p < 0.05):")
        print(worse[["noise_type", "sigma", "mean_diff", "t_pval"]].to_string(index=False))
    else:
        print("\nNo conditions where Code is significantly worse than NL by >5%")


def main(results_dir: str = "src/exps_performance/results_noise_code_vs_nl") -> None:
    """Run full analysis pipeline."""
    results_path = Path(results_dir)
    figures_dir = results_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading results from: {results_path}")
    data = load_noise_results(results_path)

    if data.empty:
        print("No data found. Run experiments first.")
        print("  bash src/exps_performance/scripts/noise_code_vs_nl.sh")
        return

    print(f"Loaded {len(data)} records")
    print(f"Models: {sorted(data['model'].unique())}")
    print(f"Arms: {sorted(data['arm'].unique())}")
    print(f"Noise types: {sorted(data['noise_type'].unique())}")
    print(f"Sigma levels: {sorted(data['sigma'].unique())}")
    print(f"Problem kinds: {sorted(data['kind'].unique())}")

    # Filter to Code and NL arms only
    data = filter_arms(data, ["nl", "code"])
    data = aggregate_by_problem_category(data)

    print(f"\nFiltered to {len(data)} records (nl + code arms only)")

    # Generate plots
    print("\nGenerating plots...")
    plot_accuracy_vs_noise_aggregated(data, figures_dir / "accuracy_vs_noise_aggregated.png")
    plot_accuracy_by_noise_type(data, figures_dir / "accuracy_by_noise_type.png")
    plot_accuracy_by_category(data, figures_dir / "accuracy_by_category.png")

    # New plots for experiment 10
    if "digit" in data.columns:
        plot_accuracy_vs_digits(data, figures_dir / "accuracy_vs_digits_under_noise.png")
    plot_degradation_slope_comparison(data, figures_dir / "degradation_curves.png")

    # Run statistical tests
    print("\nRunning statistical tests...")
    stats_df = run_statistical_tests(data)
    stats_csv_path = figures_dir / "statistical_tests.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Saved: {stats_csv_path}")

    print("\nStatistical Test Summary:")
    print(
        stats_df[
            [
                "noise_type",
                "sigma",
                "n_pairs",
                "mean_nl",
                "mean_code",
                "mean_diff",
                "t_pval",
                "noninferiority_pval",
                "code_noninferior",
            ]
        ].to_string(index=False)
    )

    plot_statistical_summary(stats_df, figures_dir / "statistical_summary.png")
    plot_effect_sizes(stats_df, figures_dir / "effect_sizes.png")

    # Summary interpretation
    print_summary(stats_df)


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "src/exps_performance/results_noise_code_vs_nl"
    main(results_dir)
