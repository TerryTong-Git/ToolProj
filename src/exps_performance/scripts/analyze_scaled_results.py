#!/usr/bin/env python3
"""
Analyze results from the scaled Code vs NL experiment.
Generates:
1. Line plot (line.png) - Accuracy across arms
2. P-value plot (pval.png) - Statistical significance
3. Main figure (main.png) - Faceted by problem type
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import wilcoxon
import itertools
from matplotlib.lines import Line2D

# Import from the project
from src.exps_performance.logger import create_big_df


def load_scaled_results(results_root: Path) -> pd.DataFrame:
    """Load all results from the scaled experiment."""
    jsonl_files = sorted(results_root.rglob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found under {results_root}")

    print(f"Found {len(jsonl_files)} JSONL files:")
    for f in jsonl_files:
        print(f"  - {f}")

    df = create_big_df(jsonl_files)
    return df


def filter_target_models(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to the 4 target models from the scaled experiment."""
    target_models = ["mistralai/codestral-2508", "mistralai/mistral-large-2411", "google/gemini-2.0-flash-001", "mistralai/mixtral-8x22b-instruct"]
    df_filtered = df[df["model"].isin(target_models)]
    print(f"Filtered to {len(df_filtered)} rows from {len(target_models)} models")
    return df_filtered


def plot_line_graph(df: pd.DataFrame, output_path: Path) -> None:
    """Plot accuracy across arms (V-graph style)."""
    from matplotlib import rcParams

    rcParams["figure.dpi"] = 500
    rcParams["savefig.dpi"] = 500
    rcParams["font.family"] = "Arial"
    rcParams["axes.labelsize"] = 18
    rcParams["axes.titlesize"] = 18
    rcParams["legend.fontsize"] = 14
    rcParams["figure.titlesize"] = 18

    fig, ax = plt.subplots(figsize=(8, 6))
    cols = ["nl_correct", "sim_correct", "controlsim_correct", "code_correct"]

    melted_df = pd.melt(df, value_vars=cols, id_vars=["model", "kind"])

    # Color palette for models
    models = sorted(melted_df["model"].unique())
    palette = dict(zip(models, sns.color_palette("husl", len(models))))

    # Plot each model
    for model in models:
        model_data = melted_df[melted_df["model"] == model]
        means = model_data.groupby("variable")["value"].mean()
        ax.plot(range(len(cols)), [means.get(c, 0) for c in cols], marker="o", label=model.split("/")[-1], linewidth=2, markersize=8)

    # Plot aggregate
    all_means = melted_df.groupby("variable")["value"].mean()
    ax.plot(range(len(cols)), [all_means.get(c, 0) for c in cols], color="black", marker="s", linewidth=3, markersize=10, label="All models")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["NL", "Sim", "ControlSim", "Code"])
    ax.set_xlabel("Arm")
    ax.set_ylabel("Accuracy")
    ax.set_ylim([0, 1])
    ax.legend(loc="best")
    ax.set_title("Code vs NL Accuracy Across Arms")

    plt.tight_layout()
    plt.savefig(output_path / "line.png", bbox_inches="tight")
    print(f"Saved line plot to {output_path / 'line.png'}")
    plt.close()


def plot_p_values(df: pd.DataFrame, output_path: Path) -> None:
    """Plot p-values from Wilcoxon tests."""
    from matplotlib import rcParams

    rcParams["figure.dpi"] = 500
    rcParams["savefig.dpi"] = 500

    cols = ["nl_correct", "sim_correct", "controlsim_correct", "code_correct"]
    col_names = ["NL", "Sim", "ControlSim", "Code"]
    name_map = dict(zip(cols, col_names))

    # Rename columns
    df_renamed = df.rename(columns=name_map)

    # Melt and aggregate
    melted = pd.melt(df_renamed, value_vars=col_names, id_vars=["model", "kind"])
    grouped = melted.groupby(["variable", "model", "kind"])["value"].mean().reset_index()
    final = grouped.drop(["model", "kind"], axis=1)
    final = final[np.isfinite(final["value"])]

    if final.empty:
        print("Warning: No finite values for p-value plot")
        return

    # Wilcoxon tests for all pairs
    arm_pairs = list(itertools.combinations(col_names, 2))
    p_values = []
    for c1, c2 in arm_pairs:
        x = final[final["variable"] == c1]["value"]
        y = final[final["variable"] == c2]["value"]
        try:
            _, p = wilcoxon(x, y)
        except Exception:
            p = 1.0
        p_values.append(p)

    # Plot boxplot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x="variable", y="value", data=final, order=col_names, palette="vlag", ax=ax)

    # Add p-value annotations
    max_val = final["value"].max()
    offset = 0.05
    for i, (pair, p_val) in enumerate(zip(arm_pairs, p_values)):
        c1, c2 = pair
        x1 = col_names.index(c1)
        x2 = col_names.index(c2)
        y = max_val + offset + i * 0.08
        h = 0.02
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color="steelblue")
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        ax.text((x1 + x2) * 0.5, y + h, f"p={p_val:.4f} {significance}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Arm")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, max(1.0, max_val + 0.5))
    ax.set_title("Accuracy Distribution with Wilcoxon Tests")

    plt.tight_layout()
    plt.savefig(output_path / "pval.png", bbox_inches="tight")
    print(f"Saved p-value plot to {output_path / 'pval.png'}")
    plt.close()


def plot_main_figure(df: pd.DataFrame, output_path: Path) -> None:
    """Plot faceted figure by problem kind."""
    sns.reset_defaults()

    # Filter to fine-grained kinds for cleaner visualization
    fg_kinds = ["add", "mul", "lcs", "rod", "knap", "ilp_assign", "ilp_prod", "ilp_partition"]
    df_fg = df[df["kind"].isin(fg_kinds)]

    if df_fg.empty:
        print("Warning: No fine-grained data for main figure")
        # Fall back to all kinds
        df_fg = df

    name_map = {
        "nl_correct": "NL",
        "sim_correct": "Sim",
        "controlsim_correct": "ControlSim",
        "code_correct": "Code",
    }
    df_renamed = df_fg.rename(columns=name_map)

    cols = list(name_map.values())
    melted = pd.melt(df_renamed, value_vars=cols, id_vars=["kind", "digit"])

    g = sns.FacetGrid(melted, col="kind", col_wrap=4, hue="variable", hue_order=cols, sharex=False, height=3, aspect=1.2)
    g.map(sns.lineplot, "digit", "value")
    g.set_titles("{col_name}")

    for ax in g.axes:
        ax.set_xlim(None, 20)
        ax.set_ylim(0, 1)

    g.axes[-1].legend(loc="best")
    g.set_xlabels("Test Length")
    g.set_ylabels("Accuracy")

    plt.tight_layout()
    plt.savefig(output_path / "main.png", bbox_inches="tight")
    print(f"Saved main figure to {output_path / 'main.png'}")
    plt.close()


def print_summary_stats(df: pd.DataFrame) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    print(f"\nTotal samples: {len(df)}")
    print(f"\nSamples by model:")
    print(df.groupby("model").size())

    print(f"\nSamples by kind:")
    print(df.groupby("kind").size())

    print(f"\nMean accuracy by arm:")
    for col in ["nl_correct", "sim_correct", "controlsim_correct", "code_correct"]:
        if col in df.columns:
            print(f"  {col}: {df[col].mean():.4f}")

    print(f"\nMean accuracy by model and arm:")
    arm_cols = ["nl_correct", "sim_correct", "controlsim_correct", "code_correct"]
    arm_cols_present = [c for c in arm_cols if c in df.columns]
    print(df.groupby("model")[arm_cols_present].mean())

    print("\n" + "=" * 60)


def main():
    results_root = Path("src/exps_performance/results/scaled_code_nl")
    output_path = Path("figures")
    output_path.mkdir(exist_ok=True)

    print("Loading results...")
    df = load_scaled_results(results_root)

    print("\nFiltering to target models...")
    df = filter_target_models(df)

    print_summary_stats(df)

    if len(df) == 0:
        print("No data to plot!")
        return

    print("\nGenerating figures...")
    plot_line_graph(df, output_path)
    plot_p_values(df, output_path)
    plot_main_figure(df, output_path)

    print(f"\nAll figures saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
