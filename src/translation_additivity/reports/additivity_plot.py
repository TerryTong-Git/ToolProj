#!/usr/bin/env python3
"""
Generate translation additivity bar plot.

Usage:
    uv run python -m src.translation_additivity.reports.additivity_plot
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.rcParams["font.family"] = "sans-serif"

# Data from the experiment (N=1000, Gemini 2.0 Flash)
conditions = ["x\n(baseline)", "x || z_NL\n(native)", "x || z̃_NL\n(translated)"]
accuracies = [22.6, 34.8, 35.1]
ci_low = [20.1, 31.9, 32.2]
ci_high = [25.3, 37.8, 38.1]


def main():
    # Calculate error bars
    errors_low = [acc - low for acc, low in zip(accuracies, ci_low)]
    errors_high = [high - acc for acc, high in zip(accuracies, ci_high)]

    # Use viridis colors
    viridis = sns.color_palette("viridis", 4)
    colors = [viridis[0], viridis[2], viridis[3]]  # Skip one for better contrast

    # Create figure (vertically compressed)
    fig, ax = plt.subplots(figsize=(10, 4.5))

    x = np.arange(len(conditions))
    width = 0.55

    bars = ax.bar(
        x,
        accuracies,
        width,
        yerr=[errors_low, errors_high],
        color=colors,
        edgecolor="black",
        linewidth=1.2,
        capsize=6,
        error_kw={"linewidth": 2, "capthick": 2},
    )

    # Add value labels above bars
    for i, (bar, acc, ci_l, ci_h) in enumerate(zip(bars, accuracies, ci_low, ci_high)):
        # Main value
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + errors_high[i] + 1.2,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )
        # CI below main value
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + errors_high[i] + 0.2,
            f"[{ci_l:.1f}%, {ci_h:.1f}%]",
            ha="center",
            va="top",
            fontsize=10,
            color="#555",
        )

    # Add improvement arrows and labels
    # Arrow from baseline to native
    ax.annotate("", xy=(1, 34.8), xytext=(0, 22.6), arrowprops=dict(arrowstyle="->", color=viridis[2], lw=2.5))
    ax.text(
        0.35,
        30,
        "+12.2%",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#1f6e4f",
        rotation=40,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
    )

    # Arrow from baseline to translated
    ax.annotate("", xy=(2, 35.1), xytext=(0, 22.6), arrowprops=dict(arrowstyle="->", color=viridis[3], lw=2.5))
    ax.text(
        0.85,
        25,
        "+12.5%",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#7fbc41",
        rotation=15,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
    )

    # Gap annotation
    gap_x = 1.5
    gap_y = 36.5
    ax.annotate(
        "Gap: 0.3%\n(not significant)",
        xy=(gap_x, gap_y),
        fontsize=11,
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5dc", edgecolor="#888", alpha=0.9),
    )

    # Baseline reference line
    ax.axhline(y=22.6, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)

    # Customize plot
    ax.set_ylabel("Accuracy (%)", fontsize=15, fontweight="bold")
    ax.set_title("Translation Additivity: Native vs Translated NL\n(N=1000, Gemini 2.0 Flash)", fontsize=17, fontweight="bold", pad=10)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=13, fontweight="bold")

    # Set y-axis limits
    ax.set_ylim(0, 45)
    ax.tick_params(axis="y", labelsize=12)

    # Add grid for readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # Remove top and right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Save
    output_dir = Path("src/translation_additivity/results")
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "translation_additivity_plot.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "translation_additivity_plot.png", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved to {output_dir}/translation_additivity_plot.pdf/png")


if __name__ == "__main__":
    main()
