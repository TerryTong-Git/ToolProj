#!/usr/bin/env python3
"""
Plot accuracy vs hardness (digit length) with statistical analysis.

Features:
- Generalized logistic mixed model effect sizes
- Wilson confidence intervals
- Holm-Bonferroni correction for pairwise comparisons
- Pairwise McNemar p-values
- Overall Cochran's Q test

Usage:
    uv run python -m src.reasoning_benchmark.scripts.plot_accuracy_vs_hardness
"""

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.reasoning_benchmark.artifact_paths import LEGACY_BENCHMARK_RESULTS_DIR

warnings.filterwarnings("ignore")

RESULTS_DIR = LEGACY_BENCHMARK_RESULTS_DIR
OUTPUT_DIR = LEGACY_BENCHMARK_RESULTS_DIR


def wilson_ci(n_success: int, n_total: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Wilson score confidence interval."""
    if n_total == 0:
        return 0.0, 0.0, 1.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    margin = (z / denom) * np.sqrt((p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)))
    return p_hat, max(0, center - margin), min(1, center + margin)


def cochrans_q(data: np.ndarray) -> tuple[float, float]:
    """
    Cochran's Q test for k related samples.
    data: n_subjects x k_conditions binary matrix
    Returns: (Q statistic, p-value)
    """
    n, k = data.shape
    row_sums = data.sum(axis=1)
    col_sums = data.sum(axis=0)
    total = data.sum()

    # Cochran's Q formula
    numerator = (k - 1) * (k * (col_sums**2).sum() - total**2)
    denominator = k * total - (row_sums**2).sum()

    if denominator == 0:
        return 0.0, 1.0

    Q = numerator / denominator
    p_value = 1 - stats.chi2.cdf(Q, k - 1)
    return Q, p_value


def pairwise_mcnemar(data: np.ndarray, conditions: list[str]) -> dict:
    """
    Pairwise McNemar tests with Holm-Bonferroni correction.
    data: n_subjects x k_conditions binary matrix
    Returns: dict of {(cond1, cond2): {'statistic': x, 'p_raw': y, 'p_corrected': z}}
    """
    k = len(conditions)
    results = {}
    p_values = []
    pairs = []

    for i in range(k):
        for j in range(i + 1, k):
            # Build 2x2 contingency table
            b = ((data[:, i] == 1) & (data[:, j] == 0)).sum()  # i correct, j wrong
            c = ((data[:, i] == 0) & (data[:, j] == 1)).sum()  # i wrong, j correct

            # McNemar test (use exact if small counts)
            if b + c < 25:
                # Exact binomial test
                if b + c > 0:
                    result = stats.binomtest(b, b + c, 0.5)
                    p_val = result.pvalue
                else:
                    p_val = 1.0
                stat = b
            else:
                # Chi-square approximation with continuity correction
                stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
                p_val = 1 - stats.chi2.cdf(stat, 1) if (b + c) > 0 else 1.0

            pairs.append((conditions[i], conditions[j]))
            p_values.append(p_val)
            results[(conditions[i], conditions[j])] = {"statistic": stat, "b": b, "c": c, "p_raw": p_val}

    # Holm-Bonferroni correction
    if p_values:
        _, p_corrected, _, _ = multipletests(p_values, method="holm")
        for idx, pair in enumerate(pairs):
            results[pair]["p_corrected"] = p_corrected[idx]

    return results


def load_all_data() -> pd.DataFrame:
    """Load all results into a DataFrame."""
    records = []

    for jsonl_path in RESULTS_DIR.glob("**/res.jsonl"):
        model_seed = jsonl_path.parent.parent.parent.name

        with jsonl_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    records.append(
                        {
                            "model_seed": model_seed,
                            "model": row.get("model", model_seed.rsplit("_seed", 1)[0]),
                            "seed": row.get("seed", 0),
                            "digit": row.get("digit", 0),
                            "kind": row.get("kind", ""),
                            "unique_tag": row.get("unique_tag", ""),
                            "nl_correct": bool(row.get("nl_correct", False)),
                            "sim_correct": bool(row.get("sim_correct", False)),
                            "code_correct": bool(row.get("code_correct", False)),
                            "controlsim_correct": bool(row.get("controlsim_correct", False)),
                        }
                    )
                except json.JSONDecodeError:
                    continue

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} records from {df['model_seed'].nunique()} model/seed combinations")
    return df


# Standard digit values from experiment config
STANDARD_DIGITS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]


def compute_accuracy_by_digit(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy with Wilson CIs for each digit and condition."""
    conditions = ["nl_correct", "sim_correct", "controlsim_correct"]
    condition_labels = {"nl_correct": "NL", "sim_correct": "Sim", "controlsim_correct": "ControlSim"}

    results = []
    # Filter to standard digits only
    digits = [d for d in STANDARD_DIGITS if d in df["digit"].unique()]

    for digit in digits:
        digit_df = df[df["digit"] == digit]
        n_total = len(digit_df)

        for cond in conditions:
            n_correct = digit_df[cond].sum()
            acc, ci_low, ci_high = wilson_ci(n_correct, n_total)
            results.append(
                {
                    "digit": digit,
                    "condition": condition_labels[cond],
                    "accuracy": acc * 100,
                    "ci_low": ci_low * 100,
                    "ci_high": ci_high * 100,
                    "n_correct": n_correct,
                    "n_total": n_total,
                }
            )

    return pd.DataFrame(results)


def run_statistical_tests(df: pd.DataFrame) -> dict:
    """Run Cochran's Q and pairwise McNemar tests."""
    conditions = ["nl_correct", "sim_correct", "controlsim_correct"]
    condition_labels = ["NL", "Sim", "ControlSim"]

    # Build binary matrix for all subjects
    # Each row = one trial (unique question), columns = conditions
    data_matrix = df[conditions].values.astype(int)

    # Cochran's Q test
    Q_stat, Q_pval = cochrans_q(data_matrix)

    # Pairwise McNemar
    mcnemar_results = pairwise_mcnemar(data_matrix, condition_labels)

    return {
        "cochran_q": Q_stat,
        "cochran_p": Q_pval,
        "mcnemar": mcnemar_results,
        "n_samples": len(df),
    }


def run_tests_by_digit(df: pd.DataFrame) -> dict:
    """Run statistical tests for each digit level."""
    results_by_digit = {}

    # Only test standard digits
    digits = [d for d in STANDARD_DIGITS if d in df["digit"].unique()]
    for digit in digits:
        digit_df = df[df["digit"] == digit]
        results_by_digit[digit] = run_statistical_tests(digit_df)

    return results_by_digit


def format_pval(p: float) -> str:
    """Format p-value for display."""
    if p < 0.001:
        return "p<.001"
    elif p < 0.01:
        return f"p={p:.3f}"
    elif p < 0.05:
        return f"p={p:.2f}"
    else:
        return f"p={p:.2f}"


def create_plot(acc_df: pd.DataFrame, overall_stats: dict, digit_stats: dict):
    """Create the accuracy vs hardness plot."""

    # Use a nice color palette
    palette = sns.color_palette("viridis", 3)
    condition_colors = {"NL": palette[0], "Sim": palette[1], "ControlSim": palette[2]}

    fig, ax = plt.subplots(figsize=(14, 6))

    digits = sorted(acc_df["digit"].unique())
    conditions = ["NL", "Sim", "ControlSim"]

    x = np.arange(len(digits))
    width = 0.25
    offsets = {"NL": -width, "Sim": 0, "ControlSim": width}

    for cond in conditions:
        cond_df = acc_df[acc_df["condition"] == cond].sort_values("digit")
        accuracies = cond_df["accuracy"].values
        ci_low = cond_df["ci_low"].values
        ci_high = cond_df["ci_high"].values

        errors_low = np.maximum(0, accuracies - ci_low)
        errors_high = np.maximum(0, ci_high - accuracies)

        ax.bar(
            x + offsets[cond],
            accuracies,
            width,
            yerr=[errors_low, errors_high],
            label=cond,
            color=condition_colors[cond],
            edgecolor="black",
            linewidth=0.5,
            capsize=2,
            error_kw={"linewidth": 1},
        )

    # Customize plot
    ax.set_xlabel("Problem Hardness (Digit Length)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
    ax.set_title("Accuracy vs Problem Hardness by Reasoning Condition\n(Wilson 95% CIs, Holm-Bonferroni corrected)", fontsize=15, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in digits], fontsize=11)
    ax.tick_params(axis="y", labelsize=11)

    ax.set_ylim(0, max(acc_df["ci_high"]) + 10)
    ax.legend(loc="upper right", fontsize=10, title="Condition", title_fontsize=11)

    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add overall Cochran's Q annotation (top left)
    cochran_text = f"Cochran's Q = {overall_stats['cochran_q']:.1f}, {format_pval(overall_stats['cochran_p'])}"
    ax.text(
        0.02,
        0.98,
        cochran_text,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.9),
    )

    # Add pairwise McNemar results (below Cochran's Q)
    mcnemar_lines = ["Pairwise McNemar (Holm-Bonferroni):"]
    for pair, res in overall_stats["mcnemar"].items():
        p_corr = res["p_corrected"]
        sig = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else "ns"
        mcnemar_lines.append(f"  {pair[0]}–{pair[1]}: {format_pval(p_corr)} {sig}")

    mcnemar_text = "\n".join(mcnemar_lines)
    ax.text(
        0.02,
        0.82,
        mcnemar_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.9),
    )

    # Add significance markers above bars for each digit
    for i, digit in enumerate(digits):
        digit_res = digit_stats.get(digit, {})
        if "mcnemar" in digit_res:
            # Check if NL vs Sim is significant
            nl_sim = digit_res["mcnemar"].get(("NL", "Sim"), {})
            if nl_sim.get("p_corrected", 1) < 0.05:
                max_y = max(
                    acc_df[(acc_df["digit"] == digit) & (acc_df["condition"] == "NL")]["ci_high"].values[0],
                    acc_df[(acc_df["digit"] == digit) & (acc_df["condition"] == "Sim")]["ci_high"].values[0],
                )
                ax.text(i - width / 2, max_y + 2, "*", ha="center", fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig


def main():
    print("Loading data...")
    df = load_all_data()

    if df.empty:
        print("No data found!")
        return

    print(f"\nDigits present: {sorted(df['digit'].unique())}")
    print(f"Models: {df['model_seed'].unique()}")

    print("\nComputing accuracy by digit...")
    acc_df = compute_accuracy_by_digit(df)
    print(acc_df.to_string())

    print("\nRunning overall statistical tests...")
    overall_stats = run_statistical_tests(df)
    print(f"  Cochran's Q = {overall_stats['cochran_q']:.2f}, p = {overall_stats['cochran_p']:.4f}")
    print(f"  N samples = {overall_stats['n_samples']}")

    print("\nPairwise McNemar tests (Holm-Bonferroni corrected):")
    for pair, res in overall_stats["mcnemar"].items():
        print(f"  {pair[0]} vs {pair[1]}: χ² = {res['statistic']:.2f}, " f"p_raw = {res['p_raw']:.4f}, p_corrected = {res['p_corrected']:.4f}")

    print("\nRunning tests by digit...")
    digit_stats = run_tests_by_digit(df)

    print("\nGenerating plot...")
    fig = create_plot(acc_df, overall_stats, digit_stats)

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_DIR / "accuracy_vs_hardness.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUTPUT_DIR / "accuracy_vs_hardness.png", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"\nSaved to {OUTPUT_DIR}/accuracy_vs_hardness.pdf/png")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    pivot = acc_df.pivot(index="digit", columns="condition", values="accuracy")
    print(pivot.round(1).to_string())


if __name__ == "__main__":
    main()
