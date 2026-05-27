#!/usr/bin/env python3
"""
Grouped bar chart for Translation Additivity results.

Shows accuracy under 3 conditions (baseline, +native NL, +translated NL)
for each source model, with significance brackets from McNemar's test.

Usage:
    uv run python -m src.translation_additivity.reports.translation_additivity
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

# --- Load trial data ---
RESULTS_DIR = Path(__file__).parent.parent / "results"

TRIAL_FILES = {
    "Haiku 4.5": "translation_claude-haiku-4.5_20260127_081757_trials.jsonl",
    "Gemini 2.5\nFlash": "translation_gemini-2.5-flash_20260127_082539_trials.jsonl",
    "Mixtral\n8x22B": "translation_mixtral_20260127_180139_trials.jsonl",
}

CONDITIONS = ["x", "x_nl_native", "x_nl_translated"]
CONDITION_LABELS = ["Baseline\n(x only)", "+Native NL", "+Translated NL"]
# "muted" palette from seaborn (colorblind-friendly)
BAR_COLORS = ["#8d8d8d", "#4878d0", "#ee854a"]


def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI."""
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - margin, centre + margin


def mcnemar_p(a, b):
    """Two-sided McNemar p-value for paired binary arrays."""
    b_better = int(((b == 1) & (a == 0)).sum())
    a_better = int(((a == 1) & (b == 0)).sum())
    total = b_better + a_better
    if total == 0:
        return 1.0
    if total < 25:
        return stats.binomtest(b_better, total, 0.5).pvalue
    chi2 = (abs(b_better - a_better) - 1) ** 2 / total
    return 1 - stats.chi2.cdf(chi2, df=1)


def load_model(fpath):
    """Return {condition: np.array of 0/1} aligned by sample_id."""
    trials = [json.loads(line) for line in open(RESULTS_DIR / fpath)]
    by_sample = defaultdict(dict)
    for t in trials:
        by_sample[t["sample_id"]][t["condition"]] = int(t["correct"])
    ids = sorted(by_sample.keys())
    return {c: np.array([by_sample[s][c] for s in ids]) for c in CONDITIONS}, len(ids)


def p_label(p):
    """Format p-value for display."""
    if p < 0.0001:
        return "p<.0001"
    if p < 0.001:
        return f"p={p:.4f}"
    if p < 0.01:
        return f"p={p:.3f}"
    return f"p={p:.2f}"


def p_color(p):
    """Green if significant, red if not."""
    return "#2a7f2a" if p < 0.05 else "#cc2222"


def draw_bracket(ax, x1, x2, y, h, p):
    """Draw a significance bracket with colored p-value."""
    color = p_color(p)
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1.2)
    ax.text(
        (x1 + x2) / 2, y + h + 0.2, p_label(p),
        ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=color,
    )


def main():
    # --- Compute stats ---
    model_names = list(TRIAL_FILES.keys())
    all_accs = []  # [model][condition]
    all_cis = []
    all_arrays = []

    for name in model_names:
        arrays, n = load_model(TRIAL_FILES[name])
        all_arrays.append(arrays)
        accs = []
        cis = []
        for c in CONDITIONS:
            k = int(arrays[c].sum())
            accs.append(k / n * 100)
            lo, hi = wilson_ci(k, n)
            cis.append((lo * 100, hi * 100))
        all_accs.append(accs)
        all_cis.append(cis)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 6))

    n_models = len(model_names)
    n_conds = len(CONDITIONS)
    group_width = 0.7
    bar_width = group_width / n_conds
    x_groups = np.arange(n_models)

    for j, (_cond, label, color) in enumerate(zip(CONDITIONS, CONDITION_LABELS, BAR_COLORS)):
        offsets = x_groups + (j - 1) * bar_width
        accs = [all_accs[i][j] for i in range(n_models)]
        err_lo = [accs[i] - all_cis[i][j][0] for i in range(n_models)]
        err_hi = [all_cis[i][j][1] - accs[i] for i in range(n_models)]

        bars = ax.bar(
            offsets,
            accs,
            bar_width * 0.9,
            yerr=[err_lo, err_hi],
            color=color,
            edgecolor="black",
            linewidth=0.8,
            capsize=3,
            error_kw={"linewidth": 1.2},
            label=label,
            zorder=3,
        )

        for i, bar in enumerate(bars):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + err_hi[i] + 0.8,
                f"{accs[i]:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    # --- Significance brackets ---
    bracket_h = 1.2
    for i, _name in enumerate(model_names):
        arrays = all_arrays[i]
        base_x = x_groups[i]

        # Find max bar height in this group for bracket placement
        max_y = max(all_cis[i][j][1] for j in range(n_conds)) + 3.5

        # Bracket 1 (lowest): baseline vs native NL (adjacent bars)
        p1 = mcnemar_p(arrays["x"], arrays["x_nl_native"])
        x1 = base_x + (0 - 1) * bar_width
        x2 = base_x + (1 - 1) * bar_width
        draw_bracket(ax, x1, x2, max_y, bracket_h, p1)

        # Bracket 2 (middle): native vs translated NL (adjacent bars)
        p3 = mcnemar_p(arrays["x_nl_translated"], arrays["x_nl_native"])
        x1c = base_x + (1 - 1) * bar_width
        x2c = base_x + (2 - 1) * bar_width
        draw_bracket(ax, x1c, x2c, max_y + 3.5, bracket_h, p3)

        # Bracket 3 (top, widest): baseline vs translated NL
        p2 = mcnemar_p(arrays["x"], arrays["x_nl_translated"])
        x1b = base_x + (0 - 1) * bar_width
        x2b = base_x + (2 - 1) * bar_width
        draw_bracket(ax, x1b, x2b, max_y + 7.0, bracket_h, p2)

    # --- Axes ---
    ax.set_ylabel("Accuracy (%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Source / Translator / Evaluator Model", fontsize=13, fontweight="bold")
    ax.set_title(
        "Translation Additivity: Are Translated and Native NL Functionally Similar?",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xticks(x_groups)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 78)
    ax.tick_params(axis="y", labelsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # Build flattened legend at bottom with annotation entries
    legend_handles = [
        Patch(facecolor=color, edgecolor="black", linewidth=0.8, label=label.replace("\n", " "))
        for color, label in zip(BAR_COLORS, CONDITION_LABELS)
    ]
    legend_handles.append(Line2D([], [], color="none", label=""))  # spacer
    legend_handles.append(Line2D([], [], color="none", label="McNemar's test (paired)"))
    legend_handles.append(Line2D([], [], color="#2a7f2a", marker="s", linestyle="none", markersize=6, label="p < .05 (sig.)"))
    legend_handles.append(Line2D([], [], color="#cc2222", marker="s", linestyle="none", markersize=6, label="p >= .05 (n.s.)"))

    ax.legend(
        handles=legend_handles, fontsize=9,
        loc="upper center", bbox_to_anchor=(0.5, -0.22),
        ncol=6, framealpha=0.9, columnspacing=1.2, handletextpad=0.5,
    )

    plt.tight_layout()

    # --- Save ---
    out = Path(__file__).parent.parent / "results"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "translation_additivity.png", bbox_inches="tight", dpi=300)
    fig.savefig(out / "translation_additivity.pdf", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved to {out}/translation_additivity.png and .pdf")


if __name__ == "__main__":
    main()
