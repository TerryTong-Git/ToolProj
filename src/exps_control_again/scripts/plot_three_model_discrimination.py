#!/usr/bin/env python3
"""
Bar plot of original distinguishability accuracies for three judge models.

Models: Gemini 2.5 Pro, Claude Opus 4.0, Grok 4.1 Fast
All using GPT-4o as translator, n≈2000 trials each.

Usage:
    uv run python src/exps_control_again/scripts/plot_three_model_discrimination.py
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- Data from judge result files ---
models = ["Claude\nOpus 4.0", "Grok 4.1\nFast", "Gemini\n2.5 Pro"]
accuracies = [49.05, 51.22, 58.45]
ci_low = [46.85, 49.01, 56.26]
ci_high = [51.24, 53.43, 60.61]
n_trials = [1990, 1960, 1964]

# Error bars (as numpy arrays to avoid float issues)
accuracies_arr = np.array(accuracies)
errors_low = np.abs(accuracies_arr - np.array(ci_low))
errors_high = np.abs(np.array(ci_high) - accuracies_arr)

# --- Plot ---
fig, ax = plt.subplots(figsize=(7, 5))

x = np.arange(len(models))
width = 0.55

# Color: green if CI contains 50% (indistinguishable), red otherwise
colors = []
for l, h in zip(ci_low, ci_high):
    if l <= 50.0 <= h:
        colors.append("#7fbf7f")  # green — indistinguishable
    else:
        colors.append("#e07070")  # red — distinguishable

bars = ax.bar(
    x, accuracies, width,
    yerr=[errors_low, errors_high],
    color=colors, edgecolor="black", linewidth=1,
    capsize=5, error_kw={"linewidth": 1.5},
)

# Value labels
for i, (bar, acc, n) in enumerate(zip(bars, accuracies, n_trials)):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + errors_high[i] + 1.0,
        f"{acc:.1f}%\n(n={n})",
        ha="center", va="bottom", fontsize=11, fontweight="bold",
    )

# Reference line at chance
ax.axhline(y=50, color="blue", linestyle="--", linewidth=1.5, label="Chance (50%)")

# Axes
ax.set_ylabel("Judge Accuracy (%)", fontsize=13, fontweight="bold")
ax.set_xlabel("Judge Model", fontsize=13, fontweight="bold")
ax.set_title(
    "Source Discrimination Accuracy\n(GPT-4o Translator, ~2 000 trials each)",
    fontsize=14, fontweight="bold",
)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylim(40, 68)
ax.legend(loc="upper left", fontsize=11)
ax.tick_params(axis="y", labelsize=11)
ax.yaxis.grid(True, linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()

# Save
out = Path("src/exps_control_again/results")
fig.savefig(out / "three_model_discrimination.png", bbox_inches="tight", dpi=300)
fig.savefig(out / "three_model_discrimination.pdf", bbox_inches="tight", dpi=300)
plt.close()
print(f"Saved to {out}/three_model_discrimination.png/.pdf")
