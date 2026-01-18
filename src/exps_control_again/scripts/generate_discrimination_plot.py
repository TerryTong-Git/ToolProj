#!/usr/bin/env python3
"""
Generate source discrimination by model bar plot.

Usage:
    uv run python src/exps_control_again/scripts/generate_discrimination_plot.py
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data from the discrimination analysis (extracted from results)
models = [
    "gemini-\n2.5-flash",
    "ministral-\n14b-2512",
    "qwen-2.5-\ncoder-32b",
    "claude-\nopus-4",
    "gemini-2.0-\nflash-001",
    "codestral-\n2508",
    "claude-\nhaiku-4.5",
    "mixtral-\n8x22b",
    "gpt-4o-\nmini",
]

accuracies = [42.7, 49.2, 54.6, 63.2, 65.4, 74.3, 76.0, 76.8, 86.3]
ci_low = [35.6, 36.4, 44.1, 50.1, 57.2, 67.7, 62.6, 70.5, 74.3]
ci_high = [50.2, 62.2, 64.8, 74.7, 72.8, 80.1, 85.8, 82.2, 93.4]
n_samples = [185, 60, 87, 57, 133, 183, 50, 194, 51]

# Calculate error bars
errors_low = [acc - low for acc, low in zip(accuracies, ci_low)]
errors_high = [high - acc for acc, high in zip(accuracies, ci_high)]

# Create figure
fig, ax = plt.subplots(figsize=(12, 5))

x = np.arange(len(models))
width = 0.7

# Color bars based on whether CI contains 50%
colors = ['#7fbf7f' if low <= 50 <= high else '#e07070'
          for low, high in zip(ci_low, ci_high)]

bars = ax.bar(x, accuracies, width, yerr=[errors_low, errors_high],
              color=colors, edgecolor='black', linewidth=1,
              capsize=4, error_kw={'linewidth': 1.5})

# Add value labels above bars
for i, (bar, acc, n) in enumerate(zip(bars, accuracies, n_samples)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + errors_high[i] + 1.5,
            f'{acc:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add reference lines
ax.axhline(y=50, color='blue', linestyle='--', linewidth=2, label='Chance (50%)')
ax.axhline(y=49.4, color='orange', linestyle='-', linewidth=2, label='Overall (49.4%)')

# Customize plot
ax.set_ylabel('Judge Accuracy on Native NL (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Source Model (generated the native NL)', fontsize=14, fontweight='bold')
ax.set_title('Source Discrimination by Model\n(Lower = Native NL more similar to Translated)',
             fontsize=16, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12, ha='center')

# Set y-axis limits (compressed to save space, but show all data)
ax.set_ylim(30, 100)

ax.legend(loc='upper left', fontsize=12)
ax.tick_params(axis='y', labelsize=12)

# Add grid for readability
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()

# Save
output_dir = Path('src/exps_control_again/results')
fig.savefig(output_dir / 'discrimination_by_model.pdf', bbox_inches='tight', dpi=300)
fig.savefig(output_dir / 'discrimination_by_model.png', bbox_inches='tight', dpi=300)
plt.close()

print(f"Saved to {output_dir}/discrimination_by_model.pdf/png")
